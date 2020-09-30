""" Prototypical networks eager implementation. This implementation is based on 
the original Prototypical networks paper J. Snell et al. 2017 
(https://arxiv.org/pdf/1703.05175).
"""
import time 
import copy 
import logging
import datetime
import pickle
import os 

import gin
import tensorflow as tf
import tensorflow_addons as tfa

from metadl.api.api import MetaLearner, Learner, Predictor
from utils import create_proto_shells, reset_proto_shell
from helper import conv_net

tf.random.set_seed(1234)
@gin.configurable
class MyMetaLearner(MetaLearner):

    def __init__(self, 
                img_size,
                N_ways,
                K_shots,
                embedding_dim,
                meta_iterations,
                distance_fn=tf.norm):
        """
        Args: 
            img_size : Integer, images are considered to be 
                        (img_size, img_size, 3)
            N_ways : Number of ways, i.e. classes in a task
            K_shots : Number of examples per class in the support set
            embedding_dim : Integer, embedding dimension
            meta_iterations : Integer, number of episodes to consider at 
                meta-train time
            distance_fn : Distance function to consider for the proto-networks
            
        """
        super().__init__()
        self.img_size = img_size
        self.N_ways = N_ways
        self.K_shots = K_shots
        self.embedding_fn = conv_net(self.N_ways, self.img_size)
        self.embedding_dim = embedding_dim
        self.meta_iterations = meta_iterations

        self.prototypes = create_proto_shells(self.N_ways, self.embedding_dim)
        self.distance_fn = distance_fn
        self.learning_rate = 0.005
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss = 0

        # Summary Writers
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = ('logs/proto/gradient_tape/' + self.current_time 
            + '/meta-train')
        self.valid_log_dir = ('logs/proto/gradient_tape/' + self.current_time 
            + '/meta-valid')
        self.train_summary_writer = tf.summary.create_file_writer(
            self.train_log_dir)
        self.valid_summary_writer = tf.summary.create_file_writer(
            self.valid_log_dir)

        # Statstics tracker
        self.train_loss = tf.keras.metrics.Mean(name = 'query_loss')
        self.query_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name = 'query_accuracy')
        self.valid_loss = tf.keras.metrics.Mean(name = 'valid_loss')
        self.valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name = 'valid_accuracy')

        self.curr_metab = tf.Variable(0, dtype = tf.int32)
        
    #@tf.function
    def compute_prototypes(self, support_dataset):
        """ 
        Computes the prototypes of the support set examples. They are computed
        as the average of the embedding projections of the examples within each
        class.
        """
        logging.debug('Computing prototypes ...')
        logging.debug('A prototype shape : {}'.format(self.prototypes[0].shape))
        for image, label in support_dataset : 
            logging.debug('Image shape : {}'.format(image.shape))
            logging.debug('Label shape : {}'.format(label.shape))
            logging.debug('Label : {}'.format(label))

            self.prototypes[tf.cast(label, dtype=tf.int32).numpy()[0]] += \
                self.embedding_fn(image, training=True)

        for i in range(self.N_ways):
            self.prototypes[i] = self.prototypes[i] / self.K_shots
        
        logging.debug('Prototypes after computing them : {}'.format(
            self.prototypes))

    def evaluate(self, learner, meta_valid_generator):
        """Evaluates the current meta-learner with episodes generated from the
        meta-validation split. The number of episodes used to compute the 
        an average accuracy is set to 20.
        Args:
            learner : MyLearner object. The current state of the meta-learner 
                    is embedded in the object via its neural network.
            meta_valid_generator : a tf.data.Dataset object that generates
                                    episodes from the meta-validation split.
        """
        count_val = 0
        for tasks_batch in meta_valid_generator : 
            sup_set = tf.data.Dataset.from_tensor_slices(
                (tasks_batch[0][1], tasks_batch[0][0]))
            que_set = tf.data.Dataset.from_tensor_slices(
                (tasks_batch[0][4],tasks_batch[0][3]))
            new_ds = tf.data.Dataset.zip((sup_set, que_set))
            for ((supp_labs, supp_img), (que_labs, que_img)) in new_ds:
                support_set = tf.data.Dataset.from_tensor_slices(
                    (supp_img, supp_labs))
                query_set = tf.data.Dataset.from_tensor_slices(que_img)
                support_set = support_set.batch(5)
                query_set = query_set.batch(95)
                predictor = learner.fit(support_set)
                preds = predictor.predict(query_set)
                self.valid_accuracy.update_state(que_labs, preds)
            
            count_val += 1 
            if count_val >= 50 :
                break
        logging.info('Meta-Valid accuracy : {:.3%}'.format(
            self.valid_accuracy.result()))

    def meta_train(self, query_set):
        """
        Computes the distance between prototypes and query examples and update 
        the loss according to each of these values. The loss we used is the one 
        derived in the original paper https://arxiv.org/pdf/1703.05175. 
        (Page 2, equation #2)

        Args: 
            query_set : a tf.data.Dataset object. The query dataset.
        """
        N_Q = 5
        cste = 1/(N_Q * self.N_ways)
        with tf.GradientTape() as tape :
            for image, label in query_set: # Potentially batch of image/labels
                proj_image = self.embedding_fn(image, training = True)
                logging.debug('Image shape : {}'.format(image.shape))
                logging.debug('Label shape : {}'.format(label.shape))
                logging.debug('Projected image shape : {}'.format(
                            proj_image.shape))

                # Distances of proj image to prototypes 
                tmp1 = tf.math.square(self.distance_fn(
                    proj_image - self.prototypes[label[0].numpy()]))
                # Log sum exp of difference between projections and prototypes
                tmp2 = tf.math.reduce_logsumexp(-tf.math.square(
                    self.distance_fn(tf.broadcast_to(tf.expand_dims(proj_image, axis=0), (self.N_ways,1,self.embedding_dim))
                    - self.prototypes, axis=2)))
                # tf.exp : [5, 1, self.embedding_dim]
                self.loss += cste * (tmp1 + tmp2)
            
            logging.info('Loss on a task : {}'.format(self.loss))
            self.train_loss.update_state(self.loss)
            
        grads = tape.gradient(self.loss, self.embedding_fn.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.embedding_fn.trainable_weights))
        
        self.loss = 0 # Reset loss after a task
    
    def meta_fit(self, meta_dataset_generator):
        """ Encapsulates the meta-learning algorithm. It generates epiosdes 
        from the meta-train split and updates the embedding function 
        (Neural network) according to the learning algorithm described in 
        the original paper. Every 50 tasks, we evaluate the current 
        meta-learner with episodes generated from the meta-validation split.
        
        Args:
            meta_dataset_generator : a DataGenerator object. We can access 
                the meta-train and meta-validation episodes via its attributes.
                Refer to the metadl/data/dataset.py for more details.
        
        Returns:
            MyLearner object : a Learner that stores the current embedding 
                function (Neural Network) of this MetaLearner.
        """
        count = 0
        meta_train_dataset = meta_dataset_generator.meta_train_pipeline
        meta_valid_dataset = meta_dataset_generator.meta_valid_pipeline

        meta_train_dataset = meta_train_dataset.batch(1)
        meta_valid_dataset = meta_valid_dataset.batch(2)
        logging.info('Starting meta-fit for the proto-net ...')
        for tasks_batch in meta_train_dataset :
            sup_set = tf.data.Dataset.from_tensor_slices(\
                (tasks_batch[0][1], tasks_batch[0][0]))
            que_set = tf.data.Dataset.from_tensor_slices(\
                (tasks_batch[0][4], tasks_batch[0][3]))

            new_ds = tf.data.Dataset.zip((sup_set, que_set))
            for ((supp_labs, supp_img), (que_labs, que_img)) in new_ds:
                supp_img, que_img = self.aug_rotation(supp_img, que_img)
                support_set = tf.data.Dataset.from_tensor_slices(\
                    (supp_img, supp_labs))
                query_set = tf.data.Dataset.from_tensor_slices(\
                    (que_img, que_labs))
                support_set = support_set.batch(1)
                query_set = query_set.batch(1)

                self.compute_prototypes(support_set)
                logging.debug('Prototypes computed : {}'.format(self.prototypes))
                self.meta_train(query_set)
                reset_proto_shell(self.prototypes) # Reset prototypes

                if count % 50 == 0:
                    self.evaluate(MyLearner(N_ways= 5,
                                            K_shots=1,
                                            img_size=28,
                                            embedding_dimension=self.embedding_dim,
                                            embedding_fn =self.embedding_fn),
                                meta_valid_dataset)
                    with self.valid_summary_writer.as_default():
                        tf.summary.scalar('Query acc',
                                        self.valid_accuracy.result(),
                                        step = count)
                    self.valid_accuracy.reset_states()
            
            with self.train_summary_writer.as_default():
                tf.summary.scalar('Query Loss',
                                    self.train_loss.result(),
                                    step = count)
            self.train_loss.reset_states()

            count += 1
            if count % 2500 == 0:
                self.learning_rate = self.learning_rate / 2
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
                logging.info('New learning rate : {}'.format(self.learning_rate))
            if(count > self.meta_iterations):
                break
        
        return MyLearner(embedding_fn=self.embedding_fn)

    def aug_rotation(self, supp_img, que_img):
        """ Rotate images from support and query set by the same angle.
        The angle is randomly generated from [0, 90, 180, 270] to reproduce
        the data augmentation performed in the original Prototypical Networks.
        Args:
            supp_img : tuple, shape (batch_size_support, img_size, img_size, 3)
            que_img : tuple, shape (batch_size_query, img_size, img_size, 3)
        
        Returns:
            supp_img :tuple, same shape as above. Augmented supp_img
            que_img :tuple, same shape as above. Augmented que_img
        """
        random_int_rotation = tf.random.uniform((), 
                                                minval=0,
                                                maxval=3,
                                                dtype=tf.int32,
                                                seed=1234)
        angle = tf.cast(random_int_rotation / 4, dtype=tf.float32)
        supp_img = tfa.image.rotate(supp_img, angle)
        que_img = tfa.image.rotate(que_img, angle)
        return supp_img, que_img

@gin.configurable
class MyLearner(Learner):

    def __init__(self,
                N_ways,
                K_shots,
                img_size,
                embedding_dimension = 64,
                embedding_fn=None):
        """ If no embedding function is provided, we create a neural network
        with randomly initialized weights.
        Args:
            N_ways : Number of classes in episodes at meta-test time.
            N_shots : Number of images per class in the support set.
            img_size : Integer, images are considered to be 
                (img_size, img_size, 3).
            embedding_dimension : Embedding space dimension
            embedding_fn : Distance funtion to consider at meta-test time.
        """
        super().__init__()
        self.N_ways = N_ways
        self.K_shots = K_shots
        self.img_size = img_size
        self.embedding_dimension = embedding_dimension
        if embedding_fn == None:
            self.embedding_fn = conv_net(self.N_ways, self.img_size)
        else:
            self.embedding_fn = embedding_fn
        
        self.prototypes = create_proto_shells(N_ways, self.embedding_dimension)
        logging.debug('[LEARNER INIT] Prototypes length : {}'.format(
                    len(self.prototypes)))
        logging.debug('[LEARNER INIT] Prototype shape : {}'.format(
                    self.prototypes[0].shape))

    def fit(self, dataset_train):
        """
        Compute the prototypes of the corresponding support set which is 
        dataset_train (support set) in this case. We need to know which 
        distance is used, as well as the number of classes (N_ways) and the 
        number of shots per class (K_shots) to compute each one of them.

        Args: 
            dataset_train : a tf.data.Dataset object. It is an iterator over 
                the support examples.
        Returns:
            ModelPredictor : a Predictor that has computed prototypes.
        """
        reset_proto_shell(self.prototypes)
        for image, label in dataset_train:
            projected_imgs = self.embedding_fn(image)
            logging.debug('Embedding space dimension : {}'.format(
                        projected_imgs.shape))
            logging.debug(('Are images projections equal in embedding' 
            + ' space ? : {}').format(tf.reduce_all(tf.math.equal(
                                                    projected_imgs[0], 
                                                    projected_imgs[1]))))
            
            for i in range(self.N_ways):
                self.prototypes[label[i].numpy()] += projected_imgs[i] 

        for i in range(self.N_ways):
            self.prototypes[i] /= self.K_shots

        return MyPredictor(self.embedding_fn, self.prototypes)

    def save(self, model_dir):
        """Saves the embedding function, i.e. the prototypical network as a 
        tensorflow checkpoint.
        """
        if(os.path.isdir(model_dir) != True):
            raise ValueError('The model directory provided is invalid. Please\
                 check that its path is valid.')
        
        ckpt_file = os.path.join(model_dir, 'learner.ckpt')
        self.embedding_fn.save_weights(ckpt_file) 

    def load(self, model_dir):
        """
        Loads the embedding function, i.e. the prototypical network from a 
        tensorflow checkpoint.
        """
        if(os.path.isdir(model_dir) != True):
            raise ValueError('The model directory provided is invalid. Please\
                    check that its path is valid.')

        ckpt_path = os.path.join(model_dir, 'learner.ckpt')
        self.embedding_fn.load_weights(ckpt_path)
    
@gin.configurable
class MyPredictor(Predictor):

    def __init__(self,
                embedding_fn,
                prototypes,
                distance_fn = tf.norm):
        """
        Args:
            embedding_fn : Distance funtion to consider at meta-test time.
            prototypes : Prototypes computed using the support set
            distance_fn : Distance function to consider for the proto-networks
        """
        super().__init__()
        self.embedding_fn = embedding_fn
        self.prototypes = prototypes
        self.distance = distance_fn

    def compute_probs(self, images):
        """ Computes probabilities of each query set examples to belong to each
        class.

        Args:
            images : tuple of length 1, containing batch_size number of images
                     ( (img1, img2, ... ) ) 
        Returns:
            probs: Probability distribution over N_ways classes for each
                image in images.
        """ 
        batch_size = 95
        projected_images = self.embedding_fn(images)
        embedding_dim = projected_images.shape[1]
        broadcast_projections = tf.broadcast_to(tf.expand_dims(
                                    projected_images, axis = 1),
                                    [batch_size,5,embedding_dim])

        logging.debug('Prototypes length : {}'.format(len(self.prototypes)))
        logging.debug('Prototype shape : {}'.format(self.prototypes[0].shape))
        logging.debug('Batch size : {}'.format(batch_size))
        logging.debug('Embedding dimension : {}'.format(embedding_dim))
        logging.debug('Broadcast embeddings shape: {}'.format(
            broadcast_projections.shape))
        
        broadcast_proto = tf.broadcast_to(
                        tf.expand_dims(tf.squeeze(self.prototypes),axis =0),
                        [batch_size,5,embedding_dim])
        dists = -tf.math.square(
            self.distance(broadcast_projections - broadcast_proto, axis =2))
        max_dists = tf.reduce_max(dists, axis =1)
        max_dists = tf.expand_dims(max_dists, axis =1)
        dists = dists - max_dists
        exps = tf.exp(dists)
        sum_exps = tf.broadcast_to(tf.expand_dims(
                    tf.reduce_sum(exps, axis = 1), axis = 1), [batch_size,5])
        
        probs = exps / sum_exps # [batch_size, 5] / [batch_size, 5]
        return probs

    def predict(self, dataset_test):
        """ Predicts the label of the examples in the query set which is the 
        dataset_test in this case. The prototypes are already computed by
        the Learner.

        Args:
            dataset_test : a tf.data.Dataset object. An iterator over the 
                unlabelled query examples.
        Returns: 
            preds : tensors, shape (num_examples, N_ways). We are using the 
            Sparse Categorical Accuracy to evaluate the predictions. Valid 
            tensors can take 2 different forms described below.

        Case 1 : The i-th prediction row contains the i-th example logits.
        Case 2 : The i-th prediction row contains the i-th example 
                probabilities.

        Since in both cases the SparseCategoricalAccuracy behaves the same way,
        i.e. taking the argmax of the row inputs, both forms are valid.

        Note : In the challenge N_ways = 5 at meta-test time.
        """

        for image in dataset_test:
            probs = self.compute_probs(image)
            logging.debug('[PREDICT] Probs for a batch : {}'.format(probs))

        return probs