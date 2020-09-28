"""Transfer baseline.
Here, we consider the transfer learning approach. We first load a model 
pre-trained on ImageNet. We freeze the layers associated to the projected images
and we fine-tune a classifer on top of this embedding function. 
"""
import os
import logging
import csv 
import datetime

import tensorflow as tf
from tensorflow import keras
import gin

from metadl.api.api import MetaLearner, Learner, Predictor

@gin.configurable
class MyMetaLearner(MetaLearner):
    """ Loads and fine-tune a model pre-trained on ImageNet. """
    def __init__(self,
                iterations=10,
                freeze_base=True,
                total_meta_train_class=883):
        super().__init__()
        self.iterations = iterations
        self.freeze_base = freeze_base
        self.total_meta_train_class = total_meta_train_class

        self.base_model = keras.applications.Xception(
            weights='imagenet',
            input_shape=(71,71,3),
            include_top=False
        )
        self.base_model.trainable = (not self.freeze_base)
        inputs = keras.Input(shape=(71,71,3))
        x = self.base_model(inputs, training=True)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(self.total_meta_train_class)(x)
        self.model = keras.Model(inputs, outputs)

        self.loss = keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = keras.optimizers.Adam()
        self.acc = keras.metrics.SparseCategoricalAccuracy()

        # Summary Writers
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = ('logs/transfer/gradient_tape/' + self.current_time 
            + '/meta-train')
        self.valid_log_dir = ('logs/transfer/gradient_tape/' + self.current_time 
            + '/meta-valid')
        self.train_summary_writer = tf.summary.create_file_writer(
            self.train_log_dir)
        self.valid_summary_writer = tf.summary.create_file_writer(
            self.valid_log_dir)

        # Statstics tracker
        self.train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name = 'train_accuracy')
        self.valid_loss = tf.keras.metrics.Mean(name = 'valid_loss')
        self.valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name = 'valid_accuracy')

    def meta_fit(self, meta_dataset_generator) -> Learner:
        """ We train the classfier created on top of the pre-trained embedding
        layers.

        Args:
            meta_dataset_generator : a DataGenerator object. We can access 
                the meta-train and meta-validation episodes via its attributes.
                Refer to the metadl/data/dataset.py for more details.
        
        Returns:
            MyLearner object : a Learner that stores the current embedding 
                function (Neural Network) of this MetaLearner.
        """
        meta_train_dataset = meta_dataset_generator.meta_train_pipeline
        meta_valid_dataset = meta_dataset_generator.meta_valid_pipeline
        meta_valid_dataset = meta_valid_dataset.batch(2)

        count = 0
        logging.info('Starting meta-fit for the transfer baseline ...')
        meta_iterator = meta_train_dataset.__iter__()
        sample_data = next(meta_iterator)
        logging.info('Images shape : {}'.format(sample_data[0][0].shape))
        logging.info('Labels shape : {}'.format(sample_data[0][1].shape))
        for (images, labels), _ in meta_train_dataset :
            with tf.GradientTape() as tape :
                preds = self.model(images)
                loss = self.loss(labels, preds)
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_weights))
            logging.info('Iteration #{} - Loss : {}'.format(count, loss.numpy()))
            self.train_accuracy.update_state(labels, preds)
            self.train_loss.update_state(loss)
            
            if count % 50 == 0 :
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('Train loss', self.train_loss.result(),
                        step=count, 
                        description='Avg train loss over 50 batches',)
                    tf.summary.scalar('Train acc', self.train_accuracy.result(),
                        step=count, 
                        description='Avg train accuracy over 50 batches')
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                self.evaluate(MyLearner(self.model), meta_valid_dataset)

                self.valid_accuracy.reset_states()
                self.valid_loss.reset_states()

            count += 1
            if count >= self.iterations :
                break

        return MyLearner(self.model)

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
            if count_val >= 20 :
                break
        logging.info('Meta-Valid accuracy : {:.3%}'.format(
            self.valid_accuracy.result()))

@gin.configurable
class MyLearner(Learner):
    def __init__(self, 
                model=None,
                N_ways = 5):
        """
        Args:
            model : A keras.Model object describing the Meta-Learner's neural
                network.
            N_ways : Integer, the number of classes to consider at meta-test
                time.
        """
        super().__init__()
        self.N_ways = N_ways
        if model == None:
            self.base_model = keras.applications.Xception(
                weights='imagenet',
                input_shape=(71,71,3),
                include_top=False
            )
            self.base_model.trainable = False
            inputs = keras.Input(shape=(71,71,3))
            x = self.base_model(inputs, training=False)
            x = keras.layers.GlobalAveragePooling2D()(x)
            outputs = keras.layers.Dense(self.N_ways, activation='softmax')(x)
            self.model = keras.Model(inputs, outputs)
        else : 
            new_model = keras.models.clone_model(model)
            new_model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)
            x = new_model.output
            outputs = keras.layers.Dense(self.N_ways, activation='softmax')(x)
            new_model = keras.Model(inputs=new_model.input, outputs = outputs)
            self.model = new_model

        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.SparseCategoricalCrossentropy()

    def fit(self, dataset_train) -> Predictor:
        """Fine-tunes the current model with the support examples of a new 
        unseen task. 

        Args:
            dataset_train : a tf.data.Dataset object. Iterates over the support
                examples. 
        Returns:
            a Predictor object that is initialized with the fine-tuned 
                Learner's neural network weights.
        """
        logging.debug('Fitting a task ...')
        for images, labels in dataset_train.repeat(5): 
            logging.debug('Image shape : {}'.format(images.shape))
            logging.debug('Labels shape : {}'.format(labels.shape))
            with tf.GradientTape() as tape :
                preds = self.model(images)
                loss = self.loss(labels, preds)
            logging.debug('[FIT] Loss on support set : {}'.format(loss))
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_weights))

        return MyPredictor(self.model)

    def save(self, model_dir):
        """
        Saves the embedding function, i.e. the prototypical network as a 
        tensorflow checkpoint.
        """
        if(os.path.isdir(model_dir) != True):
            raise ValueError('The model directory provided is invalid. Please\
                 check that its path is valid.')
        
        ckpt_file = os.path.join(model_dir, 'learner.ckpt')
        self.model.save_weights(ckpt_file) 

    def load(self, model_dir):
        """
        Loads the embedding function, i.e. the prototypical network from a 
        tensorflow checkpoint.
        """
        if(os.path.isdir(model_dir) != True):
            raise ValueError('The model directory provided is invalid. Please\
                    check that its path is valid.')

        ckpt_path = os.path.join(model_dir, 'learner.ckpt')
        self.model.load_weights(ckpt_path)

    
class MyPredictor(Predictor):
    def __init__(self, model):
        """
        Args: 
            model : a keras.Model object. The fine-tuned neural network
        """
        super().__init__()
        self.model = model
    def predict(self, dataset_test):
        """ Predicts the logits or probabilities over the different classes
        of the query examples.

        Args:
            dataset_test : a tf.data.Dataset object. Iterates over the 
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
        for images in dataset_test :
            preds = self.model(images)

        return preds

