""" This script contains the implementation of the MAML algorithms designed by 
Chelsea Finn et al. (https://arxiv.org/pdf/1703.03400).
Terminology:
------------
Support set : a set of training examples 
    (inputs+labels: iterable of (img, label) pairs)
Query set : a set of test examples 
    (inputs +labels : iterable of (img, label) pairs )
Task/Dataset : Support set + Query set.
Meta-train set: a set of datasets for meta-training
Meta-test set: a set of datasets for meta-evaluation
Meta-batch size: Number of tasks to consider for a meta-iteration
"""
import time 
import copy 
import logging
import datetime
import pickle
import numpy as np 
import os 

import gin
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.python import debug as tf_debug
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense, Conv2D

from helper import conv_net
from metadl.api.api import MetaLearner, Learner, Predictor
from utils import create_grads_shell, reset_grads, app_custom_grads

@gin.configurable
class MyMetaLearner(MetaLearner):
    """
    Replicates the fo-MAML implementation of the Model Agnostic Meta Learner 
    designed bu Chelsea Finn et al. (https://arxiv.org/pdf/1703.03400).
    The meta-learner encapsulates the neural network weights during each 
    meta-iteration. 
    Terminology : a task is defined by the pair (Support set, Query set)
    -----------

        During meta-training :

            The learner is trained on the support set for exactly one epoch.
            The updated learner is then trained again but this time on the 
            query set. The gradients of the associated loss is then computed 
            w.r.t the initial learner's parameters (2nd order opt. original 
            MAML) or w.r.t. to the updated parameters (1st order approx 
            fo-MAML).We perform the previous steps for a number of 
            (learner, tasks) pairs and aggregate the gradients from each pair 
            to perform a meta-update of the initial learner's parameters 
            (that are the same at the beginning of the process).
        
        During meta-testing :

            The pre-trained (during meta-training) learner is fine-tuned with 
            the support set. Then we evaluate the algorithm's performance by 
            predicting labels of query examples, as in standard ML/DL problems.

    """
    def __init__(self,
                meta_iterations,
                meta_batch_size,
                support_batch_size,
                query_batch_size,
                img_size,
                N_ways):
        """
        Args:
            meta_iterations : number of meta-iterations to perform, i.e. the 
            number of times the meta-learner's weights are updated.
            
            meta_batch_size : The number of (learner, task) pairs that are used
            to produce the meta-gradients, used to update the meta-learner's 
            weights after each meta-iteration.

            support_batch_size : The batch size for the support set.
            query_batch_size : The batch size for the query set.
            img_size : Integer, images are considered to be 
                        (img_size, img_size, 3)
        """
        super().__init__()
        self.meta_iterations = meta_iterations
        self.meta_batch_size = meta_batch_size
        self.support_batch_size = support_batch_size
        self.query_batch_size = query_batch_size
        self.img_size = img_size
        self.N_ways = N_ways

        self.meta_learner = conv_net(self.N_ways,img_size)
        self.learner = conv_net(self.N_ways, img_size)

        self.meta_gradients = create_grads_shell(self.meta_learner)

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.learner_optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        self.meta_learner_optimizer = tf.keras.optimizers.SGD(
                                    learning_rate= 0.01)


        # Writer 
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = ('logs/gradient_tape/' + 
                            self.current_time +
                            '/train')
        self.test_log_dir = ('logs/gradient_tape/' +
                            self.current_time +
                            '/test')
        self.train_summary_writer = tf.summary.create_file_writer(
                                    self.train_log_dir)

        self.train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                            name = 'train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name = 'test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                            name = 'test_accuracy')


    def meta_fit(self, meta_dataset_generator):
        """ Encapsulates the meta-learning procedure. In the fo-MAML case, 
        the meta-learner's weights update. 

        Args:
            meta_dataset_generator : a DataGenerator object. We can access 
                the meta-train and meta-validation episodes via its attributes.
                Refer to the metadl/data/dataset.py for more details.
        Returns:
            A Learner object initialized with the meta-learner's weights.
        """

        meta_train_dataset = meta_dataset_generator.meta_train_pipeline
        meta_valid_dataset = meta_dataset_generator.meta_valid_pipeline

        meta_dataset = meta_train_dataset.batch(self.meta_batch_size)
        meta_dataset = meta_dataset.prefetch(32)
        current_meta_iteration = 0
        for tasks_batch in meta_dataset : 
            sup_set = tf.data.Dataset.from_tensor_slices(\
                (tasks_batch[0][1], tasks_batch[0][0]))
            que_set = tf.data.Dataset.from_tensor_slices(\
                (tasks_batch[0][4], tasks_batch[0][3]))
            new_ds = tf.data.Dataset.zip((sup_set, que_set))
            for ((sup_labs, sup_imgs), (que_labs, que_imgs)) in new_ds : 
                self.reset_weights_learner() # Reset learner's weights COPY OP
                support_set = tf.data.Dataset.from_tensor_slices(
                            (sup_imgs, sup_labs))
                query_set = tf.data.Dataset.from_tensor_slices(
                            (que_imgs, que_labs))
                support_set = support_set.batch(self.support_batch_size)
                query_set = query_set.batch(self.query_batch_size)

                self.forward(support_set, query_set)
                

            # Update meta-learner
            #for grad in self.meta_gradients :
            #    grad.assign(tf.divide(grad,
            #                        tf.constant(self.meta_batch_size,
            #                        dtype = tf.float32))
            #                )

            self.meta_learner_optimizer.apply_gradients(zip(
                                        self.meta_gradients,
                                        self.meta_learner.trainable_variables))
            # Reset meta-gradients
            for layer in self.meta_gradients:
                layer.assign(tf.zeros_like(layer))
            
            with self.train_summary_writer.as_default():
                #tf.summary.scalar('Vanilla accuracy', \
                #    self.vanilla_acc.result(), step=current_meta_iteration)
                tf.summary.scalar('Loss: Query sets',
                    self.train_loss.result(), step=current_meta_iteration)
                tf.summary.scalar('Avg_accuracy_query_tasks_batch',
                    self.train_accuracy.result(), step=current_meta_iteration)
                    
            if current_meta_iteration % 20 == 0 :
                logging.info(('Meta iteration : [{}/] Loss : {:.4f}' 
                    + ' Acc : {:.3%}').format(current_meta_iteration,
                    self.train_loss.result(),
                    self.train_accuracy.result()))

            self.train_accuracy.reset_states()
            self.train_loss.reset_states()

            current_meta_iteration += 1
            if current_meta_iteration >= self.meta_iterations:
                break
        return MyLearner(self.meta_learner)

    @tf.function
    def forward(self, support_dataset, query_dataset):
        """Forward pass of the fo-MAML algorithm. Encapsulates the 
        meta-learning procedure. 
        Args:
            support_dataset : a tf.data.Dataset object. Iterates over the 
                            support examples. 
            query_dataset : a tf.data.Dataset object. Iterates over the 
                            labelled query examples.

        """
        for imgs, labels in support_dataset:
            with tf.GradientTape() as tape:
                preds = self.learner(imgs)
                support_loss = self.loss(labels, preds)
            self.learner_optimizer.apply_gradients(zip(
                tape.gradient(support_loss, self.learner.trainable_variables),
                self.learner.trainable_variables))

        for imgs, labels in query_dataset:
            with tf.GradientTape() as tape:
                que_preds = self.learner(imgs)
                query_loss = self.loss(labels, que_preds)

            self.train_loss.update_state(query_loss)
            self.train_accuracy.update_state(labels, que_preds)
            # Add task gradient to meta-gradient shell
            for meta_grad, learner_grad in zip(self.meta_gradients, 
                    tape.gradient(query_loss, self.learner.trainable_variables)):
                meta_grad.assign_add(learner_grad)

    def reset_weights_learner(self):  
        """Reset the learners' weights. The goal is to reuse the same models
        that were already defined and don't create new ones.
        """
        for meta_layer, layer in \
                zip(self.meta_learner.trainable_variables,
                    self.learner.trainable_variables):
            layer.assign(copy.copy(meta_layer))


@gin.configurable
class MyLearner(Learner):
    """ In the case of fo-MAML, encapsulates a neural network and its training 
    methods.
    """
    def __init__(self, 
                neural_net = None,
                num_epochs=3,
                lr=0.1,
                img_size=32):
        """
        Args:
            neural_net : a keras.Sequential object. A neural network model to 
                        copy as Learner.
            num_epochs : Integer, the number of epochs to consider for the 
                        training on support examples.
            lr : Float, the learning rate associated to the learning procedure
                (Adaptation).
            img_size : Integer, images are considered to be 
                        (img_size,img_size,3)
        """
        super().__init__()
        if neural_net == None :
            self.learner = conv_net(5, img_size=img_size)
        else : 
            self.learner = neural_net

        # Learning procedure parameters
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

        # Gradient placeholder: 
        self.task_gradient = create_grads_shell(self.learner)

    def __call__(self, imgs):
        return self.learner(imgs)

    @tf.function
    def train(self, dataset_train):
        """ Fit beta-level algo from support set of a task. It is essentially
        the adaptation part of the MAML Learner.
        
        Args:
            dataset_train : a tf.data.Dataset object. Iterates over the 
                training examples (support set).
        """  

        for images, labels in dataset_train.repeat(3) : 
            with tf.GradientTape() as tape :
                preds = self.learner(images)
                support_loss = self.loss(labels, preds)
            gradients = tape.gradient(support_loss,
                                    self.learner.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients,self.learner.trainable_variables))
        
    def fit(self, dataset_train):
        """ The learner's fit function over the train set of a task.

        Args:
            dataset_train : a tf.data.Dataset object. Iterates over the training 
                            examples (support set).
        Returns:
            predictor : An instance of MyPredictor that is initilialized with 
                the fine-tuned learner's weights in this case.
        """
        self.train(dataset_train)
        predictor = MyPredictor(self.learner)
        return predictor

    def load(self, model_dir):
        """Loads the learner model from a pickle file.

        Args:
            model_dir: the directory name in which the participant's code and 
                their saved/serialized learner are stored.
        """

        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))

        ckpt_path = os.path.join(model_dir, 'learner.ckpt')
        self.learner.load_weights(ckpt_path)
        
    def save(self, model_dir):
        """Saves the learner model into a pickle file.

        Args:
            model_dir: the directory name from which the participant's code and 
                their saved/serialized learner are loaded.
        """

        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))
        
        ckpt_file = os.path.join(model_dir, 'learner.ckpt')
        self.learner.save_weights(ckpt_file)
        
####### Predictor ########
@gin.configurable
class MyPredictor(Predictor):
    """ The predictor is meant to predict labels of the query examples at 
    meta-test time.
    """
    def __init__(self,
                 learner):
        """
        Args:
            learner : a MyLearner object that encapsulates the fine-tuned 
                neural network.
        """
        super().__init__()
        self.learner = learner
    
    def predict(self, dataset_test):
        """ Predicts labels of the query set examples associated to a task.
        Note that the query set is a tf.data.Dataset containing 50 examples for
        the Omniglot dataset.

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
        for images_test in dataset_test:
            preds = self.learner(images_test)
        return preds

