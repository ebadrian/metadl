""" This is a dummy baseline. It is just supposed to check if ingestion and 
scoring are called properly.
"""
import os
import logging
import csv 

import tensorflow as tf

from metadl.api.api import MetaLearner, Learner, Predictor


class MyMetaLearner(MetaLearner):

    def __init__(self):
        super().__init__()

    def meta_fit(self, meta_dataset_generator) -> Learner:
        """
        Args:
            meta_dataset_generator : a DataGenerator object. We can access 
                the meta-train and meta-validation episodes via its attributes.
                Refer to the metadl/data/dataset.py for more details.
        
        Returns:
            MyLearner object : a Learner that stores the meta-learner's 
                learning object. (e.g. a neural network trained on meta-train
                episodes)
        """
        return MyLearner()


class MyLearner(Learner):

    def __init__(self):
        super().__init__()

    def fit(self, dataset_train) -> Predictor:
        """
        Args: 
            dataset_train : a tf.data.Dataset object. It is an iterator over 
                the support examples.
        Returns:
            ModelPredictor : a Predictor.
        """
        return MyPredictor()

    def save(self, model_dir):
        """ Saves the learning object associated to the Learner. It could be 
        a neural network for example. 

        Note : It is mandatory to write a file in model_dir. Otherwise, your 
        code won't be available in the scoring process (and thus it won't be 
        a valid submission).
        """
        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))
        
        # Save a file for the code submission to work correctly.
        with open(os.path.join(model_dir,'dummy_sample.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Dummy example'])
            
    def load(self, model_dir):
        """ Loads the learning object associated to the Learner. It should 
        match the way you saved this object in save().
        """
        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))
        
    
class MyPredictor(Predictor):

    def __init__(self):
        super().__init__()

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
        dummy_pred = tf.constant([[1.0, 0, 0, 0 ,0]], dtype=tf.float32)
        dummy_pred = tf.broadcast_to(dummy_pred, (95, 5))
        return dummy_pred

