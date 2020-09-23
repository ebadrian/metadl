"""
MAML Test.
In this script, we test the following actions/functions : 

    - We ensure that we correctly define a model
    - test_meta_iteration : we ensure the meta-iteration is changing weights
        value.
    - test_meta_copy_learners : we ensure that the learner's copy is well 
        executed

##############################################################################

Important information about the implementation :
-----------------------------------------------
It is worth to mention that we made the choice of representing the whole MAML
learning procedure as follows : 
    - We define a meta-learner with the base architecture of choice. 
    - We define a list of learners, essentially a list of models created via 
        the Keras API here. The number of learners we consider is the 
        meta-batch size. 
    
    - The meta-learner's weights are updated using an aggregation of the
        gradients computed for each learner. 
    - After each meta-iteration, the learners' weights are re-initialized with
        the new meta-learner's weights. 

"""

import gin
from absl.testing import parameterized
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from metadl.data.dataset import EpisodeGenerator
from metadl.api.api import MetaLearner, Learner, Predictor
from utils import create_grads_shell, reset_grads, app_custom_grads
from helper import conv_net
from model import Model, ModelLearner, ModelPredictor



FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_name', 'omniglot',
                       'Dataset to perform the tests on.')

flags.DEFINE_integer('meta_batch_size', 10, 'Number of tasks to consider to \
     to perform a meta-iteration.')


def test_meta_iteration():
    logging.info('\n Executing test_meta_iteration ... \n')
    test = tf.test.TestCase()
    
    meta_learner = Model()
    logging.info('\n MAML meta-learner is created, starting the learning \
         procedure ...')
    
    # Here we create a batch from the episode generator so that we can 
    learner = meta_learner.meta_fit(dataset_fixed)
    logging.info('Done!')

def test_meta_copy_learners(): 
    """
    This test verifies if after a meta-iteration, the learner's weights are
    well re-initialized, i.e. with the meta-learner's weights. 
    For more information about the MAML algorithm and why we even have a list
    of learners, please refer to the original paper.
    """
    logging.info('\n Executing test_meta_copy_learners ... \n')
    test = tf.test.TestCase()

    meta_learner = Model()
    logging.info('\n MAML meta-learner is created, starting the learning \
        procedure ...')
    
    meta_dataset = meta_dataset.take(FLAGS.meta_batch_size)
    learner = meta_learner.meta_fit(meta_dataset)
    

def main(argv):
    del argv
    logging.info('\n Testing suites - MAML learning procedure\n \
         you can find the implementation details in the script\'s header \n')
    logging.info('Starting the meta-iteration test ... ')
    test_meta_iteration()   
    logging.info('=========> Meta-iteration test PASSED!')



    
if __name__ == '__main__':
    app.run(main)
