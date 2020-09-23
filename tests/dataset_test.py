"""
Test script for the DataGenerator object. Focus on the episodes and batch
shapes, and the unicity of classes in episodes.

Usage example : 
    python -m metadl.data.dataset_test --path_to_dataset=<path/Omniglot/>
 
"""
import time
import os 

import gin
import numpy as np
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from metadl.data.dataset import DataGenerator

FLAGS = flags.FLAGS

flags.DEFINE_string('path_to_dataset', './', 
    ('Path to the dataset folder. It should include 2 folders : '
        + 'meta_train and meta_test'))

def check_shape_episode(episode, episode_config):
    """ Checking the shape of every element of an episode, according to the 
    episode_config.
    """
    assert len(episode) == 2
    assert episode[1].numpy() == 0
    # Support set 
    assert (episode[0][0].shape == (episode_config[1]*episode_config[2],
                                episode_config[0],
                                episode_config[0],
                                3))
    assert episode[0][1].shape == (episode_config[1]*episode_config[2],)
    assert episode[0][2].shape == (episode_config[1]*episode_config[2],)
    # Query set
    assert (episode[0][3].shape == (episode_config[3]*episode_config[1],
                                episode_config[0],
                                episode_config[0],
                                3))
    assert episode[0][4].shape == (episode_config[3]*episode_config[1],)
    assert episode[0][5].shape == (episode_config[3]*episode_config[1],)

    # Checking unique classes in support and query set
    assert len(np.unique(episode[0][1])) == episode_config[1]
    assert len(np.unique(episode[0][4])) == episode_config[1]

def check_shape_batch(batch, batch_config):
    """ Checking the shape of every element of a batch, according to the 
    batch_config.
    """
    assert len(batch) == 2
    assert batch[1].numpy() == 0
    assert batch[0][0].shape == (batch_config[1], 
                                batch_config[0],
                                batch_config[0],
                                3)
    assert batch[0][1].shape == (batch_config[1],)
    
def main(argv):
    """ Checks the shape of episodes and batches generated via the DataGenerator
    API. 
    We test the 3 different type of generator settings : 
        - Meta-train phase : meta-train : Episode mode, meta-valid : Episode mode 
        - Meta-test phase : meta-test : Episode mode
        - Meta-train phase : meta-train : Batch mode, meta-valid : Episode mode
    """
    del argv
    path_to_dataset = FLAGS.path_to_dataset
    path_to_meta_train_records = os.path.join(path_to_dataset, 'meta_train')
    path_to_meta_test_records = os.path.join(path_to_dataset, 'meta_test')
    batch_config = [50, 100]
    episode_config = [28, 20, 1, 19]
    
    # Meta-train phase / Episode mode
    generator = DataGenerator(path_to_records=path_to_meta_train_records,
                            batch_config=batch_config,
                            episode_config=episode_config,
                            pool='train',
                            mode='episode')

    meta_train_generator = generator.meta_train_pipeline
    meta_valid_generator = generator.meta_valid_pipeline

    meta_train_iterator = meta_train_generator.__iter__()
    meta_valid_iterator = meta_valid_generator.__iter__()
    meta_train_data = next(meta_train_iterator)
    meta_valid_data = next(meta_valid_iterator)

    logging.info('Verifying episode shapes for the meta-train phase ...\n')
    check_shape_episode(meta_train_data, episode_config)
    check_shape_episode(meta_valid_data, episode_config)
    logging.info('[1/3] Correct episode shape !\n')

    # Meta-test phase
    generator_test = DataGenerator(path_to_records=path_to_meta_test_records,
                                batch_config=batch_config,
                                episode_config=episode_config,
                                pool='test',
                                mode='episode')
    meta_test_generator = generator_test.meta_test_pipeline
    meta_test_iterator = meta_test_generator.__iter__()
    meta_test_data = next(meta_test_iterator)
    logging.info('Verifying episode shapes for the meta-test phase ...\n')
    check_shape_episode(meta_test_data, episode_config)
    logging.info('[2/3] Correct episode shape! \n')

    # Meta-train phase / Batch mode
    generator = DataGenerator(path_to_records=path_to_meta_train_records,
                            batch_config=batch_config,
                            episode_config=episode_config,
                            pool='train',
                            mode='batch')

    meta_train_generator = generator.meta_train_pipeline
    meta_valid_generator = generator.meta_valid_pipeline

    meta_train_iterator = meta_train_generator.__iter__()
    meta_valid_iterator = meta_valid_generator.__iter__()
    meta_train_data = next(meta_train_iterator)
    meta_valid_data = next(meta_valid_iterator)

    logging.info(('Verifying batch/episode shapes for meta-train and meta-valid'
        + 'generator in meta-train phase ...\n'))
    check_shape_batch(meta_train_data, batch_config)
    check_shape_episode(meta_valid_data, episode_config)
    logging.info('[3/3] Correct batch/episode shape!')


if __name__ == '__main__':
    app.run(main)
