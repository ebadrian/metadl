"""
Creates a data generation API.
This script is using the Meta-dataset pipeline than can be found :
https://github.com/google-research/meta-dataset

"""
import os 
import time 
import sys 

import gin
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline

## Flags 
FLAGS = flags.FLAGS

@gin.configurable
class DataGenerator():
    """ This class is meant to define data generators. 

    Usage example : 
        At meta-train time : 
            generator = DataGenerator(path_to_records=path)
            meta_train_generator = generator.meta_train_pipeline()
            meta_valid_generator = generator.meta_valid_pipeline()

        At meta-test time :
            generator = DataGenerator(path_to_records=path,
                                        pool='test')
            meta_test_generator = generator.meta_test_pipeline()
    """
    def __init__(self,
                 path_to_records,
                 batch_config=None,
                 episode_config=[28,5,1,19],
                 valid_episode_config=[28,5,1,19],
                 pool='train',
                 mode='episode'
                 ):
        """
        Args:
            path_to_records: Absolute path of the tfrecords from which data
                will be generated. Should have the form ~/meta_train or 
                ~/meta_test
            batch_config: Array-like. In batch mode, controls 
                the size of batches and decoded images generated.Ignored 
                otherwise. [image_size, batch_size]
            episode_config: Array-like. Describes the episode configuration. If
                pool='train' and mode='episode', it sets the meta-train 
                episodes configuration generator. If pool='test', it sets the
                meta-test episodes configuration generator.
                [image_size, num_ways, num_examples_per_class_in_support, 
                    num_total_query_examples] 
            valid_episode_config: Array-like. Sets the episode configuration 
                for the meta-valid episodes generator. It is only relevant in 
                the pool='train' and mode='episode' setting, since
                both meta-train and meta-valid split could have their own 
                episodes configuration.
            pool: The split from which images are taken. Only 'train' and 'test'
                pool are allowed. Automatically create a meta-validation 
                generator if pool='train'.
            mode: The configuration of the data coming from the meta-train 
                generator. 'batch' and 'episode' are available.
        """
        self.episode_config = episode_config
        self.valid_episode_config = valid_episode_config
        self.pool = pool
        self.mode = mode

        if self.pool not in ['train', 'test']:
            raise ValueError(('In DataGenerator, only \'train\' or \'test\' '
                + 'are valid arguments for pool. Received :{}').format(self.pool))
        if self.mode not in ['episode', 'batch']:
            raise ValueError(('In DataGenerator, only \'episode\' or \'batch\' ' 
                + 'are valid arguments for mode. Received :{}').format(self.mode))
        if(self.pool == 'test' and self.mode == 'batch'):
            raise ValueError(('In DataGenerator, batch mode is only available '
                + 'at meta-train time. Received pool : {} and mode : {}').format(
                    self.pool, self.mode))
        if self.mode == 'batch':
            try:
                self.image_size_batch = batch_config[0]
                self.batch_size = batch_config[1]
            except: 
                raise ValueError(('The batch_config argument in DataGenerator '
                    + 'is not defined properly. Make sure it has the form '
                    + '[img_size, batch_size]. '
                    + 'Received batch_config : {}').format(batch_config))
        if self.mode == 'episode':
            try:
                _, _, _, _ = (self.episode_config[0], self.episode_config[1],
                    self.episode_config[2], self.episode_config[3])
            except:
                raise ValueError(('The episode config argument in DataGenerator '
                    + 'is not defined properly. Make sure it has the form '
                    + '[img_size, num_ways, num_shots, num_query]. '
                    + 'Received episode_config : {}').format(episode_config))

            try:
                _, _, _, _ = (self.valid_episode_config[0], 
                    self.valid_episode_config[1], self.valid_episode_config[2],
                    self.valid_episode_config[3])
            except:
                raise ValueError(('The episode config argument in DataGenerator '
                    + 'is not defined properly. Make sure it has the form '
                    + '[img_size, num_ways, num_shots, num_query]. '
                    + 'Received episode_config : {}').format(
                        valid_episode_config))     
        
        self.dataset_spec = dataset_spec_lib.load_dataset_spec(path_to_records)
        
        # Loading root path.
        root_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        gin_path = os.path.join(root_path, 'metadl/gin/default/decoders.gin')
        gin.parse_config_file(gin_path)

        self.meta_train_pipeline = None
        self.meta_test_pipeline = None
        self.meta_valid_pipeline = None
        
        logging.info('Creating {} generator for meta-{} dataset.'.format(
            self.mode, self.pool))

        self.set_fixed_episode_config() 
        if self.pool == 'train':
            if self.mode == 'episode':
                self.generate_meta_train_episodes_pipeline()
            else :
                self.generate_meta_train_batch_pipeline()
        else :
            self.generate_meta_test_episodes_pipeline()


    def set_fixed_episode_config(self):
        """ Set the episode description configuration. """
        if self.episode_config is not None:
            self.fixed_ways_shots = config.EpisodeDescriptionConfig(
                num_ways=self.episode_config[1], 
                num_support=self.episode_config[2],
                num_query=self.episode_config[3],
                min_ways=self.episode_config[1],
                max_ways_upper_bound=self.episode_config[1],
                max_num_query=20,
                max_support_set_size=20,
                max_support_size_contrib_per_class=200,
                min_log_weight=-0.69314718055994529,
                max_log_weight=0.69314718055994529,
                ignore_dag_ontology=True,
                ignore_bilevel_ontology=True,
                min_examples_in_class=0
                )

        if self.valid_episode_config is not None :
            self.fixed_ways_shots_valid = config.EpisodeDescriptionConfig(
                num_ways=self.valid_episode_config[1], 
                num_support=self.valid_episode_config[2],
                num_query=self.valid_episode_config[3],
                min_ways=self.valid_episode_config[1],
                max_ways_upper_bound=self.valid_episode_config[1],
                max_num_query=20,
                max_support_set_size=20,
                max_support_size_contrib_per_class=200,
                min_log_weight=-0.69314718055994529,
                max_log_weight=0.69314718055994529,
                ignore_dag_ontology=True,
                ignore_bilevel_ontology=True,
                min_examples_in_class=0
                )

    def generate_meta_test_episodes_pipeline(self):
        """Creates the episode generator for the meta-test dataset. 
        
        Notice that at meta-test time, the meta-learning algorithms always
        receive data in the form of episodes. Also, participantscan't control 
        these episodes' setting.

        ----------------------------------------------------------------------
        Details of some arguments inside the function : 

        The following arguments are ignored since we are using fixed episode
        description : 
            - max_ways_upper_bound, max_num_query, max_support_{}, 
                min/max_log_weight

        dag_ontology : only relevant for ImageNet dataset. Ignored.
        bilevel_ontology : Whether to ignore Omniglot's DAG ontology when
            sampling classes from it. Ignored.
        min_examples : 0 means that no class is discarded from having a too 
            small number of examples.
        """
        self.meta_test_pipeline = pipeline.make_one_source_episode_pipeline(
            dataset_spec=self.dataset_spec,
            use_dag_ontology=False,
            use_bilevel_ontology=False,
            split=learning_spec.Split.TEST,
            episode_descr_config=self.fixed_ways_shots,
            pool=None,
            shuffle_buffer_size=3000,
            read_buffer_size_bytes=None,
            num_prefetch=0,
            image_size=self.episode_config[0],
            num_to_take=None
        )
        logging.info('Meta-test episode config : {}'.format(
            self.episode_config))
    def generate_meta_train_episodes_pipeline(self):
        """ Creates an episode generator for both meta-train and meta-valid 
        splits.

        ----------------------------------------------------------------------
        Details of some arguments inside the function : 

        The following arguments are ignored since we are using fixed episode
        description : 
            - max_ways_upper_bound, max_num_query, max_support_{}, 
                min/max_log_weight

        dag_ontology : only relevant for ImageNet dataset. Ignored.
        bilevel_ontology : Whether to ignore Omniglot's DAG ontology when
            sampling classes from it. Ignored.
        min_examples : 0 means that no class is discarded from having a too 
            small number of examples.
        """
        self.meta_train_pipeline = pipeline.make_one_source_episode_pipeline(
            dataset_spec=self.dataset_spec,
            use_dag_ontology=False,
            use_bilevel_ontology=False,
            split=learning_spec.Split.TRAIN,
            episode_descr_config=self.fixed_ways_shots,
            pool=None,
            shuffle_buffer_size=3000,
            read_buffer_size_bytes=None,
            num_prefetch=0,
            image_size=self.episode_config[0],
            num_to_take=None
        )
        self.meta_valid_pipeline = pipeline.make_one_source_episode_pipeline(
            dataset_spec=self.dataset_spec,
            use_dag_ontology=False,
            use_bilevel_ontology=False,
            split=learning_spec.Split.VALID,
            episode_descr_config=self.fixed_ways_shots_valid,
            pool=None,
            shuffle_buffer_size=3000,
            read_buffer_size_bytes=None,
            num_prefetch=0,
            image_size=self.valid_episode_config[0],
            num_to_take=None
        )
        logging.info('Meta-Valid episode config : {}'.format(
            self.valid_episode_config))
        logging.info('Meta-Train episode config : {}'.format(
            self.episode_config))

    def generate_meta_train_batch_pipeline(self):
        """ Creates a batch generator for examples coming from the meta-train 
        split. Also creates an episode generator for examples coming from the 
        meta-valid split. Indeed, the way data comes from the meta-valid split
        should match its meta-test split counter-part.

        The meta-valid data generator will use the episode_config for the 
        episodes description if its own configuration is not provided.
        ----------------------------------------------------------------------
        Details of some arguments inside the function : 

        The following arguments are ignored since we are using fixed episode
        description : 
            - max_ways_upper_bound, max_num_query, max_support_{}, 
                min/max_log_weight

        dag_ontology : only relevant for ImageNet dataset. Ignored.
        bilevel_ontology : Whether to ignore Omniglot's DAG ontology when
            sampling classes from it. Ignored.
        min_examples : 0 means that no class is discarded from having a too 
            small number of examples.
        """
        self.meta_train_pipeline = pipeline.make_one_source_batch_pipeline(
            dataset_spec=self.dataset_spec,
            split=learning_spec.Split.TRAIN,
            batch_size= self.batch_size,
            pool=None,
            shuffle_buffer_size=3000,
            read_buffer_size_bytes=None,
            num_prefetch=0,
            image_size=self.image_size_batch,
            num_to_take=None
        )
        self.meta_valid_pipeline = pipeline.make_one_source_episode_pipeline(
            dataset_spec=self.dataset_spec,
            use_dag_ontology=False,
            use_bilevel_ontology=False,
            split=learning_spec.Split.VALID,
            episode_descr_config=self.fixed_ways_shots_valid,
            pool=None,
            shuffle_buffer_size=3000,
            read_buffer_size_bytes=None,
            num_prefetch=0,
            image_size=self.valid_episode_config[0],
            num_to_take=None
        )
        logging.info('Meta-valid episode config : {}'.format(
            self.valid_episode_config))
        logging.info('Meta-train batch config : [{}, {}]'.format(
            self.batch_size, self.image_size_batch))


