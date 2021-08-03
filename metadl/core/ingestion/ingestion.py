""" The ingestion process handles the meta-fit part of the meta-algorithm.
Participants have no direct interaction with this script.

A dataset folder should look like the following :  
- omniglot
| --> meta_train
| --> meta_test

Usage example for a local run (root : metadl/): 
    python -m metadl.core.ingestion.ingestion \
            --meta_train_dir=../omniglot/meta_train \ 
            --code_dir=./baselines/zero/

"""
import os 
import shutil
from sys import path

import gin
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from metadl.data.dataset import DataGenerator

FLAGS = flags.FLAGS

flags.DEFINE_string('meta_train_dir',
                    '../omniglot/meta_train',
                    ('Directory of the meta-train dataset. This directory'
                       +'should contain tfrecords and a datasetspec.json '
                       +'file.'))

flags.DEFINE_string('code_dir', 
                    './baselines/zero/',
                    'Path to the directory containing the algorithm to use.')

flags.DEFINE_string('model_dir', 
                    './model_dir',
                    'Directory path to save the participants code, along with '
                    + 'the serialized learner returned by meta-fit().')

flags.DEFINE_boolean('debug_mode',
                    False,
                    'Whether to use debug verbosity.')

tf.random.set_seed(1234)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)
tf.config.experimental.set_memory_growth(gpus[2], True)
tf.config.experimental.set_memory_growth(gpus[3], True)

def get_gin_path():
    """ Get the absolute path of a gin file in a compute_worker. This method is 
    necessary since we can't know the directory names in advance (temporary 
    folders handled by CodaLab docker images).
    """
    ingestion_path = __file__
    dir_name_par = os.path.dirname(ingestion_path)
    for _ in range(2):
        dir_name_par = os.path.abspath(os.path.join(dir_name_par, os.pardir))
    gin_file_path = os.path.join(dir_name_par, 'gin/setups/data_config.gin')

    logging.info('Dir name parent: {}'.format(dir_name_par))
    logging.info('gin file path : {}'.format(gin_file_path))
    
    return gin_file_path


flags.DEFINE_string('gin_config_path', 
                    get_gin_path(),
                    'Gin file path for data augmentation config.')

def run_callbacks(model, method, remaining_time_budget: float):
  """Run callbacks of model.
  Args:
      model (metadl.api.MetaLearner): model on witch to run callbacks.
      method (callable): method of callbacks to run.
  """
  for cb in model.callbacks:
    if hasattr(cb, method):
      func = getattr(cb, method)
      func(remaining_time_budget=remaining_time_budget)

def show_dir(d, n_bytes=100, depth=1):
    """Helper function for debugging: print the content of a directory."""
    if not os.path.isdir(d):
        print("The directory {} doesn't exist. Nothing to show.".format(d))
        return

    print("Showing the content of directory: {}".format(d))
    print(os.listdir(d))
    for filename in os.listdir(d):
        filepath = os.path.join(d, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                print(filepath)
                print(' '*4, f.read(n_bytes))
        if depth > 1 and os.path.isdir(filepath):
            show_dir(filepath, n_bytes=n_bytes, depth=depth-1)


def ingestion(argv):
    """The ingestion process achieves 2 things. First, it uses the meta-fit() 
    method defined by the participant in code_dir/model.py to create a Learner.
    Then, it saves/serialized this Learner in model_dir/ along with the content
    of code_dir. This way, we are able to access both the participant's code 
    and the saved Learner in the scoring process. 
    """
    del argv
    debug_mode = FLAGS.debug_mode
    meta_train_dir = FLAGS.meta_train_dir 
    code_dir = FLAGS.code_dir 
    model_dir = FLAGS.model_dir
    gin_path = FLAGS.gin_config_path
    if debug_mode : 
        logging.set_verbosity(logging.DEBUG)
    if meta_train_dir.endswith('run/input') and\
       code_dir.endswith('run/program'):
        logging.debug("Since meta_train_dir ends with 'run/input' and code_dir "
            "ends with 'run/program', suppose running on the AutoDL instance " +
            "of CodaLab platform. Modify meta_train_dir to 'run/input_data' "
            "and code_dir to 'run/submission'. " +
            "Directory parsing should be more flexible in the code of " +
            "compute worker: we need explicit directories for " +
            "dataset_dir and code_dir.")
        meta_train_dir = meta_train_dir.replace('run/input', 'run/input_data')
        code_dir = code_dir.replace('run/program', 'run/submission')

    logging.debug("Using meta_train_dir={}".format(meta_train_dir))
    logging.debug("Using code_dir={}".format(code_dir))
    logging.debug(str(os.listdir(code_dir)))
    logging.debug("Using model_dir={}".format(model_dir))
    logging.debug("Using gin_path={}".format(gin_path))
    
    path.append(code_dir)
    try:
        from model import MyMetaLearner 

        # Loading model.gin parameters if specified
        if(os.path.exists(os.path.join(code_dir, 'model.gin'))):
            gin.parse_config_file(os.path.join(code_dir, 'model.gin'))

        # Loading data generation config.gin if specified
        if(os.path.exists(os.path.join(code_dir, 'config.gin'))):
            gin.parse_config_file(os.path.join(code_dir, 'config.gin'))

        logging.debug('Gin file path : {}'.format(gin_path))
        logging.debug("Files in meta-train directory are:\n{}".format(
            os.listdir(meta_train_dir)))
        
        logging.info('Creating the episode generator ...')
        # Creating DataGenerator with default params or loaded from config.gin
        generator = DataGenerator(path_to_records=meta_train_dir)  

        logging.info('Generator created !')
        logging.info("#"*50)
        meta_learner = MyMetaLearner()
        logging.info('Starting meta-fit ... \n')
        learner = meta_learner.meta_fit(generator)
        logging.info('Meta-fit done.')

        if(not os.path.isdir(model_dir)):
            os.mkdir(model_dir)
        
        logging.info('Saving the learner in {} ...'.format(model_dir))
        learner.save(model_dir)
        logging.debug(str(os.listdir(model_dir)))
        logging.info('Done! \n ')
    except Exception as e:
        logging.error("Error encountered :\n" + str(e), exc_info=True)

    # Copy the baseline file into model_dir
    model_code_dir = os.path.join(model_dir, 'code_dir')
    if os.path.isdir(model_code_dir):
        shutil.rmtree(model_code_dir)
    shutil.copytree(code_dir, model_code_dir)


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    logging.set_verbosity('ERROR')
    app.run(ingestion)

