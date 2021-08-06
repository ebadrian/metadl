""" Combine the ingestion and scoring processes. 
The data folder should be in this format :
    --- omniglot
        omniglot.meta_train
            |--> 0.tfrecord
            |--> 1.tfrecord
            | ...
            | ...
            |--> dataset_spec.json
        meta_test
            |--> 0.tfrecord
            |--> 1.tfrecord
            | ...
            | ...
            |--> dataset_spec.json
 
Usage example : 
    python run.py --meta_dataset_dir=<dir> --code_dir=<code_dir_path> 
"""
import os 
import time 
import shutil
import webbrowser
import shlex, subprocess

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from metadl.data.dataset import DataGenerator

FLAGS = flags.FLAGS

flags.DEFINE_string('meta_dataset_dir', '../../../omniglot',
                       'Path to the dataset directory containing a meta_train \
                         and meta_test folders.')

flags.DEFINE_string('code_dir', '../../baselines/zero/',
                       'Path to the directory containing algorithm to use.')

flags.DEFINE_string('gin_config_path', './',
                       'Path to the gin configuration files associated to a \
                           specific run.')

flags.DEFINE_string('model_dir', 'model_dir/', 'Directory for storing ' + 
                        'meta-trained model.')

flags.DEFINE_string('score_dir', 'scoring_output/', 'Directory for storing '+
                        'predictions scores on meta-test dataset.')

flags.DEFINE_boolean('open_browser', False, 'Whether to open the detailled ' + 
                        ' results page in a browser.')

def remove_dir(output_dir):
    """Removes the directory output_dir, to clean existing output of last run 
    of local test.
    Args:
        output_dir: path, the directory to remove.
    """
    if os.path.isdir(output_dir):
        logging.info("Cleaning existing output directory of last run: {}"\
                    .format(output_dir))
        shutil.rmtree(output_dir)

def main(argv):
    """ Runs the ingestion and scoring programs sequentially, as they are 
    handled in CodaLab.
    """
    del argv
    meta_dataset_dir = FLAGS.meta_dataset_dir
    code_dir = FLAGS.code_dir
    gin_config_path = FLAGS.gin_config_path
    model_dir = FLAGS.model_dir
    score_dir = FLAGS.score_dir
    open_browser = FLAGS.open_browser
    omniglot_query_size = 15 # Compatible with 5-shot (max = 20 examples)
    remove_dir(model_dir)
    remove_dir(score_dir)
    meta_train_dir = os.path.join(meta_dataset_dir, 'meta_train')
    meta_test_dir = os.path.join(meta_dataset_dir, 'meta_test')

    command_ingestion =\
         "python -m metadl.core.ingestion.ingestion --meta_train_dir={}\
             --code_dir={} --model_dir={}".format(
                                                meta_train_dir,
                                                code_dir,
                                                model_dir)

    command_scoring = "python -m metadl.core.scoring.scoring --meta_test_dir={} \
         --model_dir={} --score_dir={} --query_size_per_class={}".format(
                                            meta_test_dir,
                                            model_dir,
                                            score_dir,
                                            omniglot_query_size)

    cmd_ing = shlex.split(command_ingestion)
    cmd_sco = shlex.split(command_scoring)

    p1 = subprocess.Popen(cmd_ing)
    p1.wait()
    p2 = subprocess.Popen(cmd_sco)
    p2.wait()
    logging.info('Run finished ! ')

if __name__ == '__main__':
    app.run(main)