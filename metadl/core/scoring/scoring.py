""" Runs the scoring procedure for the challenge.
It assumes that there exists a ./model_dir folder containing both the 
submission code and the saved learner. 
It will create a folder named ./scoring_output (default) in which a txt file 
will contain the average score over 600 episodes. You can change the folder 
name via the score_dir flag.

Usage example executed from the metadl/ repository : 

python -m metadl.core.scoring.scoring --meta_test_dir=<path_dataset.meta_test>                     
""" 
import os 
from sys import path

import base64
import gin
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import time
from absl import app
from absl import flags 
from absl import logging
from glob import glob
import tensorflow as tf

from metadl.data.dataset import DataGenerator
from metadl.core.ingestion.ingestion import get_gin_path, show_dir

FLAGS = flags.FLAGS

flags.DEFINE_string('meta_test_dir', 
                    '/Users/adrian/GitInria/meta-dataset/records/',
                    ('Directory of the meta-test dataset. This directory '
                        + 'should contain records and a json spec file.'))

flags.DEFINE_string('saved_model_dir',
                    './model_dir',
                    ('Directory path that contains the participant\'s code '
                        + 'along with the serialized learner from meta-fit.'))

flags.DEFINE_string('score_dir',
                    './scoring_output',
                    'Path to the score directory.')

flags.DEFINE_string('evaltype',
                    'test',
                    'Data type on which to perform evaluation. [train, val, test]')

flags.DEFINE_integer('ingestion_time_budget',
                    7200,
                    'Data type on which to perform evaluation. [train, val, test]')
flags.DEFINE_integer('query_size_per_class', 19, 
    'Number of query example per episode at meta-test time.')

tf.random.set_seed(1234)
def NwayKshot_accuracy(predictions, ground_truth, metric):
    """ N-way, K-shot accuracy which corresponds to the accuracy in a
    multi-classification context with N classes.

    Args:
        predictions : tensors, sparse tensors corresponding to the predicted 
            labels.
        ground_truth : tensors, sparse tensors corresponding the ground truth 
            labels.
        metric : keras.metrics , the metric we use to evaluate the 
            classification performance of the meta-learning algorithm. We use 
            the SparseCategoricalAccuracy in this challenge.

    Retruns:
        score : Float, the resulting performance using the given metric.
    """
    ground_truth = tf.expand_dims(ground_truth, axis = 1)
    predictions = tf.expand_dims(predictions, axis = 1)
    logging.debug('Predictions shape : {} - Ground truth shape : {}'.format(
        predictions.shape, ground_truth.shape))

    metric.update_state(ground_truth, predictions)
    score = metric.result()
    logging.debug('An episode score: {}'.format(score))
    metric.reset_states()
    return score
    
def is_one_hot_vector(x, axis=None, keepdims=False):
  """Check if a vector 'x' is one-hot (i.e. one entry is 1 and others 0)."""
  norm_1 = np.linalg.norm(x, ord=1, axis=axis, keepdims=keepdims)
  norm_inf = np.linalg.norm(x, ord=np.inf, axis=axis, keepdims=keepdims)
  return np.logical_and(norm_1 == 1, norm_inf == 1)

def write_score(score, conf_int, file_score, duration=-1):
    """Write score of the k-th task in the given file_score."""
    file_score.write('score: {:.6f}\n'.format(float(score)))
    file_score.write('conf_int: {:.3f}\n'.format(float(conf_int)))
    file_score.write('Duration: {:.6f}\n'.format(float(duration)))
    
def extract_elapsed_time(saved_model_dir):
    """ Extracts elapsed time from the metadata file. It corresponds to the 
    meta-training time, the duration of the ingestion process.
    """
    if not os.path.isdir(saved_model_dir): 
        raise ValueError('Saved model directory does not exists.')

    if os.path.isfile(os.path.join(saved_model_dir, 'metadata')):
        with open(os.path.join(saved_model_dir, 'metadata'), 'r') as f : 
            lines = f.readlines()
            for line in lines : 
                splitted_line = line.split(' ')
                for k, word in enumerate(splitted_line): 
                    if 'elapsed' in splitted_line[k]:
                        elapsed_time = float(splitted_line[k+1])
                        return elapsed_time
        
    return -1

def process_task(task, query_size_per_class):
    """We are using the meta-dataset code to generate episodes from a dataset. 
    Generated episodes have a specific format. Each is processed such that the 
    the support and query sets are ready to be used by the participants. Each
    set is returned as a tf.data.Dataset object.
    The que_labs are kept hidden.

    Returns : 
        support_dataset : tf.data.Dataset containing the support examples and 
            labels.
        query_dataset : tf.data.Dataset containing the query examples
        que_labs : tuple (query_batch_size, 1), the query examples labels 
            i.e. the ground truth labels.
    """
    sup_set = tf.data.Dataset.from_tensor_slices(\
        (task[0][1], task[0][0]))
    dim = task[0][4].shape[1]
    arr = np.arange(dim)
    np.random.shuffle(arr) # shuffling arr
    query_labs = task[0][4]
    query_imgs = task[0][3]
    
    query_labs_s = tf.gather(query_labs, arr, axis=1)
    query_imgs_s = tf.gather(query_imgs, arr, axis=1)

    que_set = tf.data.Dataset.from_tensor_slices(
            (query_labs_s, query_imgs_s)
    )
    new_ds = tf.data.Dataset.zip((sup_set, que_set))
    for ((supp_labs, supp_img), (que_labs, que_img)) \
            in new_ds :

        logging.debug('Supp labs : {}'.format(supp_labs))
        logging.debug('Query labs : {}'.format(que_labs))

        support_set = tf.data.Dataset.from_tensor_slices(\
            (supp_img, supp_labs))
        query_set = tf.data.Dataset.from_tensor_slices(\
            (que_img,))
        support_set = support_set.batch(25)
        query_set = query_set.batch(query_size_per_class * 5)

    return support_set, query_set, que_labs


def list_files(startpath):
    """List a tree structure of directories and files from startpath"""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        logging.debug('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            logging.debug('{}{}'.format(subindent, f))


def get_fig_name(task_name):
    """Helper function for getting learning curve figure name."""
    fig_name = "learning-curve-" + str(task_name) + ".png"
    return fig_name


def initialize_detailed_results_page(score_dir):
    """Initialize detailed results page with a message for waiting."""
    # Create the output directory, if it does not already exist
    if not os.path.isdir(score_dir):
        os.mkdir(score_dir)
    # Initialize detailed_results.html
    detailed_results_filepath = os.path.join(
        score_dir,
        'detailed_results.html'
    )
    html_head = '<html><head> <meta http-equiv="refresh" content="5"> ' +\
                '</head><body><pre>'
    html_end = '</pre></body></html>'
    with open(detailed_results_filepath, 'a') as html_file:
        html_file.write(html_head)
        html_file.write(
            "Starting meta-training process... <br> Please be patient. " +
            "Note that no learning curves will be generated for this " +
            "particular challenge. The name 'Learning Curve' is a legacy " +
            "of the AutoDL challenge."
        )
        html_file.write(html_end)


def plot_detailed_results_figure(results, score_dir, task_name=None):
    """Plot figure for one task and save to `score_dir`."""
    fig_name = get_fig_name(task_name)
    path_to_fig = os.path.join(score_dir, fig_name)
    mean_acc = np.mean(results)
    results = np.array(results)
    # 95 is to avoid empty bins as we have 95 examples in meta-test
    bins = np.linspace(0, 1, 95)
    plt.hist(results, bins=bins, range=(0, 1))
    plt.axvline(mean_acc, linestyle='dashed', linewidth=2, c='r')
    plt.text(mean_acc, 6, 'Mean acc: {:.4f}'.format(mean_acc))
    plt.title("Distribution of accuracies obtained on {} meta-test tasks."\
        .format(len(results)))
    plt.savefig(path_to_fig)
    plt.close()


def write_scores_html(score_dir, auto_refresh=False, append=False):
    filename = 'detailed_results.html'
    image_paths = sorted(glob(os.path.join(score_dir, '*.png')))
    if auto_refresh:
      html_head = '<html><head> <meta http-equiv="refresh" content="5"> ' +\
                  '</head><body><pre>'
    else:
      html_head = """<html><body><pre>"""
    html_end = '</pre></body></html>'
    if append:
      mode = 'a'
    else:
      mode = 'w'
    filepath = os.path.join(score_dir, filename)
    with open(filepath, mode) as html_file:
        html_file.write(html_head)
        for image_path in image_paths:
          with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            encoded_string = encoded_string.decode('utf-8')
            s = '<img src="data:image/png;charset=utf-8;base64,{}"/>'\
                .format(encoded_string)
            html_file.write(s + '<br>')
        html_file.write(html_end)
    logging.debug("Wrote learning curve page to {}".format(filepath))

def set_device():
    """Automatically chooses the right device to run on TF.
    Reason: Conflict with 2 different processes running TF (metric from ingestion
    is cached when scoring starts).
    Returns:
        str: device identifier which is best suited 
            (most free GPU, or CPU in case GPUs are unavailable)
    """
    gpus = tf.config.list_physical_devices(device_type="GPU")
    dev = None
    try:
      if gpus is not None:
        current_device = gpus[-1].name[-1]
        dev = f"GPU:{current_device}"
    finally:
      return dev

def scoring(argv):
    """ 
    For each task, load and fit the Learner with the support set and evaluate
    the submission performance with the query set. 
    A directory 'scoring_output' is created and contains a txt file that 
    contains the submission score and duration. Note that the former is the 
    time elapsed during the ingestion program and hence the meta-fit() 
    duration.

    The metric considered here is the Sparse Categorical Accuracy for a 
    5 classes image classification problem.
    """
    del argv
    saved_model_dir = FLAGS.saved_model_dir
    meta_test_dir = FLAGS.meta_test_dir
    eval_type = FLAGS.evaltype
    ingestion_time_budget = FLAGS.ingestion_time_budget
    query_size_per_class = FLAGS.query_size_per_class 
    
    # Making eval type compatible with DataGenerator specs
    if eval_type == 'train' or eval_type == 'val':
        data_generator_eval_type = 'train'
    elif eval_type == 'test':
        data_generator_eval_type = 'test'
    # Use CodaLab's path `run/input/ref` in parallel with `run/input/res`
    if not os.path.isdir(meta_test_dir): 
        meta_test_dir = os.path.join(saved_model_dir, os.pardir, 'ref')

    # Evaluation type scenario: if meta_test is specified -> act as normal 
    # scoring on meta_test data
    if (eval_type == 'train' or eval_type == 'val') and 'meta_test' in meta_test_dir:
        raise ValueError('Cannot perform train/val evaluation on meta-test data!')
    #if 'meta_test' not in meta_test_dir:
    #    if eval_type == 'test':
    #        meta_test_dir = os.path.join(meta_test_dir, 'meta_test')
    #    else:
    #        meta_test_dir = os.path.join(meta_test_dir, 'meta_train')

    code_dir = os.path.join(saved_model_dir, 'code_dir')
    score_dir = FLAGS.score_dir

    logging.debug("Using meta_test_dir={}".format(meta_test_dir))
    logging.debug("Using code_dir={}".format(code_dir))
    logging.debug("Using saved_model_dir={}".format(saved_model_dir))
    logging.debug("Using score_dir={}".format(score_dir))
    list_files(os.path.join(score_dir, os.pardir, os.pardir))

    # Initialize detailed results page
    initialize_detailed_results_page(score_dir)
    ########################################
    # IMPORTANT: 
    # Wait until code_dir is created
    ########################################
    t = 0
    while (not os.path.isdir(code_dir)) and t < ingestion_time_budget:
        time.sleep(1)
        t += 1
    
    path.append(code_dir)
    from model import MyLearner
    if(os.path.exists(os.path.join(code_dir, 'model.gin'))):
        gin.parse_config_file(os.path.join(code_dir, 'model.gin'))

    logging.info('Ingestion done! Starting scoring process ... ')
    logging.info('Creating the meta-test episode generator ... \n ')
    generator = DataGenerator(path_to_records=meta_test_dir,
                            batch_config=None,
                            episode_config=[128, 5, 5, query_size_per_class],
                            pool= data_generator_eval_type,
                            mode='episode')
    
    if eval_type == 'test':
        meta_test_dataset = generator.meta_test_pipeline
    elif eval_type == 'train':
        meta_test_dataset = generator.meta_train_pipeline
    elif eval_type == 'val':
        meta_test_dataset = generator.meta_valid_pipeline
    else:
        raise ValueError('Wrong eval_type : {}'.format(eval_type))

    logging.info('Evaluating performance on episodes ... ')

    meta_test_dataset = meta_test_dataset.batch(1)
    meta_test_dataset = meta_test_dataset.prefetch(5)
    learner = MyLearner()
    
    if (not os.path.isdir(score_dir)):
        os.mkdir(score_dir)
    score_file = os.path.join(score_dir, 'scores.txt')
    results = []

    dev = set_device()
    if dev is not None:
        with tf.device(f'/device:{dev}'):
            metric = tf.metrics.SparseCategoricalAccuracy(name="test_sparse_categorical_accuracy")
    else:
        metric = tf.metrics.SparseCategoricalAccuracy(name="test_sparse_categorical_accuracy")

    nbr_episodes = 600

    for k , task in enumerate(meta_test_dataset):
        support_set, query_set, ground_truth = process_task(task, query_size_per_class)
        learner.load(saved_model_dir)
        predictor = learner.fit(support_set)
        predictions = predictor.predict(query_set)
        score = NwayKshot_accuracy(predictions, ground_truth, metric)
        results.append(score)

        logging.debug('Score on {} : {}'.format(k, score))
        logging.debug('Results : {}'.format(results[:20]))
        if(k == nbr_episodes - 1):
            break
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, h

    m, conf_int = mean_confidence_interval(results)
    with open(score_file, 'w') as f :
        write_score(m,
                    conf_int,
                    f, 
                    extract_elapsed_time(saved_model_dir))

    # Update detailed results page
    task_name = None
    plot_detailed_results_figure(results, score_dir, task_name=task_name)
    write_scores_html(score_dir)

    logging.info(('Scoring done! The average score over {} '
        + 'episodes is : {:.3%}').format(nbr_episodes,
                                        sum(results)/len(results))
    )

if __name__ == '__main__':
    np.random.seed(seed=1234)
    tf.get_logger().setLevel('ERROR')
    logging.set_verbosity('INFO')
    app.run(scoring)


    
