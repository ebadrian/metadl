""" 
TODO:
    - learning_curve.png: think of what we can add
    - Add CI (leaderboard) for average accuracy on 600 episodes on a particular
        meta-test set
    - Load data zips in CodaLab so we can directly refer to them in the 
        competition yaml file.

"""
import os
from os.path import join
import sys
import yaml
import argparse
import base64
import metadl
from shutil import copyfile
from glob import glob
import logging

################################################################################
# USER DEFINED CONSTANTS
################################################################################

# Verbosity level of logging.
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
verbosity_level = 'INFO'

# Number of children phases/datasets (as defined in competition bundle)
DEFAULT_NUM_DATASET = 5
current_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SCORE = join(current_path, 'default_scores.txt')
DEFAULT_CURVE = join(current_path, 'default_curve.png')

print (current_path)
print (DEFAULT_SCORE)
print (DEFAULT_CURVE)

################################################################################
# FUNCTIONS
################################################################################

def get_logger(verbosity_level, use_error_log=False):
    """Set logging format to something like:
        2019-04-25 12:52:51,924 INFO score.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger

def validate_full_res(args):
    """
        Check if we have DEFAULT_NUM_DATASET results in the args.input_dir.
        Replace by defaulta values otherwise.
    """
    for i in range(DEFAULT_NUM_DATASET):
        # Check whether res_i/ exists
        check_path = join(args.input_dir, "res_"+str(i+2))
        logger.info("Checking " + str(check_path))
        if not os.path.exists(check_path):
            # Replace both learning curve and score by default:
            logging.warning(str(check_path) +
                            " does not exist. Default values will be used.")
            # Create this folder and copy default values
            os.mkdir(check_path)
            copyfile(DEFAULT_SCORE, join(check_path,"scores.txt"))
            copyfile(DEFAULT_CURVE, join(check_path,"learning-curve-default.png"))
        else:
            # Replace either learning curve or score by default, depending...
            if not os.path.exists(join(check_path,"scores.txt")):
                logging.warning("Score file" +
                            " does not exist. Default values will be used.")
                copyfile(DEFAULT_SCORE, join(check_path,"scores.txt"))
            is_curve_exist = False
            for f in os.listdir(check_path):
                if f[-4:] == ".png":
                    is_curve_exist = True
                    break
            if not is_curve_exist:
                logging.warning("Learning curve" +
                            " does not exist. Default values will be used.")
                copyfile(DEFAULT_CURVE, join(check_path,"learning-curve-default.png"))
    return

def read_score(args):
    """
        Fetch scores from scores.txt
    """
    # TODO: should not be hard coded: figure out which phase you are in.
    score_ls = []
    for i in range(DEFAULT_NUM_DATASET):
        score_dir = args.input_dir + "/res_"+str(i+2)
        score_file = join(score_dir, "scores.txt")
        try:
            with open(score_file, 'r') as f:
                score_info = yaml.safe_load(f)
            score_ls.append(float(score_info['set1_score']))
        except Exception as e:
            logging.exception("Failed to load score in: {}".format(score_dir))
            logging.exception(e)
    return score_ls

def read_curve(args):
    """
        Fetch learning curve from learning-curve-*.png
    """
    curve_ls = []
    try:
        for i in range(DEFAULT_NUM_DATASET):
            curve_dir = join(args.input_dir, 'res_'+str(i+2))
            _img = glob(os.path.join(curve_dir,'learning-curve-*.png'))
            curve_ls.append(_img[0])
    except Exception as e:
        logging.exception("Failed to read curves.")
        logging.exception(e)
    return curve_ls

def write_score(score_ls, args):
    """
        Write scores to master phase scores.txt, as setj_score, where j = 1 to DEFAULT_NUM_DATASET
    """
    output_file = join(args.output_dir, 'scores.txt')
    try:
        with open(output_file, 'w') as f:
            f.write("score: \n")
            for i in range(DEFAULT_NUM_DATASET):
                score_name = 'set{}_score'.format(i+1)
                score = score_ls[i]
                f.write("{}: {}\n".format(score_name, score))
    except Exception as e:
        logging.exception("Failed to write to" + output_file)
        logging.exception(e)
    return

def write_curve(curve_ls, args):
    """
        Write learning curves concatenated
    """
    filename = 'detailed_results.html'
    detailed_results_path = join(args.output_dir, filename)
    html_head = '<html><body><pre>'
    html_end = '</pre></body></html>'
    try:
        with open(detailed_results_path, 'w') as html_file:
            html_file.write(html_head)
            for id,image_path in enumerate(curve_ls):
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                    encoded_string = encoded_string.decode('utf-8')
                    t = '<font size="7">Meta-dataset ' + str(id+1) + '.</font>'
                    html_file.write(t + '<br>')
                    s = '<img src="data:image/png;charset=utf-8;base64,%s"/>'%encoded_string
                    html_file.write(s + '<br>')
            html_file.write(html_end)
    except Exception as e:
        logging.exception("Failed to write to" + detailed_results_path)
        logging.exception(e)
    return

if __name__ == "__main__":
    logger = get_logger(verbosity_level)
    try:
        VERSION = metadl.__version__
        # Logging version information and description
        logger.info('#' * 50)
        logger.info("Version: " + VERSION)
        # logger.info(DESCRIPTION)
        logger.info('#' * 50)

        # Get input and output dir from input arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_dir', type=str, default='./test_input',
                            help='where input results are stored')
        parser.add_argument('--output_dir', type=str, default='./test_output',
                            help='where to store aggregated outputs')
        args = parser.parse_args()

        print ("Copying input folder....")
        os.system("cp -R {} {}".format(join(args.input_dir, '*'), args.output_dir))

        if not os.path.exists(args.input_dir):
            logging.error("No input folder! Exit!")
            sys.exit()
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        # List the contents of the input directory (should be a bunch of res_i/ subdirectories)
        input_ls = sorted(os.listdir(args.input_dir))
        logger.debug("Input dir contains: " + str(input_ls))

        # Check if we have correct results in input_dir/res_i/ and copy default values otherwise
        validate_full_res(args)
        logger.info("[+] Results validation done.")
        logger.debug("-" * 50)
        logger.debug("Start aggregation...")

        # Read all scores from input_dir/res_i/ subdirectories
        score_ls = read_score(args)
        logger.info("[+] Score reading done.")
        logger.info("Score list: " + str(score_ls))

        # Aggregate all scores and write to output
        write_score(score_ls, args)
        logger.info("[+] Score writing done.")

        # Read all learning curves
        curve_ls = read_curve(args)
        logger.info("[+] Learning curve reading done.")
        logger.debug("Curve list: " + str(curve_ls))

        # Aggregate all learning curves and write to output
        write_curve(curve_ls, args)
        logger.info("[+] Learning curve writing done.")

        logger.info("[+] Parent scoring program finished!")

    except Exception as e:
        logger.exception("Unexpected exception raised! Check parent scoring program!")
        logger.exception(e)