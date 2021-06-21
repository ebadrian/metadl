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
import pandas as pd
import os 

import gin
import higher
import torch 
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
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

        self.meta_learner = conv_net(self.N_ways, img_size)
        
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        np.random.seed(1234)

        self.device = torch.device('cuda:1')
        self.meta_learner.to(device=self.device)
        self. meta_opt = optim.Adam(self.meta_learner.parameters(), lr=1e-3)


        # Writer 
        #self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #self.train_log_dir = ('logs/gradient_tape/' + 
        #                    self.current_time +
        #                    '/train')
        #self.test_log_dir = ('logs/gradient_tape/' +
        #                    self.current_time +
        #                    '/test')
        #self.train_summary_writer = tf.summary.create_file_writer(
        #                            self.train_log_dir)
#
        #self.train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        #self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        #                    name = 'train_accuracy')
        #self.test_loss = tf.keras.metrics.Mean(name = 'test_loss')
        #self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        #                    name = 'test_accuracy')


    def dataloader(self, dataset_episodic):
        to_torch_labels = lambda a: torch.from_numpy(a.numpy()).long()
        to_torch_imgs = lambda a: torch.from_numpy(np.transpose(a.numpy(), (0, 3, 1, 2)))
        # 2
        
        def data_loader(n_batches):
            for i, (e, _) in enumerate(dataset_episodic):
                if i == n_batches:
                    break
                logging.info('e shape: {}'.format(len(e)))
                yield (to_torch_imgs(e[0]), to_torch_labels(e[1]),
                    to_torch_imgs(e[3]), to_torch_labels(e[4]))

        datal = data_loader(n_batches=1)
        for i, batch in enumerate(datal):
            #3
            data_support, labels_support, data_query, labels_query = [x.to(device=self.device) for x in batch]
            logging.info('Supp imgs: {} | Supp labs : {} | Query imgs : {} | Query labs \n \n'.format(data_support.shape, labels_support.shape, data_query.shape, labels_query.shape))


    def process_task(self, batch):
        """
        batch : [sup_imgs, sup_labs, sup_tidx, qry_imgs, qry_labs, qry_tidx]
        sup_imgs : [batch_idx, nbr_imgs, H, W, C]
        """
        to_torch_labels = lambda a: torch.from_numpy(a.numpy()).long()
        to_torch_imgs = lambda a: torch.from_numpy(np.transpose(a.numpy(), (0, 1, 4, 2, 3)))
        return (to_torch_imgs(batch[0]), to_torch_labels(batch[1]),
                    to_torch_imgs(batch[3]), to_torch_labels(batch[4]))

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
        # Load dataset in db
        meta_train_dataset = meta_dataset_generator.meta_train_pipeline
        meta_valid_dataset = meta_dataset_generator.meta_valid_pipeline
        meta_train_dataset = meta_train_dataset.batch(32)
        mtrain_iterator = meta_train_dataset.__iter__()

        #batch = next(mtrain_iterator)
        #logging.info('len batch : {}'.format(len(batch)))
        #batch_1 = batch[0]
        #logging.info('len batch 1: {}'.format(len(batch_1)))
        #batch_1 = self.process_task(batch_1)
        #data_support, labels_support, data_query, labels_query = [x.to(device=self.device) for x in batch_1]
        #logging.info('Supp imgs: {} | Supp labs : {} | Query imgs : {} | Query labs \n \n'.format(data_support.shape, labels_support.shape, data_query.shape, labels_query.shape))

        
        #logging.info('Batch :  \n {}'.format(batch))
        
        #self.dataloader(meta_train_dataset)

        log = []
        for epoch in range(self.meta_iterations):
            if epoch % 20 == 0 : 
                tmp_learner = MyLearner(self.meta_learner)
                tmp_learner.save(os.path.join('trained_models/feedback/maml_torch/models', 'epoch{}'.format(epoch)))
            self.train(mtrain_iterator, self.meta_learner, self.device, self.meta_opt, epoch, log)


        return MyLearner(self.meta_learner)

    def train(self, db, net, device, meta_opt, epoch, log):
        net.train()
        #n_train_iter = db.x_train.shape[0] // db.batchsz
        n_train_iter = 10
        for batch_idx in range(n_train_iter):
            start_time = time.time()
            # Sample a batch of support and query images and labels.
            #x_spt, y_spt, x_qry, y_qry = db.next()
            batch = next(db)
            batch = batch[0]
            batch = self.process_task(batch)
            x_spt, y_spt, x_qry, y_qry = [x.to(device=self.device) for x in batch]
            
            #task_num = self.meta_batch_size
            task_num, setsz, c_, h, w = x_spt.size()
            #logging.info('Task num: {} | Setsz: {} | c_ : {} | h : {} | w : {}'.format(task_num, setsz, c_, h, w))
            querysz = x_qry.size(1)

            #logging.info(f'sup_x : {x_spt[0].shape} | sup_y : {y_spt[0].shape} | qry_x : {x_qry[0].shape} | qry_y : {y_qry[0].shape}')
            # TODO: Maybe pull this out into a separate module so it
            # doesn't have to be duplicated between `train` and `test`?

            # Initialize the inner optimizer to adapt the parameters to
            # the support set.
            n_inner_iter = 5
            inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

            qry_losses = []
            qry_accs = []
            meta_opt.zero_grad()
            for i in range(task_num):
                with higher.innerloop_ctx(
                    net, inner_opt, copy_initial_weights=False
                ) as (fnet, diffopt):
                    # Optimize the likelihood of the support set by taking
                    # gradient steps w.r.t. the model's parameters.
                    # This adapts the model's meta-parameters to the task.
                    # higher is able to automatically keep copies of
                    # your network's parameters as they are being updated.
                    for _ in range(n_inner_iter):
                        spt_logits = fnet(x_spt[i])
                        spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                        diffopt.step(spt_loss)

                    # The final set of adapted parameters will induce some
                    # final loss and accuracy on the query dataset.
                    # These will be used to update the model's meta-parameters.
                    qry_logits = fnet(x_qry[i])
                    qry_loss = F.cross_entropy(qry_logits, y_qry[i])
                    qry_losses.append(qry_loss.detach())
                    qry_acc = (qry_logits.argmax(
                        dim=1) == y_qry[i]).sum().item() / querysz
                    
                    qry_accs.append(qry_acc)

                    # Update the model's meta-parameters to optimize the query
                    # losses across all of the tasks sampled in this batch.
                    # This unrolls through the gradient steps.
                    qry_loss.backward()

            meta_opt.step()
            qry_losses = sum(qry_losses) / task_num
            qry_accs = 100. * sum(qry_accs) / task_num
            i = epoch + float(batch_idx) / n_train_iter
            iter_time = time.time() - start_time
            if batch_idx % 4 == 0:
                logging.info(
                    f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}'
                )

            log.append({
                'epoch': i,
                'loss': qry_losses,
                'acc': qry_accs,
                'mode': 'train',
                'time': time.time(),
            })


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
        self.optimizer = torch.optim.SGD(self.learner.parameters(), lr=1e-1) # inner opt
        self.n_inner_iter = num_epochs

    def __call__(self, imgs):
        return self.learner(imgs)

    def process_task(self, images, labels):
        """
        batch : [sup_imgs, sup_labs, sup_tidx, qry_imgs, qry_labs, qry_tidx]
        sup_imgs : [batch_idx, nbr_imgs, H, W, C]
        """
        to_torch_labels = lambda a: torch.from_numpy(a.numpy()).long()
        to_torch_imgs = lambda a: torch.from_numpy(np.transpose(a.numpy(), (0, 3, 1, 2)))
        return to_torch_imgs(images), to_torch_labels(labels)

    def fit(self, dataset_train):
        """ The learner's fit function over the train set of a task.

        Args:
            dataset_train : a tf.data.Dataset object. Iterates over the training 
                            examples (support set).
        Returns:
            predictor : An instance of MyPredictor that is initilialized with 
                the fine-tuned learner's weights in this case.
        """
        self.learner.train()
        for images, labels in dataset_train:
            images, labels = self.process_task(images, labels)
            with higher.innerloop_ctx(self.learner, self.optimizer, track_higher_grads=False) as (fnet, diffopt):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.
                for _ in range(self.n_inner_iter):
                    spt_logits = fnet(images)
                    spt_loss = F.cross_entropy(spt_logits, labels)
                    diffopt.step(spt_loss)

                predictor = MyPredictor(fnet)
            break
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

        ckpt_path = os.path.join(model_dir, 'learner.pt')
        self.learner.load_state_dict(torch.load(ckpt_path))
        
    def save(self, model_dir):
        """Saves the learner model into a pickle file.

        Args:
            model_dir: the directory name from which the participant's code and 
                their saved/serialized learner are loaded.
        """

        if(os.path.isdir(model_dir) != True):
            os.mkdir(model_dir)
            #raise ValueError(('The model directory provided is invalid. Please'
            #    + ' check that its path is valid.'))

        ckpt_file = os.path.join(model_dir, 'learner.pt')
        torch.save(self.learner.state_dict(), ckpt_file)

        
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

    def process_imgs(self, images):
        to_torch_imgs = lambda a: torch.from_numpy(np.transpose(a.numpy(), (0, 3, 1, 2)))
        return to_torch_imgs(images)

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
        self.learner.eval()
        for images in dataset_test:
            #logging.info('Images shape : {}'.format(images))
            images = self.process_imgs(images[0])

            qry_logits = self.learner(images).detach()
        return qry_logits


