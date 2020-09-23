""" Defines the API used in the few-shot learning challenge. Please check the 
dedicated notebook tutorial for details.
"""

class MetaLearner():
    """Define the meta-learning algorithm we want to use, through its methods.
    It is an abstract class so one has to overide the core methods depending 
    on the algorithm.
    """
    def __init__(self):
        """Defines the meta-learning algorithm's parameters. For example, one 
        has to define what would be the learner meta-learner's architecture. 
        For instance, one could use the Keras API to define models.
        """
        pass

    def meta_fit(self, meta_dataset_generator):
        """ Uses the datasets in the meta-dataset (training) to fit the 
        meta-learner's parameters. A dataset reprensents a task here.
        Args:
            meta_dataset_generator : a DataGenerator object, encapsulating the 
                data generators for the meta-train and meta-validation splits.
                Each of these generators is a tf.data.Dataset object.
        
        Returns:
            Learner : The resulting learner ready to be trained and evaluated 
                on new unseen tasks.
        """
        raise NotImplementedError(('You should implement the save method for '
            + 'the MetaLearner class.'))
    

class Learner():
    """ This class represents the learner returned at the end of the 
    meta-learning procedure.
    """
    def __init__(self):
        pass

    def fit(self, dataset_train):
        """ Fit the Learner to a new unseen task. Note that this function could
        take various forms. Please check the difference between the different
        implementation of Proto-Networks and fo-MAML baselines if this is 
        unclear.
        Args:
            dataset_train : the training set of a task. The data arrive in the 
                following format : 
            ([batch_size, img_height, img_width, channels], [batch_size, label])
        
        Returns:
            Predictor : The resulting predictor ready to predict unlabelled
                query image examples from the new unseen task.
        """
        raise NotImplementedError(('You should implement the fit method for '
            + 'the Learner class.'))
    
    def save(self, path_to_save):
        raise NotImplementedError(('You should implement the save method for '
            + 'the Learner class.'))

    def load(self, path_to_model):
        raise NotImplementedError(('You should implement the load method for '
            + 'the Learner class.'))


class Predictor():
    """ This class represents the predictor returned at the end of the Learner's
    fit method. 
    """
    def __init__(self):
        pass

    def predict(self, dataset_test):
        """ Given a dataset_test, predicts the labels associated to the 
        provided images.
        Args:
            dataset_test : a tf.data.Dataset containing unlabelled 
                image examples. Images arrive in the following format : 
                [batch_size, img_height, img_width, channels]
        Returns:
            tf.Tensor, shape : (batch_size, 1)

        Notice that the labels are expected to be given in their 'Sparse' form.
        Also the tensor returned should contain the labels of ALL images that 
        were in dataset_test.  
        """
        raise NotImplementedError(('You should implement the predict method for '
            + 'the Predictor class.'))

