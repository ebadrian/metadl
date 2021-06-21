import tensorflow as tf

def create_grads_shell(model):
    """ Create list of gradients associated to each trainable layer in model.
    
    Returns:
    -------
    list_grads, array-like : each element of this list is tensor representing 
        the associated layer's gradient.
    """


    list_grads = []
    for layer in model.trainable_variables :
        list_grads.append(tf.Variable(tf.zeros_like(layer)))

    return list_grads

def reset_grads(meta_grads):
    """Reset the variable that contains the meta-learner gradients.
    Arguments:
    ----------
    meta_grads : list of tf.Variable

    Note : Each element is guaranteed to remain a tf.Variable. Using
    tf.zeros_like on tf.Variable does not transform the element to 
    tf.Tensor
    """
    for ele in meta_grads :
        ele.assign(tf.zeros_like(ele))


def app_custom_grads(model, inner_gradients, lr):
    """ Apply gradient update to the model's parameters using inner_gradients.
    """
    i = 0
    #print(inner_gradients)
    for k, layer in enumerate(model.layers) :
        if 'kernel' in dir(layer) : 
            #print(layer.kernel.shape)
            layer.kernel.assign_sub(tf.multiply(lr, inner_gradients[i]))
            i+=1
        elif 'normalization' in layer.name:
            layer.trainable_weights[0].assign_sub(\
                tf.multiply(lr, inner_gradients[i]))
            
            i+=1
        if 'bias' in dir(layer):
            layer.bias.assign_sub(tf.multiply(lr, inner_gradients[i]))
            i+=1
        elif 'normalization' in layer.name:
            layer.trainable_weights[1].assign_sub(\
                tf.multiply(lr, inner_gradients[i]))
            i+=1
