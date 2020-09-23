import tensorflow as tf

def create_proto_shells(N_ways:int, d:int):
    """
    Create a prototype shell. In an episode, there are N_ways prototypes, i.e.
    one for each class. For each class, the associated prototype is a 
    d-dimensional vector. 'd' is the embedding dimension.
    Args:
        N_ways : integer, number of class in an episode
        d : interger, embedding dimension
    """
    proto_shell = [tf.zeros((1, d)) for _ in range(N_ways)]
    return proto_shell

def reset_proto_shell(proto_shell):
    """Resets the prototypes. Used when we are changing tasks.
    Args:
        proto_shell : list of tensors. Each element of the list represents a 
                    prototype. A prototype has shape : (1, embedding_dim)
    """
    for i in range(len(proto_shell)):
        proto_shell[i] = tf.zeros_like(proto_shell[i])
    # Graph mode
    #for element in proto_shell:
    #    element.assign(tf.zeros_like(element))