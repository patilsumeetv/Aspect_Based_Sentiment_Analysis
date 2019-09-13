import tensorflow as tf

def build_word2vec():
    # pivot/input words and target/output words
    x = tf.placeholder(tf.int32, shape = [None,], name = "x_pivot_idxs")
    y = tf.placeholder(tf.int32, shape = [None,], name = "y_target_idxs")

    # Word Embedding Matrix
    embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
                            minval=-1.0, maxval=1.0), name = "word_embedding")

    # Biases and Weights for NCE Loss Function
    nce_biases = tf.Variable(tf.zeros([vocab_size]), name = "nce_biases")
    nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                              stddev = tf.sqrt(1/embedding_size)),
                              name = "nce_weights")

    # pivot word lookup in word embedding
    pivot = tf.nn.embedding_lookup(embedding, x, name = "word_embedding_lookup")

    # reshaping training labels
    y_train = tf.reshape(y, [tf.shape(y)[0], 1])

    # NCE Loss Function
    nce_loss = tf.reduce_mean(tf.nn.nce_loss(biases = nce_biases, labels = y_train,
                                             weights = nce_weights, inputs = pivot,
                                             num_classes = vocab_size, num_true = 1,
                                             num_sampled = num_samples),
                                             name = "nce_loss_function")

    return
