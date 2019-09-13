import tensorflow as tf

def build_word2vec():
    # pivot/input words and target/output words
    x = tf.placeholder(tf.int32, shape = [None,], name = "x_pivot_idxs")
    y = tf.placeholder(tf.int32, shape = [None,], name = "y_target_idxs")

    # Word Embedding Matrix
    embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
                            minval=-1.0, maxval=1.0), name = "word_embedding")

    return
