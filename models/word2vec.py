import tensorflow as tf
from sklearn.model_selection import train_test_split

def build_word2vec(vocab_size, embedding_size=128, num_samples=64, learning_rate=0.001):
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

    # NCE Optimizer Function
    nce_optimizer = tf.contrib.layers.optimize_loss(global_step = tf.train.get_global_step(),
                                                    loss = nce_loss, name = "nce_optimizer",
                                                    optimizer = "Adam", clip_gradients=5.0,
                                                    learning_rate = learning_rate)

    # tensorflow session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    return nce_optimizer, nce_loss, x, y, sess


def train_word2vec(input, output, x, y, optimizer, loss, sess, batch_size=32, num_epochs=5):
    # creating train test data
    X_train, X_test, y_train, y_test = train_test_split(input, output)

    # calculating number of batches required for training
    num_batches = len(X_train) // batch_size

    # weight saver object
    saver = tf.train.Saver()

    for epoch in range(num_epochs):
        print("Starting Epoch {}".format(epoch))
        for batch in range(num_batches):
            if batch != range(num_batches-1):
                x_batch = X_train[batch*batch_size:batch*batch_size+batch_size]
                y_batch = y_train[batch*batch_size:batch*batch_size+batch_size]
            else:
                x_batch = X_train[batch*batch_size:]
                y_batch = y_train[batch*batch_size:]

            # run word2vec neural network
            _ , l = sess.run([optimizer, loss], feed_dict={x:x_batch, y:y_batch})

            if batch > 0 and batch % 1000 == 0:
                print("Step {} of {}, LOSS: {}".format(batch, num_batches, l))
        # saving the weights
        save_path = saver.save(sess, "logdir\\word2vec_model.ckpt")

    return
