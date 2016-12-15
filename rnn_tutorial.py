import tensorflow as tf

SEQ_LENGTH = 5
VOCA_SIZE = 10
BATCH_SIZE = 4
RNN_SIZE = 10

tf.set_random_seed(0)

def get_batch():
    input = tf.random_uniform([BATCH_SIZE, SEQ_LENGTH], minval=0, maxval=VOCA_SIZE-1, dtype=tf.int64)
    target = tf.random_uniform([BATCH_SIZE, SEQ_LENGTH], minval=0, maxval=VOCA_SIZE-1, dtype=tf.int64)
    return input, target

def build_model(labels):
    with tf.variable_scope("RNN"):
        # word embedding
        embedding = tf.random_normal([VOCA_SIZE, RNN_SIZE], name="embedding/weight")

        # RNN Cell
        cell = tf.nn.rnn_cell.LSTMCell(
            num_units=RNN_SIZE
        )

        lstm_h = tf.zeros(shape=[BATCH_SIZE, RNN_SIZE], name="lstm_h")
        lstm_c = tf.zeros(shape=[BATCH_SIZE, RNN_SIZE], name="lstm_c")
        lstm_state = [lstm_h, lstm_c]

        # FC
        fc1_w = tf.random_normal([RNN_SIZE, VOCA_SIZE], name="embedding/weight")
        fc1_h = tf.random_normal([VOCA_SIZE], name="embedding/weight")

        train_logits = []
        for seq_idx in range(SEQ_LENGTH):
            if seq_idx > 0:
                tf.get_variable_scope().reuse_variables()
            lstm_input = tf.nn.embedding_lookup(embedding, labels[:,seq_idx])
            lstm_h, lstm_state = cell(
                inputs=lstm_input,
                state=lstm_state
            )
            train_logits.append(tf.matmul(lstm_h, fc1_w) + fc1_h)
        train_logits = tf.pack(train_logits)
        return train_logits

labels, target = get_batch()
logits = build_model(labels)

loss = 0
for seq_idx in range(SEQ_LENGTH):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits[seq_idx], labels[:,seq_idx])
    loss += tf.reduce_mean(cross_entropy)

learning_rate = tf.placeholder(tf.float32)
train_op = train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicts = tf.arg_max(tf.transpose(logits, perm=[1, 0, 2]), 2)
print predicts

accuracy = tf.mul(tf.reduce_sum(tf.cast(tf.equal(predicts, labels), tf.float32)), 100 / (SEQ_LENGTH*BATCH_SIZE))

with tf.Session() as sess:
    tf.initialize_all_variables().run()


    input_value, target_value, output_value, accuracy_value\
        = sess.run([labels, target, predicts, accuracy], feed_dict={
        learning_rate : 0.01
    })

    print "input_value"
    print input_value
    print input_value.shape

    print "target_value"
    print target_value
    print target_value.shape