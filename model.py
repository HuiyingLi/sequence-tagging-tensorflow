import tensorflow as tf
import reader
import pdb
import numpy as np

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def charemb_initializer(emb_size):
    return tf.random_uniform_initializer(minval=-np.sqrt(3.0/emb_size), maxval=np.sqrt(3.0/emb_size))


def weight_initializer(r, c):
    return tf.random_uniform_initializer(minval=-np.sqrt(6.0/(r+c)), maxval=np.sqrt(6.0/(r+c)))


def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])
    return tf.nn.conv2d(input_, w, strides=[1,1,1,1], padding='VALID')+b


def tdnn(input_, filter_scopes, nfilters, scope='TDNN'):
    '''
    input: float tensor of shape [batch_size x num_steps x max_word_len x embed_size]
    kernels: arrays of kernel sizes (window sizes, scopes of kernels, etc.)
    kernel_features: output features, dim of output
    '''
    max_word_len = input_.get_shape()[1]
    embed_size = input_.get_shape()[-1]
    input_ = tf.expand_dims(input_,1)##
    layers = []
    with tf.variable_scope(scope):
        for window_size, nfilter in zip(filter_scopes, nfilters):
            reduced_length = max_word_len-window_size+1
            conv = conv2d(input_, nfilter, 1, window_size, name = "kernel_%d"%window_size)
            #conv=tf.layers.conv2d(inputs=input_, filters=nfilter, kernel_size=window_size, padding="VALID", activation=tf.nn.relu, name="kernel_%d"%window_size)
            #pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[1,1,reduced_length,1], stride=1)
            pool = tf.nn.max_pool(tf.tanh(conv), [1,1,reduced_length, 1],[1,1,1,1],'VALID')
            layers.append(tf.squeeze(pool,[1,2])) #
        if len(filter_scopes)>1:
            output=tf.concat(layers,1)
        else:
            output=layers[0]
    return output

def linear(input_, output_size, scope=None):
    shape = input_.get_shape().as_list()
    input_size = shape[1]
    with tf.variable_scope(scope or "SimpleLinear"):
        W = tf.get_variable("W",
                shape=[output_size, input_size],
                initializer=weight_initializer(output_size, input_size), dtype=tf.float32)
        b = tf.get_variable("b", shape=[output_size], initializer=tf.zeros_initializer(), dtype=tf.float32)
    return tf.matmul(input_, tf.transpose(W)) + b

def inference_graph(
    char_vocab_size,
    pretrain_embedding,
    max_word_len,
    ntags,
    batch_size=10,
    num_steps=30,
    char_emb_size=30,
    lstm_state_size=200,
    dropout=0.5,
    filter_sizes=[3],
    nfilters=[30],
    is_training=True):

    char_input = tf.placeholder(tf.int32,shape=[batch_size, num_steps, max_word_len],name="char_input")
    
    with tf.variable_scope('char_embedding'):
        char_embedding = tf.get_variable("char_embedding",
            [char_vocab_size,char_emb_size], 
            initializer=charemb_initializer(char_emb_size),
            dtype=tf.float32)
        input_embedded = tf.nn.embedding_lookup(char_embedding, char_input)
        input_embedded = tf.reshape(input_embedded, [-1, max_word_len, char_emb_size])

    '''Apply convolution
        input: [batch_size*num_steps, max_word_len, char_emb_size]
        output: [batch_size, num_steps, sum(nfilters)]
    '''
    char_rep = tdnn(input_embedded, filter_sizes, nfilters)
    char_rep = tf.reshape(char_rep, [batch_size, num_steps, -1])    
    
    '''load and concatenate with pretrained embeddings
        output: [batch_size, num_steps, char_embedding_size + pretrained_embedding_size]
    '''
    word_input = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name='word_input')
    
    L = tf.Variable(pretrain_embedding, dtype=tf.float32, trainable=False)
    word_embedding = tf.nn.embedding_lookup(L, word_input)

    word_rep = tf.concat([word_embedding, char_rep], axis=-1)
    word_rep2 = [tf.squeeze(x, [1]) for x in tf.split(word_rep, num_steps, 1)]

    '''LSTM
        intput size: [batch_size, num_steps, char_emb_size + pretrained_emb_size]
        output size: a list of length num_step, containing tensors shaped [batch_size, lstm_size*2] 
    '''
    def create_rnn_cell():
        cell = tf.contrib.rnn.BasicLSTMCell(lstm_state_size, forget_bias=1.0, state_is_tuple=True, reuse=False)
        if dropout > 0.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
        return cell

    cell_fw = create_rnn_cell()
    cell_bw = create_rnn_cell()
    initial_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)
    bilstm_outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, word_rep2, initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw, dtype=tf.float32)
   
    '''Linear layer
        input: a list of length num_step, containing tensors shaped [batch_size, lstm_size*2]
    '''
    
    logits = []
    with tf.variable_scope("linear") as scope:
        for idx, output in enumerate(bilstm_outputs):
            if idx > 0:
                scope.reuse_variables()
            pred = linear(output,ntags)
            logits.append(pred)
    return adict(
            charinput=char_input,
            wordinput=word_input,
            logits=logits,
            initial_lstm_state_fw=initial_state_fw,
            initial_lstm_state_bw=initial_state_bw,
            final_lstm_state_fw=output_state_fw,
            final_lstm_state_bw=output_state_bw)


def loss_graph(logits, batch_size, num_steps):
    with tf.variable_scope('Loss'):
        targets = tf.placeholder(tf.int64, [batch_size, num_steps], name='targets')
        target_list = [tf.squeeze(x, [1]) for x in tf.split(targets, num_steps, 1)]
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target_list), name='loss')
    return adict(targets=targets, target_list=target_list,loss=loss)


def training_graph(loss, learning_rate=0.01, max_grad_norm=5.0):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    with tf.variable_scope('SGD_Training'):
        learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
        tvars = tf.trainable_variables()
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads,tvars), global_step=global_step)
    
    return adict(learning_rate=learning_rate,
            global_step=global_step,
            global_norm=global_norm,
            train_op=train_op)

    
def main():
    max_word_len = 35
    pretrain_path = "glove.840B.300d.txt"
    train_path = "eng.train"
    validate_path = "eng.dev"
    test_path = "eng.test"
    pretrain_word2id, pretrain_emb = reader.load_pretrain(pretrain_path, [train_path, validate_path, test_path])
    vocabs = reader.build_vocab("eng.train")
    traindata = reader.DataSet(train_path, max_word_len, pretrain_word2id, pretrain_emb, vocabs)
    traindata.load_data()
    validate = reader.DataSet(validate_path, max_word_len, pretrain_word2id, pretrain_emb, vocabs)
    validate.load_data()
    test = reader.DataSet(test_path, max_word_len, pretrain_word2id, pretrain_emb, vocabs)
    test.load_data()
    with tf.Graph().as_default(), tf.Session() as sess:
        train_model = inference_graph(
                char_vocab_size=len(traindata.char2id),
                pretrain_embedding=traindata.pretrain_emb,
                max_word_len=max_word_len,
                ntags=len(traindata.tag2id))

        train_model.update(loss_graph(train_model.logits))
        train_model.update(training_graph(train_model.loss * 20))    
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        lstm_state_fw = sess.run(train_model.initial_lstm_state_fw)
        lstm_state_bw = sess.run(train_model.initial_lstm_state_bw)
        print "Start Training..."
        for epoch in range(1000):
            for i,(xc, xw, y) in enumerate(traindata.iterator(10, 20)):
                logits, target_list, loss, _, lstm_state_fw, lstm_state_bw, gradient_norm, step = sess.run([
                    train_model.logits,
                    train_model.target_list,
                    train_model.loss,
                    train_model.train_op,
                    train_model.final_lstm_state_fw,
                    train_model.final_lstm_state_bw,
                    train_model.global_norm,
                    train_model.global_step],
                    {train_model.charinput : xc,
                    train_model.wordinput : xw,
                    train_model.targets : y,
                    train_model.initial_lstm_state_fw: lstm_state_fw,
                    train_model.initial_lstm_state_bw: lstm_state_bw})
#                target_list=[tf.squeeze(x,[1]) for x in tf.split(y, 20, 1)]
                '''
                for res in zip(logits, target_list):
                    for i in range(10):
                        pred=np.argmax(res[0][i])
                        ans=res[1][i]
                        print pred, ans 
                '''
            print "loss:",loss


if __name__== '__main__':
   main() 
