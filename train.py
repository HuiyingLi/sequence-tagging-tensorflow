import tensorflow as tf
import model
import reader
import numpy as np
import pdb
import subprocess
import time
tf.app.flags.DEFINE_string('pretrain_path', "embeddings/glove.6B.100d.txt", "")
tf.app.flags.DEFINE_string('train_path', "eng.train.bioes", "")
tf.app.flags.DEFINE_string('validate_path', "eng.dev.bioes", "")
tf.app.flags.DEFINE_string('test_path', "eng.test.bioes", "")
tf.app.flags.DEFINE_string('eval_path', "/home/huiying/workspace/ner-project/tmp/", "")
tf.app.flags.DEFINE_string('eval_script_path', "/home/huiying/workspace/ner-project/tmp/bioeseval.py", "")
tf.app.flags.DEFINE_string('checkpoint_path', "/home/huiying/workspace/ner-project/tmp/chkpnt", "")


tf.app.flags.DEFINE_integer('batch_size', 1, 'number of sentences in a batch')
tf.app.flags.DEFINE_integer('num_steps', 30, 'number of steps unrolled in RNN')
tf.app.flags.DEFINE_integer('max_word_len', 35, 'maximum number of characters contained in a word')
tf.app.flags.DEFINE_integer('lstm_state_size', 200, 'number of lstm cells')
tf.app.flags.DEFINE_integer('num_rnn_layers', 1, 'number of BiLSTM layers')
tf.app.flags.DEFINE_integer('nfilter', 30, 'number of filters in characterlevel cnn')
tf.app.flags.DEFINE_integer('filter_size', 3, 'the windows size each filter scans')
tf.app.flags.DEFINE_integer('char_emb_size', 30, 'the character embedding dimension')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout probability')
#tf.app.flags.DEFINE_boolean('batch_normalization', True, 'if add batch normalization layer')

tf.app.flags.DEFINE_integer('total_epoch', 200, 'total number of epochs to run')
tf.app.flags.DEFINE_float('max_grad_norm', 5.0, 'gradient clipping')
tf.app.flags.DEFINE_float('learning_rate', 0.015, 'initial learning rate')
tf.app.flags.DEFINE_float('decay_rate', 0.05, 'the decay rate of learning rate')
tf.app.flags.DEFINE_boolean('crf', True, 'if crf layer is used')

FLAGS = tf.app.flags.FLAGS

def run_epoch(sess, traindata, train_model, lstm_state_fw, lstm_state_bw, batch_size, num_steps):
    avg_loss = 0
    for i, (xc, xw, y) in enumerate(traindata.iterator(batch_size, num_steps)):
        logits, transition_params, loss, _, learning_rate, lstm_state_fw, lstm_state_bw, gradient_norm, step = sess.run([
            train_model.logits,
            train_model.transition_params,
            train_model.loss,
            train_model.train_op,
            train_model.learning_rate,
            train_model.final_lstm_state_fw,
            train_model.final_lstm_state_bw,
            train_model.global_norm,
            train_model.global_step
        ],{
            train_model.charinput: xc,
            train_model.wordinput: xw,
            train_model.targets: y,
            train_model.initial_lstm_state_fw: lstm_state_fw,
            train_model.initial_lstm_state_bw: lstm_state_bw})
        avg_loss += loss
    avg_loss /= (i+1)
    print "learning rate:", learning_rate, "average training loss:", avg_loss
    return avg_loss


def externaleval(evaldir_path, script_path):
    """
    :Use the script provided by CoNLL 2013 shared task to evaluate
    """
    cmd = "cat " + evaldir_path + "/eval.txt |" + script_path
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p.wait()
    result = output.split("\n")[1]
    print result
    return float(result.split(" ")[-1])


def crf_eval(sess, validate_data, validate_model, batch_size, num_steps, tmpdir, eval_script_path):
    lstm_state_fw = sess.run(validate_model.initial_lstm_state_fw)
    lstm_state_bw = sess.run(validate_model.initial_lstm_state_bw)
    evalres = []
    predlist = []
    goldlist = []
    #print "Done with an epoch, evaluating on", validate_model.name, "set..."
    for i, (xc, xw, y) in enumerate(validate_data.iterator(batch_size, num_steps)):
        loss, stackedlogits, transition_params, lstm_state_fw, lstm_state_bw = sess.run([
            validate_model.loss,
            validate_model.stackedlogits,
            validate_model.transition_params,
            validate_model.final_lstm_state_fw,
            validate_model.final_lstm_state_bw
        ], {
            validate_model.charinput: xc,
            validate_model.wordinput: xw,
            validate_model.targets: y,
            validate_model.initial_lstm_state_fw: lstm_state_fw,
            validate_model.initial_lstm_state_bw: lstm_state_bw})
        ''' Collect predictions.'''
        viterbi_seqs = []
        #pdb.set_trace()

        for sent, logit in zip(xw, stackedlogits):
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, transition_params)
            viterbi_seqs += [viterbi_seq]


        '''Prepares file for evluation'''
        for i in range(batch_size):
            for j in range(num_steps):
                pred = viterbi_seqs[i][j]
                ans = y[i][j]
                line = "random random random"
                predtag = validate_data.id2tag[pred]
                anstag = validate_data.id2tag[ans]
                evalline = line + " " + anstag + " " + predtag
                evalres.append(evalline)

        predlist.extend(viterbi_seqs)
        goldlist.extend(y)

    fp = open(tmpdir + "eval.txt", "w")
    for l in evalres:
        fp.write(l + "\n")
    fp.close()
    return externaleval(tmpdir, eval_script_path)

def evaluate(sess, validate_data, validate_model, batch_size, num_steps, tmpdir):
    lstm_state_fw = sess.run(validate_model.initial_lstm_state_fw)
    lstm_state_bw = sess.run(validate_model.initial_lstm_state_bw)
    evalres = []

    print "Done with an epoch, evaluating on validation set..."
    for i, (xc, xw, y) in enumerate(validate_data.iterator(batch_size,num_steps)):
        loss, logits, target_list, lstm_state_fw, lstm_state_bw = sess.run([
            validate_model.loss,
            validate_model.logits,
            validate_model.target_list,
            validate_model.final_lstm_state_fw,
            validate_model.final_lstm_state_bw
        ],{
            validate_model.charinput: xc,
            validate_model.wordinput: xw,
            validate_model.targets: y,
            validate_model.initial_lstm_state_fw: lstm_state_fw,
            validate_model.initial_lstm_state_bw: lstm_state_bw})
        ''' Collect predictions'''
        #pdb.set_trace()
        #tmpxl = [tf.squeeze(x, [1]) for x in tf.split(xl, num_steps, 1)]
        #xl2 = sess.run(tmpxl)
        for i in range(batch_size):
            for j in range(num_steps):
                pred = np.argmax(logits[j][i])
                ans = target_list[j][i]
                line = "random random random"
                predtag = validate_data.id2tag[pred]
                anstag = validate_data.id2tag[ans]
                evalline = line + " " + anstag + " " + predtag
                evalres.append(evalline)

    fp = open(tmpdir + "eval.txt", "w")
    for l in evalres:
        fp.write(l+"\n")
    fp.close()
    return externaleval(tmpdir)


def main():
    pretrain_word2id, pretrain_id2word, pretrain_emb = reader.load_pretrain(
        FLAGS.pretrain_path,
        [FLAGS.train_path, FLAGS.validate_path, FLAGS.test_path])
    vocabs = reader.build_vocab(FLAGS.train_path)
    traindata = reader.DataSet(FLAGS.train_path, FLAGS.max_word_len,
                               pretrain_word2id, pretrain_id2word, pretrain_emb, vocabs)
    traindata.load_data()
    validate = reader.DataSet(FLAGS.validate_path, FLAGS.max_word_len,
                              pretrain_word2id, pretrain_id2word, pretrain_emb, vocabs)
    validate.load_data()
    test = reader.DataSet(FLAGS.test_path, FLAGS.max_word_len,
                          pretrain_word2id, pretrain_id2word, pretrain_emb, vocabs)
    test.load_data()
    seq_lens = FLAGS.num_steps * np.ones(FLAGS.batch_size)

    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("Model"):
            train_model = model.inference_graph(
                char_vocab_size=len(traindata.char2id),
                pretrain_embedding=traindata.pretrain_emb,
                max_word_len=FLAGS.max_word_len,
                ntags=len(traindata.tag2id),
                batch_size=FLAGS.batch_size,
                num_steps=FLAGS.num_steps,
                char_emb_size=FLAGS.char_emb_size,
                lstm_state_size=FLAGS.lstm_state_size,
                num_rnn_layers=FLAGS.num_rnn_layers,
                dropout=FLAGS.dropout,
                filter_sizes=[FLAGS.filter_size],
                nfilters=[FLAGS.nfilter])
            train_model.update(model.loss_graph(train_model.logits, FLAGS.batch_size, FLAGS.num_steps, FLAGS.crf, seq_lens))
            train_model.update(model.training_graph(train_model.loss * FLAGS.num_steps, FLAGS.learning_rate, FLAGS.max_grad_norm))
            #train_model.update(model.training_graph(train_model.loss))
        saver = tf.train.Saver()

        '''Validate model'''
        with tf.variable_scope("Model", reuse=True):
            validate_model=model.inference_graph(
                char_vocab_size=len(validate.char2id),
                pretrain_embedding=validate.pretrain_emb,
                max_word_len=FLAGS.max_word_len,
                ntags=len(validate.tag2id),
                batch_size=FLAGS.batch_size,
                num_steps=FLAGS.num_steps,
                char_emb_size=FLAGS.char_emb_size,
                lstm_state_size=FLAGS.lstm_state_size,
                num_rnn_layers=FLAGS.num_rnn_layers,
                dropout=0,  #No dropout when testing!
                filter_sizes=[FLAGS.filter_size],
                nfilters=[FLAGS.nfilter])
            validate_model.update(model.loss_graph(validate_model.logits, FLAGS.batch_size, FLAGS.num_steps, FLAGS.crf, seq_lens))
            validate_model.update(model.adict(name="validation"))
        '''Test model'''
        with tf.variable_scope("Model", reuse=True):
            test_model=model.inference_graph(
                char_vocab_size=len(test.char2id),
                pretrain_embedding=test.pretrain_emb,
                max_word_len=FLAGS.max_word_len,
                ntags=len(test.tag2id),
                batch_size=FLAGS.batch_size,
                num_steps=FLAGS.num_steps,
                char_emb_size=FLAGS.char_emb_size,
                lstm_state_size=FLAGS.lstm_state_size,
                num_rnn_layers=FLAGS.num_rnn_layers,
                dropout=0,
                filter_sizes=[FLAGS.filter_size],
                nfilters=[FLAGS.nfilter])
            test_model.update(model.loss_graph(test_model.logits, FLAGS.batch_size, FLAGS.num_steps, FLAGS.crf, seq_lens))
            test_model.update(model.adict(name="test"))
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        lstm_state_fw = sess.run(train_model.initial_lstm_state_fw)
        lstm_state_bw = sess.run(train_model.initial_lstm_state_bw)
        print "Start Training..."


        current_best_Fscore = 0.0
        for epoch in range(FLAGS.total_epoch):
            print "epoch", epoch
            start_time = time.time()
            loss = run_epoch(sess, traindata, train_model, lstm_state_fw, lstm_state_bw, FLAGS.batch_size, FLAGS.num_steps)

            if FLAGS.crf:
                Fscore = crf_eval(sess, validate, validate_model, FLAGS.batch_size, FLAGS.num_steps, FLAGS.eval_path, FLAGS.eval_script_path)
            else:
                Fscore = evaluate(sess, validate, validate_model, FLAGS.batch_size, FLAGS.num_steps, FLAGS.eval_path)
            if Fscore > current_best_Fscore:
                current_best_Fscore = Fscore
                print "**Results on test set with current best F:", current_best_Fscore
                crf_eval(sess, test, test_model, FLAGS.batch_size, FLAGS.num_steps, FLAGS.eval_path,
                         FLAGS.eval_script_path)
                saver.save(sess, FLAGS.checkpoint_path)
                print "Model saved!"

            new_learning_rate = FLAGS.learning_rate / (1 + FLAGS.decay_rate * (epoch + 1))
            sess.run(train_model.learning_rate.assign(new_learning_rate))
            end_time = time.time()
            print "Epoch training time:", end_time - start_time

if __name__ == '__main__':
   main()