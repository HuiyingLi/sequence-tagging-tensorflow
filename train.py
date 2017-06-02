import tensorflow as tf
import model
import reader
import numpy as np
import pdb
import subprocess
import time

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
    print "Done with an epoch, evaluating on", validate_model.name, "set..."
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
    config = reader.read_parameters("config.txt")
    max_word_len = config['max_word_len']
    total_epoch = config['total_epoch']
    batch_size = config['batch_size']
    num_steps = config['num_steps']
    crf = config['crf']
    learning_rate = config['learning_rate']
    decay_rate = config['decay_rate']
    max_grad_norm = config['max_grad_norm']
    pretrain_word2id, pretrain_id2word, pretrain_emb = reader.load_pretrain(
        config['pretrain_path'],
        [config['train_path'], config['validate_path'], config['test_path']])
    vocabs = reader.build_vocab(config['train_path'])
    traindata = reader.DataSet(config['train_path'], max_word_len,
                               pretrain_word2id, pretrain_id2word, pretrain_emb, vocabs)
    traindata.load_data()
    validate = reader.DataSet(config['validate_path'], max_word_len,
                              pretrain_word2id, pretrain_id2word, pretrain_emb, vocabs)
    validate.load_data()
    test = reader.DataSet(config['test_path'], max_word_len,
                          pretrain_word2id, pretrain_id2word, pretrain_emb, vocabs)
    test.load_data()
    seq_lens = num_steps * np.ones(batch_size)
    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("Model"):
            train_model = model.inference_graph(
                char_vocab_size=len(traindata.char2id),
                pretrain_embedding=traindata.pretrain_emb,
                max_word_len=max_word_len,
                ntags=len(traindata.tag2id),
                batch_size=batch_size,
                num_steps=num_steps,
                char_emb_size=config['char_emb_size'],
                lstm_state_size=config['lstm_state_size'],
                dropout=config['dropout'],
                filter_sizes=[config['filter_size']],
                nfilters=[config['nfilter']])
            train_model.update(model.loss_graph(train_model.logits, batch_size, num_steps, crf, seq_lens))
            train_model.update(model.training_graph(train_model.loss * num_steps, learning_rate, max_grad_norm))
            #train_model.update(model.training_graph(train_model.loss))
        '''Validate model'''
        with tf.variable_scope("Model", reuse=True):
            validate_model=model.inference_graph(
                char_vocab_size=len(validate.char2id),
                pretrain_embedding=validate.pretrain_emb,
                max_word_len=max_word_len,
                ntags=len(validate.tag2id),
                batch_size=batch_size,
                num_steps=num_steps,
                char_emb_size=config['char_emb_size'],
                lstm_state_size=config['lstm_state_size'],
                dropout=config['dropout'],
                filter_sizes=[config['filter_size']],
                nfilters=[config['nfilter']])
            validate_model.update(model.loss_graph(validate_model.logits, batch_size, num_steps, crf, seq_lens))
            validate_model.update(model.adict(name="validation"))
        '''Test model'''
        with tf.variable_scope("Model", reuse=True):
            test_model=model.inference_graph(
                char_vocab_size=len(test.char2id),
                pretrain_embedding=test.pretrain_emb,
                max_word_len=max_word_len,
                ntags=len(test.tag2id),
                batch_size=batch_size,
                num_steps=num_steps,
                char_emb_size=config['char_emb_size'],
                lstm_state_size=config['lstm_state_size'],
                dropout=config['dropout'],
                filter_sizes=[config['filter_size']],
                nfilters=[config['nfilter']])
            test_model.update(model.loss_graph(test_model.logits, batch_size, num_steps, crf, seq_lens))
            test_model.update(model.adict(name="test"))
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        lstm_state_fw = sess.run(train_model.initial_lstm_state_fw)
        lstm_state_bw = sess.run(train_model.initial_lstm_state_bw)
        print "Start Training..."

        current_best_Fscore = 0.0
        for epoch in range(total_epoch):
            print "epoch", epoch
            start_time = time.time()
            loss = run_epoch(sess, traindata, train_model, lstm_state_fw, lstm_state_bw, batch_size, num_steps)

            if crf:
                Fscore = crf_eval(sess, validate, validate_model, batch_size, num_steps, config['eval_path'], config['eval_script_path'])
            else:
                Fscore = evaluate(sess, validate, validate_model, batch_size, num_steps, config['eval_path'])
            if Fscore>current_best_Fscore:
                current_best_Fscore = Fscore
                if epoch > 10:
                    print "**Results on test set with current best F:", current_best_Fscore
                    crf_eval(sess, test, test_model, batch_size, num_steps, config['eval_path'],
                             config['eval_script_path'])

            new_learning_rate = learning_rate / (1 + decay_rate * (epoch + 1))
            sess.run(train_model.learning_rate.assign(new_learning_rate))
            end_time = time.time()
            print "Epoch training time:", end_time-start_time

if __name__== '__main__':
   main()