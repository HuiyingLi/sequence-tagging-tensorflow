import tensorflow as tf
import model
import reader
import numpy as np
import pdb
import subprocess
def run_epoch(sess, traindata, train_model, lstm_state_fw, lstm_state_bw, batch_size, num_steps):

    for i, (xc, xw, y) in enumerate(traindata.iterator(batch_size, num_steps)):
        logits, target_list, loss, _, lstm_state_fw, lstm_state_bw, gradient_norm, step = sess.run([
            train_model.logits,
            train_model.target_list,
            train_model.loss,
            train_model.train_op,
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
    return loss

def conlleval(path):
    p = subprocess.Popen("cat " + path + "/eval.txt |" + path + "/conlleval", stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p.wait()
    result = output.split("\n")[1]
    print result
    return float(result.split(" ")[-1])

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
    conlleval(tmpdir)


def main():
    max_word_len = 35
    total_epoch = 1000
    batch_size = 10
    num_steps = 30
    eval_path = "/home/huiying/workspace/ner-project/tmp/"
    pretrain_path = "glove.840B.300d.txt"
    train_path = "eng.train"
    validate_path = "eng.dev"
    test_path = "eng.test"
    pretrain_word2id, pretrain_id2word, pretrain_emb = reader.load_pretrain(pretrain_path, [train_path, validate_path, test_path])
    vocabs = reader.build_vocab("eng.train")
    traindata = reader.DataSet(train_path, max_word_len, pretrain_word2id, pretrain_id2word, pretrain_emb, vocabs)
    traindata.load_data()
    validate = reader.DataSet(validate_path, max_word_len, pretrain_word2id, pretrain_id2word, pretrain_emb, vocabs)
    validate.load_data()
    test = reader.DataSet(test_path, max_word_len, pretrain_word2id, pretrain_id2word, pretrain_emb, vocabs)
    test.load_data()
    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("Model"):
            train_model = model.inference_graph(
                char_vocab_size=len(traindata.char2id),
                pretrain_embedding=traindata.pretrain_emb,
                max_word_len=max_word_len,
                ntags=len(traindata.tag2id))
            train_model.update(model.loss_graph(train_model.logits, batch_size, num_steps))
            train_model.update(model.training_graph(train_model.loss * num_steps))
        with tf.variable_scope("Model", reuse=True):
            validate_model=model.inference_graph(
                char_vocab_size=len(validate.char2id),
                pretrain_embedding=validate.pretrain_emb,
                max_word_len=max_word_len,
                ntags=len(validate.tag2id))
            validate_model.update(model.loss_graph(validate_model.logits, batch_size, num_steps))

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        lstm_state_fw = sess.run(train_model.initial_lstm_state_fw)
        lstm_state_bw = sess.run(train_model.initial_lstm_state_bw)
        print "Start Training..."
        for epoch in range(total_epoch):
            print "epoch", epoch
            loss = run_epoch(sess, traindata, train_model, lstm_state_fw, lstm_state_bw, batch_size, num_steps)
            evaluate(sess, validate, validate_model, batch_size, num_steps, eval_path)

if __name__== '__main__':
   main()