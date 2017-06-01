import random
import numpy as np
import pdb
import sys


def read_parameters(config_path):
    config={}
    with open(config_path,"r") as f:
        for l in f:
            l=l.strip()
            if len(l) > 0 and l[0] != "#":
                l=l.split(" ")
                if l[2][0] == '"':
                    l[2] = l[2][1:-1]
                elif l[2] == "true":
                    l[2] = True
                elif l[2] == "false":
                    l[2] = False
                else:
                    l[2] = float(l[2]) if '.' in l[2] else int(l[2])
                config[l[0]] = l[2]
    return config


def build_vocab(train_path):
    worddict = {}
    tagdict = {}
    chardict = {}
    char2id = {}
    id2char = []
    word2id = {}
    id2word = []
    tagset = {}
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            print line
            if len(line) == 0:
                continue
            line = line.split(" ")
            worddict[line[0]] = 1
            for c in line[0]:
                chardict[c] = 1
            tagdict[line[3]] = 1
        chardict[' '] = 1
        zc = zip(chardict, range(len(chardict)))
        char2id = dict(zc)
        id2char = zip(*zc)[0]
        zw = zip(worddict, range(len(worddict)))
        word2id = dict(zw)
        id2word = zip(*zw)[0]
        zt = zip(tagdict, range(len(tagdict)))
        tag2id = dict(zt)
        id2tag = zip(*zt)[0]
    return char2id, id2char, tag2id, id2tag


'''
    training set vocabulary is passed to this function, because I don't want
    to load words that are not in the training dataset. System is running out
    of memory.
'''
def load_pretrain(path, datasets):
    '''Collect vocab from train, dev and test. Cannot load all embedding because of memory issue'''
    big_vocab= set()   #OOV presented as ' '
    for dataset in datasets:
        with open(dataset, "r") as f:
            for l in f:
                l = l.strip().split(" ")[0]
                big_vocab.add(l.lower())

    print "Loading pretrained embeddings:", path

    with open(path, "r") as f:
        vocab = []
        emb = []
        i = 0

        for l in f:
            l = l.strip().split(" ")
            if l[0] not in big_vocab:
                continue
            vocab.append(l[0])
            emb.append(np.array([float(x) for x in l[1:]]))
            i += 1
            if i%1000 == 0:
                sys.stdout.write(".")
        f.close()


    print "\ndone reading pretained vectors! "
    print len(vocab), "words can find their embeddings in the pretrain."
    vdict = set(vocab)
    '''now add random embeddings'''

    def generate_random_embedding(emb_size):
        scale = np.sqrt(3.0 / emb_size)
        return np.random.uniform(-scale, scale, emb_size)

    for w in big_vocab:
        if w not in vdict:
            vocab.append(w)
            emb.append(generate_random_embedding(len(emb[0])))
    pretrain_word2id = dict(zip(vocab, range(len(vocab))))
    pretrain_id2word = dict(zip(range(len(vocab)), vocab))
    pretrain_emb = np.asarray(emb)
    return pretrain_word2id, pretrain_id2word, pretrain_emb


class DataSet:
    def __init__(self, data_path, max_word_len, pretrain_word2id, pretrain_id2word, pretrain_emb, vocabs, keep_origin=False):
        #self.char2id,self.charlist,self.max_word_len,self.word2id,self.wordlist=load_vocab(vocab_path)
        self.data_path = data_path
        self.max_word_len = max_word_len
        self.pretrain_word2id = pretrain_word2id
        self.pretrain_id2word = pretrain_id2word
        self.pretrain_emb = pretrain_emb
        self.pretrain_dim = len(pretrain_emb[0])
        self.char2id, self.id2char, self.tag2id, self.id2tag = vocabs
        self.sents = []
        self.words = []
        self.chars = []
        self.tags = []
        self.keep_origin=keep_origin
        if keep_origin:
            self.lines = []


    def pad_word(self, word, padding_char=" "):
        if len(word) > self.max_word_len-2:
            word = word[:self.max_word_len-2]
        n_padding = self.max_word_len-len(word)-1
        new_word = padding_char + word + ''.join([padding_char]*n_padding)
        return new_word

    #def inspect(self, m):

    def load_data(self):
        sent = []
        with open(self.data_path, "r") as f:
            for l in f:
                l = l.strip()
                if l == "":
                    self.sents.append(sent)
                    sent = []
                    continue
                sent.append(l.split(" "))

        #random.shuffle(self.sents)
        for sent in self.sents:
            for p in sent:
                if self.keep_origin:
                    self.lines.append(" ".join(p[:3]))
                self.words.append(p[0].lower())  #so that when looking for embeddings, words are uncased
                self.tags.append(p[3])
                self.chars.append(list(self.pad_word(p[0])))



        self.word_array = [self.pretrain_word2id[w] for w in self.words]
        self.tag_array = [self.tag2id[w] for w in self.tags]
        self.char_array = []
        for i in range(len(self.chars)):
            self.char_array.append([self.char2id[c] if c in self.char2id else self.char2id[' '] for c in self.chars[i]])
 
    def iterator(self, batch_size, num_steps):
        char_data = np.array(self.char_array, dtype=np.int32)
        word_data = np.array(self.word_array, dtype=np.int32)
        tag_data = np.array(self.tag_array, dtype=np.int32)
        if self.keep_origin:
            line_data = np.array(self.lines, dtype=object)
        datalen = len(tag_data)
        nbatch = datalen//batch_size
        xcarr = np.zeros((batch_size, nbatch, self.max_word_len), dtype=np.int32)
        xwarr = np.zeros((batch_size, nbatch), dtype=np.int32)
        if self.keep_origin:
            xlarr = np.empty((batch_size, nbatch), dtype=object)
        yarr = np.zeros((batch_size, nbatch), dtype=np.int32)
        for i in range(batch_size):
            xcarr[i] = char_data[nbatch*i : nbatch*(i+1), :]
            xwarr[i] = word_data[nbatch*i : nbatch*(i+1)]
            if self.keep_origin:
                xlarr[i] = line_data[nbatch*i : nbatch*(i+1)]
            yarr[i] = tag_data[nbatch*i : nbatch*(i+1)]

        nepoch = nbatch//num_steps
        #pdb.set_trace()
        for i in range(nepoch):
            xc = xcarr[:, i*num_steps:(i+1)*num_steps, :]   #x.shape=batch_size x num_steps x max_word_len
            xw = xwarr[:, i*num_steps:(i+1)*num_steps]
            if self.keep_origin:
                xl = xlarr[:, i*num_steps:(i+1)*num_steps]
            y = yarr[:, i*num_steps:(i+1)*num_steps]
            if self.keep_origin:
                yield (xc, xw, xl, y)
            else:
                yield (xc, xw, y)


def main():
    pretrain_path = "glove.840B.300d.txt"
    train_path = "eng.train"
    validate_path = "eng.dev"
    test_path = "eng.test"
    pretrain_word2id, pretrain_id2word, pretrain_emb = load_pretrain(pretrain_path, [train_path, validate_path, test_path])
    vocabs = build_vocab("eng.train")
    train = DataSet(train_path, 35, pretrain_word2id, pretrain_id2word, pretrain_emb, vocabs)
    train.load_data()
    validate = DataSet(validate_path, 35, pretrain_word2id, pretrain_id2word, pretrain_emb, vocabs, keep_origin=True)
    validate.load_data()
    test = DataSet(test_path, 35, pretrain_word2id, pretrain_id2word, pretrain_emb, vocabs)

    count = 0
    batch_size = 10
    num_steps = 30
    for step, (xc, xw, xl, y) in enumerate(validate.iterator(10, 30)):
        count += 1
        pdb.set_trace()
        print count

if __name__ == '__main__':
    main()
