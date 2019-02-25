import argparse
import math
import sys
import time
import os
from functools import reduce
import json
import numpy as np
# from multiprocessing import Pool, Value, Array
import torch
import torch.optim as optim
import torch.nn as nn


STRUCTURES = {'无':0,
 '左右结构': 1,
 '单一结构': 2,
 '上下结构': 3,
 '半包围结构': 4,
 '品字结构': 5,
 '上中下结构': 6,
 '左中右结构': 7,
 '全包围结构': 8,
 '独体': 9,
 '嵌套结构': 10,
 '下包围结构': 11,
 '右上包围结构': 12,
 '左上包围结构': 13,
 '单体结构': 14,
 '上包围结构': 15,
 '左下包围结构': 16,
 '在右结构': 17,
 '左包围结构': 18,
 '独体字': 19,
 '上下下结构': 20,
 '半包围': 21}

class Characters:
    def __init__(self):
        self.corpus = {}

    def __contains__(self, key):
        return key in self.corpus

    def __getitem__(self, i):
        return self.corpus[i]

class CharacterItem:        #字
    def __init__(self, char, strokes = ''):
        self.char = char
        self.strokes = strokes
        self.structures = 0
        self.redicals = ''

class VocabItem:            #词
    def __init__(self, word):
        self.word = word
        self.is_name_entity = False
        self.count = 0          #词出现次数
        self.characters = []
        self.path = None # Path (list of indices) from the root to the word (leaf)
        self.code = None # Huffman encoding
        self.n_gram = []
        self.component_sequence = []
        self.component_sequence_nodivision = []

    # def compute_components(self):
    #     char_list,redical_list,components_1_list,components_2_list = [],[],[],[]
    #     if not self.characters:
    #         self.components = []
    #         return
    #     for char in self.characters:
    #         char_self, redical,components_1,components_2 = char['reuse_all_components'].split('\t')
    #         char_list += char_self
    #         redical_list += redical
    #         components_1_list += components_1
    #         components_2_list += components_2
    #     self.components = [char_list,redical_list,components_1_list,components_2_list]

    def compute_strokes_sequences(self):
        self.strokes_sequences = ''
        if not self.characters:
            return
        for char in self.characters:
            self.strokes_sequences += char['stroke_numbers']

class Vocab:                #词库
    def __init__(self, fi, min_count,char_corpus, name_entity_dict, mig=3,mag=12):
        self.char_corpus = char_corpus
        self.name_entity_dict = name_entity_dict
        self.n_gram = {}
        self.name_entity = []
        self.components2index = {}
        self.index2components = {}
        self.min_gram = mig
        self.max_gram = mag
        self.unk_word = {}
        vocab_items = []
        vocab_hash = {}         #词 => index
        word_count = 0
        fi_file = open(fi, 'r')
        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        for token in ['<bol>', '<eol>']:
            vocab_hash[token] = len(vocab_items)
            vocab_items.append(VocabItem(token))

        for line in fi_file:
            tokens = line.split()
            for token in tokens:
                if token not in vocab_hash:
                    vocab_hash[token] = len(vocab_items)
                    vocab_items.append(VocabItem(token))

                #assert vocab_items[vocab_hash[token]].word == token, 'Wrong vocab_hash index'
                vocab_items[vocab_hash[token]].count += 1
                word_count += 1
            
                if word_count % 10000 == 0:
                    sys.stdout.write("\rReading word %d" % word_count)
                    sys.stdout.flush()

            # Add special tokens <bol> (beginning of line) and <eol> (end of line)
            vocab_items[vocab_hash['<bol>']].count += 1
            vocab_items[vocab_hash['<eol>']].count += 1
            word_count += 2

        self.bytes = fi_file.tell()
        self.vocab_items = vocab_items         # List of VocabItem objects
        self.vocab_hash = vocab_hash           # Mapping from each token to its index in vocab
        self.word_count = word_count           # Total number of words in train file

        # Add special token <unk> (unknown),
        # merge words occurring less than min_count into <unk>, and
        # sort vocab in descending order by frequency in train file
        self.__sort(min_count)
        self.init_attention_matrix = torch.zeros((len(self.vocab_hash), 5))
        fi_file.close()
        #assert self.word_count == sum([t.count for t in self.vocab_items]), 'word_count and sum of t.count do not agree'
        print ('Total words in training file: %d' % self.word_count)
        print ('Total bytes in training file: %d' % self.bytes)
        print ('Vocab size: %d' % len(self))

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

    def __sort(self, min_count):
        tmp = []
        tmp.append(VocabItem('<unk>'))
        unk_hash = 0
        unk_char = 0
        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                for char in token.word:
                    if char in self.char_corpus:
                        token.characters.append(self.char_corpus[char])
                    else:
                        unk_char += 1
                if token.word in self.name_entity_dict:
                    self.name_entity.append(token.word)
                    token.is_name_entity = True
                tmp.append(token)
        print ('\nname_entity:%d'%len(self.name_entity))
        print ('unchar:',unk_char)
        tmp.sort(key=lambda token : token.count, reverse=True)
        # Update vocab_hash
        vocab_hash = {}
        self.index2word = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i
            self.index2word[i] = token

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash

        print ('')
        print ('Unknown vocab size:', count_unk)

    def compute_word_strokes(self):
        for i in self.vocab_items:
            i.compute_strokes_sequences()
        self.get_n_grams()

    def compute_components(self):
        self.max_char_length = 0
        self.max_redical_length = 0
        self.max_com1_length = 0
        self.max_com2_length = 0
        self.maxcount = 100
        self.decount = 0
        self.using_name_entity = True
        self.entity_count = 0
        for index,i in enumerate(self.vocab_items):
            if not i.characters:
                if i.word not in self.unk_word:
                    self.unk_word[i.word] = len(self.unk_word)
                i.components = [[],[],[],[]]
                i.components_nodivision = []
                self.init_attention_matrix[index] = - 1/torch.zeros(1)
                continue
            if self.maxcount > 0:
                if i.count >= self.maxcount:
                    self.decount += 1
                    i.components = [[], [], [], []]
                    i.components_nodivision = []
                    self.init_attention_matrix[index] = - 1 / torch.zeros(1)
                    continue
            if self.using_name_entity:
                if i.is_name_entity:
                    self.entity_count += 1
                    # print ('using name_entity filter')
                    i.components = [[], [], [], []]
                    i.components_nodivision = []
                    self.init_attention_matrix[index] = - 1 / torch.zeros(1)
                    continue
            char_list, redical_list, components_1_list, components_2_list = [], [], [], []
            for char in i.characters:
                char_self, redical, components_1, components_2 = char['reuse_all_components'].split('\t')
                char_list += char_self
                redical_list += redical
                components_1_list += components_1
                components_2_list += components_2
            self.max_char_length = max(self.max_char_length,len(char_list))
            self.max_redical_length = max(self.max_redical_length,len(redical_list))
            self.max_com1_length = max(self.max_com1_length,len(components_1_list))
            self.max_com2_length = max(self.max_com2_length,len(components_2_list))
            i.components = [char_list, redical_list, components_1_list, components_2_list]
            for ii,jj in enumerate(i.components):
                if not jj : self.init_attention_matrix[index][ii+1] = - 1 / torch.zeros(1)
            i.components_nodivision = char_list + redical_list + components_1_list + components_2_list
        if self.using_name_entity:
            print ('using name entity total:%d'%self.entity_count)
        if self.maxcount > 0:
            print ('using count %d total: %d' %(self.maxcount,self.decount))
        self.components2index['pad'] = 0
        self.index2components[0] = 'pad'
        for i in self.vocab_items:
            for j in i.components:
                for k in j:
                    if k not in self.components2index:
                        self.components2index[k] = len(self.components2index)
                        self.index2components[self.components2index[k]] = k
        self.init_attention_matrix[:, 0] = torch.log(torch.FloatTensor([4]))
        print ('components:%d' % len(self.components2index))
        self.init_mask_char = torch.zeros((len(self.vocab_hash), self.max_char_length))
        self.init_mask_redical = torch.zeros((len(self.vocab_hash), self.max_redical_length))
        self.init_mask_com1 = torch.zeros((len(self.vocab_hash), self.max_com1_length))
        self.init_mask_com2 = torch.zeros((len(self.vocab_hash), self.max_com2_length))
        for index,i in enumerate(self.vocab_items):
            init_char = [0] * self.max_char_length
            init_redical = [0] * self.max_redical_length
            init_com1 = [0] * self.max_com1_length
            init_com2 = [0] * self.max_com2_length
            for char_i,char in enumerate(i.components[0]):
                init_char[char_i] = self.components2index[char]
            for redical_i,redical in enumerate(i.components[1]):
                init_redical[redical_i] = self.components2index[redical]
            for com1_i,com1 in enumerate(i.components[2]):
                init_com1[com1_i] = self.components2index[com1]
            for com2_i,com2 in enumerate(i.components[3]):
                init_com2[com2_i] = self.components2index[com2]
            if i.components[0]:
                self.init_mask_char[index][len(i.components[0]):] = -1 / torch.zeros(1)
            if i.components[1]:
                self.init_mask_redical[index][len(i.components[1]):] = -1 / torch.zeros(1)
            if i.components[2]:
                self.init_mask_com1[index][len(i.components[2]):] = -1 / torch.zeros(1)
            if i.components[3]:
                self.init_mask_com2[index][len(i.components[3]):] = -1 / torch.zeros(1)
            i.component_sequence = [init_char,init_redical,init_com1,init_com2]
        return [self.max_char_length,self.max_redical_length,self.max_com1_length,self.max_com2_length,self.init_mask_char,self.init_mask_redical,self.init_mask_com1,self.init_mask_com2]

    def get_n_grams(self):
        count = 0
        for i in self.vocab_items:
            count += 1
            # if count == 37:
            #     s = 0
            if not i.strokes_sequences or len(i.strokes_sequences)<self.min_gram:
                continue
            else:
                for n in range(self.min_gram,1+min(self.max_gram,len(i.strokes_sequences))):
                    for k in range(len(i.strokes_sequences)-n+1):
                        if i.strokes_sequences[k:k+n] not in self.n_gram:
                            self.n_gram[i.strokes_sequences[k:k+n]] = len(self.n_gram)
                        i.n_gram.append(self.n_gram[i.strokes_sequences[k:k+n]])
        print ('n_gram:',len(self.n_gram))
        for i in self.vocab_items:
            if not i.n_gram:
                self.n_gram[i] = len(self.n_gram)
                i.n_gram.append(self.n_gram[i])

    def indices(self, tokens):
        return [self.vocab_hash[token.decode('utf-8')] if token.decode('utf-8') in self else self.vocab_hash['<unk>'] for token in tokens]

class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """

    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab]) # Normalizing constant

        table_size = int(1e8) # Length of the unigram table
        table = []

        print ('Filling unigram table')
        for j,unigram in enumerate(vocab):
            p = round(float(math.pow(unigram.count, power))/norm * table_size)
            table += [j] * p
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

class SkipGramModel(nn.Module):
    def __init__(self, component_size, word_size, dim, init_attention_matrix,para_list,attention_mode=1):
        super(SkipGramModel, self).__init__()
        self.emb_size = dim
        self.component_size = component_size
        self.word_size = word_size
        self.atten_char = nn.Embedding(word_size,para_list[0])
        self.atten_redical = nn.Embedding(word_size,para_list[1])
        self.atten_com1 = nn.Embedding(word_size,para_list[2])
        self.atten_com2 = nn.Embedding(word_size,para_list[3])
        self.attention_mode = attention_mode
        if attention_mode == 1:
            self.atten_in_char = nn.Linear(2*dim, 1, bias=False)
            self.atten_in_redical = nn.Linear(2*dim, 1, bias=False)
            self.atten_in_com1 = nn.Linear(2*dim, 1, bias=False)
            self.atten_in_com2 = nn.Linear(2*dim, 1, bias=False)

        elif attention_mode == 2:
            self.atten_in_char = nn.Linear(dim, dim, bias=False)
            self.atten_in_redical = nn.Linear(dim, dim, bias=False)
            self.atten_in_com1 = nn.Linear(dim, dim, bias=False)
            self.atten_in_com2 = nn.Linear(dim, dim, bias=False)

        self.mask_char = nn.Embedding(word_size,para_list[4].shape[1])
        self.mask_redical = nn.Embedding(word_size,para_list[5].shape[1])
        self.mask_com1 = nn.Embedding(word_size,para_list[6].shape[1])
        self.mask_com2 = nn.Embedding(word_size,para_list[7].shape[1])

        self.atten_layers = nn.Embedding(word_size,5)
        self.u_embeddings = nn.Embedding(component_size,dim,padding_idx=0)
        self.word_embeddings = nn.Embedding(word_size,dim,sparse=True)
        self.v_embeddings = nn.Embedding(word_size,dim,sparse=True)
        self.m = nn.Sigmoid()
        self.init_emb(init_attention_matrix,para_list)

    def init_emb(self,init_attention_matrix,para_list):
        self.mask_char.weight.data = para_list[4]
        self.mask_char.weight.requires_grad = False
        self.mask_redical.weight.data = para_list[5]
        self.mask_redical.weight.requires_grad = False
        self.mask_com1.weight.data = para_list[6]
        self.mask_com1.weight.requires_grad = False
        self.mask_com2.weight.data = para_list[7]
        self.mask_com2.weight.requires_grad = False
        initrange = 0.5 / self.emb_size
        self.word_embeddings.weight.data.uniform_(-initrange,initrange)
        self.u_embeddings.weight.data[1:].uniform_(-initrange, initrange)
        self.atten_layers.weight.data = init_attention_matrix

        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, word_in,component_in, word_out):

        char_in = torch.cuda.LongTensor(component_in[0])
        redical_in = torch.cuda.LongTensor(component_in[1])
        com1_in = torch.cuda.LongTensor(component_in[2])
        com2_in = torch.cuda.LongTensor(component_in[3])

        emb_uword = self.word_embeddings(word_in)
        emb_char = self.u_embeddings(char_in)
        emb_redical = self.u_embeddings(redical_in)
        emb_com1 = self.u_embeddings(com1_in)
        emb_com2 = self.u_embeddings(com2_in)

        attention = torch.softmax(self.atten_layers(word_in),dim=-1).unsqueeze(1)
        mask_char = self.mask_char(word_in)
        mask_redical = self.mask_redical(word_in)
        mask_com1 = self.mask_com1(word_in)
        mask_com2 = self.mask_com2(word_in)
        if self.attention_mode == 1:
            atten_char = torch.softmax(self.atten_in_char(torch.cat((emb_char,emb_uword.unsqueeze(1).expand(-1,emb_char.shape[1],-1)),-1)).squeeze(2)+mask_char,dim=1)
            atten_redical = torch.softmax(self.atten_in_redical(torch.cat((emb_redical,emb_uword.unsqueeze(1).expand(-1,emb_redical.shape[1],-1)),-1)).squeeze(2)+mask_redical,dim=1)
            atten_com1 = torch.softmax(self.atten_in_com1(torch.cat((emb_com1,emb_uword.unsqueeze(1).expand(-1,emb_com1.shape[1],-1)),-1)).squeeze(2)+mask_com1,dim=1)
            atten_com2 = torch.softmax(self.atten_in_com2(torch.cat((emb_com2,emb_uword.unsqueeze(1).expand(-1,emb_com2.shape[1],-1)),-1)).squeeze(2)+mask_com2,dim=1)
        elif self.attention_mode == 2:
            atten_char = torch.softmax(torch.bmm(self.atten_in_char(emb_char),emb_uword.unsqueeze(2)).squeeze(2)+mask_char,dim=1)
            atten_redical = torch.softmax(torch.bmm(self.atten_in_redical(emb_redical), emb_uword.unsqueeze(2)).squeeze(2) + mask_redical, dim=1)
            atten_com1 = torch.softmax(torch.bmm(self.atten_in_com1(emb_com1), emb_uword.unsqueeze(2)).squeeze(2) + mask_com1, dim=1)
            atten_com2 = torch.softmax(torch.bmm(self.atten_in_com2(emb_com2), emb_uword.unsqueeze(2)).squeeze(2) + mask_com2, dim=1)

        # atten_char = torch.softmax(torch.bmm(emb_char, emb_uword.unsqueeze(2)).squeeze(2)+mask_char, dim=1)
        # atten_redical = torch.softmax(torch.bmm(emb_redical, emb_uword.unsqueeze(2)).squeeze(2)+mask_redical, dim=1)
        # atten_com1 = torch.softmax(torch.bmm(emb_com1, emb_uword.unsqueeze(2)).squeeze(2)+mask_com1, dim=1)
        # atten_com2 = torch.softmax(torch.bmm(emb_com2, emb_uword.unsqueeze(2)).squeeze(2)+mask_com2, dim=1)

        weighted_char = torch.bmm(atten_char.unsqueeze(1),emb_char).squeeze(1)
        weighted_redical = torch.bmm(atten_redical.unsqueeze(1),emb_redical).squeeze(1)
        weighted_com1 = torch.bmm(atten_com1.unsqueeze(1),emb_com1).squeeze(1)
        weighted_com2 = torch.bmm(atten_com2.unsqueeze(1),emb_com2).squeeze(1)

        emb_all = torch.stack((emb_uword,weighted_char,weighted_redical,weighted_com1,weighted_com2),1)
        emb_vword = self.v_embeddings(word_out)
        emb_mixin = torch.bmm(attention,emb_all).squeeze(1)
        score = torch.mul(emb_mixin, emb_vword)
        score = torch.sum(score, dim=-1)
        score = self.m(score)
        return score

    def save_attention(self, file_name):
        embedding = self.m(self.atten_layers.weight).data.cpu().numpy()
        with open(file_name,'w') as fout:
            for i in range(embedding.size):
                e = [embedding[i][0]]
                e = ' '.join(map(lambda x: str(x), e))
                fout.write('%s\n' % (e))
        fout.close()

class BatchPair(object):
    def __init__(self, fi):
        self.fi = open(fi,'rb')
        self.word_count = 0
        self.last_word_count = 0
        self.word_in = []
        self.word_out = []
        self.label_pool = []
        self.negtive_pool = []
        self.get_data_time = 0

    def batch_pair(self):
        def get_offset(iterable):
            offset = [0]
            a = 0
            for i in iterable[:-1]:
                a += i
                offset.append(a)
            return offset
        while len(self.label_pool) < batch and self.fi.tell() < vocab.bytes:
            line = self.fi.readline().strip()
            sent = vocab.indices(['<bol>'.encode('utf-8')] + line.split() + ['<eol>'.encode('utf-8')])
            for sent_pos, token in enumerate(sent):
                current_win = np.random.randint(low=1, high=win + 1)
                context_start = max(sent_pos - current_win, 0)
                context_end = min(sent_pos + current_win + 1, len(sent))
                context = sent[context_start:sent_pos] + sent[sent_pos + 1:context_end]
                self.word_count += 1
                for context_word in context:
                    getdata_start = time.time()
                    label = [1]+[0] * neg
                    if not self.negtive_pool:
                        self.negtive_pool += table.sample(neg*10000)
                    self.word_in += [token] * (neg+1)
                    self.word_out += [context_word]+self.negtive_pool[:neg]
                    del self.negtive_pool[:neg]
                    self.label_pool += label
                    getdata_end = time.time()
                    self.get_data_time += (getdata_end - getdata_start)
                    if len(self.label_pool) >= batch:
                        getdata_start = time.time()
                        label_batch,self.label_pool = self.label_pool[:batch],self.label_pool[batch:]
                        word_in_batch,self.word_in = self.word_in[:batch],self.word_in[batch:]
                        word_out_batch,self.word_out = self.word_out[:batch],self.word_out[batch:]
                        char,redi,com1,com2 = list(zip(*[vocab.index2word[i].component_sequence for i in word_in_batch]))
                        word_in_batch_data = [char,redi,com1,com2]
                        getdata_end = time.time()
                        self.get_data_time += (getdata_end - getdata_start)
                        yield word_in_batch,word_in_batch_data,word_out_batch,label_batch

def train_process(alpha,iter_index):
    # Set fi to point to the right chunk of training file
    global global_word_count
    train_time = 0
    attention = 0
    starting_alpha = alpha
    pid_dataset = BatchPair(fi)
    sum_loss = 0
    index,total_index,total_loss = 0,0,0
    print_index = 0
    for (word_in,component_in,word_out,train_label) in pid_dataset.batch_pair():
        if pid_dataset.word_count - pid_dataset.last_word_count > 10000:
            print_index += 1
            global_word_count += (pid_dataset.word_count - pid_dataset.last_word_count)
            pid_dataset.last_word_count = pid_dataset.word_count
            # Recalculate alpha
            alpha = starting_alpha * (1 - float(global_word_count) / (iter_num * vocab.word_count))
            if alpha < 0.0001: alpha = 0.0001
            for param_group in optimizer.param_groups:
                param_group['lr'] = alpha
            # Print progress info
            mean_loss = 1.0*sum_loss / index
            sys.stdout.write("\rPr:%d of %d (%.2f%%) ge:%.2f tr:%.2f lo:%.2f" %
                             (global_word_count, vocab.word_count * iter_num,
                              float(global_word_count) / vocab.word_count * 100 / iter_num,
                              pid_dataset.get_data_time / 60,
                              train_time / 60, mean_loss))
            if print_index % 10 == 0:
                print('\t')
                print(torch.mean(torch.softmax(model.atten_layers.weight.data,dim=-1),dim=0))
                attention_matrix = torch.softmax(model.atten_layers.weight.data.cpu(),-1)
                topk_redical = attention_matrix.topk(20,0)[1][:,-3]
                topk_char = attention_matrix.topk(20,0)[1][:,-4]
                topk_com1 = attention_matrix.topk(20,0)[1][:,-2]
                redical_word_list = [vocab[ii].word for ii in topk_redical]
                redical_list = [vocab[ii].components[1] for ii in topk_redical]
                char_word_list = [vocab[ii].word for ii in topk_char]
                char_list = [vocab[ii].components[0] for ii in topk_char]
                com1_word_list = [vocab[ii].word for ii in topk_com1]
                com1_list = [vocab[ii].components[2] for ii in topk_com1]
                redical_attention_list = attention_matrix[topk_redical]
                char_attention_list = attention_matrix[topk_char]
                com1_attention_list = attention_matrix[topk_com1]
                print('top 20 redical:')
                for tmpp in list(zip(redical_word_list,redical_list,redical_attention_list)):
                    print (tmpp)
                print('top 20 char')
                for tmpp in list(zip(char_word_list,char_list,char_attention_list)):
                    print (tmpp)
                print('top 20 com1')
                for tmpp in list(zip(com1_word_list, com1_list,com1_attention_list)):
                    print(tmpp)
            # sys.stdout.flush()
            index = 0
            sum_loss = 0
        train_time_start = time.time()
        word_in = torch.cuda.LongTensor(word_in)
        word_out = torch.cuda.LongTensor(word_out)
        label = torch.cuda.FloatTensor(train_label)
        outs = model.forward(word_in,component_in, word_out)
        loss = Lossfunc(outs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_time += time.time() - train_time_start
        index += 1
        total_index += 1
        total_loss += loss.data
        sum_loss += loss.data
    global_word_count += (pid_dataset.word_count - pid_dataset.last_word_count)
    sys.stdout.write("\rPr:%d of %d (%.2f%%) ge:%.2f tr:%.2f to:%f" %
                     (global_word_count, vocab.word_count * iter_num,
                      float(global_word_count) / vocab.word_count * 100 / iter_num,
                      pid_dataset.get_data_time / 60, train_time / 60,total_loss/total_index))
    sys.stdout.flush()

def save2(vocab, syn0, fo, binary):
    fo2 = fo+'_second'
    print ('saving second model to',fo2)
    syn0 = np.asarray(syn0)
    dim = len(syn0[0])
    fo = open(fo2, 'w')
    fo.write('%d %d\n' %(len(syn0), dim))
    for token in vocab.vocab_items:
        word = token.word.decode('utf-8')
        x_ = token.component_sequence_nodivision
        vector = syn0[x_].sum(0)/len(x_)
        vector_str = ' '.join([str(s) for s in vector])
        fo.write('%s %s\n' % (word, vector_str))
    fo.close()

def save(vocab, model, fo, iter_index, save_u=False,save_mix=False, attention_mode=1):
    fo2 = fo
    fo_mix = fo
    syn1 = model.v_embeddings.weight.data.cpu().numpy()
    print ('Saving model to', fo)
    dim = len(syn1[0])
    fo = open(fo, 'w')
    fo.write('%d %d\n' % (len(syn1), dim))
    for token, vector in zip(vocab, syn1):
        word = token.word
        vector_str = ' '.join([str(s) for s in vector])
        fo.write('%s %s\n' % (word, vector_str))
    fo.close()
    if save_u:
        syn2 = model.word_embeddings.weight.data.cpu().numpy()
        print('Saving model to', fo2+'_v')
        fo2 = open(fo2+'_v','w')
        fo2.write('%d %d\n' % (len(syn2), dim))
        for token, vector in zip(vocab, syn2):
            word = token.word
            vector_str = ' '.join([str(s) for s in vector])
            fo2.write('%s %s\n' % (word, vector_str))
        fo2.close()
    if save_mix:
        print('Saving momdel to', fo_mix+'_mix')
        fo_mix = open(fo_mix+'_mix','w')
        fo_mix.write('%d %d\n' % (len(syn1), dim))
        for index,word_item in enumerate(vocab.vocab_items):
            word_in = torch.cuda.LongTensor([index])
            component_in = word_item.component_sequence
            char_in = torch.cuda.LongTensor(component_in[0])
            redical_in = torch.cuda.LongTensor(component_in[1])
            com1_in = torch.cuda.LongTensor(component_in[2])
            com2_in = torch.cuda.LongTensor(component_in[3])

            emb_uword = model.word_embeddings(word_in)
            emb_char = model.u_embeddings(char_in)
            emb_redical = model.u_embeddings(redical_in)
            emb_com1 = model.u_embeddings(com1_in)
            emb_com2 = model.u_embeddings(com2_in)

            attention = torch.softmax(model.atten_layers(word_in), dim=-1).unsqueeze(1)
            if attention_mode == 1:
                atten_char = torch.softmax(model.atten_in_char(
                    torch.cat((emb_uword.expand(emb_char.shape[0], -1), emb_char), -1)).squeeze(
                    -1) + model.mask_char.weight[index], dim=0).unsqueeze(-1)
                atten_redical = torch.softmax(model.atten_in_redical(
                    torch.cat((emb_uword.expand(emb_redical.shape[0], -1), emb_redical), -1)).squeeze(
                    -1) + model.mask_redical.weight[index], dim=0).unsqueeze(-1)
                atten_com1 = torch.softmax(model.atten_in_com1(
                    torch.cat((emb_uword.expand(emb_com1.shape[0], -1), emb_com1), -1)).squeeze(
                    -1) + model.mask_com1.weight[index], dim=0).unsqueeze(-1)
                atten_com2 = torch.softmax(model.atten_in_com2(
                    torch.cat((emb_uword.expand(emb_com2.shape[0], -1), emb_com2), -1)).squeeze(
                    -1) + model.mask_com2.weight[index], dim=0).unsqueeze(-1)
            elif attention_mode == 2:
                atten_char = torch.softmax(torch.mm(model.atten_in_char(emb_char),emb_uword.squeeze(0).unsqueeze(1)).squeeze(-1)+model.mask_char.weight[index],dim=0).unsqueeze(-1)
                atten_redical = torch.softmax(torch.mm(model.atten_in_redical(emb_redical), emb_uword.squeeze(0).unsqueeze(1)).squeeze(-1)+model.mask_redical.weight[index],dim=0).unsqueeze(-1)
                atten_com1 = torch.softmax(torch.mm(model.atten_in_com1(emb_com1), emb_uword.squeeze(0).unsqueeze(1)).squeeze(-1)+model.mask_com1.weight[index],dim=0).unsqueeze(-1)
                atten_com2 = torch.softmax(torch.mm(model.atten_in_com2(emb_com2), emb_uword.squeeze(0).unsqueeze(1)).squeeze(-1)+model.mask_com2.weight[index],dim=0).unsqueeze(-1)
            # atten_char = torch.softmax(torch.mm(emb_char, emb_uword.t()) + model.mask_char.weight[index].unsqueeze(1), dim=0)
            # atten_redical = torch.softmax(torch.mm(emb_redical, emb_uword.t()) + model.mask_redical.weight[index].unsqueeze(1), dim=0)
            # atten_com1 = torch.softmax(torch.mm(emb_com1, emb_uword.t()) +model.mask_com1.weight[index].unsqueeze(1) , dim=0)
            # atten_com2 = torch.softmax(torch.mm(emb_com2, emb_uword.t())+model.mask_com2.weight[index].unsqueeze(1), dim=0)

            weighted_char = torch.mm(atten_char.t(),emb_char)
            weighted_redical = torch.mm(atten_redical.t(),emb_redical)
            weighted_com1 = torch.mm(atten_com1.t(),emb_com1)
            weighted_com2 = torch.mm(atten_com2.t(),emb_com2)

            emb_all = torch.stack((emb_uword, weighted_char, weighted_redical, weighted_com1, weighted_com2), 1)
            word_vector = torch.bmm(attention, emb_all).squeeze(1)
            word_vector = word_vector[0].data.cpu().numpy()
            word = word_item.word
            vector_str = ' '.join([str(s) for s in word_vector])
            fo_mix.write('%s %s\n' % (word, vector_str))
        fo_mix.close()

def read_characters():
    file_name = '../data/subcharacter/complete_word_information.json'
    f = open(file_name, 'r')
    char_corpus = json.load(f)
    file_name = '../data/subcharacter/name_entity_word.json'
    # file_name = '../data/subcharacter/filterBIE_name_entity_word.json'
    f = open(file_name, 'r')
    name_entity_dict = json.load(f)
    return char_corpus,name_entity_dict

def __init_process(*args):
    global model, Lossfunc, optimizer, fi, batch, vocab, table, neg, win ,total_time,global_word_count
    model, Lossfunc, optimizer, fi, batch, vocab, table, neg, win, global_word_count = args

def train(fi, fo, cbow, neg, dim, alpha, win, min_count, num_processes, binary, batch, mode):
    # Read train file to init vocab
    attention_mode = 2
    fi_text = fi
    using_components = True
    char_corpus, name_entity_dict = read_characters()
    vocab = Vocab(fi, min_count, char_corpus, name_entity_dict)
    if not using_components:
        vocab.compute_word_strokes()
    else:
        para_list = vocab.compute_components()
    model = SkipGramModel(len(vocab.components2index), len(vocab.vocab_items), dim, vocab.init_attention_matrix, para_list,attention_mode).cuda()
    print (model)
    Lossfunc = nn.BCELoss(reduction='sum').cuda()
    optimizer = optim.SGD(model.parameters(), lr=alpha)
    print('Initializing unigram table')
    table = UnigramTable(vocab)
    global_word_count = 0
    __init_process(model, Lossfunc, optimizer, fi, batch, vocab, table, neg, win, global_word_count)
    '/data/users/mabing/QAsystem/cw2vec/data'
    dirname =  '/data/users/mabing/QAsystem/cw2vec/data/' + 'my-atten22-newattention_mode' + 'comp2vec' + '-' + \
               (fi_text.split('data/')[1].split('.txt')[0] if len(fi_text.split('data/'))>1 else fi_text.split('data/')[0].split('.txt')[0]) + '-ba%d' % (batch) + '-format' + '-mode%d'%(mode)
    while os.path.isdir(dirname):
        dirname = dirname + '_1'
        if not os.path.isdir(dirname):
            break
    os.mkdir(dirname)
    print ('mkdir %s' %dirname)
    # Begin training using num_processes workers
    for iter_index in range(iter_num):
        t0 = time.time()
        if mode == 1:
            for param in model.atten_layers.parameters():
                param.requires_grad = False
        elif mode == 3:
            pass
        elif iter_index in [0]:
            print("train embedding and attention")
            for param in model.atten_layers.parameters():
                param.requires_grad = True
            for param in model.u_embeddings.parameters():
                param.requires_grad = True
            for param in model.v_embeddings.parameters():
                param.requires_grad = True
            for param in model.word_embeddings.parameters():
                param.requires_grad = True
        elif iter_index in [1,3,4,5,7,8] or iter_index > 8:
            print ("train embedding")
            for param in model.atten_layers.parameters():
                param.requires_grad = False
            for param in model.u_embeddings.parameters():
                param.requires_grad = True
            for param in model.v_embeddings.parameters():
                param.requires_grad = True
            for param in model.word_embeddings.parameters():
                param.requires_grad = True
        elif iter_index in [2,6]:
            print('train attention')
            for param in model.atten_layers.parameters():
                param.requires_grad = True
            for param in model.u_embeddings.parameters():
                param.requires_grad = False
            for param in model.v_embeddings.parameters():
                param.requires_grad = False
            for param in model.word_embeddings.parameters():
                param.requires_grad = False
        train_process(alpha,iter_index)
        t1 = time.time()
        # mode 1: 0.5 0.5 mode 2 :a ,b

        if using_components:
            cc = 'comp2vec'
        else: cc = 'cw2vec'
        print ('\n')
        print (iter_index+1,'th: Completed training. Training took', (t1 - t0) / 60, 'minutes')
        fo = dirname +'/' + 'my-atten-torch' + cc + '-' + (fi_text.split('data/')[1].split('.txt')[0] if len(fi_text.split('data/'))>1 else fi_text.split('data/')[0].split('.txt')[0]) \
             + '-iter%d' % (iter_index+1) + '-ba%d' % (batch) + '-format' + '-mode%d'%(mode)
        save(vocab, model, fo, iter_index, save_u=True, save_mix=True,attention_mode=attention_mode)
        # model.save_attention(fo+'attention_matrixiter')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training file', dest='fi', required=True)
    parser.add_argument('-model', help='Output model file', dest='fo', required=False)
    parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=0, type=int)
    parser.add_argument('-negative', help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax', dest='neg', default=5, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=100, type=int)
    parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.025, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int)
    parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=10, type=int)
    parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=40, type=int)
    parser.add_argument('-binary', help='1 for output model in binary format, 0 otherwise', dest='binary', default=0, type=int)
    parser.add_argument('-iter', help='iter num', dest='iter', default=1, type=int)
    parser.add_argument('-batch', help='batch_size', dest='batch', default=128, type=int)
    parser.add_argument('-mode', help='mode', dest='mode',default=3, type = int)

    #TO DO: parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)
    args = parser.parse_args()
    iter_num = args.iter
    fi = args.fi
    fo = args.fo
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not fo:
        print('will save to:'+fi.split('data/')[0] + 'data/' + 'my-torchword2comp2vec-' + fi.split('data/')[1].split('.txt')[0] + '-iter%d' % (iter_num ) + '-format')
    # fo = 'my-comp2vec-'+args.fi.split('.')[0]+'-iter%d'%(args.iter)
    train(args.fi, args.fo, bool(args.cbow), args.neg, args.dim, args.alpha, args.win,
          args.min_count, args.num_processes, bool(args.binary), args.batch, args.mode)
