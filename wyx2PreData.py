import numpy as np
import collections
import math
import gensim
import numpy as np
import torch
from torch import nn

import torch



#%%定义处理数据方法
coden_dict = {'GCU': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0,                             # alanine<A>
              'UGU': 1, 'UGC': 1,                                                 # systeine<C>
              'GAU': 2, 'GAC': 2,                                                 # aspartic acid<D>
              'GAA': 3, 'GAG': 3,                                                 # glutamic acid<E>
              'UUU': 4, 'UUC': 4,                                                 # phenylanaline<F>
              'GGU': 5, 'GGC': 5, 'GGA': 5, 'GGG': 5,                             # glycine<G>
              'CAU': 6, 'CAC': 6,                                                 # histidine<H>
              'AUU': 7, 'AUC': 7, 'AUA': 7,                                       # isoleucine<I>
              'AAA': 8, 'AAG': 8,                                                 # lycine<K>
              'UUA': 9, 'UUG': 9, 'CUU': 9, 'CUC': 9, 'CUA': 9, 'CUG': 9,         # leucine<L>
              'AUG': 10,                                                          # methionine<M>
              'AAU': 11, 'AAC': 11,                                               # asparagine<N>
              'CCU': 12, 'CCC': 12, 'CCA': 12, 'CCG': 12,                         # proline<P>
              'CAA': 13, 'CAG': 13,                                               # glutamine<Q>
              'CGU': 14, 'CGC': 14, 'CGA': 14, 'CGG': 14, 'AGA': 14, 'AGG': 14,   # arginine<R>
              'UCU': 15, 'UCC': 15, 'UCA': 15, 'UCG': 15, 'AGU': 15, 'AGC': 15,   # serine<S>
              'ACU': 16, 'ACC': 16, 'ACA': 16, 'ACG': 16,                         # threonine<T>
              'GUU': 17, 'GUC': 17, 'GUA': 17, 'GUG': 17,                         # valine<V>
              'UGG': 18,                                                          # tryptophan<W>
              'UAU': 19, 'UAC': 19,                                               # tyrosine(Y)
              'UAA': 20, 'UAG': 20, 'UGA': 20,                                    # STOP code
              }
def get_1_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**1
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        nucle_com.append(ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return  word_index

def get_2_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**2
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        nucle_com.append(ch0 + ch1)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return  word_index

def get_3_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        n=n//base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return  word_index

def coden(seq,kmer,tris):
    coden_dict = tris
    freq_dict = frequency(seq,kmer,coden_dict)
    vectors = np.zeros((101, len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        value = freq_dict[coden_dict[seq[i:i+kmer].replace('T', 'U')]]
        vectors[i][coden_dict[seq[i:i+kmer].replace('T', 'U')]] = value/100
    return vectors

def codenKNF(seq):
    vectors = np.zeros((len(seq), 21))
    for i in range(len(seq)):
        if i < len(seq)-2:
            vectors[i][coden_dict[seq[i:i+3].replace('T', 'U')]] = 1
    return vectors.tolist()


def dpcp(seq):
    phys_dic = {
        # Shift Slide Rise Tilt Roll Twist Stacking_energy Enthalpy Entropy Free_energy Hydrophilicity
        #DELECT5.6.7.8.9+电子离子作用伪势
        'AA': [-0.08, -1.27, 3.18, -0.8,-0.93, 0.04,0.126],
        'AT': [-0.06, -1.36, 3.24, 1.1, -1.1, 0.14,0.126],
        'AC': [0.23, -1.43, 3.24, 0.8, -2.24, 0.14,0.126],
        'AG': [-0.04, -1.5, 3.3, 0.5,-2.08, 0.08,0.126],
        'TA': [0.07, -1.7, 3.38, 1.3, -2.35, 0.1,0.1335],
        'TT': [0.23, -1.43, 3.24, 0.8, -2.24, 0.27,0.1335],
        'TC': [0.07, -1.39, 3.22, 0, -3.42, 0.26,0.1335],
        'TG': [-0.01, -1.78, 3.32, 0.3, -3.26, 0.17,0.1335],
        'CA': [0.11, -1.46, 3.09, 1, -2.11, 0.21,0.134],
        'CT': [-0.04, -1.5, 3.3, 0.5, -2.08, 0.52,0.134],
        'CC': [-0.01, -1.78, 3.32, 0.3,-3.26, 0.49,0.134],
        'CG': [0.3, -1.89, 3.3, -0.1,-2.36, 0.35,0.134],
        'GA': [-0.02, -1.45, 3.26, -0.2,-1.33, 0.21,0.08],
        'GT': [-0.08, -1.27, 3.18, -0.8,-0.93, 0.44,0.08],
        'GC': [0.07, -1.7, 3.38, 1.3,-2.35, 0.48,0.08],
        'GG': [0.11, -1.46, 3.09, 1,-2.11, 0.34,0.08]}

    seqLength = len(seq)
    sequence_vector = np.zeros([101, 7])
    k = 2
    for i in range(0, seqLength - 1):
        sequence_vector[i, 0:7] = phys_dic[seq[i:i + k]]
    return sequence_vector


def frequency(seq,kmer,coden_dict):
    Value = []
    k = kmer
    coden_dict = coden_dict
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i:i+k]
        kmer_value = coden_dict[kmer.replace('T', 'U')]
        Value.append(kmer_value)
    freq_dict = dict(collections.Counter(Value))
    return freq_dict

def creatmat(data):
    mat = np.zeros([len(data),len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add <len(data):
                    score = paired(data[i - add],data[j + add])
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * Gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1,30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(data[i + add],data[j - add])
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * Gaussian(add)
                    else:
                        break
            mat[[i],[j]] = coefficient
    return mat


def Gaussian(x):
    return math.exp(-0.5*(x*x))

def paired(x,y):
    if x == 'A' and y == 'U':
        return 2
    elif x == 'G' and y == 'C':
        return 3
    elif x == "G"and y == 'U':
        return 0.8
    elif x == 'U' and y == 'A':
        return 2
    elif x == 'C' and y == 'G':
        return 3
    elif x == "U"and y == 'G':
        return 0.8
    else:
        return 0

# %%处理三级结构，输出为101×100
def threeshape(protein):
    dataE = []
    dataH = []
    dataM = []
    dataP = []
    dataR = []
    with open(r'./Datasets/EP') as f:
        for line in f:
            dataE.append([float(x) for x in line.split(',')])
    dataE = np.array(dataE)
    with open(r'./Datasets/Helt') as f:
        for line in f:
            dataH.append([float(x) for x in line.split(',')])
    dataH = np.array(dataH)
    with open(r'./Datasets/MGW') as f:
        for line in f:
            dataM.append([float(x) for x in line.split(',')])
    dataM = np.array(dataM)
    with open(r'./Datasets/Prot') as f:
        for line in f:
            dataP.append([float(x) for x in line.split(',')])
    dataP = np.array(dataP)
    with open(r'./Datasets/Roll') as f:
        for line in f:
            dataR.append([float(x) for x in line.split(',')])
    dataR = np.array(dataR)
    return dataE,dataH,dataM,dataP,dataR
#%%处理Kmer，输出为101×84
def dealwithdata1(protein):
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    tris3 = get_3_trids()
    dataX = []
    with open(r'./Datasets/positive') as f:
        for line in f:
            if '>' not in line:
                kmer1 = coden(line.strip(),1,tris1)
                kmer2 = coden(line.strip(),2,tris2)
                kmer3 = coden(line.strip(),3,tris3)
                Kmer = np.hstack((kmer1,kmer2,kmer3))
                dataX.append(Kmer.tolist())
    with open(r'./Datasets/negative') as f:
        for line in f:
            if '>' not in line:
                kmer1 = coden(line.strip(),1,tris1)
                kmer2 = coden(line.strip(),2,tris2)
                kmer3 = coden(line.strip(),3,tris3)
                Kmer = np.hstack((kmer1,kmer2,kmer3))
                dataX.append(Kmer.tolist())

    dataX = np.array(dataX)
    return dataX
#%%处理结构特征，输出为101×101
def creatmat(protein):
    #data_pair = np.arange(10201).reshape(1,101,101)
    pair_data = []
    with open(r'./Datasets/positive') as f:
        num = 0
        for data in f:
            if '>' not in data:
                data = data[:-1]
                mat = np.zeros([len(data),len(data)])
                for i in range(len(data)):
                    for j in range(len(data)):
                        coefficient = 0
                        for add in range(30):
                            if i - add >= 0 and j + add <len(data):
                                score = paired(data[i - add].replace('T', 'U'),data[j + add].replace('T', 'U'))
                                if score == 0:
                                    break
                                else:
                                    coefficient = coefficient + score * Gaussian(add)
                            else:
                                break
                        if coefficient > 0:
                            for add in range(1,30):
                                if i + add < len(data) and j - add >= 0:
                                    score = paired(data[i + add],data[j - add])
                                    if score == 0:
                                        break
                                    else:
                                        coefficient = coefficient + score * Gaussian(add)
                                else:
                                    break
                        mat[[i],[j]] = coefficient
                if len(pair_data)==0:
                    pair_data = torch.from_numpy(mat).unsqueeze(0)
                else:
                    matt = torch.from_numpy(mat).unsqueeze(0)
                    pair_data = torch.cat((pair_data,matt),0)
                num=num+1
                print(num)
    with open(r'./Datasets/negative') as f:
        for data in f:
            if '>' not in data:
                data = data[:-1]
                mat = np.zeros([len(data), len(data)])
                for i in range(len(data)):
                    for j in range(len(data)):
                        coefficient = 0
                        for add in range(30):
                            if i - add >= 0 and j + add < len(data):
                                score = paired(data[i - add], data[j + add])
                                if score == 0:
                                    break
                                else:
                                    coefficient = coefficient + score * Gaussian(add)
                            else:
                                break
                        if coefficient > 0:
                            for add in range(1, 30):
                                if i + add < len(data) and j - add >= 0:
                                    score = paired(data[i + add], data[j - add])
                                    if score == 0:
                                        break
                                    else:
                                        coefficient = coefficient + score * Gaussian(add)
                                else:
                                    break
                        mat[[i], [j]] = coefficient

                matt = torch.from_numpy(mat).unsqueeze(0)
                pair_data = torch.cat((pair_data, matt), 0)
                num = num + 1
                print(num)
    return pair_data
#%%处理KNF特征，输出为99×21
def dealwithdataKNF(protein):
    dataXKNF = []
    dataYKNF = []
    with open(r'./Datasets/positive') as f:
            for line in f:
                if '>' not in line:
                    dataYKNF.append(1)
                else:
                    # list=[]
                    # line1 =line[10:][:7]
                    # line2 = line[24:]
                    # lst = line2.split(",end:")
                    # line2_strat = lst[0]
                    # line2_end = lst[1][:-1]
                    # list.append(int(line1))
                    # list.append(int(line2_strat))
                    # list.append(int(line2_end))
                    dataXKNF.append(1)

    with open(r'./Datasets/negative') as f:
            for line in f:
                if '>' not in line:
                    dataYKNF.append(0)
                else:
                    # list = []
                    # line1 = line[10:][:7]
                    # line2 = line[24:]
                    # lst = line2.split(",end:")
                    # line2_strat = lst[0]
                    # line2_end = lst[1][:-1]
                    # list.append(int(line1))
                    # list.append(int(line2_strat))
                    # list.append(int(line2_end))
                    dataXKNF.append(1)
    #indexes = np.random.choice(len(dataYKNF), len(dataYKNF), replace=False)
    # dataX = np.array(dataXKNF)[indexes]
    # dataY = np.array(dataYKNF)[indexes]
    dataX = np.array(dataXKNF)
    dataY = np.array(dataYKNF)
    return dataX,dataY
#%%处理DPCP特征，输出为99×21
def dealwithdataDPCP(protein):
    dataXDPCP = []
    dataYDPCP = []
    with open(r'./Datasets/positive') as f:
            for line in f:
                if '>' not in line:
                    dataXDPCP.append(dpcp(line.strip()))
                    dataYDPCP.append(1)
    with open(r'./Datasets/negative') as f:
            for line in f:
                if '>' not in line:
                    dataXDPCP.append(dpcp(line.strip()))
                    dataYDPCP.append(0)
    #indexes = np.random.choice(len(dataYKNF), len(dataYKNF), replace=False)
    # dataX = np.array(dataXKNF)[indexes]
    # dataY = np.array(dataYKNF)[indexes]
    dataX = np.array(dataXDPCP)
    dataY = np.array(dataYDPCP)
    return dataX
#%%处理RNA2Vec，输出为92×30
def Vec(protein):
    model = './circRNA2Vec/circRNA2Vec_model'
    seqpos_path = './Datasets/positive'
    seqneg_path = './Datasets/negative'

    def Generate_Embedding(seq_posfile, seq_negfile, model):
        seqpos = read_fasta_file(seq_posfile)
        seqneg = read_fasta_file(seq_negfile)

        X, y, embedding_matrix = circRNA2Vec(10, 1, 30, model, 101, seqpos, seqneg)
        return X, y, embedding_matrix

    def read_fasta_file(fasta_file):
        seq_dict = {}
        bag_sen = list()
        fp = open(fasta_file, 'r')
        name = ''
        for line in fp:
            line = line.rstrip()
            if line[0] == '>':
                name = line[1:]
                seq_dict[name] = ''
            else:
                seq_dict[name] = seq_dict[name] + line.upper()
        fp.close()

        for seq in seq_dict.values():
            seq = seq.replace('T', 'U')
            bag_sen.append(seq)

        return np.asarray(bag_sen)

    def circRNA2Vec(k, s, vector_dim, model, MAX_LEN, pos_sequences, neg_sequences):
        model1 = gensim.models.Doc2Vec.load(model)
        pos_list = seq2ngram(pos_sequences, k, s, model1.wv)
        neg_list = seq2ngram(neg_sequences, k, s, model1.wv)
        seqs = pos_list + neg_list

        # X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
        X = seqs
        y = np.array([1] * len(pos_list) + [0] * len(neg_list))
        # y = to_categorical(y)
        indexes = np.random.choice(len(y), len(y), replace=False)

        embedding_matrix = np.zeros((len(model1.wv.vocab), vector_dim))
        for i in range(len(model1.wv.vocab)):
            embedding_vector = model1.wv[model1.wv.index2word[i]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return X, y, embedding_matrix

    def seq2ngram(seqs, k, s, wv):
        list = []
        print('need to n-gram %d lines' % len(seqs))

        for num, line in enumerate(seqs):
            if num < 3000000:
                line = line.strip()
                l = len(line)
                list2 = []
                for i in range(0, l, s):
                    if i + k >= l + 1:
                        break
                    list2.append(line[i:i + k])
                list.append(convert_data_to_index(list2, wv))
        return list

    def convert_data_to_index(string_data, wv):
        index_data = []
        last = 1
        for word in string_data:
            if word in wv:
                index_data.append(wv.vocab[word].index)
                last = wv.vocab[word].index
            else:
                index_data.append(last)
        return index_data

    Embedding, dataY, embedding_matrix = Generate_Embedding(seqpos_path, seqneg_path, model)
    embedding_matrixT = torch.from_numpy(embedding_matrix)
    embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                             _weight=embedding_matrixT)
    Embedding = torch.tensor(Embedding)
    e = embedding(Embedding)
    return e
#%%读取主函数
def all_data(protein):
    KNF = dealwithdataKNF(protein)
    DPCP = dealwithdataDPCP(protein)
    Kmer = dealwithdata1(protein)
    Vec_embedding = Vec(protein)
    dataE,dataH,dataM,dataP,dataR =threeshape(protein)
    dataE = torch.from_numpy(dataE)
    dataH = torch.from_numpy(dataH)
    dataM = torch.from_numpy(dataM)
    dataP = torch.from_numpy(dataP)
    dataR = torch.from_numpy(dataR)
    shape = torch.cat((dataE,dataH,dataM,dataP,dataR), dim=1)
    Y = KNF[1]
    name = KNF[0]
    return Kmer,DPCP,Y,Vec_embedding,name,shape





