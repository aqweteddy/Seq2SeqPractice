from typing import *
from tqdm import tqdm
import csv


class Vocab:
    BOS = '[BOS]'
    UNK = '[UNK]'
    EOS = '[EOS]'
    PAD = '[PAD]'

    def __init__(self, from_file=None, word_filter=None):
        self.word_filter = word_filter if word_filter else self.word_filter
        self.word2idx = dict()
        self.idx2word = dict()
        self.idx_freq = dict()  # idx_freq[idx] = cnt

        if from_file:
            with open(from_file, 'r') as f:
                rows = csv.reader(f)
                for idx, word, freq in rows:
                    idx, freq = int(idx), int(freq)

                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    self.idx_freq[idx] = freq
        else:
            # add special token
            self.__add_special_token()

        self.bos_id = self.word2idx[self.BOS]
        self.unk_id = self.word2idx[self.UNK]
        self.eos_id = self.word2idx[self.EOS]
        self.pad_token_id = self.word2idx[self.PAD]

    @staticmethod
    def from_pretrained(file):
        return Vocab(from_file=file)

    def __add_special_token(self):
        self.add_word(self.UNK, ignore_filter=True)  # unknown token
        self.add_word(self.BOS, ignore_filter=True)  # begin of sentence
        self.add_word(self.EOS, ignore_filter=True)  # end of sentence
        self.add_word(self.PAD, ignore_filter=True)  # pad token

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def word_filter(word: str):
        punctuation = "＃＄％＆＇（）◆＊+——！\n，。[]？、~@#￥%……&.+: -*/`~*（＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
        for char in word:
            for p in punctuation:
                if char == p:
                    return False
        return True if len(word) <= 5 else False

    def add_word(self, word: str, ignore_filter=True):
        if not self.word_filter(word) and not ignore_filter:
            return
        if word not in self.word2idx.keys():
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.idx_freq[idx] = 1
        else:
            self.idx_freq[self.word2idx[word]] += 1

    def add_docs(self, docs: List[List[str]]):
        for doc in tqdm(docs):
            for word in doc:
                self.add_word(word.strip())

    def add_doc(self, doc: List[str]):
        for word in doc:
            self.add_word(word.strip())

    def encode(self, doc: str, add_special_token=True):
        result = [self.bos_id] if add_special_token else []
        for word in doc:
            result.append(self.word2idx.get(word, self.unk_id))
        if add_special_token:
            result.append(self.eos_id)
        return result

    def decode(self, idx_list: List[int], remove_special_token=False):
        result = []
        for idx in idx_list:
            if not remove_special_token or not idx in [self.eos_id, self.bos_id, self.pad_token_id, self.unk_id]:
                result.append(self.idx2word.get(idx, ''))
        return result

    def re_index(self):
        idx2word, word2idx, idx_freq = {}, {}, {}
        for new_idx, (idx, word) in enumerate(self.idx2word.items()):
            idx2word[new_idx] = word
            word2idx[word] = new_idx
            idx_freq[new_idx] = self.idx_freq[idx]
        self.idx2word, self.word2idx, self.idx_freq = idx2word, word2idx, idx_freq

    def remove_idx(self, idx):
        if idx < 4:
            raise (ValueError, "can't remove special token")

        del self.word2idx[self.idx2word[idx]]
        del self.idx_freq[idx]
        del self.idx2word[idx]

    def get_topn_freq_idx(self, n=10, reverse=True):
        tmp = sorted([(idx, freq) for idx, freq in self.idx_freq.items()],
                     key=lambda x: x[1], reverse=reverse)
        return tmp[:n]

    def save(self, path: str):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            for idx, word in self.idx2word.items():
                writer.writerow([idx, word, self.idx_freq[idx]])


if __name__ == '__main__':
    import json
    with open('../../corpus/ettoday_2017.json', 'r') as f:
        data = json.load(f)

    text = [[c] for t in tqdm(data) for c in t['content']]
    title = [[c] for t in tqdm(data) for c in t['title']]
    vocab = Vocab()
    vocab.add_docs(text)
    vocab.add_docs(title)
    # vocab.save('model/vocab.txt')

    # vocab = Vocab.from_pretrained('model/vocab.txt')
    for idx, freq in vocab.get_topn_freq_idx(100000, reverse=False):
        if freq > 10:
            break
        try:
            vocab.remove_idx(idx)
        except:
            pass
    vocab.re_index()
    
    enc = vocab.encode(['你', '在', '麻'])
    print(enc)
    print(vocab.decode(enc))
    
    vocab.save('model/vocab.txt')
    print(len(vocab))
