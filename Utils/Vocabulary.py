import re
from collections import Counter

class vocabulary:
    def __init__(self):
        self.freq_thres = 5
        self.itos = {0 : '<pad>', 1 : '<sos>', 2 : '<eos>', 3 : '<unk>'}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.word_freq = Counter()
    
    def __len__(self):
        return len(self.itos)
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip().split()

    def build_vocab(self, sentences):
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            self.word_freq.update(tokens)

        for word, freq in self.word_freq.items():
            if freq >= self.freq_thres:
                idx = len(self.itos)
                self.itos[idx] = word
                self.stoi[word] = idx
    
    def numeric(self, text):
        tokens = self.tokenize(text)
        return [self.stoi.get(token, self.stoi['<unk>']) for token in tokens]