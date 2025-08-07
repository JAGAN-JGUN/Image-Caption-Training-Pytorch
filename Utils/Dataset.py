import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import random

class ImageCaptionDataset(Dataset):
    def __init__(self, directory, captions_dict, vocab, transform=None, maxlen=50):
        self.directory = directory
        self.captions_dict = captions_dict
        self.vocab = vocab
        self.transform = transform
        self.maxlen = maxlen
        self.image_names = list(captions_dict.keys())

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        caption = random.choice(self.captions_dict[image_name])

        image_path = os.path.join(self.directory, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        tokens = [self.vocab.stoi["<sos>"]] + \
                 self.vocab.numeric(caption) + \
                 [self.vocab.stoi["<eos>"]]
        
        if len(tokens) < self.maxlen:
            tokens += [self.vocab.stoi["<pad>"]] * (self.maxlen - len(tokens))
        else:
            tokens = tokens[:self.maxlen]

        caption_tensor = torch.tensor(tokens)

        return image, caption_tensor