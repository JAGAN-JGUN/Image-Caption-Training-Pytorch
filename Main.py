import pandas as pd
from collections import defaultdict    
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pickle

from Models.ICM import ImageCaptionModel
from Models.Encoder import ImageEncoder
from Models.Decoder import ImageDecoder
from Utils.Vocabulary import vocabulary
from Utils.Dataset import ImageCaptionDataset
from Utils.Collate import collate_fn
from Utils.Train import train

def build_caption_dict(df):
    d = defaultdict(list)
    for image, caption in zip(df['image_name'], df['comment']):
        d[image].append(caption.strip())
    return dict(d)

if __name__ == "__main__":
    train_data = pd.read_csv('Data/train.csv')
    val_data   = pd.read_csv('Data/val.csv')

    train_dict = build_caption_dict(train_data)
    val_dict   = build_caption_dict(val_data)

    img = list(train_dict.keys())[0]
    print(f"Image: {img}")
    print("Captions:")
    for caption in train_dict[img]:
        print(f"- {caption}")

    all_captions = []
    for captions in train_dict.values():
        all_captions.extend(captions)

    vocab = vocabulary()
    vocab.build_vocab(all_captions)
    print(f"Vocabulary size: {len(vocab)}")
    print(vocab.numeric("A man in a black shirt"))

    pickle.dump(vocab, open('Data/vocab.pkl', 'wb'))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageCaptionDataset(
        directory='Data/Images',
        captions_dict=train_dict,
        vocab=vocab,
        transform=transform,
    )

    val_dataset = ImageCaptionDataset(
    directory='Data/Images',
    captions_dict=val_dict,
    vocab=vocab,
    transform=transform
    )

    img,caption = train_dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Caption tensor: {caption[:10]}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
    )

    for images, captions in train_loader:
        print(f"Batch image shape: {images.shape}")
        print(f"Batch captions shape: {captions.shape}")
        break

    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)

    encoder = ImageEncoder(embed_size)
    decoder = ImageDecoder(embed_size, hidden_size, vocab_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImageCaptionModel(encoder, decoder).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    train(model, train_loader, val_loader, optimizer, criterion, num_epochs=50, device=device)