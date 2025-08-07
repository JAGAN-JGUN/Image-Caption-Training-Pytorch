import torch

def collate_fn(batch):
    images, captions = [],[]

    for img, cap in batch:
        images.append(img)
        captions.append(cap)

    images = torch.stack(images)
    captions = torch.stack(captions)

    return images, captions