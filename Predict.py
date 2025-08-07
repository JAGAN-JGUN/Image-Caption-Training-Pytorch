import torch
from torchvision import transforms
from PIL import Image
from Models.ICM import ImageCaptionModel
from Models.Encoder import ImageEncoder
from Models.Decoder import ImageDecoder
import heapq
import argparse

embed_size = 256
hidden_size = 512
vocab_path = "Data/vocab.pkl"
model_path = "best_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 50

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)

def generate_caption(model, image_tensor, vocab, beam_width=5, max_len=50):
    model.eval()
    with torch.no_grad():
        device = image_tensor.device
        features = model.encoder(image_tensor).unsqueeze(1)

        start_token = vocab.stoi["<sos>"]
        end_token = vocab.stoi["<eos>"]

        beams = [(0.0, [start_token], None, features)]
        completed = []

        for _ in range(max_len):
            new_beams = []
            for score, seq, hidden, input_token in beams:
                if seq[-1] == end_token:
                    completed.append((score, seq))
                    continue

                if len(seq) == 1:
                    inputs = input_token
                else:
                    last_word = torch.tensor([seq[-1]]).to(device)
                    inputs = model.decoder.embed(last_word).unsqueeze(1)

                hiddens, hidden = model.decoder.lstm(inputs, hidden)
                output = model.decoder.linear(hiddens.squeeze(1))
                log_probs = torch.log_softmax(output, dim=1)

                top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    next_token = top_indices[0][i].item()
                    next_score = score + top_log_probs[0][i].item()
                    new_seq = seq + [next_token]
                    new_beams.append((next_score, new_seq, hidden, inputs))

            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        completed += beams
        best = max(completed, key=lambda x: x[0])
        final_ids = best[1][1:]

        caption = [vocab.itos[idx] for idx in final_ids if idx != end_token]
        return " ".join(caption)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to input image")
    args = parser.parse_args()

    import pickle
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    encoder = ImageEncoder(embed_size)
    decoder = ImageDecoder(embed_size, hidden_size, len(vocab))
    model = ImageCaptionModel(encoder, decoder).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    image_tensor = load_image(args.image_path)

    caption = generate_caption(model, image_tensor, vocab)
    print(f"\n Caption: {caption}")

if __name__ == "__main__":
    main()
