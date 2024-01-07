from glob import glob

import torch
from tqdm import tqdm

wav_root = '/home/ai_music/data'

from inference.DAV import *
device = 'cuda:0'
weights_path = "runs/baseline/best/dac/weights.pth"

model = load_model(weights_path, device)

all_wavs = glob(f'{wav_root}/**/*.wav', recursive=True)

for wav_path in tqdm(all_wavs):
    latent_path = wav_path.replace('.wav', '.latent.pt')
    z = encode_from_file(model, wav_path)
    torch.save(z, latent_path)


