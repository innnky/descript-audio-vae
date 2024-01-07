import librosa
import soundfile
import torch

from inference.DAV import *

if __name__ == "__main__":

    weights_path = "runs/baseline/best/dac/weights.pth"
    audio_path = "test.wav"
    device = 'cuda:0'
    model = load_model(weights_path, device)

    z = encode_from_file(model, audio_path)
    print("z shape:", z.shape)
    decode_to_file(model, z, 'out.wav')
