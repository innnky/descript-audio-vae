from inference.DAV import *
import torch.nn.functional as F
if __name__ == "__main__":

    weights_path = "runs/vqfixed/best/dac/weights.pth"
    audio_path = "测试音频.wav"
    device = 'cuda:1'
    model = load_model(weights_path, device)

    z = encode_from_file(model, audio_path)
    z = F.interpolate(z[None], size=int(z.shape[-1]*1.1), mode='linear', align_corners=True)[0]
    print("z shape:", z.shape)
    decode_to_file(model, z, 'out.wav')
