from inference.dav import *

if __name__ == "__main__":

    weights_path = "vae_dac_44100_87hz_64dim.pth"
    audio_path = "part.mp3"
    device = 'cuda:1'
    model = load_model(weights_path, device)

    z = encode_from_file(model, audio_path)
    print("z shape:", z.shape)
    decode_to_file(model, z, 'out.wav')
