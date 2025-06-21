import torch
import math
from tqdm import tqdm
import numpy as np
from scipy.linalg import sqrtm
import pretty_midi
from scipy.io.wavfile import write
import os
from frechet_audio_distance import FrechetAudioDistance


@torch.no_grad()
def embed_real_data(ds, encoder, fid_sample_size, batch_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = math.ceil(fid_sample_size / batch_size)
    real_latents = []

    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    for x in tqdm(dl, desc="Embedding", total=iterations):
        if len(real_latents)*batch_size >= fid_sample_size:
            real_latents = real_latents[:fid_sample_size]
            break

        x = x.to(device)
        z = encoder(x)
        real_latents.append(z)


    real_latents = torch.cat(real_latents, dim=0).numpy()
    return real_latents


@torch.no_grad()
def embed_generated_data(model, sampler, num_timesteps, encoder, fid_sample_size, batch_size):
    """
    Function that generates samples using the model (denoising network) and sampler (DDPM or DDIM) and embeds them in the latent space of the encoder model.

    :model: denoising network responsible for generating data
    :sampler: sampler (DDPM or DDIM) that generates data according to the reverse process
    :num_timesteps: number of diffusion steps in the generation process
    :encoder: network that embeds data in the latent space
    :fid_sample_size: number of samples to generate and embed
    :batch_size: size of the batch for generation

    :return: embedded data in the latent space
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = math.ceil(fid_sample_size / batch_size)
    fake_latents = []

    for _ in range(iterations):
        noise = torch.randn(batch_size, 3, 32, 32, device=device)
        generated = sampler.p_sample_loop(model, noise, num_inference_steps=num_timesteps, clip=True, quiet=True)
        generated = torch.from_numpy(generated).to(device)
        z = encoder(generated)
        fake_latents.append(z)
        if len(fake_latents) * batch_size >= fid_sample_size:
            fake_latents = fake_latents[:fid_sample_size]
            break

    fake_latents = torch.cat(fake_latents, dim=0)[:fid_sample_size].cpu().numpy()

    return fake_latents

def fit_n_dimensional_gaussian(latents):
    """
    Function that fits a Gaussian distribution with mean vector mu and covariance matrix sigma to data in the latent space.

    :latents: data in the latent space to which the Gaussian distribution is fitted

    :return: mu, sigma - parameters of the Gaussian distribution - mean vector and covariance matrix
    """
    mu, sigma = None, None
    mu = np.mean(latents, axis=0)
    sigma = np.cov(latents, rowvar=False)
    return mu, sigma


def wasserstein_distance(mu1, sigma1, mu2, sigma2):
    """
    Function that computes the Wasserstein distance between two Gaussian distributions with parameters mu and sigma.

    :mu1: mean of the first distribution
    :sigma1: covariance matrix of the first distribution
    :mu2: mean of the second distribution
    :sigma2: covariance matrix of the second distribution
    """
    distance = 0
    mean_diff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)
    trace = np.trace(sigma1 + sigma2 - 2 * covmean)
    distance = mean_diff + trace

    return distance.real

def calculate_fid_from_latents(real_latents, fake_latents):
    """
    Function that calculates the FID score between two sets of latents.

    :real_latents: latents of real data
    :fake_latents: latents of generated data

    :return: FID score
    """
    mu_real, sigma_real = fit_n_dimensional_gaussian(real_latents)
    mu_fake, sigma_fake = fit_n_dimensional_gaussian(fake_latents)

    fid_score = wasserstein_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_score

def calculate_fid_encoder(model, sampler, encoder, ds, fid_sample_size=1000, batch_size=64, num_timesteps=1000):
    """
    Function that calculates the FID score between real and generated data.

    :model: denoising network responsible for generating data
    :sampler: sampler (DDPM or DDIM) that generates data according to the reverse process
    :encoder: network that embeds data in the latent space
    :ds: dataset with real data
    :fid_sample_size: number of samples to generate and embed
    :batch_size: size of the batch for generation
    :num_timesteps: number of diffusion steps in the generation process

    :return: FID score
    """
    real_latents = embed_real_data(ds, encoder, fid_sample_size, batch_size)
    fake_latents = embed_generated_data(model, sampler, num_timesteps, encoder, fid_sample_size, batch_size)
    
    fid_score = calculate_fid_from_latents(real_latents, fake_latents)
    
    return fid_score


@torch.inference_mode
def note_events_to_pretty_midi(note_array: torch.Tensor | np.ndarray, path="eg.mid", default_program=0):
    if isinstance(note_array, torch.Tensor):
        note_array = note_array.detach().cpu().numpy()
    #scale  and velocities
    note_array[:, 1] *= 127
    # clamp pitches and velocities
    note_array[:, 0] = np.clip(note_array[:, 0], 0, 127)
    note_array[:, 1] = np.clip(note_array[:, 1], 0, 127)
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=default_program)

    current_time = 0.0
    print(note_array.shape)
    for row in note_array:
        pitch, velocity, delta, duration = row
        current_time += delta
        start = current_time
        end = start + duration

        note = pretty_midi.Note(
            velocity=int(velocity),
            pitch=int(pitch),
            start=start,
            end=end
        )
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(path)



# def calculate_fad(real_midi_path, output_data_path, model, sampler, fid_sample_size=1000, batch_size=64, num_timesteps=1000):
#     """
#     Function that calculates the FAD score between real and generated data.

#     :model: denoising network responsible for generating data
#     :sampler: sampler (DDPM or DDIM) that generates data according to the reverse process
#     :ds: dataset with real data
#     :fad_sample_size: number of samples to generate and embed
#     :batch_size: size of the batch for generation
#     :num_timesteps: number of diffusion steps in the generation process

#     :return: FAD score
#     """
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'


#     # calculate FAD score
#     frechet = FrechetAudioDistance(
#         model_name="vggish",
#         sample_rate=16000,
#         use_pca=False, 
#         use_activation=False,
#         verbose=False
#     )

#     fad_score = frechet.score(
#         REAL_PAHT, 
#         GENERATED_PATH, 
#         dtype="float32"
#     )

    
#     return fad_score