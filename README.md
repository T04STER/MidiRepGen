# MIDI Music Generation with Generative Models
This project explores automatic generation of simple musical compositions in MIDI format using modern generative models. We implement and compare three different approaches: Variational Autoencoder (VAE) with LSTM layers, Denoising Diffusion Probabilistic Model (DDPM), and its deterministic variant DDIM.
## 🎵 Overview
The project addresses the complex problem of music generation by converting musical data into sequential MIDI token format and comparing the quality of generated compositions in terms of structure, rhythm, and diversity. Our research shows that diffusion models, despite higher computational requirements, offer significantly better music quality compared to classical sequential solutions.

## 📊 Dataset
We use the GiantMIDI-Piano dataset consisting of 10,855 MIDI songs (.mid format):

Training split: 90% of the data
Evaluation split: 10% of the data
Preprocessing: Songs divided into non-overlapping 8-note windows, resulting in ~4.3M training samples
Library: pretty_midi for MIDI processing

## 🏗️ Architecture
Condensed Note Events
Our novel representation encodes each note as: X_i ∈ P × V × S × D where:

P = {1, ..., 127}: pitch/tone of the note
V = (0, 1]: velocity/dynamics
S = (0, s_max]: time in seconds between previous and current key press
D = (0, d_max]: note duration in seconds

Empirically set limits: s_max = 2s, d_max = 2s
Models
1. Variational Autoencoder (VAE)

Seq2Seq architecture with BiLSTM encoder
Autoregressive LSTM decoder with memory overwrite module
Annealing Teacher Forcing for stable training
Loss function combines cross-entropy, MSE, and KL divergence

2. DDPM (Denoising Diffusion Probabilistic Model)

Forward diffusion process adds Gaussian noise step-by-step
Reverse process learns to denoise and generate samples
Trained to predict added noise rather than direct reconstruction

3. DDIM (Denoising Diffusion Implicit Model)

Deterministic variant of DDPM
Faster sampling without quality loss
Same training procedure as DDPM but deterministic generation

## 🚀 Getting Started
Create .venv environment
```
    python -m venv .venv 
```
Activate and install requirements
```
    pip install -r requirements.txt
```
Reproduce DVC pipeline (downloads data and runs VAE training)
```
    make
```


## Generation:
-  Generate 8-note sequences
-  Use random walk or interpolation for longer compositions


## 📈 Results
Performance Metrics

| Method | Accuracy | Average IoU | Generation Speed (10k samples) | FAD (↓)       |
|--------|----------|-------------|-------------------------------|---------------|
| VAE    | 0.903    | 0.316       | ~3:40 (fastest)               | **0.895**     |
| DDIM   | –        | –           | ~9:00+                        | **0.603**     |
| DDPM   | –        | –           | ~9:00+                        | **0.565**     |

- **VAE** achieves the highest speed with moderate quality.
- **DDIM** produces the best overall quality (lowest FAD).
- **DDPM** slightly outperforms DDIM in FAD but is still slow.

## Key Findings

Diffusion models produce higher quality music despite computational overhead
VAE struggles with sequence alignment (snowball effect)
DDIM offers best quality-speed tradeoff among diffusion approaches

## 🔧 Training Details

Optimizer: LAMB (lr=1e-3, weight_decay=1e-4)
Batch Size: 4096
Training Time: ~8 hours on RTX 4070/5070 SUPER TI
Model Size: ~2M parameters each
Regularization: Weight decay and dropout in LSTM layers
