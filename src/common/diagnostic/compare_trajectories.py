import matplotlib.pyplot as plt
import torch

def _plot_trajectories(ds, trajectories, name, limit=1000):
    plt.figure(figsize=(12, 5))

    subset = [ds[i] for i in range(limit)]
    x = [item[0] for item in subset]
    y = [item[1] for item in subset]

    plt.subplot(1, 2, 1)
    plt.scatter(x, y)
    plt.plot(trajectories[0, 0, 0], trajectories[0, 0, 1], color='black')
    plt.title(f"Jedna Trajektoria dla {name}")

    plt.subplot(1, 2, 2)
    plt.scatter(x, y)
    plt.plot(trajectories[0, 1:11, 0], trajectories[0, 1:11, 1], color='black')
    plt.title(f"Dziesięć Trajektorii dla {name}")

    plt.show()

def compare_trajectories(ds, model_1, model_2, model_1_name, model_2_name, model, samples_number, seq_len = 128, feature_dim = 4):
    noise = torch.randn(samples_number, seq_len, feature_dim, device=model_2.device)
    _, trajectories_1 =model_1.p_sample_loop(model, noise, return_trajectory=True, quiet=False)
    _, trajectories_2 =model_2.p_sample_loop(model, noise, return_trajectory=True, quiet=False)
    _plot_trajectories(ds, trajectories_1, model_1_name)
    _plot_trajectories(ds, trajectories_2, model_2_name)
