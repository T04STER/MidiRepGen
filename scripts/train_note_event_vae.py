from datetime import datetime
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch_optimizer
from src.common.config_utils import get_config, seed_all
from src.common.diagnostic.latent.latent_visualization import plot_tsne, plot_tsne_3d
from src.common.diagnostic.misc import get_random_reconstruction_true_pair_from_note_event
from src.common.diagnostic.summary import show_summary
from src.common.diagnostic.visualize_note_events import compare_reco_true
from src.common.metrics.metrics import get_vae_metrics
from src.models.representation.vae.decoder.mom_vae_decoder import MomVaeDecoder
from src.models.representation.vae.decoder.recurrent_decoder import LSTMVaeDecoderWithTeacherForcingPitchEmbeddedResidualMemory
from src.models.representation.vae.encoder.recurrent_encoder import LSTMVaeEncoderPitchEmbedding
from src.models.representation.vae.vae import RecurrentVaeWithTeacherForcing
from src.models.representation.vae.vae_loss import VaeLossWithCrossEntropy
from src.models.representation.vae.vae_trainer import VaeTrainer


RUN_NAME = f"default_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LATENT_DIM = 1_024 # othervise fails to learn
EPOCHS = 100
ENC_HIDDEN_DIM = 16 * 4
DEC_HIDDEN_DIM = 16 * 4
DEC_FC_OUTPUT = 32
ENC_DEC_LAYERS = 4
BATCH_SIZE = 4096
MAX_DELTA_TIME = 2  # maximum time between notes in seconds and durations helps with stability
MIN_NOTE_DURATION = 0.1 # minimum note duration in seconds, otherwise colapses to zero
MIN_NOTE_DELTA = 0.001 # minimum delta time between notes in seconds, otherwise colapses to zero
SEQUENCE_LENGTH = 8
BETA_STEP = 0.01 # step size for beta value in the VAE loss starting from 0
BETA_MAX = 0.1  # maximum beta value for the VAE loss

OPTIM_WEIGHT_DECAY = 1e-4
OPTIM_LR = 1e-3

DEFAULT_HPARAMETERS = {
    "latent_dim": LATENT_DIM,
    "enc_hidden_dim": ENC_HIDDEN_DIM,
    "dec_hidden_dim": DEC_HIDDEN_DIM,
    "dec_fc_output": DEC_FC_OUTPUT,
    "enc_dec_layers": ENC_DEC_LAYERS,
    "epochs": EPOCHS,
}


def get_model():
    encoder = LSTMVaeEncoderPitchEmbedding(
        4,
        ENC_HIDDEN_DIM,
        LATENT_DIM,
        ENC_DEC_LAYERS,
        linear_scaling_dim=4*3,
        pitch_embedding_dim=4,
        dropout=0.5,
    )
    decoder = MomVaeDecoder(
        LATENT_DIM, 
        DEC_HIDDEN_DIM,
        DEC_FC_OUTPUT,
        ENC_DEC_LAYERS,
        teacher_forcing_ratio=0.6,
        teacher_forcing_decrease=0.05
    )
    vae = RecurrentVaeWithTeacherForcing(encoder, decoder).cuda()
    return vae

def get_dataloader(dataset_path=None) -> DataLoader:
    if dataset_path is not None:
        dataset = pickle.load(open(dataset_path, "rb"))
    else:
        dataset = pickle.load(open("data/preprocessed_note_events.pkl", "rb"))

    def collate_fn(batch):
        batch = torch.stack(batch, dim=0)
        # batch = batch[:,:SEQUENCE_LENGTH,:]
        batch[:, :, 2] = batch[:, :, 2].clamp(min=MIN_NOTE_DELTA, max=MAX_DELTA_TIME)
        batch[:, :, 3] = batch[:, :, 3].clamp(min=MIN_NOTE_DURATION, max=MAX_DELTA_TIME)
        return batch

    midi_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=0,
        shuffle=True,
        collate_fn=collate_fn
    )
    return midi_dataloader

def main():
    config = get_config()
    if "seed" in config:    
        seed_all(config["seed"])
    
    dataloader = get_dataloader()
    vae = get_model()
    optimizer = torch_optimizer.Lamb(
        vae.parameters(),
        lr=OPTIM_LR,
        weight_decay=OPTIM_WEIGHT_DECAY,
    )

    loss_fn = VaeLossWithCrossEntropy(
        reduction="mean",
        beta_max=BETA_MAX,
        beta_step=BETA_STEP,
    )
    with torch.no_grad():
        test_batch = next(iter(dataloader))
        test_batch = test_batch.cuda()
        vae(test_batch)

    show_summary(vae, (BATCH_SIZE, SEQUENCE_LENGTH, 4), batch_size=BATCH_SIZE, dataset=dataloader.dataset)
    
    run_name = config["train_note_event_vae"].get("run_name", RUN_NAME)
    trainer = VaeTrainer(
        vae,
        optimizer,
        loss_fn,
        run_name,
    )
    try: 
        trainer.train(dataloader, epochs=EPOCHS)
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model...")
    trainer.save_full_model(f"models/{run_name}.pt")
    
    writter = trainer.writter

    test_dataloader = get_dataloader("data/preprocessed_note_events_test.pkl")
    metrics = test_model(run_name, vae, config, writter, test_dataloader)

    trainer.log_model_hyperparameters(
        DEFAULT_HPARAMETERS,
        metric_dict=metrics,
    )

@torch.inference_mode()
def eval_representation(model: RecurrentVaeWithTeacherForcing, dataloader, device="cuda"):
    samples = []
    for batch in dataloader:
        batch = batch.to(device)
        reparam = model.encode_with_reparameterization(batch)
        reparam = reparam.detach()
        samples.append(reparam)
        if len(samples*batch.shape[0]) > 20_000:
            break
    fig, ax = plot_tsne_3d(
        samples,
    )
    return fig, ax


def compare_n_random_samples(run_name, vae, dataloader, writter, n=5, strip_to_seq_length=None):
    """
    Compare n random samples from the dataset and their reconstructions using the VAE.
    Args:
        vae (RecurrentVae): The VAE model to use for reconstruction.
        dataloader (DataLoader): DataLoader containing the data to evaluate on.
        n (int): Number of random samples to compare.
        strip_to_seq_length (int, optional): If provided, strips the sequences to this length.
    """
    for i in range(n):
        first_sample, reconstructed_sample = get_random_reconstruction_true_pair_from_note_event(vae, dataloader, strip_to_seq_length)
        fig, ax = compare_reco_true(first_sample, reconstructed_sample)
        ax.set_title(f"Random Sample Comparison {run_name} - Sample {i + 1}")
        writter.add_figure(f"Random Sample Comparison {run_name} - Sample {i + 1}", fig)
        writter.flush()


def test_model(run_name, model, dvc_config, writter, test_dataloader):
    model.eval()
    metrics = get_vae_metrics(
        model,
        test_dataloader,
        device="cuda",
    )
    print(f"Model: {model.__class__.__name__}, Avg IoU: {metrics['avg_iou']:.4f}, Avg Accuracy: {metrics['avg_accuracy']:.4f}, Total Batches: {metrics['total_batches']}")
    try:
        fig, ax = eval_representation(model, test_dataloader, device="cuda")
        ax.set_title(f"TSNE of {run_name} representation")
        writter.add_figure("TSNE Representation", fig)
        writter.close()
        print("TSNE plot saved.")
    except KeyboardInterrupt:
        print("TSNE evaluation interrupted. Skipping...")

    compare_n_random_samples(run_name, model, test_dataloader, writter, n=5)
    return metrics

if __name__ == "__main__":
    main()
