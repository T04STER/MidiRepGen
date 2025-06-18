from datetime import datetime
import pickle
import torch
from torch.utils.data import DataLoader
import torch_optimizer
from src.common.diagnostic.summary import show_summary
from src.dataloader.dataset import PianoRollMidiDataset
from src.models.representation.vae.decoder.recurrent_decoder import LSTMVaeDecoderWithTeacherForcingPitchEmbeddedResidualMemory
from src.models.representation.vae.encoder.recurrent_encoder import LSTMVaeEncoderPitchEmbedding
from src.models.representation.vae.vae import RecurrentVaeWithTeacherForcing
from src.models.representation.vae.vae_loss import VaeLossWithCrossEntropy
from src.models.representation.vae.vae_trainer import VaeTrainer


RUN_NAME = f"default_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LATENT_DIM = 1_024 # sequence length of 4, so we have 16 dimensions per step 
EPOCHS = 200
ENC_HIDDEN_DIM = 16 * 4
DEC_HIDDEN_DIM = 16 * 4
DEC_FC_OUTPUT = 32
ENC_DEC_LAYERS = 4
BATCH_SIZE = 4096
MAX_DELTA_TIME = 2  # maximum time between notes in seconds and durations helps with stability
MIN_NOTE_DURATION = 0.1 # minimum note duration in seconds, otherwise colapses to zero
SEQUENCE_LENGTH = 4

DEFAULT_HPARAMETERS = {
    "latent_dim": LATENT_DIM,
    "enc_hidden_dim": ENC_HIDDEN_DIM,
    "dec_hidden_dim": DEC_HIDDEN_DIM,
    "dec_fc_output": DEC_FC_OUTPUT,
    "enc_dec_layers": ENC_DEC_LAYERS,
    "run_name": RUN_NAME,
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
    decoder = LSTMVaeDecoderWithTeacherForcingPitchEmbeddedResidualMemory(
        LATENT_DIM, 
        DEC_HIDDEN_DIM,
        DEC_FC_OUTPUT,
        ENC_DEC_LAYERS,
        teacher_forcing_ratio=0.6,
        teacher_forcing_decrease=0.05
    )
    vae = RecurrentVaeWithTeacherForcing(encoder, decoder).cuda()
    return vae

def get_dataloader() -> DataLoader:
    dataset = pickle.load(open("data/preprocessed_note_events.pkl", "rb"))

    def collate_fn(batch):
        batch = torch.stack(batch, dim=0)
        # batch = batch[:,:SEQUENCE_LENGTH,:]
        batch[:, :, 2:] = batch[:, :, 2:].clamp(min=MIN_NOTE_DURATION, max=MAX_DELTA_TIME)
        return batch

    midi_dataloader = DataLoader(
        dataset,
        batch_size=4096,
        pin_memory=True,
        num_workers=0,
        shuffle=True,
        collate_fn=collate_fn
    )
    return midi_dataloader

def main():
    dataloader = get_dataloader()
    vae = get_model()
    optimizer = torch_optimizer.Lamb(
        vae.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    loss_fn = VaeLossWithCrossEntropy(
        reduction="mean",
        beta_max=1,
        beta_step=0.01,
    )
    with torch.no_grad():
        test_batch = next(iter(dataloader))
        test_batch = test_batch.cuda()
        vae(test_batch)

    show_summary(vae, (BATCH_SIZE, SEQUENCE_LENGTH, 4), batch_size=BATCH_SIZE)
    
    trainer = VaeTrainer(
        vae,
        optimizer,
        loss_fn,
        RUN_NAME,
    )
    trainer.train(dataloader, epochs=EPOCHS)
    trainer.save_model(f"models/{RUN_NAME}.pt")
    trainer.log_model_hyperparameters(
        DEFAULT_HPARAMETERS,
    )


if __name__ == "__main__":
    main()
