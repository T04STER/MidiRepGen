

from datetime import datetime


RUN_NAME = f"default_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LATENT_DIM = 1_024 # sequence length of 4, so we have 16 dimensions per step 
EPOCHS = 200
ENC_HIDDEN_DIM = 16 * 4
DEC_HIDDEN_DIM = 16 * 4
DEC_FC_OUTPUT = 32
ENC_DEC_LAYERS = 4
BATCH_SIZE = 4096


DEFAULT_HPARAMETERS = {
    "latent_dim": LATENT_DIM,
    "enc_hidden_dim": ENC_HIDDEN_DIM,
    "dec_hidden_dim": DEC_HIDDEN_DIM,
    "dec_fc_output": DEC_FC_OUTPUT,
    "enc_dec_layers": ENC_DEC_LAYERS,
    "run_name": RUN_NAME,
    "epochs": EPOCHS,
}

def main():