


import torch
from tqdm import tqdm
from src.common import torch_writter
from src.models.diffusion.ddpm import DDPM


class DDPMTrainer:
    def __init__(self, diffusion: DDPM, model, optimizer, run_name=None, use_tqdm=True, device='cuda'):
        """
        Initializes the DDPMTrainer with a diffusion model, a neural network model, and a dataset.
        
        Args:
            diffusion (DDPM|DDIM): The diffusion model used for training.
            model (torch.nn.Module): The neural network model to be trained.
            
            batch_size (int): The size of the batches for training. Default is 5000.
            device (str): The device to run the training on ('cuda' or 'cpu'). Default is 'cuda'.
        """
        self.diffusion = diffusion
        self.model = model.to(device)
        self.device = device
        self.use_tqdm = use_tqdm
        self.run_name = run_name if run_name else model.__class__.__name__
        self.optimizer = optimizer
        self.writter = torch_writter.get_writter(self.run_name, prefix="vae")
        # self.clip_grad_norm = 1.0  # Default gradient clipping value


    def train_step(self, data):
        self.optimizer.zero_grad()
        loss = self.diffusion.train_losses(self.model, data)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), None

    def train(self, dataloader, epochs):
        iter_epochs = tqdm.tqdm(range(epochs))  if self.use_tqdm else range(epochs)
        for epoch in iter_epochs:
            self.model.train()
            total_loss = 0
            for i, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                loss, others = self.train_step(batch)
                total_loss += loss
            avg_loss = total_loss / len(dataloader)
            self.writter.add_scalar("Loss/Train", avg_loss, epoch)
            self.step_epoch()
            if self.use_tqdm:
                iter_epochs.set_description(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            
        print("Training complete.")
        self.writter.flush()
        self.writter.close()

    
    def step_epoch(self):
        """
        Perform any necessary operations at the end of an epoch.
        This can be used to update the model's state or perform logging.
        """
        pass