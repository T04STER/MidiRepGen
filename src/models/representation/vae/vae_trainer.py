import os
import os.path
import torch
import tqdm
from src.common import torch_writter
from src.models.representation.vae.vae_loss import VaeLoss
from pathlib import Path

class VaeTrainer:
    def __init__(self, model, optimizer, loss_fn: VaeLoss, run_name=None, log_every_n_epochs=1, use_tqdm=True):
        self.use_tqdm = use_tqdm
        self.model = model
        self.run_name = run_name if run_name else model.__class__.__name__
        self.log_every_n_epochs = log_every_n_epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.writter = torch_writter.get_writter(self.run_name, prefix="vae")
        self.device = next(model.parameters()).device


    def train_step(self, data):
        self.optimizer.zero_grad()
        reconstructed, mu, log_var = self.model(data)
        loss, others = self.loss_fn(reconstructed, data, mu, log_var)        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), others

    def train(self, dataloader, epochs):
        other_losses = {
            0: "CE Loss",
            1: "MSE Loss",
            2: "KL Divergence",
        }
        
        iter_epochs = tqdm.tqdm(range(epochs))  if self.use_tqdm else range(epochs)
        for epoch in iter_epochs:
            self.model.train()
            total_loss = 0
            total_others = [0] * len(other_losses)
            for i, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                loss, others = self.train_step(batch)
                total_loss += loss
                for j, other in enumerate(others):
                    total_others[j] += other
            self.model.decoder.step_teacher_forcing()
            avg_loss = total_loss / len(dataloader)
            self.writter.add_scalar("Loss/Train", avg_loss, epoch)
            for j, other in enumerate(total_others):
                name = other_losses.get(j, f"Other Loss {j}")
                other /= len(dataloader)
                self.writter.add_scalar(f"Loss/Train/{name}", other, epoch)
            if self.use_tqdm:
                iter_epochs.set_description(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            
        print("Training complete.")
        self.writter.flush()
        self.writter.close()
    
    def graph_model(self, input_tensor):
        """
        Visualize the model's architecture using tensorboard.
        
        :param input_tensor: A sample input tensor to trace the model.
        """
        self.model.eval()
        with torch.no_grad():
            self.writter.add_graph(self.model, input_tensor)
    
    def log_model_hyperparameters(self, hyperparams: dict):
        self.writter.add_hparams(
            hparam_dict=hyperparams,
            metric_dict={}
        )

    def save_model(self, path_to_save):
        """
        Save the model's state dictionary to the specified path.
        :param path: Path to directory to save model.
        """
        path_to_save = Path(path_to_save)
        print(f"Saving model to {path_to_save}")
        path_to_save = path_to_save / f"{self.run_name}.pth"
        if not path_to_save.parent.exists():
            path_to_save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path_to_save)
        print(f"Model saved to {path_to_save}")
    
    def load_model(self, path):
        """
        Load the model's state dictionary from the specified path.
        :param path: Path to load the model from.
        """
        path_to_load = Path(path)
        if not path_to_load.exists():
            raise FileNotFoundError(f"Model file not found: {path_to_load}")
        print(f"Loading model from {path_to_load}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")
