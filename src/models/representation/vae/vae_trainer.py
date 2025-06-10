from src.common import torch_writter
from src.models.representation.vae.vae_loss import VaeLoss


class VaeTrainer:
    def __init__(self, model, optimizer, loss_fn: VaeLoss, run_name=None, log_every_n_epochs=1):
        self.model = model
        self.run_name = run_name if run_name else model.__class__.__name__
        self.log_every_n_epochs = log_every_n_epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.writter = torch_writter.get_writter(self.run_name, prefix="vae")
        self.device = next(model.parameters()).device

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        reconstructed, mu, log_var = self.model(data)
        loss = self.loss_fn(reconstructed, data, mu, log_var)        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                loss = self.train_step(batch)
                total_loss += loss
                if (i + 1) % self.log_every_n_epochs == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss/i:.4f}, Run: {self.run_name}")
                    self.writter.add_scalar("Loss/Train", total_loss/i, i + epoch * len(dataloader))
            
            avg_loss = total_loss / len(dataloader)
            
        print("Training complete.")
        self.writter.flush()
        self.writter.close()