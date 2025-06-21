

from pathlib import Path
import torch


class TrainerBase:
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from {path_to_load}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

    
    def save_full_model(self, path_to_save):
        """
        Save the entire model (including architecture) to the specified path.
        :param path: Path to directory to save model.
        """
        path_to_save = Path(path_to_save)
        print(f"Saving full model to {path_to_save}")
        if not path_to_save.parent.exists():
            path_to_save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, path_to_save)
        print(f"Full model saved to {path_to_save}")

    @staticmethod
    def load_full_model(path):
        """
        Load the entire model (including architecture) from the specified path.
        :param path: Path to load the model from.
        :return: The loaded model.
        """
        path_to_load = Path(path)
        if not path_to_load.exists():
            raise FileNotFoundError(f"Full model file not found: {path_to_load}")
        print(f"Loading full model from {path_to_load}")
        model = torch.load(path_to_load)
        print(f"Full model loaded from {path}")
        return model
