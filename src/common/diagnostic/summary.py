


import torch


__pretty_int = lambda n: f"{n:,}".replace(",", " ")


def show_summary(model: torch.nn.Module, input_shape=None, batch_size=None, dataset=None, outfile=None):
    """
        Show summary of the model, its parameters, and dataset information.
        :param model: The model to summarize.
        :param input_shape: Shape of the input data (optional).
        :param batch_size: Batch size used for training (optional).
        :param dataset: Dataset used for training (optional).
        :param outfile: File to save the summary (optional).
        :return: None
    """
    print(model)
    p_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}")
    
    print(f"Number of parameters: {__pretty_int(p_count)}")
    print(f"Number of trainable parameters: {__pretty_int(sum(p.numel() for p in model.parameters() if p.requires_grad))}")
    total_param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Total parameter memory: {total_param_memory / (1024 ** 2):.2f} MB")
    if input_shape is not None:
        print(f"Input shape: {input_shape}")
    if batch_size is not None:
        print(f"Batch size: {batch_size}")
    if dataset is not None and input_shape is not None and batch_size is not None:
        print(f"Dataset size: {__pretty_int(len(dataset))} samples")
        ratio = p_count / len(dataset)
        if p_count > len(dataset):
            print("Warning: Model has more parameters than samples in the dataset. This may lead to overfitting.")
        print(f"Parameter to sample ratio: {ratio:.2f}")
    
    if outfile is not None:
        with open(outfile, "w") as f:
            f.write(f"Model: {model.__class__.__name__}\n")
            f.write(f"Number of parameters: {__pretty_int(p_count)}\n")
            f.write(f"Number of trainable parameters: {__pretty_int(sum(p.numel() for p in model.parameters() if p.requires_grad))}\n")
            f.write(f"Total parameter memory: {total_param_memory / (1024 ** 2):.2f} MB\n")
            if input_shape is not None:
                f.write(f"Input shape: {input_shape}\n")
            if batch_size is not None:
                f.write(f"Batch size: {batch_size}\n")
            if dataset is not None and input_shape is not None and batch_size is not None:
                f.write(f"Dataset size: {__pretty_int(len(dataset))} samples\n")
                f.write(f"Parameter to sample ratio: {ratio:.2f}\n")