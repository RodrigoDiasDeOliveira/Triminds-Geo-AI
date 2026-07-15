import torch


class EarlyStopping:
    def __init__(self, patience=5):

        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss):

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0

            return False

        self.counter += 1

        return self.counter >= self.patience


def save_checkpoint(model, path):

    torch.save(model.state_dict(), path)
