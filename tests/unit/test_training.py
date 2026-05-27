import torch
from src.training.trainer import Trainer


def test_training_step(dummy_image_batch, dummy_labels):

    model = torch.nn.Linear(10, 10)

    trainer = Trainer(
        model=model,
        train_loader=[(dummy_image_batch, dummy_labels)],
        val_loader=[(dummy_image_batch, dummy_labels)],
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters()),
        device="cpu"
    )

    loss = trainer.train_epoch()

    assert loss >= 0