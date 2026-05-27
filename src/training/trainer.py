import torch
from tqdm import tqdm

from src.mlops.tracking.mlflow_logger import MLflowLogger
from src.mlops.logging.logger import Logger
from src.mlops.registry.model_registry import ModelRegistry


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        experiment_name="satellite-exp"
    ):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # MLOps layer
        self.logger = Logger()
        self.mlflow = MLflowLogger(experiment_name)
        self.registry = ModelRegistry()

    def train_epoch(self):

        self.model.train()
        total_loss = 0

        for images, labels in tqdm(self.train_loader):

            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):

        self.model.eval()
        total_loss = 0

        with torch.no_grad():

            for images, labels in self.val_loader:

                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, epochs):

        with self.mlflow.start_run():

            self.logger.info("Training started")

            for epoch in range(epochs):

                train_loss = self.train_epoch()
                val_loss = self.validate()

                # MLflow tracking
                self.mlflow.log_metric("train_loss", train_loss, step=epoch)
                self.mlflow.log_metric("val_loss", val_loss, step=epoch)

                self.logger.info(
                    f"Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f}"
                )

            # Save model
            model_path = "model.pth"
            torch.save(self.model.state_dict(), model_path)

            # MLflow model log
            self.mlflow.log_model(self.model)

            # Registry
            self.registry.register_model(
                model_name="satellite-model",
                version=1,
                metadata={
                    "epochs": epochs,
                    "framework": "pytorch"
                }
            )

            self.mlflow.end_run()

            self.logger.info("Training completed")