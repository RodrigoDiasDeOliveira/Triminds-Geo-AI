import torch
from pathlib import Path
from tqdm import tqdm

from src.mlops.tracking.mlflow_logger import MLflowLogger
from src.mlops.logging.logger import Logger
from src.mlops.registry.model_registry import ModelRegistry


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, config=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config or {}

        # MLOps
        self.logger = Logger()
        self.mlflow = MLflowLogger(self.config.get("experiment_name", "satellite-exp"))
        self.registry = ModelRegistry()

        # Paths - Cria diretório artifacts/
        self.artifacts_dir = Path(self.config.get("paths", {}).get("artifacts_dir", "artifacts"))
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        """Treinamento de uma época"""
        self.model.train()
        total_loss = 0.0

        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        """Validação completa com loss + métricas"""
        self.model.eval()
        total_loss = 0.0
        
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        # Concatena batches
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        avg_loss = total_loss / len(self.val_loader)
        metrics = self._compute_metrics(all_outputs, all_labels)

        # Log das métricas
        self.logger.info(
            f"Validation - Loss: {avg_loss:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"F1-Score: {metrics['f1']:.4f}"
        )

        return avg_loss, metrics

    def _compute_metrics(self, outputs, labels):
        """Calcula todas as métricas de classificação"""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        import numpy as np

        preds = torch.argmax(outputs, dim=1).numpy()
        labels_np = labels.numpy()

        return {
            "accuracy": float(accuracy_score(labels_np, preds)),
            "f1": float(f1_score(labels_np, preds, average='weighted', zero_division=0)),
            "precision": float(precision_score(labels_np, preds, average='weighted', zero_division=0)),
            "recall": float(recall_score(labels_np, preds, average='weighted', zero_division=0))
        }

    def train(self, epochs: int):
        """Loop principal de treinamento com Early Stopping"""
        best_val_loss = float('inf')
        patience = self.config.get("training", {}).get("early_stopping_patience", 8)
        patience_counter = 0

        with self.mlflow.start_run():
            self.logger.info("🚀 Training started")

            for epoch in range(epochs):
                train_loss = self.train_epoch()
                val_loss, val_metrics = self.validate()

                # Early Stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model("best_model.pth")
                    self.logger.info(f"✅ New best model saved at epoch {epoch}")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.logger.info(f"⏹️ Early stopping triggered at epoch {epoch}")
                    break

                # Logging no MLflow
                self.mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                    "epoch": epoch
                })

                self.logger.info(
                    f"Epoch {epoch:2d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

            # Salva modelo final
            final_path = self.save_model("model_final.pth")
            self.logger.info(f"💾 Final model saved at: {final_path}")

            # Log e registro do modelo
            self.mlflow.log_model(self.model, "model")
            self.registry.register_model(
                model_name="satellite-model",
                version=1,
                metadata={
                    "epochs": epochs,
                    "framework": "pytorch",
                    "best_val_loss": best_val_loss
                }
            )

            self.logger.info("🎉 Training completed successfully!")

    def save_model(self, filename: str):
        """Salva modelo no diretório artifacts/"""
        model_path = self.artifacts_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'epoch': 'best' if 'best' in filename else 'final'
        }, model_path)
        return model_path