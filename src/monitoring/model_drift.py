import numpy as np


class ModelDriftDetector:

    def check(self, y_true, y_pred, threshold=0.1):

        accuracy = np.mean(np.array(y_true) == np.array(y_pred))

        return {
            "accuracy": accuracy,
            "drift_detected": accuracy < (1 - threshold)
        }