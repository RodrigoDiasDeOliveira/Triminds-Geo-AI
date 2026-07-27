import numpy as np
from scipy.stats import ks_2samp


class DataDriftDetector:
    """Detect data drift using the Kolmogorov-Smirnov test."""

    def check(
        self,
        reference_data: np.ndarray,
        new_data: np.ndarray,
        threshold: float = 0.05,
    ) -> list[dict[str, int | float | bool]]:
        """Compare reference and new datasets feature by feature."""

        drift_results: list[dict[str, int | float | bool]] = []

        for i in range(reference_data.shape[1]):
            _, p_value = ks_2samp(
                reference_data[:, i],
                new_data[:, i],
            )

            drift_results.append(
                {
                    "feature": i,
                    "p_value": p_value,
                    "drift": p_value < threshold,
                }
            )

        return drift_results
