from scipy.stats import ks_2samp


class DataDriftDetector:
    def check(self, reference_data, new_data, threshold=0.05):

        drift_results = []

        for i in range(reference_data.shape[1]):
            stat, p_value = ks_2samp(reference_data[:, i], new_data[:, i])

            drift_results.append({"feature": i, "p_value": p_value, "drift": p_value < threshold})

        return drift_results
