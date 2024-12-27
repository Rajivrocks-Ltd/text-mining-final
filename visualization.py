import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Plotter:
    def __init__(self, file_path):
        """
        Initialize the Plotter with the dataset from an Excel file.
        :param file_path: Path to the Excel file containing the data.
        """
        self.data = self._load_data(file_path)

    @staticmethod
    def _load_data(file_path):
        """
        Load the Excel file into a pandas DataFrame.
        :param file_path: Path to the Excel file.
        :return: pandas DataFrame with the data.
        """
        try:
            data = pd.read_excel(file_path)
            print("Columns in the Excel sheet:", data.columns.tolist())
            required_columns = ['Train Size', 'K-Fold', 'Test F1', 'Model']
            if not all(column in data.columns for column in required_columns):
                raise ValueError(f"The input file must contain the following columns: {required_columns}")
            return data
        except Exception as e:
            raise ValueError(f"Failed to load data from {file_path}: {e}")

    def check_folds(self):
        """
        Check that each train size and model combination has exactly 5 entries (folds).
        """
        grouped = self.data.groupby(['Model', 'Train Size'])['K-Fold'].count()
        invalid_entries = grouped[grouped != 5]

        if invalid_entries.empty:
            print("All train size and model combinations have exactly 5 folds.")
        else:
            print("The following model and train size combinations do not have exactly 5 folds:")
            print(invalid_entries)

    def plot_model_performance(self, save=False):
        """
        Plot the performance of models with shaded variance.
        :param save: Save the generated plot as a PNG file, yes or no.
        """
        plt.figure(figsize=(12, 8))

        models = self.data['Model'].unique()
        training_size = sorted(self.data['Train Size'].unique())

        for model in models:
            model_data = self.data[self.data['Model'] == model]

            means = []
            mins = []
            maxs = []

            for size in training_size:
                size_data = model_data[model_data['Train Size'] == size]['Test F1']

                means.append(size_data.mean())
                mins.append(size_data.min())
                maxs.append(size_data.max())

            means = np.array(means)
            mins = np.array(mins)
            maxs = np.array(maxs)

            plt.plot(training_size, means, label=f'{model}')
            plt.fill_between(training_size, mins, maxs, alpha=0.2)

        plt.xlabel('Train Size', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.title('Model Performance vs. Train Size', fontsize=16)
        plt.legend(title='Model', fontsize=12)
        plt.grid(True)
        if save:
            plt.savefig('experimental_results/model_performance.png')
        else:
            plt.show()

    def _calculate_differences(self, metric="percentage", include_spread=False):
        """
        Helper function to calculate differences (percentage or absolute) between consecutive sample sizes.
        :param metric: "percentage" or "raw" to calculate the respective differences.
        :param include_spread: Boolean to include the spread (min and max) of the k-folds.
        :return: A dictionary of results for plotting.
        """
        results = {}
        models = self.data['Model'].unique()
        sample_sizes = sorted(self.data['Train Size'].unique())

        for model in models:
            model_data = self.data[self.data['Model'] == model]

            means = []
            mins = []
            maxs = []

            for size in sample_sizes:
                size_data = model_data[model_data['Train Size'] == size]['Test F1']
                means.append(size_data.mean())
                if include_spread:
                    mins.append(size_data.min())
                    maxs.append(size_data.max())

            means = np.array(means)

            if metric == "percentage":
                differences = (np.diff(means) / means[:-1]) * 100
            elif metric == "raw":
                differences = np.diff(means)
            else:
                raise ValueError("Invalid metric. Use 'percentage' or 'raw'.")

            result = {
                "sample_sizes": sample_sizes[1:],
                "differences": differences,
            }

            if include_spread:
                if metric == "percentage":
                    min_diff = (np.diff(np.array(mins)) / np.array(mins[:-1])) * 100
                    max_diff = (np.diff(np.array(maxs)) / np.array(maxs[:-1])) * 100
                elif metric == "absolute":
                    min_diff = np.abs(np.diff(np.array(mins)))
                    max_diff = np.abs(np.diff(np.array(maxs)))

                result["min_diff"] = min_diff
                result["max_diff"] = max_diff

            results[model] = result

        return results

    @staticmethod
    def _plot_differences(results, metric, save=False):
        """
        Helper function to plot differences.
        :param results: Dictionary of results from _calculate_differences.
        :param metric: The type of difference being plotted (e.g., 'percentage' or 'raw').
        """
        plt.figure(figsize=(12, 8))

        for model, data in results.items():
            plt.plot(data["sample_sizes"], data["differences"], label=f'{model}')
            if "min_diff" in data and "max_diff" in data:
                plt.fill_between(data["sample_sizes"], data["min_diff"], data["max_diff"], alpha=0.2)

        plt.xlabel('Sample Size', fontsize=14)
        y_label = 'Percentage Difference (%)' if metric == "percentage" else 'Raw Difference in F1 Score'
        plt.ylabel(y_label, fontsize=14)
        title = 'Percentage Difference in Performance vs. Sample Size' if metric == "percentage" else 'Raw Difference in Performance vs. Sample Size'
        plt.title(title, fontsize=16)
        plt.legend(title='Model', fontsize=12)
        plt.grid(True)
        if save:
            plt.savefig(f'experimental_results/{metric}.png')
        else:
            plt.show()

    def plot_percentage_difference(self, include_spread=False, save=False):
        """
        Plot the percentage difference in performance between consecutive sample sizes.
        :param save: Save the generated plot as a PNG file, yes or no.
        :param include_spread: Boolean to include the spread (min and max) of the k-folds.
        """
        results = self._calculate_differences(metric="percentage", include_spread=include_spread)
        self._plot_differences(results, metric="percentage", save=save)

    def plot_raw_difference(self, include_spread=False, save=False):
        """
        Plot the absolute difference in performance between consecutive sample sizes.
        :param save: Save the generated plot as a PNG file, yes or no.
        :param include_spread: Boolean to include the spread (min and max) of the k-folds.
        """
        results = self._calculate_differences(metric="raw", include_spread=include_spread)
        self._plot_differences(results, metric="raw", save=save)


if __name__ == '__main__':
    path = "experimental_results/Experiments_full_labeled.xlsx"
    plotter = Plotter(path)
    plotter.check_folds()
    # plotter.plot_model_performance(save=True)
    # plotter.plot_percentage_difference(include_spread=False, save=True)
    # plotter.plot_raw_difference(include_spread=False, save=True)

