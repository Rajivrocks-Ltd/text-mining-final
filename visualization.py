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

    @staticmethod
    def combine_data(file_path_1, file_path_2, file_name=None):
        """
        Combine two Excel files into a single pandas DataFrame.
        :param file_path_1: Path to the first Excel file.
        :param file_path_2: Path to the second Excel file.
        :param file_name: Optional path to save the combined data.
        :return: Combined pandas DataFrame.
        """
        try:
            data1 = pd.read_excel(file_path_1)
            data2 = pd.read_excel(file_path_2)

            # Ensure both files have the required columns
            required_columns = ['Train Size', 'K-Fold', 'Test F1', 'Model']
            if not all(column in data1.columns for column in required_columns):
                raise ValueError(f"The first input file must contain the following columns: {required_columns}")
            if not all(column in data2.columns for column in required_columns):
                raise ValueError(f"The second input file must contain the following columns: {required_columns}")

            # Concatenate the two datasets
            combined_data = pd.concat([data1, data2], ignore_index=True)
            print("Combined data successfully.")

            # Save the combined data if a save path is provided
            combined_data.to_excel(f"experimental_results/sheets/{file_name}.xlsx", index=False)
            print(f"Combined data saved to {file_name}")

            save_path = f"experimental_results/sheets/{file_name}.xlsx"

            return save_path

        except Exception as e:
            raise ValueError(f"Failed to combine data from the files: {e}")

    def check_folds(self, expected_folds=None):
        """
        Check the number of entries (folds) for each train size and model combination.
        :param expected_folds: Expected number of folds. If specified, validate against this number.
        """
        grouped = self.data.groupby(['Model', 'Train Size'])['K-Fold'].count()

        if expected_folds is not None:
            # Check for mismatches against the expected number of folds
            invalid_entries = grouped[grouped != expected_folds]
            if invalid_entries.empty:
                print(f"All train size and model combinations have exactly {expected_folds} folds.")
            else:
                print(f"The following model and train size combinations do not have exactly {expected_folds} folds:")
                print(invalid_entries)
        else:
            # Display the fold counts for all combinations
            print("Fold counts for each model and train size combination:")
            print(grouped)

    def plot_model_performance(self, save=False, name=None):
        """
        Plot the performance of models with shaded variance.
        :param name: Specify the name of the plot using the K-Fold and Train Size.
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
        plt.title(f'Model Performance vs. Train Size: {name}', fontsize=16)
        plt.legend(title='Model', fontsize=12)
        plt.grid(True)
        if save:
            plt.savefig(f'experimental_results/model_performance_{name}.png')
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
    def _plot_differences(results, metric, save=False, name=None):
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
        title = f'Percentage Difference in Performance vs. Sample Size: {name}' if metric == "percentage" else f'Raw Difference in Performance vs. Sample Size: {name}'
        plt.title(title, fontsize=16)
        plt.legend(title='Model', fontsize=12)
        plt.grid(True)
        if save:
            plt.savefig(f'experimental_results/{metric}_{name}.png')
        else:
            plt.show()

    def plot_percentage_difference(self, include_spread=False, save=False, name=None):
        """
        Plot the percentage difference in performance between consecutive sample sizes.
        :param name: Specify the name of the plot using the K-Fold and Train Size.
        :param save: Save the generated plot as a PNG file, yes or no.
        :param include_spread: Boolean to include the spread (min and max) of the k-folds.
        """
        results = self._calculate_differences(metric="percentage", include_spread=include_spread)
        self._plot_differences(results, metric="percentage", save=save, name=name)

    def plot_raw_difference(self, include_spread=False, save=False, name=None):
        """
        Plot the absolute difference in performance between consecutive sample sizes.
        :param name: Specify the name of the plot using the K-Fold and Train Size.
        :param save: Save the generated plot as a PNG file, yes or no.
        :param include_spread: Boolean to include the spread (min and max) of the k-folds.
        """
        results = self._calculate_differences(metric="raw", include_spread=include_spread)
        self._plot_differences(results, metric="raw", save=save, name=name)


if __name__ == '__main__':
    path = "experimental_results/sheets/Experiments_full_labeled.xlsx"
    path2 = "experimental_results/sheets/Experiments_moreksplits10_lesssteps20_for_smoother_graphh.xlsx"

    path_biobert = "experimental_results/sheets/Experiments_full_labeled_biobert.xlsx"
    path_biobert2 = "experimental_results/sheets/Experiments_moreksplits10_lesssteps20_for_smoother_graphh_biobert.xlsx"

    # Plotting 5-Fold 5 Train
    name = "5-Fold 5 Train Size"
    combined_data = Plotter.combine_data(path, path_biobert, f"combined_data_{name}")
    plotter = Plotter(combined_data)
    plotter.check_folds(expected_folds=5)
    plotter.plot_model_performance(save=True, name=name)
    plotter.plot_percentage_difference(include_spread=False, save=True, name=name)
    plotter.plot_raw_difference(include_spread=False, save=True, name=name)

    # Plotting 10-Fold 20 Train
    name = "10-Fold 20 Train Size"
    combined_data = Plotter.combine_data(path2, path_biobert2, f"combined_data_{name}")
    plotter = Plotter(combined_data)
    plotter.check_folds(expected_folds=10)
    plotter.plot_model_performance(save=True, name=name)
    plotter.plot_percentage_difference(include_spread=False, save=True, name=name)
    plotter.plot_raw_difference(include_spread=False, save=True, name=name)

