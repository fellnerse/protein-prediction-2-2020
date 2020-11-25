#%%
import json
from pathlib import Path

results_file = Path("sebastian/results/linear_search_results.json")

with open(results_file, "r") as f:
    results = json.load(f)

#%% binary case
import matplotlib.pyplot as plt


def extract_data(class_mode: str, total_set_size):
    parameter_values = []
    accuracies = []
    recall = []
    precision = []
    unmatched_total = []
    unmatched_e = []
    unmatched_non_e = []
    for result in results["performances"]:
        parameter_values += [result[0]]

        binary = result[2][class_mode]

        accuracies += [binary["accuracy"] * 100]
        try:
            recall += [binary["recall"] * 100]
            precision += [binary["precision"] * 100]

            unmatched_e += [binary["unmatched"]["enzyme"] / total_set_size * 100]
            unmatched_non_e += [
                binary["unmatched"]["non-enzyme"] / total_set_size * 100
            ]

            unmatched_total += [unmatched_non_e[-1] + unmatched_e[-1]]
        except KeyError:
            pass

    return (
        parameter_values,
        accuracies,
        recall,
        precision,
        unmatched_total,
        unmatched_e,
        unmatched_non_e,
    )


#%%
def plot_metric(metric_list, y_axis_description: str):
    try:
        plt.plot(parameter_values, metric_list, label=y_axis_description)
        plt.xlabel("sensitivityof 'mmseq -s'")
        plt.ylabel("value in %")
    except ValueError:
        pass


def plot_binary_metrics(title: str):
    plot_metric(accuracies, "accuracy in %")
    plot_metric(recall, "recall in %")
    plot_metric(precision, "precision in %")
    plt.ylim(ymin=0, ymax=100)
    plt.legend()
    plt.title(title)
    plt.show()
    plt.close()


def plot_unmatched(title: str):
    plot_metric(unmatched_e, "unmatched in %")
    plot_metric(unmatched_non_e, "unmatched non enzymes in %")
    plot_metric(unmatched_total, "unmatched total in %")
    plt.ylim(ymin=0, ymax=100)
    plt.legend()
    plt.title(title)
    plt.show()
    plt.close()


#%%
parameter_values, accuracies, recall, precision, unmatched_total, unmatched_e, unmatched_non_e, = extract_data(
    "binary", total_set_size=500.0
)

plot_binary_metrics("Binary Metrics")
plot_unmatched("Binary Unmatched")

# %%
parameter_values, accuracies, recall, precision, unmatched_total, unmatched_e, unmatched_non_e, = extract_data(
    "multiclass", total_set_size=500.0
)

plot_binary_metrics("Multiclass Metrics")
plot_unmatched("Multiclass Unmatched")
