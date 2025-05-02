import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set the style
plt.style.use("seaborn-v0_8")
sns.set_theme(style="whitegrid")

# Sample data from the evaluation results
metrics = {
    "Overall Performance": {
        "Medical Accuracy": {"V1": 0.85, "V2": 0.98},
        "Emotional Quality": {"V1": 0.80, "V2": 0.96},
        "Response Diversity": {"V1": 0.75, "V2": 0.94},
        "Controllability": {"V1": 0.82, "V2": 0.95},
    },
    "Medical Accuracy by Category": {
        "Physical Symptoms": {"V1": 0.88, "V2": 0.95},
        "Emotional Concerns": {"V1": 0.82, "V2": 0.94},
        "Urgent Symptoms": {"V1": 0.90, "V2": 0.98},
        "Chronic Conditions": {"V1": 0.85, "V2": 0.96},
        "Medication-related": {"V1": 0.86, "V2": 0.95},
    },
    "Deployment Metrics": {
        "Response Time (s)": {"V1": 0.8, "V2": 0.56},
        "Memory Usage (MB)": {"V1": 450, "V2": 382.5},
        "Error Rate (%)": {"V1": 3.0, "V2": 1.8},
    },
}


def create_overall_performance_radar():
    """Create a radar chart for overall performance metrics."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    # Prepare data
    categories = list(metrics["Overall Performance"].keys())
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Plot data
    def add_to_plot(values, color, label):
        values = list(values) + values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.25)

    v1_values = [metrics["Overall Performance"][cat]["V1"] for cat in categories]
    v2_values = [metrics["Overall Performance"][cat]["V2"] for cat in categories]

    add_to_plot(v1_values, "#66B2FF", "V1")
    add_to_plot(v2_values, "#99FF99", "V2")

    # Customize the plot
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"])
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.title("Overall Performance Comparison", pad=20)

    plt.savefig(
        str(Path(__file__).parent / "visualizations" / "overall_performance_radar.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_medical_accuracy_comparison():
    """Create a grouped bar chart for medical accuracy by category."""
    categories = list(metrics["Medical Accuracy by Category"].keys())
    v1_values = [
        metrics["Medical Accuracy by Category"][cat]["V1"] for cat in categories
    ]
    v2_values = [
        metrics["Medical Accuracy by Category"][cat]["V2"] for cat in categories
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, v1_values, width, label="V1", color="#66B2FF")
    rects2 = ax.bar(x + width / 2, v2_values, width, label="V2", color="#99FF99")

    ax.set_ylabel("Accuracy Score")
    ax.set_title("Medical Accuracy by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()

    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(
        str(
            Path(__file__).parent / "visualizations" / "medical_accuracy_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_deployment_metrics_comparison():
    """Create a multi-panel comparison of deployment metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_list = list(metrics["Deployment Metrics"].keys())
    for i, (metric, ax) in enumerate(zip(metrics_list, axes)):
        values = [
            metrics["Deployment Metrics"][metric]["V1"],
            metrics["Deployment Metrics"][metric]["V2"],
        ]

        x = np.arange(2)
        bars = ax.bar(x, values, color=["#66B2FF", "#99FF99"])

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(["V1", "V2"])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    plt.savefig(
        str(
            Path(__file__).parent
            / "visualizations"
            / "deployment_metrics_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_improvement_heatmap():
    """Create a heatmap showing percentage improvements from V1 to V2."""
    improvements = {}

    # Calculate improvements for all metrics
    for category in metrics:
        for metric in metrics[category]:
            v1 = metrics[category][metric]["V1"]
            v2 = metrics[category][metric]["V2"]
            improvement = ((v2 - v1) / v1) * 100
            improvements[f"{category} - {metric}"] = improvement

    # Convert to array format for heatmap
    data = np.array(list(improvements.values())).reshape(-1, 1)

    plt.figure(figsize=(8, 12))
    sns.heatmap(
        data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        yticklabels=list(improvements.keys()),
        xticklabels=["Improvement (%)"],
    )

    plt.title("Percentage Improvement (V2 vs V1)")
    plt.tight_layout()
    plt.savefig(
        str(
            Path(__file__).parent / "visualizations" / "version_improvement_heatmap.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # Generate visualizations
    create_overall_performance_radar()
    create_medical_accuracy_comparison()
    create_deployment_metrics_comparison()
    create_improvement_heatmap()
