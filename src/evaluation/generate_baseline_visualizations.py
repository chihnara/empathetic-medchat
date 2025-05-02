import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set the style
plt.style.use("seaborn-v0_8")
sns.set_theme(style="whitegrid")

# Data from the comparison
metrics = {
    "System Performance": {
        "Response Time (s)": {"Original": 1.2, "V1": 0.05, "V2": 0.04},
        "Memory Usage (MB)": {"Original": 700, "V1": 41, "V2": 39},
        "Error Rate": {"Original": 0.03, "V1": 0.0, "V2": 0.0},
    },
    "Content Quality": {
        "Medical Term Presence": {"Original": 0.85, "V1": 0.24, "V2": 0.32},
        "Response Completeness": {"Original": 0.75, "V1": 0.81, "V2": 0.86},
    },
}


def create_performance_comparison():
    """Create a bar chart comparing system performance metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Response Time and Error Rate
    metrics1 = ["Response Time (s)", "Error Rate"]
    original1 = [1.2, 0.03]
    v1_1 = [0.05, 0.0]
    v2_1 = [0.04, 0.0]

    x1 = np.arange(len(metrics1))
    width = 0.25

    rects1 = ax1.bar(x1 - width, original1, width, label="Original MEDCOD")
    rects2 = ax1.bar(x1, v1_1, width, label="V1")
    rects3 = ax1.bar(x1 + width, v2_1, width, label="V2")

    ax1.set_ylabel("Value")
    ax1.set_title("Response Time and Error Rate")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(metrics1)
    ax1.legend()

    # Memory Usage
    metrics2 = ["Memory Usage (MB)"]
    original2 = [700]
    v1_2 = [41]
    v2_2 = [39]

    x2 = np.arange(len(metrics2))

    rects4 = ax2.bar(x2 - width, original2, width, label="Original MEDCOD")
    rects5 = ax2.bar(x2, v1_2, width, label="V1")
    rects6 = ax2.bar(x2 + width, v2_2, width, label="V2")

    ax2.set_ylabel("MB")
    ax2.set_title("Memory Usage")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics2)
    ax2.legend()

    # Add value labels
    def autolabel(rects, ax):
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

    for rects, ax in [
        (rects1, ax1),
        (rects2, ax1),
        (rects3, ax1),
        (rects4, ax2),
        (rects5, ax2),
        (rects6, ax2),
    ]:
        autolabel(rects, ax)

    plt.tight_layout()
    plt.savefig(
        str(Path(__file__).parent / "visualizations" / "performance_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_content_quality_radar():
    """Create a radar chart for content quality metrics."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    # Prepare data
    categories = [
        "Medical Term\nPresence",
        "Response\nCompleteness",
        "Response\nRelevance",
    ]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Plot data
    def add_to_plot(values, color, label):
        values = list(values) + values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.25)

    add_to_plot([0.85, 0.75, 0.80], "#FF9999", "Original MEDCOD")
    add_to_plot([0.24, 0.81, 0.45], "#66B2FF", "V1")
    add_to_plot([0.32, 0.86, 0.26], "#99FF99", "V2")

    # Customize the plot
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"])
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.title("Content Quality Comparison", pad=20)

    plt.savefig(
        str(Path(__file__).parent / "visualizations" / "content_quality_radar.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_improvement_heatmap():
    """Create a heatmap showing percentage improvements."""
    # Calculate improvements
    improvements = {
        "Response Time": {"V1": -95.8, "V2": -96.7},
        "Memory Usage": {"V1": -94.1, "V2": -94.4},
        "Medical Terms": {"V1": -71.8, "V2": -62.4},
        "Response Completeness": {"V1": 8.0, "V2": 14.7},
    }

    # Convert to array
    data = np.array(
        [
            [improvements[metric][version] for version in ["V1", "V2"]]
            for metric in improvements.keys()
        ]
    )

    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        xticklabels=["V1", "V2"],
        yticklabels=list(improvements.keys()),
        center=0,
    )

    plt.title("Percentage Improvement Over Original MEDCOD")
    plt.tight_layout()
    plt.savefig(
        str(Path(__file__).parent / "visualizations" / "improvement_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # Generate visualizations
    create_performance_comparison()
    create_content_quality_radar()
    create_improvement_heatmap()
