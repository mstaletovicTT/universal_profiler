"""
Visualization Generation Module for Universal TTNN Profiler
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional, Any
import datetime


class VisualizationConfig:
    """Configuration for plot generation"""

    def __init__(self):
        self.figure_size = (12, 8)
        self.dpi = 300
        self.format = "svg"
        self.grid_alpha = 0.3
        self.marker_size = 8
        self.line_width = 2
        self.font_size = 12
        self.title_font_size = 14
        self.legend_font_size = 10

        self.colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        self.markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]


def create_performance_plots(
    viz_data: Dict[str, Dict[str, Any]],
    output_dir: str,
    config: Optional[VisualizationConfig] = None,
    save_plots: bool = True,
    static_params: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate performance visualizations for each operation"""
    if config is None:
        config = VisualizationConfig()

    plot_files = []

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    for operation_name, plot_data in viz_data.items():
        plt.figure(figsize=config.figure_size)

        if "x_values" in plot_data:
            # Single parameter case
            x_values = np.array(plot_data["x_values"])
            y_values = np.array(plot_data["y_data"])
            core_counts = np.array(plot_data.get("core_counts", []))

            sort_indices = np.argsort(x_values)
            x_sorted = x_values[sort_indices]
            y_sorted = y_values[sort_indices]
            cores_sorted = core_counts[sort_indices] if len(core_counts) > 0 else []

            plt.plot(
                x_sorted,
                y_sorted,
                marker=config.markers[0],
                color=config.colors[0],
                linewidth=config.line_width,
                markersize=config.marker_size,
            )

            # Add core count annotations
            if len(cores_sorted) > 0:
                for i, (x, y, cores) in enumerate(zip(x_sorted, y_sorted, cores_sorted)):
                    plt.annotate(
                        f"{cores}",
                        (x, y),
                        xytext=(5, 5),  # 5 points offset from the point
                        textcoords="offset points",
                        fontsize=8,
                        color="gray",
                        alpha=0.7,
                    )

            plt.xlabel(plot_data["x_label"])
            plt.ylabel("Device Kernel Duration [ns]")

        elif "grouped_data" in plot_data:
            # Two parameter case
            color_idx = 0
            for series_name, series_data in plot_data["grouped_data"].items():
                x_values = np.array(series_data["x_values"])
                y_values = np.array(series_data["y_values"])
                core_counts = np.array(series_data.get("core_counts", []))

                sort_indices = np.argsort(x_values)
                x_sorted = x_values[sort_indices]
                y_sorted = y_values[sort_indices]
                cores_sorted = core_counts[sort_indices] if len(core_counts) > 0 else []

                color = config.colors[color_idx % len(config.colors)]
                marker = config.markers[color_idx % len(config.markers)]

                plt.plot(
                    x_sorted,
                    y_sorted,
                    marker=marker,
                    color=color,
                    linewidth=config.line_width,
                    markersize=config.marker_size,
                    label=f"{plot_data['legend_label']}={series_name}",
                )

                # Add core count annotations for this series
                if len(cores_sorted) > 0:
                    for i, (x, y, cores) in enumerate(zip(x_sorted, y_sorted, cores_sorted)):
                        plt.annotate(
                            f"{cores}",
                            (x, y),
                            xytext=(5, 5),  # 5 points offset from the point
                            textcoords="offset points",
                            fontsize=8,
                            color="gray",
                            alpha=0.7,
                        )

                color_idx += 1

            plt.xlabel(plot_data["x_label"])
            plt.ylabel("Device Kernel Duration [ns]")
            plt.legend()

        plt.title(f"Kernel Performance: {operation_name}")
        plt.grid(True, alpha=config.grid_alpha)
        
        # Add static parameters as text box on the plot
        if static_params:
            static_text_lines = ["Static Parameters:"]
            for param_name, param_value in static_params.items():
                # Format the value nicely
                if isinstance(param_value, (tuple, list)):
                    value_str = str(param_value)
                elif isinstance(param_value, dict):
                    value_str = str(param_value)
                else:
                    value_str = str(param_value)
                
                # Truncate long values
                if len(value_str) > 40:
                    value_str = value_str[:37] + "..."
                
                static_text_lines.append(f"  {param_name}: {value_str}")
            
            static_text = "\n".join(static_text_lines)
            
            # Add text box in upper right corner
            plt.text(
                0.98, 0.98, static_text,
                transform=plt.gca().transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace'
            )
        
        plt.tight_layout()

        if save_plots:
            plot_file = os.path.join(output_dir, f"{operation_name}_performance.{config.format}")
            plt.savefig(plot_file, format=config.format, dpi=config.dpi, bbox_inches="tight")
            plot_files.append(plot_file)
            print(f"✅ Saved plot: {plot_file}")

        plt.show()
        plt.close()

    return plot_files


def create_summary_report(param_info_list, performance_data, output_dir: str, static_params: Optional[Dict[str, Any]] = None) -> str:
    """Generate a text summary report"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(output_dir, "profiling_summary.txt")

    with open(report_path, "w") as f:
        f.write("UNIVERSAL TTNN OPERATION PROFILING REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {timestamp}\n\n")

        f.write("PARAMETER ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total varying parameters detected: {len(param_info_list)}\n\n")

        for param in param_info_list:
            f.write(f"Parameter: {param.name}\n")
            f.write(f"  Display Name: {param.display_name}\n")
            f.write(f"  Type: {'Numeric' if param.is_numeric else 'Categorical'}\n")
            f.write(f"  Values Count: {param.count}\n")
            if hasattr(param, "changing_dimension") and param.changing_dimension is not None:
                f.write(f"  Changing Dimension: {param.changing_dimension}\n")
            f.write("\n")
        
        # Add static parameters section
        if static_params:
            f.write("STATIC PARAMETERS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total static parameters: {len(static_params)}\n\n")
            for param_name, param_value in static_params.items():
                f.write(f"Parameter: {param_name}\n")
                f.write(f"  Value: {param_value}\n\n")

        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 20 + "\n")

        for op_type, data_points in performance_data.items():
            f.write(f"Operation: {op_type}\n")
            f.write(f"  Total measurements: {len(data_points)}\n")

            durations = [dp[1] for dp in data_points]
            if durations:
                f.write(f"  Min duration: {min(durations):.0f} ns\n")
                f.write(f"  Max duration: {max(durations):.0f} ns\n")
                f.write(f"  Avg duration: {np.mean(durations):.0f} ns\n")
            f.write("\n")

    print(f"✅ Generated summary report: {report_path}")
    return report_path


def generate_output_folder_name(
    operation_name: str = "TTNN_Op", hw_arch: str = "WH", custom_suffix: str = "", output_base_dir: str = "generated"
) -> str:
    """Generate output folder name with incremental numbering"""
    timestamp = datetime.datetime.now()
    date_str = timestamp.strftime("%Y%m%d")

    base_name = f"{operation_name}_{hw_arch}_{date_str}"
    ordinal = 0

    # Find next available ordinal by checking if folder exists
    while True:
        folder_name = f"{base_name}_{ordinal}"
        if custom_suffix:
            folder_name += f"_{custom_suffix}"

        # Check if this folder would exist in the output directory
        full_path = os.path.join(output_base_dir, folder_name)
        if not os.path.exists(full_path):
            break
        ordinal += 1

    return folder_name


# Interactive plots removed per user request


def generate_all_visualizations(
    viz_data: Dict[str, Dict[str, Any]],
    param_info_list,
    performance_data,
    output_dir: str,
    config: Optional[VisualizationConfig] = None,
    csv_source_path: Optional[str] = None,
    static_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:
    """Generate all types of visualizations"""
    results = {}

    plot_files = create_performance_plots(viz_data, output_dir, config, save_plots=True, static_params=static_params)
    results["static_plots"] = plot_files

    report_file = create_summary_report(param_info_list, performance_data, output_dir, static_params)
    results["summary_report"] = [report_file]

    # Copy original CSV to output directory
    if csv_source_path and os.path.exists(csv_source_path):
        csv_dest_path = os.path.join(output_dir, "original_ops_perf_results.csv")
        import shutil

        shutil.copy2(csv_source_path, csv_dest_path)
        results["original_csv"] = [csv_dest_path]
        print(f"✅ Copied original CSV: {csv_dest_path}")

    return results
