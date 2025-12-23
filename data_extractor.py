"""
Data Extraction Module for Universal TTNN Profiler

This module handles parsing tracy profiling CSV files, extracting performance data,
and aligning it with test parameter combinations. It's operation-agnostic and works
with any operation type found in the tracy output.
"""

import csv
import os
from typing import Dict, List, Tuple, Any, Optional
from parameter_detector import ParameterInfo


class OperationPerformanceData:
    """Container for performance data of a single operation"""

    def __init__(self, operation_type: str):
        self.operation_type = operation_type
        self.entries: List[Dict[str, Any]] = []

    def add_entry(self, csv_row: Dict[str, Any]):
        """Add a performance entry for this operation type"""
        self.entries.append(csv_row)

    def get_kernel_durations(self) -> List[float]:
        """Extract DEVICE KERNEL DURATION values in nanoseconds"""
        durations = []
        for entry in self.entries:
            duration_str = entry.get("DEVICE KERNEL DURATION [ns]", "0")
            try:
                duration = float(duration_str) if duration_str else 0.0
            except ValueError:
                duration = 0.0
            durations.append(duration)
        return durations

    def get_core_counts(self) -> List[int]:
        """Extract CORE COUNT values"""
        core_counts = []
        for entry in self.entries:
            core_count_str = entry.get("CORE COUNT", "0")
            try:
                core_count = int(core_count_str) if core_count_str else 0
            except ValueError:
                core_count = 0
            core_counts.append(core_count)
        return core_counts

    def get_entry_count(self) -> int:
        """Get number of performance entries for this operation"""
        return len(self.entries)


def find_latest_tracy_folder(base_path: str = "generated/profiler/reports") -> Optional[str]:
    """
    Find the most recent tracy profiling output folder.

    Args:
        base_path: Base directory to search for tracy reports

    Returns:
        Path to the most recent tracy folder, or None if not found
    """
    if not os.path.exists(base_path):
        return None

    # List all subdirectories with timestamp pattern YYYY_MM_DD_HH_MM_SS
    folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            # Check if it looks like a timestamp folder
            parts = item.split("_")
            if len(parts) == 6 and all(part.isdigit() for part in parts):
                folders.append((item, item_path))

    if not folders:
        return None

    # Sort by folder name (timestamp) and return the most recent
    folders.sort(key=lambda x: x[0], reverse=True)
    return folders[0][1]


def parse_ops_perf_csv(csv_file_path: str) -> Dict[str, OperationPerformanceData]:
    """
    Parse ops_perf_results.csv and group by operation type.

    Args:
        csv_file_path: Path to the ops_perf_results.csv file

    Returns:
        Dictionary mapping operation type to OperationPerformanceData

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Tracy CSV file not found: {csv_file_path}")

    operations_data = {}

    try:
        with open(csv_file_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                op_code = row.get("OP CODE", "").strip()

                if not op_code:
                    continue  # Skip rows without operation code

                # Initialize operation data if not seen before
                if op_code not in operations_data:
                    operations_data[op_code] = OperationPerformanceData(op_code)

                # Add this entry to the operation's data
                operations_data[op_code].add_entry(row)

    except Exception as e:
        raise ValueError(f"Error parsing CSV file {csv_file_path}: {e}")

    return operations_data


def filter_operations_by_type(
    operations_data: Dict[str, OperationPerformanceData],
    include_types: Optional[List[str]] = None,
    exclude_types: Optional[List[str]] = None,
) -> Dict[str, OperationPerformanceData]:
    """
    Filter operations by type with include/exclude lists.

    Args:
        operations_data: Dictionary of operation type -> performance data
        include_types: If provided, only include these operation types
        exclude_types: If provided, exclude these operation types

    Returns:
        Filtered dictionary of operations
    """
    filtered_data = {}

    for op_type, perf_data in operations_data.items():
        # Apply include filter
        if include_types and op_type not in include_types:
            continue

        # Apply exclude filter
        if exclude_types and op_type in exclude_types:
            continue

        filtered_data[op_type] = perf_data

    return filtered_data


def align_performance_with_parameters(
    operations_data: Dict[str, OperationPerformanceData],
    param_combinations: List[Dict[str, Any]],
    target_operation: Optional[str] = None,
) -> Dict[str, List[Tuple[Dict[str, Any], float, int]]]:
    """
    Align performance data with parameter combinations.

    Args:
        operations_data: Dictionary of operation type -> performance data
        param_combinations: List of parameter combinations from test parsing
        target_operation: Specific operation type to focus on (if None, process all)

    Returns:
        Dictionary mapping operation type -> list of (parameter_combination, kernel_duration, core_count) tuples

    Raises:
        ValueError: If data alignment fails
    """
    aligned_data = {}

    # Filter to target operation if specified
    if target_operation:
        if target_operation not in operations_data:
            raise ValueError(
                f"Target operation '{target_operation}' not found in performance data. Available: {list(operations_data.keys())}"
            )
        ops_to_process = {target_operation: operations_data[target_operation]}
    else:
        ops_to_process = operations_data

    for op_type, perf_data in ops_to_process.items():
        kernel_durations = perf_data.get_kernel_durations()
        core_counts = perf_data.get_core_counts()

        # Check if we have the right number of performance entries
        if len(kernel_durations) != len(param_combinations):
            print(
                f"Warning: Operation '{op_type}' has {len(kernel_durations)} performance entries but {len(param_combinations)} parameter combinations"
            )
            # Take the minimum to avoid index errors
            min_count = min(len(kernel_durations), len(param_combinations))
            kernel_durations = kernel_durations[:min_count]
            core_counts = core_counts[:min_count]
            combinations_subset = param_combinations[:min_count]
        else:
            combinations_subset = param_combinations

        # Align data
        operation_aligned_data = []
        for i, (params, duration, core_count) in enumerate(zip(combinations_subset, kernel_durations, core_counts)):
            operation_aligned_data.append((params, duration, core_count))

        aligned_data[op_type] = operation_aligned_data

    return aligned_data


def extract_visualization_data(
    aligned_data: Dict[str, List[Tuple[Dict[str, Any], float, int]]],
    x_axis_param: ParameterInfo,
    legend_param: Optional[ParameterInfo],
) -> Dict[str, Dict[str, Any]]:
    """
    Extract and organize data for visualization plotting.

    Args:
        aligned_data: Aligned performance data by operation type
        x_axis_param: Parameter info for X-axis
        legend_param: Parameter info for legend (None for single parameter)

    Returns:
        Dictionary with visualization-ready data structure:
        {
            operation_type: {
                'x_values': [...],
                'y_data': {...} or [...],  # Dict if legend_param, list if single param
                'x_label': str,
                'legend_label': str or None
            }
        }
    """
    viz_data = {}

    for op_type, data_points in aligned_data.items():
        if not data_points:
            continue

        if legend_param is None:
            # Single parameter case
            x_values = []
            y_values = []
            core_counts = []

            for params, duration, core_count in data_points:
                # Extract X value
                if x_axis_param.changing_dimension is not None:
                    # Tuple parameter with specific dimension
                    param_value = params[x_axis_param.name]
                    x_val = param_value[x_axis_param.changing_dimension]
                else:
                    # Simple parameter
                    x_val = params[x_axis_param.name]

                x_values.append(x_val)
                y_values.append(duration)
                core_counts.append(core_count)

            viz_data[op_type] = {
                "x_values": x_values,
                "y_data": y_values,
                "core_counts": core_counts,
                "x_label": x_axis_param.display_name,
                "legend_label": None,
            }

        else:
            # Two parameter case - need to group by legend parameter
            grouped_data = {}

            for params, duration, core_count in data_points:
                # Extract X value
                if x_axis_param.changing_dimension is not None:
                    param_value = params[x_axis_param.name]
                    x_val = param_value[x_axis_param.changing_dimension]
                else:
                    x_val = params[x_axis_param.name]

                # Extract legend value
                if legend_param.changing_dimension is not None:
                    param_value = params[legend_param.name]
                    legend_val = param_value[legend_param.changing_dimension]
                else:
                    legend_val = params[legend_param.name]

                # Convert to string for consistent grouping
                legend_key = str(legend_val)

                if legend_key not in grouped_data:
                    grouped_data[legend_key] = {"x_values": [], "y_values": [], "core_counts": []}

                grouped_data[legend_key]["x_values"].append(x_val)
                grouped_data[legend_key]["y_values"].append(duration)
                grouped_data[legend_key]["core_counts"].append(core_count)

            viz_data[op_type] = {
                "grouped_data": grouped_data,
                "x_label": x_axis_param.display_name,
                "legend_label": legend_param.display_name,
            }

    return viz_data


def get_hardware_architecture() -> str:
    """
    Detect hardware architecture for output folder naming.

    Returns:
        "WH" for Wormhole, "BH" for Blackhole, "GS" for Grayskull, or "UNK" for unknown
    """
    try:
        import ttnn
        arch_name = ttnn.get_arch_name().lower()
        
        if "wormhole" in arch_name:
            return "WH"
        elif "blackhole" in arch_name:
            return "BH"
        elif "grayskull" in arch_name:
            return "GS"
        else:
            return "UNK"
    except Exception:
        return "UNK"


def generate_output_folder_name(
    operation_name: str, date_str: Optional[str] = None, output_base_dir: str = "generated"
) -> str:
    """
    Generate output folder name following the convention:
    {operation_name}_{HW_arch}_{date}_{ordinal}

    Args:
        operation_name: Name of the primary operation being profiled
        date_str: Date string in YYYYMMDD format (if None, use current date)
        output_base_dir: Base directory to check for existing folders

    Returns:
        Folder name string
    """
    import datetime
    import os

    if date_str is None:
        date_str = datetime.datetime.now().strftime("%Y%m%d")

    hw_arch = get_hardware_architecture()

    # Find existing folders with same base name to determine ordinal
    base_name = f"{operation_name}_{hw_arch}_{date_str}"
    ordinal = 0

    # Find next available ordinal by checking if folder exists
    while True:
        folder_name = f"{base_name}_{ordinal}"
        full_path = os.path.join(output_base_dir, folder_name)
        if not os.path.exists(full_path):
            break
        ordinal += 1

    return folder_name


# Main entry points for the data extraction pipeline
def extract_performance_data(
    tracy_folder_path: str, param_combinations: List[Dict[str, Any]], target_operation: Optional[str] = None
) -> Dict[str, OperationPerformanceData]:
    """
    Main entry point for extracting performance data from tracy output.

    Args:
        tracy_folder_path: Path to tracy output folder
        param_combinations: Parameter combinations from test parsing
        target_operation: Specific operation to focus on (None for all)

    Returns:
        Dictionary of operation performance data
    """
    # Find ops_perf_results.csv file
    csv_files = [f for f in os.listdir(tracy_folder_path) if f.startswith("ops_perf_results") and f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(f"No ops_perf_results CSV file found in {tracy_folder_path}")

    # Use the first (and typically only) CSV file found
    csv_path = os.path.join(tracy_folder_path, csv_files[0])

    # Parse CSV and extract operations data
    operations_data = parse_ops_perf_csv(csv_path)

    # Filter to target operation if specified
    if target_operation:
        operations_data = filter_operations_by_type(operations_data, include_types=[target_operation])

    return operations_data
