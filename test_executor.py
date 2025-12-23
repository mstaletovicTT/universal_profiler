"""
Tracy Profiling Test Executor

This module handles the execution of pytest with Tracy profiling,
captures output, and manages the profiling artifacts.
"""

import subprocess
import os
import re
import time
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


class TracyExecutionResult:
    """Container for tracy execution results"""

    def __init__(self, pytest_output: str, tracy_folder: str, exit_code: int, execution_time: float):
        self.pytest_output = pytest_output
        self.tracy_folder = tracy_folder
        self.exit_code = exit_code
        self.execution_time = execution_time
        self.success = exit_code == 0


def run_profiling_test(test_file_path: str, verbose: bool = True, timeout: int = 600) -> TracyExecutionResult:
    """
    Execute any pytest file with tracy profiling.

    Args:
        test_file_path: Path to the pytest file
        verbose: Enable verbose output
        timeout: Timeout in seconds

    Returns:
        TracyExecutionResult containing execution results

    Raises:
        FileNotFoundError: If test file doesn't exist
        TimeoutError: If execution times out
        subprocess.CalledProcessError: If tracy command fails
    """
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test file not found: {test_file_path}")

    # Prepare tracy profiling command with color support
    cmd = ["python", "-m", "tracy", "-r", "-m", "pytest", test_file_path, "--color=yes"]
    if verbose:
        cmd.append("-v")

    print(f"Executing: {' '.join(cmd)}")
    print("=" * 60)

    start_time = time.time()
    captured_output = []

    try:
        # Execute command with real-time output streaming
        # Set FORCE_COLOR to ensure pytest outputs color codes
        env = os.environ.copy()
        env["FORCE_COLOR"] = "1"
        env["PYTEST_CURRENT_TEST"] = ""
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=os.getcwd(),
            env=env,
        )

        # Stream output in real-time while capturing it
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())  # Show output immediately
                captured_output.append(output)

        # Wait for process to complete
        exit_code = process.wait(timeout=timeout)
        execution_time = time.time() - start_time

        print("=" * 60)
        print(f"✅ Test execution completed in {execution_time:.2f}s (exit code: {exit_code})")
        print()

        # Combine captured output
        pytest_output = "".join(captured_output)

        # Extract tracy folder from output
        tracy_folder = _extract_tracy_folder_from_output(pytest_output)

        if tracy_folder is None:
            # Try to find the latest folder
            tracy_folder = _find_latest_tracy_folder()

        return TracyExecutionResult(
            pytest_output=pytest_output,
            tracy_folder=tracy_folder,
            exit_code=exit_code,
            execution_time=execution_time,
        )

    except subprocess.TimeoutExpired:
        process.kill()
        execution_time = time.time() - start_time
        raise TimeoutError(f"Tracy execution timed out after {timeout} seconds")
    except Exception as e:
        execution_time = time.time() - start_time
        raise RuntimeError(f"Tracy execution failed: {e}")


def _extract_tracy_folder_from_output(output: str) -> Optional[str]:
    """
    Extract tracy output folder path from command output.

    Args:
        output: Combined stdout and stderr from tracy execution

    Returns:
        Path to tracy folder or None if not found
    """
    # Look for pattern in tracy output that indicates report generation
    patterns = [
        r"OPs csv generated at: (.+?\.csv)",
        r"generated at (.+?reports/[^/\s]+)",
        r"reports/(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})",
        r"generated/profiler/reports/([^/\s]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            csv_path = match.group(1)
            # Extract folder path from CSV path
            if csv_path.endswith(".csv"):
                tracy_folder = os.path.dirname(csv_path)
            else:
                tracy_folder = csv_path

            if os.path.exists(tracy_folder):
                return tracy_folder

    return None


def _find_latest_tracy_folder(base_path: str = "generated/profiler/reports") -> Optional[str]:
    """
    Find the most recent tracy output folder by timestamp.

    Args:
        base_path: Base directory to search for tracy reports

    Returns:
        Path to the most recent tracy folder or None if not found
    """
    if not os.path.exists(base_path):
        return None

    # List all subdirectories with timestamp pattern
    folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            # Check if it looks like a timestamp folder (YYYY_MM_DD_HH_MM_SS)
            if re.match(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}", item):
                folders.append((item, item_path))

    if not folders:
        return None

    # Sort by folder name (timestamp) and return the most recent
    folders.sort(key=lambda x: x[0], reverse=True)
    return folders[0][1]


def validate_tracy_output(tracy_folder: str) -> Dict[str, bool]:
    """
    Validate that tracy output folder contains expected files.

    Args:
        tracy_folder: Path to tracy output folder

    Returns:
        Dictionary with validation results
    """
    validation_results = {}

    if not os.path.exists(tracy_folder):
        validation_results["folder_exists"] = False
        return validation_results

    validation_results["folder_exists"] = True

    # Check for ops_perf_results CSV file
    csv_files = [f for f in os.listdir(tracy_folder) if f.startswith("ops_perf_results") and f.endswith(".csv")]
    validation_results["ops_perf_csv_exists"] = len(csv_files) > 0

    if csv_files:
        csv_path = os.path.join(tracy_folder, csv_files[0])
        validation_results["ops_perf_csv_readable"] = os.access(csv_path, os.R_OK)

        # Check if CSV file has content
        try:
            with open(csv_path, "r") as f:
                content = f.read().strip()
                validation_results["ops_perf_csv_has_content"] = len(content) > 0
        except Exception:
            validation_results["ops_perf_csv_has_content"] = False
    else:
        validation_results["ops_perf_csv_readable"] = False
        validation_results["ops_perf_csv_has_content"] = False

    # Check for other tracy artifacts
    expected_files = ["tracy_ops_times.csv", "tracy_ops_data.csv"]
    for expected_file in expected_files:
        file_path = os.path.join(tracy_folder, expected_file)
        validation_results[f"{expected_file}_exists"] = os.path.exists(file_path)

    return validation_results


def get_test_parameter_count(test_file_path: str) -> int:
    """
    Estimate the number of test parameter combinations by parsing pytest output.

    Args:
        test_file_path: Path to the test file

    Returns:
        Estimated number of parameter combinations
    """
    try:
        # Run pytest with collect-only to get test count without execution
        cmd = ["python", "-m", "pytest", test_file_path, "--collect-only", "-q"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            # Parse output to count tests
            lines = result.stdout.split("\n")
            for line in lines:
                if "test" in line and "collected" in line:
                    # Extract number from "collected X items"
                    match = re.search(r"collected (\d+)", line)
                    if match:
                        return int(match.group(1))

        # Fallback: try to parse test file directly for parametrize decorators
        return _estimate_combinations_from_file(test_file_path)

    except Exception:
        # Fallback: return conservative estimate
        return 1


def _estimate_combinations_from_file(test_file_path: str) -> int:
    """
    Estimate parameter combinations by parsing the test file directly.

    Args:
        test_file_path: Path to the test file

    Returns:
        Estimated number of parameter combinations
    """
    try:
        with open(test_file_path, "r") as f:
            content = f.read()

        # Look for parametrize decorators and count values
        parametrize_pattern = r'@pytest\.mark\.parametrize\s*\(\s*["\']([^"\']+)["\']\s*,\s*(\[[^\]]+\])'
        matches = re.findall(parametrize_pattern, content)

        total_combinations = 1
        for param_name, param_values in matches:
            # Count items in the list (rough estimate)
            value_count = param_values.count(",") + 1
            total_combinations *= value_count

        return max(total_combinations, 1)

    except Exception:
        return 1


def cleanup_old_tracy_folders(base_path: str = "generated/profiler/reports", keep_recent: int = 5) -> int:
    """
    Clean up old tracy output folders, keeping only the most recent ones.

    Args:
        base_path: Base directory containing tracy reports
        keep_recent: Number of recent folders to keep

    Returns:
        Number of folders cleaned up
    """
    if not os.path.exists(base_path):
        return 0

    # Get all tracy folders with timestamps
    folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            if re.match(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}", item):
                folders.append((item, item_path))

    if len(folders) <= keep_recent:
        return 0

    # Sort by timestamp and remove old folders
    folders.sort(key=lambda x: x[0], reverse=True)
    folders_to_remove = folders[keep_recent:]

    removed_count = 0
    for folder_name, folder_path in folders_to_remove:
        try:
            import shutil

            shutil.rmtree(folder_path)
            removed_count += 1
            print(f"Cleaned up old tracy folder: {folder_name}")
        except Exception as e:
            print(f"Failed to remove {folder_path}: {e}")

    return removed_count


def create_execution_summary(execution_result: TracyExecutionResult, param_count: int) -> Dict[str, Any]:
    """
    Create a summary of the tracy execution.

    Args:
        execution_result: TracyExecutionResult object
        param_count: Number of parameter combinations

    Returns:
        Dictionary with execution summary
    """
    summary = {
        "success": execution_result.success,
        "execution_time": execution_result.execution_time,
        "exit_code": execution_result.exit_code,
        "tracy_folder": execution_result.tracy_folder,
        "parameter_combinations": param_count,
        "validation": None,
    }

    if execution_result.tracy_folder:
        summary["validation"] = validate_tracy_output(execution_result.tracy_folder)

    return summary


# Hardware detection utilities
def detect_hardware_architecture() -> str:
    """
    Detect the current hardware architecture.

    Returns:
        Hardware architecture string ("WH" for Wormhole, "BH" for Blackhole, "GS" for Grayskull, or "UNK")
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


def generate_profiling_report_path(operation_name: str = "TTNN_Op", custom_suffix: str = "") -> str:
    """
    Generate a path for saving profiling reports.

    Args:
        operation_name: Name of the operation being profiled
        custom_suffix: Additional suffix for the folder name

    Returns:
        Path where profiling reports should be saved
    """
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hw_arch = detect_hardware_architecture()

    folder_name = f"{operation_name}_{hw_arch}_{timestamp}"
    if custom_suffix:
        folder_name += f"_{custom_suffix}"

    return os.path.join("profiling_results", folder_name)
