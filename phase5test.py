#!/usr/bin/env python3
"""
Phase 5 Testing: CLI Integration

MANDATORY TEST FILE for Phase 5 - CLI Integration and End-to-End Testing.
This file tests ALL CLI scenarios outlined in the plan.

NO PHASE ADVANCEMENT WITHOUT 100% TEST COMPLETION
"""

import pytest
import tempfile
import os
import subprocess
import sys
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the CLI module functions directly
# We'll import the module by file path to avoid naming conflicts
import importlib.util
import os

# Load the CLI module directly
cli_module_path = os.path.join(os.path.dirname(__file__), "universal_profiler.py")
spec = importlib.util.spec_from_file_location("cli_module", cli_module_path)
cli_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli_module)

# Import functions from the loaded module
setup_argument_parser = cli_module.setup_argument_parser
validate_arguments = cli_module.validate_arguments
detect_operation_name = cli_module.detect_operation_name
main = cli_module.main


class TestArgumentParsing:
    """Test CLI argument parsing functionality"""

    def test_basic_argument_parsing(self):
        """Test parsing of basic required arguments"""
        parser = setup_argument_parser()

        # Test with just the required test file
        args = parser.parse_args(["test_file.py"])

        assert args.test_file == "test_file.py"
        assert args.pltsave is True  # Default enabled
        assert args.output is None
        assert args.timeout == 600
        assert args.verbose is False

    def test_all_cli_flags(self):
        """Test parsing of all CLI flags"""
        parser = setup_argument_parser()

        args = parser.parse_args(
            [
                "test_maxpool_simple.py",
                "--output",
                "./results/maxpool_analysis",
                "--no-pltsave",
                "--x-param",
                "input_shape",
                "--legend-param",
                "kernel_size",
                "--extract-config",
                "custom_extractors.json",
                "--timeout",
                "300",
                "--verbose",
            ]
        )

        assert args.test_file == "test_maxpool_simple.py"
        assert args.output == "./results/maxpool_analysis"
        assert args.pltsave is False  # Disabled with --no-pltsave
        assert args.x_param == "input_shape"
        assert args.legend_param == "kernel_size"
        assert args.extract_config == "custom_extractors.json"
        assert args.timeout == 300
        assert args.verbose is True

    def test_pltsave_flag_variations(self):
        """Test both --pltsave and --no-pltsave flags"""
        parser = setup_argument_parser()

        # Test --pltsave (explicit enable)
        args1 = parser.parse_args(["test_file.py", "--pltsave"])
        assert args1.pltsave is True

        # Test --no-pltsave (disable)
        args2 = parser.parse_args(["test_file.py", "--no-pltsave"])
        assert args2.pltsave is False

    def test_short_flag_aliases(self):
        """Test short flag aliases"""
        parser = setup_argument_parser()

        args = parser.parse_args(["test_file.py", "-o", "output_dir", "-v"])

        assert args.output == "output_dir"
        assert args.verbose is True

    def test_invalid_timeout(self):
        """Test error handling for invalid timeout values"""
        # This will be caught by validate_arguments, not argparse
        parser = setup_argument_parser()
        args = parser.parse_args(["test_file.py", "--timeout", "-1"])
        assert args.timeout == -1  # Parsed successfully, but will fail validation


class TestArgumentValidation:
    """Test validation of parsed arguments"""

    def create_mock_test_file(self) -> str:
        """Helper to create a temporary test file"""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        temp_file.write("# Mock test file\npass")
        temp_file.close()
        return temp_file.name

    def test_valid_arguments(self):
        """Test validation of valid arguments"""
        test_file = self.create_mock_test_file()

        try:
            # Create mock args object
            args = MagicMock()
            args.test_file = test_file
            args.extract_config = None
            args.timeout = 600

            # Should not raise any exceptions
            validate_arguments(args)

        finally:
            os.unlink(test_file)

    def test_missing_test_file(self):
        """Test handling of missing test file"""
        args = MagicMock()
        args.test_file = "/nonexistent/test_file.py"
        args.extract_config = None
        args.timeout = 600

        with pytest.raises(FileNotFoundError, match="Test file not found"):
            validate_arguments(args)

    def test_non_python_test_file(self):
        """Test handling of non-Python test files"""
        # Create a non-Python file
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        temp_file.write("Not a Python file")
        temp_file.close()

        try:
            args = MagicMock()
            args.test_file = temp_file.name
            args.extract_config = None
            args.timeout = 600

            with pytest.raises(ValueError, match="Test file must be a Python file"):
                validate_arguments(args)
        finally:
            os.unlink(temp_file.name)

    def test_missing_extract_config(self):
        """Test handling of missing extract config file"""
        test_file = self.create_mock_test_file()

        try:
            args = MagicMock()
            args.test_file = test_file
            args.extract_config = "/nonexistent/config.json"
            args.timeout = 600

            with pytest.raises(FileNotFoundError, match="Extract config file not found"):
                validate_arguments(args)
        finally:
            os.unlink(test_file)

    def test_invalid_timeout(self):
        """Test handling of invalid timeout values"""
        test_file = self.create_mock_test_file()

        try:
            args = MagicMock()
            args.test_file = test_file
            args.extract_config = None
            args.timeout = -1

            with pytest.raises(ValueError, match="Timeout must be positive"):
                validate_arguments(args)
        finally:
            os.unlink(test_file)


class TestOperationNameDetection:
    """Test automatic operation name detection"""

    def test_maxpool_detection(self):
        """Test detection of MaxPool operations"""
        assert detect_operation_name("test_maxpool_simple.py") == "MaxPool2D"
        assert detect_operation_name("maxpool_test.py") == "MaxPool2D"
        assert detect_operation_name("pool2d_performance.py") == "MaxPool2D"

    def test_conv_detection(self):
        """Test detection of Conv operations"""
        assert detect_operation_name("test_conv2d_basic.py") == "Conv2D"
        assert detect_operation_name("conv_performance_test.py") == "Conv2D"

    def test_linear_detection(self):
        """Test detection of Linear operations"""
        assert detect_operation_name("test_linear_layer.py") == "Linear"
        assert detect_operation_name("linear_performance.py") == "Linear"

    def test_generic_fallback(self):
        """Test fallback for generic operation names"""
        assert detect_operation_name("test_custom_operation.py") == "Custom_Operation"
        assert detect_operation_name("my_special_test.py") == "My_Special"


class TestErrorReporting:
    """Test error reporting and user feedback"""

    def create_mock_test_file(self) -> str:
        """Helper to create a temporary test file"""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        temp_file.write(
            """
import pytest

@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("channels", [32, 64])
def test_operation(batch_size, channels):
    pass
"""
        )
        temp_file.close()
        return temp_file.name

    def test_missing_file_error_message(self):
        """Test clear error message for missing test file"""
        with patch("sys.argv", ["universal_profiler.py", "/nonexistent/file.py"]):
            with patch("builtins.print") as mock_print:
                exit_code = main()

                # Check that error exit code was returned
                assert exit_code == 1
                # Should have printed an error message about file not found
                error_printed = any("not found" in str(call) for call in mock_print.call_args_list)
                assert error_printed

    def test_parameter_detection_error_handling(self):
        """Test error handling in parameter detection"""
        test_file = self.create_mock_test_file()

        try:
            with patch("sys.argv", ["universal_profiler.py", test_file]):
                with patch.object(cli_module, "detect_and_analyze_parameters") as mock_detect:
                    mock_detect.side_effect = Exception("Parameter detection failed")

                    with patch("builtins.print") as mock_print:
                        exit_code = main()

                        # Check that error was handled gracefully
                        assert exit_code == 1
                        # Should print parameter detection error
                        error_printed = any(
                            "Parameter detection failed" in str(call) for call in mock_print.call_args_list
                        )
                        assert error_printed
        finally:
            os.unlink(test_file)


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow"""

    def create_realistic_test_file(self) -> str:
        """Create a realistic test file for end-to-end testing"""
        content = '''
import pytest
import time

@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("input_shape", [(1, 32, 64, 64), (1, 64, 64, 64)])
@pytest.mark.parametrize("kernel_size", [(2, 2)])
def test_pool2d_operation(input_shape, kernel_size, device_params):
    """Mock Pool2D test that simulates a TTNN operation"""
    # Simulate some computation time
    time.sleep(0.01)

    # Mock assertions
    assert input_shape[0] == 1
    assert kernel_size == (2, 2)
    assert isinstance(device_params, dict)
'''

        test_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        test_file.write(content)
        test_file.close()
        return test_file.name

    @patch.object(cli_module, "run_profiling_test")
    @patch.object(cli_module, "parse_ops_perf_csv")
    def test_successful_end_to_end_workflow(self, mock_parse_csv, mock_run_profiling):
        """Test successful end-to-end workflow with mocked Tracy execution"""
        test_file = self.create_realistic_test_file()

        try:
            # Setup mocks
            from test_executor import TracyExecutionResult
            from data_extractor import OperationPerformanceData

            # Mock Tracy execution result
            mock_exec_result = TracyExecutionResult(
                pytest_output="2 passed in 1.23s",
                tracy_folder="/tmp/mock_tracy_folder",
                exit_code=0,
                execution_time=15.5,
            )
            mock_run_profiling.return_value = mock_exec_result

            # Mock CSV parsing
            mock_pool_data = OperationPerformanceData("Pool2D")
            mock_pool_data.add_entry({"DEVICE KERNEL DURATION [ns]": "2000"})
            mock_pool_data.add_entry({"DEVICE KERNEL DURATION [ns]": "3000"})
            mock_parse_csv.return_value = {"Pool2D": mock_pool_data}

            # Create mock CSV file
            with tempfile.TemporaryDirectory() as temp_tracy_dir:
                csv_path = os.path.join(temp_tracy_dir, "ops_perf_results.csv")
                with open(csv_path, "w") as f:
                    f.write("OP TYPE,DEVICE KERNEL DURATION [ns]\nPool2D,2000\nPool2D,3000\n")

                # Update mock to return our temp directory
                mock_exec_result.tracy_folder = temp_tracy_dir

                # Create temporary output directory
                with tempfile.TemporaryDirectory() as temp_output_dir:
                    # Test the CLI with mocked components
                    with patch(
                        "sys.argv", ["universal_profiler.py", test_file, "--output", temp_output_dir, "--verbose"]
                    ):
                        exit_code = main()

                        # Should exit successfully
                        assert exit_code == 0

                        # Check that Tracy was called
                        mock_run_profiling.assert_called_once()

                        # Check that CSV parsing was called
                        mock_parse_csv.assert_called_once()

        finally:
            os.unlink(test_file)

    def test_output_directory_creation(self):
        """Test that output directories are created correctly"""
        test_file = self.create_realistic_test_file()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                custom_output = os.path.join(temp_dir, "custom_output")

                # Mock all the heavy components
                with patch.object(cli_module, "detect_and_analyze_parameters") as mock_detect:
                    with patch.object(cli_module, "run_profiling_test") as mock_tracy:
                        with patch.object(cli_module, "parse_ops_perf_csv") as mock_parse:
                            with patch.object(cli_module, "generate_all_visualizations") as mock_viz:
                                # Setup minimal mocks
                                from parameter_detector import ParameterInfo

                                mock_param = ParameterInfo(
                                    name="input_shape",
                                    values=[(1, 32, 64, 64), (1, 64, 64, 64)],
                                    is_numeric=True,
                                    display_name="input_shape dim: 1",
                                    extracted_values=[32, 64],
                                    changing_dimension=1,
                                )
                                mock_detect.return_value = ([mock_param], mock_param, None, [])

                                from test_executor import TracyExecutionResult

                                mock_tracy.return_value = TracyExecutionResult(
                                    pytest_output="passed", tracy_folder=temp_dir, exit_code=0, execution_time=1.0
                                )

                                mock_parse.return_value = {}
                                mock_viz.return_value = {"static_plots": [], "summary_report": []}

                                # Create mock CSV
                                csv_path = os.path.join(temp_dir, "ops_perf_results.csv")
                                with open(csv_path, "w") as f:
                                    f.write("header\n")

                                with patch("sys.argv", ["universal_profiler.py", test_file, "--output", custom_output]):
                                    exit_code = main()

                                    # Should create the output directory
                                    assert os.path.exists(custom_output)
                                    assert exit_code == 0
        finally:
            os.unlink(test_file)

    def test_manual_parameter_specification(self):
        """Test manual parameter specification override"""
        test_file = self.create_realistic_test_file()

        try:
            with patch.object(cli_module, "detect_and_analyze_parameters") as mock_detect:
                # Setup mock parameters
                from parameter_detector import ParameterInfo

                param1 = ParameterInfo(
                    name="input_shape",
                    values=[(1, 32, 64, 64), (1, 64, 64, 64)],
                    is_numeric=True,
                    display_name="input_shape dim: 1",
                    extracted_values=[32, 64],
                    changing_dimension=1,
                )
                param2 = ParameterInfo(
                    name="kernel_size",
                    values=[(2, 2), (3, 3)],
                    is_numeric=True,
                    display_name="kernel_size dim: 0",
                    extracted_values=[2, 3],
                    changing_dimension=0,
                )
                mock_detect.return_value = ([param1, param2], param1, param2, [])

                # Mock other components
                with patch.object(cli_module, "run_profiling_test") as mock_tracy:
                    with patch.object(cli_module, "parse_ops_perf_csv"):
                        with patch.object(cli_module, "generate_all_visualizations"):
                            with tempfile.TemporaryDirectory() as temp_dir:
                                from test_executor import TracyExecutionResult

                                mock_tracy.return_value = TracyExecutionResult(
                                    pytest_output="passed", tracy_folder=temp_dir, exit_code=0, execution_time=1.0
                                )

                                # Create mock CSV
                                csv_path = os.path.join(temp_dir, "ops_perf_results.csv")
                                with open(csv_path, "w") as f:
                                    f.write("header\n")

                                with patch(
                                    "sys.argv",
                                    [
                                        "universal_profiler.py",
                                        test_file,
                                        "--x-param",
                                        "kernel_size",
                                        "--legend-param",
                                        "input_shape",
                                    ],
                                ):
                                    exit_code = main()
                                    assert exit_code == 0
        finally:
            os.unlink(test_file)


class TestOutputValidation:
    """Test validation of generated outputs"""

    def test_folder_naming_convention(self):
        """Test that folder names follow the correct convention"""
        # Test auto-generated folder names
        from visualizer import generate_output_folder_name

        folder_name = generate_output_folder_name("Pool2D")
        parts = folder_name.split("_")

        # Should follow: {operation_name}_{HW_arch}_{date}_{ordinal}
        assert len(parts) >= 4  # Pool2D is kept as one part
        assert parts[0] == "Pool2D"
        assert parts[1] in ["WH", "BH"]  # Hardware architecture
        assert len(parts[2]) == 8  # Date in YYYYMMDD format
        assert parts[3].isdigit()  # Ordinal

    def test_operation_name_detection_accuracy(self):
        """Test accuracy of operation name detection from test files"""
        test_cases = [
            ("test_maxpool_simple.py", "MaxPool2D"),
            ("conv2d_performance_test.py", "Conv2D"),
            ("linear_layer_test.py", "Linear"),
            ("test_custom_op.py", "Custom_Op"),
        ]

        for test_file, expected_name in test_cases:
            detected_name = detect_operation_name(test_file)
            assert detected_name == expected_name


def run_phase5_tests():
    """Run all Phase 5 tests and report results"""
    print("=" * 80)
    print("PHASE 5 MANDATORY TESTING: CLI Integration")
    print("=" * 80)

    # Run pytest on this file
    exit_code = pytest.main([__file__, "-v"])

    if exit_code == 0:
        print("\n" + "=" * 80)
        print("✅ PHASE 5 TESTS PASSED - CLI Integration is Ready")
        print("✅ ALL MANDATORY REQUIREMENTS MET")
        print("✅ UNIVERSAL PROFILER COMPLETE")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ PHASE 5 TESTS FAILED")
        print("❌ CLI INTEGRATION BLOCKED")
        print("❌ FIX ALL ISSUES BEFORE DEPLOYMENT")
        print("=" * 80)

    return exit_code == 0


if __name__ == "__main__":
    success = run_phase5_tests()
    exit(0 if success else 1)
