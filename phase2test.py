#!/usr/bin/env python3
"""
Phase 2 Testing: Operation Separation and Data Extraction

MANDATORY TEST FILE for Phase 2 - Operation Detection and Data Parsing.
This file tests ALL operation separation scenarios outlined in the plan.

NO PHASE ADVANCEMENT WITHOUT 100% TEST COMPLETION
"""

import pytest
import tempfile
import os
import csv
from data_extractor import (
    parse_ops_perf_csv,
    filter_operations_by_type,
    align_performance_with_parameters,
    extract_visualization_data,
    find_latest_tracy_folder,
    generate_output_folder_name,
    OperationPerformanceData,
)
from parameter_detector import ParameterInfo


class TestCSVParsing:
    """Test parsing of ops_perf_results.csv files"""

    def create_mock_csv(self, data_rows: list, filename: str = None) -> str:
        """Helper to create mock CSV files for testing"""
        headers = [
            "OP CODE",
            "OP TYPE",
            "GLOBAL CALL COUNT",
            "DEVICE ID",
            "ATTRIBUTES",
            "MATH FIDELITY",
            "CORE COUNT",
            "PARALLELIZATION STRATEGY",
            "HOST START TS",
            "HOST END TS",
            "HOST DURATION [ns]",
            "DEVICE FW START CYCLE",
            "DEVICE FW END CYCLE",
            "OP TO OP LATENCY [ns]",
            "OP TO OP LATENCY BR/NRISC START [ns]",
            "DEVICE FW DURATION [ns]",
            "DEVICE KERNEL DURATION [ns]",  # This is the key column we need
        ]

        if filename is None:
            temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
            filename = temp_file.name
        else:
            temp_file = open(filename, "w")

        writer = csv.writer(temp_file)
        writer.writerow(headers)

        for row in data_rows:
            writer.writerow(row)

        temp_file.close()
        return filename

    def test_parse_single_operation_type(self):
        """Test parsing CSV with single operation type"""
        data_rows = [
            [
                "Pool2D",
                "tt_dnn_device",
                4096,
                0,
                "{}",
                "HiFi4",
                64,
                "",
                5116326497,
                5116448086,
                121589,
                3588321096872,
                3588321100366,
                13870942,
                13871225,
                3494,
                2812,
            ],  # DEVICE KERNEL DURATION = 2812
            [
                "Pool2D",
                "tt_dnn_device",
                8192,
                0,
                "{}",
                "HiFi4",
                64,
                "",
                5882185713,
                5882319613,
                133900,
                3589075849177,
                3589075853540,
                12532674,
                12532934,
                4363,
                3664,
            ],  # DEVICE KERNEL DURATION = 3664
        ]

        csv_file = self.create_mock_csv(data_rows)

        try:
            operations_data = parse_ops_perf_csv(csv_file)

            assert "Pool2D" in operations_data
            pool_data = operations_data["Pool2D"]
            assert pool_data.get_entry_count() == 2

            durations = pool_data.get_kernel_durations()
            assert durations == [2812.0, 3664.0]

        finally:
            os.unlink(csv_file)

    def test_parse_multiple_operation_types(self):
        """Test parsing CSV with multiple operation types"""
        data_rows = [
            [
                "InterleavedToShardedDeviceOperation",
                "tt_dnn_device",
                1024,
                0,
                "{}",
                "",
                64,
                "",
                5086794404,
                5086935254,
                140850,
                3588291991445,
                3588291995746,
                0,
                0,
                4301,
                3372,
            ],
            [
                "Pool2D",
                "tt_dnn_device",
                4096,
                0,
                "{}",
                "HiFi4",
                64,
                "",
                5116326497,
                5116448086,
                121589,
                3588321096872,
                3588321100366,
                13870942,
                13871225,
                3494,
                2812,
            ],
            [
                "MoveDeviceOperation",
                "tt_dnn_device",
                3072,
                0,
                "{}",
                "",
                64,
                "",
                5102251489,
                5102360199,
                108710,
                3588307225065,
                3588307226592,
                6566826,
                6566826,
                1527,
                656,
            ],
            [
                "Pool2D",
                "tt_dnn_device",
                8192,
                0,
                "{}",
                "HiFi4",
                64,
                "",
                5882185713,
                5882319613,
                133900,
                3589075849177,
                3589075853540,
                12532674,
                12532934,
                4363,
                3664,
            ],
        ]

        csv_file = self.create_mock_csv(data_rows)

        try:
            operations_data = parse_ops_perf_csv(csv_file)

            # Should have distinct operation types (using OP CODE)
            assert len(operations_data) == 3  # InterleavedToSharded, Pool2D, MoveDevice

            # Check that we have correct operations by name
            assert "InterleavedToShardedDeviceOperation" in operations_data
            assert "Pool2D" in operations_data
            assert "MoveDeviceOperation" in operations_data

            # Check entry counts
            assert operations_data["InterleavedToShardedDeviceOperation"].get_entry_count() == 1
            assert operations_data["Pool2D"].get_entry_count() == 2
            assert operations_data["MoveDeviceOperation"].get_entry_count() == 1

        finally:
            os.unlink(csv_file)

    def test_parse_missing_file(self):
        """Test handling of missing CSV file"""
        with pytest.raises(FileNotFoundError, match="Tracy CSV file not found"):
            parse_ops_perf_csv("/nonexistent/file.csv")

    def test_parse_empty_csv(self):
        """Test parsing empty CSV file"""
        csv_file = self.create_mock_csv([])

        try:
            operations_data = parse_ops_perf_csv(csv_file)
            assert len(operations_data) == 0
        finally:
            os.unlink(csv_file)

    def test_parse_malformed_csv(self):
        """Test handling malformed CSV data"""
        # Create a malformed CSV file
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write("invalid,csv,data\nwith,missing,headers")
        temp_file.close()

        try:
            operations_data = parse_ops_perf_csv(temp_file.name)
            # Should handle gracefully - may return empty or partial data
            assert isinstance(operations_data, dict)
        finally:
            os.unlink(temp_file.name)


class TestOperationDetection:
    """Test detection and filtering of different operation types"""

    def test_filter_by_include_types(self):
        """Test filtering operations by include list"""
        # Create mock operation data
        operations_data = {
            "Pool2D": OperationPerformanceData("Pool2D"),
            "Conv2D": OperationPerformanceData("Conv2D"),
            "MoveDeviceOperation": OperationPerformanceData("MoveDeviceOperation"),
        }

        # Add some mock entries
        for op_data in operations_data.values():
            op_data.add_entry({"DEVICE KERNEL DURATION [ns]": "1000"})

        # Filter to only include Pool2D and Conv2D
        filtered = filter_operations_by_type(operations_data, include_types=["Pool2D", "Conv2D"])

        assert len(filtered) == 2
        assert "Pool2D" in filtered
        assert "Conv2D" in filtered
        assert "MoveDeviceOperation" not in filtered

    def test_filter_by_exclude_types(self):
        """Test filtering operations by exclude list"""
        operations_data = {
            "Pool2D": OperationPerformanceData("Pool2D"),
            "Conv2D": OperationPerformanceData("Conv2D"),
            "MoveDeviceOperation": OperationPerformanceData("MoveDeviceOperation"),
        }

        for op_data in operations_data.values():
            op_data.add_entry({"DEVICE KERNEL DURATION [ns]": "1000"})

        # Exclude data movement operations
        filtered = filter_operations_by_type(operations_data, exclude_types=["MoveDeviceOperation"])

        assert len(filtered) == 2
        assert "Pool2D" in filtered
        assert "Conv2D" in filtered
        assert "MoveDeviceOperation" not in filtered

    def test_filter_include_and_exclude(self):
        """Test filtering with both include and exclude lists"""
        operations_data = {
            "Pool2D": OperationPerformanceData("Pool2D"),
            "Conv2D": OperationPerformanceData("Conv2D"),
            "MaxPool": OperationPerformanceData("MaxPool"),
            "MoveDeviceOperation": OperationPerformanceData("MoveDeviceOperation"),
        }

        for op_data in operations_data.values():
            op_data.add_entry({"DEVICE KERNEL DURATION [ns]": "1000"})

        # Include pool operations but exclude MaxPool specifically
        filtered = filter_operations_by_type(
            operations_data, include_types=["Pool2D", "MaxPool", "Conv2D"], exclude_types=["MaxPool"]
        )

        assert len(filtered) == 2
        assert "Pool2D" in filtered
        assert "Conv2D" in filtered
        assert "MaxPool" not in filtered
        assert "MoveDeviceOperation" not in filtered


class TestDataAlignment:
    """Test alignment of performance data with parameter combinations"""

    def test_align_single_operation_perfect_match(self):
        """Test alignment when counts match perfectly"""
        # Create mock operation data
        pool_data = OperationPerformanceData("Pool2D")
        pool_data.add_entry({"DEVICE KERNEL DURATION [ns]": "2000"})
        pool_data.add_entry({"DEVICE KERNEL DURATION [ns]": "3000"})
        pool_data.add_entry({"DEVICE KERNEL DURATION [ns]": "2500"})

        operations_data = {"Pool2D": pool_data}

        # Create parameter combinations
        param_combinations = [
            {"input_shape": (1, 32, 64, 64), "kernel_size": (2, 2)},
            {"input_shape": (1, 64, 64, 64), "kernel_size": (2, 2)},
            {"input_shape": (1, 128, 64, 64), "kernel_size": (2, 2)},
        ]

        aligned_data = align_performance_with_parameters(operations_data, param_combinations, target_operation="Pool2D")

        assert "Pool2D" in aligned_data
        pool_aligned = aligned_data["Pool2D"]
        assert len(pool_aligned) == 3

        # Check alignment
        assert pool_aligned[0][0]["input_shape"] == (1, 32, 64, 64)
        assert pool_aligned[0][1] == 2000.0
        assert pool_aligned[1][1] == 3000.0
        assert pool_aligned[2][1] == 2500.0

    def test_align_count_mismatch_handling(self):
        """Test handling when performance data count doesn't match parameter combinations"""
        # Create operation data with 2 entries
        pool_data = OperationPerformanceData("Pool2D")
        pool_data.add_entry({"DEVICE KERNEL DURATION [ns]": "2000"})
        pool_data.add_entry({"DEVICE KERNEL DURATION [ns]": "3000"})

        operations_data = {"Pool2D": pool_data}

        # Create 3 parameter combinations (mismatch)
        param_combinations = [
            {"input_shape": (1, 32, 64, 64)},
            {"input_shape": (1, 64, 64, 64)},
            {"input_shape": (1, 128, 64, 64)},
        ]

        aligned_data = align_performance_with_parameters(operations_data, param_combinations, target_operation="Pool2D")

        # Should take minimum count (2)
        assert "Pool2D" in aligned_data
        pool_aligned = aligned_data["Pool2D"]
        assert len(pool_aligned) == 2

    def test_align_multiple_operations(self):
        """Test aligning data for multiple operations"""
        # Create mock data for multiple operations
        pool_data = OperationPerformanceData("Pool2D")
        pool_data.add_entry({"DEVICE KERNEL DURATION [ns]": "2000"})
        pool_data.add_entry({"DEVICE KERNEL DURATION [ns]": "3000"})

        move_data = OperationPerformanceData("MoveDeviceOperation")
        move_data.add_entry({"DEVICE KERNEL DURATION [ns]": "500"})
        move_data.add_entry({"DEVICE KERNEL DURATION [ns]": "600"})

        operations_data = {"Pool2D": pool_data, "MoveDeviceOperation": move_data}

        param_combinations = [
            {"input_shape": (1, 32, 64, 64)},
            {"input_shape": (1, 64, 64, 64)},
        ]

        aligned_data = align_performance_with_parameters(operations_data, param_combinations)

        assert "Pool2D" in aligned_data
        assert "MoveDeviceOperation" in aligned_data
        assert len(aligned_data["Pool2D"]) == 2
        assert len(aligned_data["MoveDeviceOperation"]) == 2

    def test_align_target_operation_not_found(self):
        """Test error when target operation is not in data"""
        operations_data = {"Pool2D": OperationPerformanceData("Pool2D")}
        param_combinations = [{"input_shape": (1, 32, 64, 64)}]

        with pytest.raises(ValueError, match="Target operation 'Conv2D' not found"):
            align_performance_with_parameters(operations_data, param_combinations, target_operation="Conv2D")


class TestVisualizationDataExtraction:
    """Test extraction of visualization-ready data"""

    def create_mock_param_info(
        self,
        name: str,
        is_numeric: bool,
        changing_dim: int = None,
        display_name: str = None,
        extracted_values: list = None,
    ) -> ParameterInfo:
        """Helper to create mock ParameterInfo objects"""
        if display_name is None:
            display_name = name if changing_dim is None else f"{name} dim: {changing_dim}"
        if extracted_values is None:
            extracted_values = [1, 2, 3] if is_numeric else ["a", "b", "c"]

        param_info = ParameterInfo(
            name=name,
            values=extracted_values,  # This will be processed by ParameterInfo
            is_numeric=is_numeric,
            display_name=display_name,
            extracted_values=extracted_values,
            changing_dimension=changing_dim,
        )
        return param_info

    def test_extract_single_parameter_data(self):
        """Test extracting visualization data for single parameter"""
        # Mock aligned data
        aligned_data = {
            "Pool2D": [
                ({"input_shape": (1, 32, 64, 64)}, 2000.0),
                ({"input_shape": (1, 64, 64, 64)}, 3000.0),
                ({"input_shape": (1, 128, 64, 64)}, 2500.0),
            ]
        }

        # Mock parameter info for input_shape (changing dimension 1)
        x_axis_param = self.create_mock_param_info(
            name="input_shape",
            is_numeric=True,
            changing_dim=1,
            display_name="input_shape dim: 1",
            extracted_values=[32, 64, 128],
        )

        viz_data = extract_visualization_data(aligned_data, x_axis_param, None)

        assert "Pool2D" in viz_data
        pool_viz = viz_data["Pool2D"]

        assert pool_viz["x_values"] == [32, 64, 128]
        assert pool_viz["y_data"] == [2000.0, 3000.0, 2500.0]
        assert pool_viz["x_label"] == "input_shape dim: 1"
        assert pool_viz["legend_label"] is None

    def test_extract_two_parameter_data(self):
        """Test extracting visualization data for two parameters"""
        # Mock aligned data with two varying parameters
        aligned_data = {
            "Pool2D": [
                ({"input_shape": (1, 32, 64, 64), "kernel_size": (2, 2)}, 2000.0),
                ({"input_shape": (1, 32, 64, 64), "kernel_size": (3, 3)}, 2200.0),
                ({"input_shape": (1, 64, 64, 64), "kernel_size": (2, 2)}, 3000.0),
                ({"input_shape": (1, 64, 64, 64), "kernel_size": (3, 3)}, 3200.0),
            ]
        }

        # X-axis: kernel_size (changing dimension 0)
        x_axis_param = self.create_mock_param_info(
            name="kernel_size", is_numeric=True, changing_dim=0, display_name="kernel_size dim: 0"
        )

        # Legend: input_shape (changing dimension 1)
        legend_param = self.create_mock_param_info(
            name="input_shape", is_numeric=True, changing_dim=1, display_name="input_shape dim: 1"
        )

        viz_data = extract_visualization_data(aligned_data, x_axis_param, legend_param)

        assert "Pool2D" in viz_data
        pool_viz = viz_data["Pool2D"]

        assert "grouped_data" in pool_viz
        grouped = pool_viz["grouped_data"]

        # Should have two groups: one for each input_shape channel count
        assert "32" in grouped  # input_shape dim 1 = 32
        assert "64" in grouped  # input_shape dim 1 = 64

        # Check data structure
        assert grouped["32"]["x_values"] == [2, 3]  # kernel sizes
        assert grouped["32"]["y_values"] == [2000.0, 2200.0]
        assert grouped["64"]["x_values"] == [2, 3]
        assert grouped["64"]["y_values"] == [3000.0, 3200.0]

        assert pool_viz["x_label"] == "kernel_size dim: 0"
        assert pool_viz["legend_label"] == "input_shape dim: 1"

    def test_extract_simple_parameter_data(self):
        """Test extracting data with simple (non-tuple) parameters"""
        aligned_data = {
            "Pool2D": [
                ({"batch_size": 1, "channels": 32}, 1000.0),
                ({"batch_size": 2, "channels": 32}, 1200.0),
                ({"batch_size": 1, "channels": 64}, 1500.0),
                ({"batch_size": 2, "channels": 64}, 1700.0),
            ]
        }

        # X-axis: batch_size (simple parameter)
        x_axis_param = self.create_mock_param_info(
            name="batch_size", is_numeric=True, changing_dim=None, display_name="batch_size"
        )

        # Legend: channels (simple parameter)
        legend_param = self.create_mock_param_info(
            name="channels", is_numeric=True, changing_dim=None, display_name="channels"
        )

        viz_data = extract_visualization_data(aligned_data, x_axis_param, legend_param)

        assert "Pool2D" in viz_data
        pool_viz = viz_data["Pool2D"]

        grouped = pool_viz["grouped_data"]
        assert "32" in grouped
        assert "64" in grouped

        assert grouped["32"]["x_values"] == [1, 2]
        assert grouped["64"]["x_values"] == [1, 2]


class TestUtilityFunctions:
    """Test utility functions for folder management and naming"""

    def test_generate_output_folder_name(self):
        """Test output folder name generation"""
        folder_name = generate_output_folder_name("Pool2D", "20241210")

        # Should follow pattern: {operation_name}_{HW_arch}_{date}_{ordinal}
        parts = folder_name.split("_")
        assert len(parts) == 4
        assert parts[0] == "Pool2D"
        assert parts[1] in ["WH", "BH"]  # Hardware architecture
        assert parts[2] == "20241210"
        assert parts[3].isdigit()  # Ordinal

    def test_find_latest_tracy_folder_nonexistent(self):
        """Test finding latest folder when directory doesn't exist"""
        result = find_latest_tracy_folder("/nonexistent/path")
        assert result is None

    def test_find_latest_tracy_folder_empty(self):
        """Test finding latest folder in empty directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = find_latest_tracy_folder(temp_dir)
            assert result is None


class TestMultipleOperationTypes:
    """Test parsing of multiple distinct operation types from real CSV data"""

    def test_four_operation_types_parsing(self):
        """Test parsing CSV with 4 distinct operation types like in real maxpool test"""
        # Create realistic CSV data based on actual maxpool_simple.py output
        data_rows = [
            # InterleavedToShardedDeviceOperation
            [
                "InterleavedToShardedDeviceOperation",
                "tt_dnn_device",
                1024,
                0,
                "{}",
                "",
                64,
                "",
                5342485903,
                5342612463,
                126560,
                8188306371280,
                8188306375636,
                0,
                0,
                4356,
                3417,
                3417,
            ],
            # HaloDeviceOperation
            [
                "HaloDeviceOperation",
                "tt_dnn_device",
                2048,
                0,
                "{}",
                "",
                64,
                "",
                5352030553,
                5352154382,
                123829,
                8188315777513,
                8188315779392,
                9402809,
                9402809,
                1879,
                941,
                941,
            ],
            # MoveDeviceOperation
            [
                "MoveDeviceOperation",
                "tt_dnn_device",
                3072,
                0,
                "{}",
                "",
                64,
                "",
                5358922048,
                5359033137,
                111089,
                8188322570528,
                8188322572052,
                6792018,
                6792018,
                1524,
                645,
                645,
            ],
            # Pool2D (the main operation)
            [
                "Pool2D",
                "tt_dnn_device",
                4096,
                0,
                "{}",
                "HiFi4",
                64,
                "",
                5370728582,
                5370840002,
                111420,
                8188334205919,
                8188334209430,
                11634547,
                11634820,
                3511,
                2811,
                2811,
            ],
            # Second set for different input size
            [
                "InterleavedToShardedDeviceOperation",
                "tt_dnn_device",
                5120,
                0,
                "{}",
                "",
                64,
                "",
                7222348014,
                7222477484,
                129470,
                8190158966027,
                8190158970523,
                1824757553,
                1824757553,
                4496,
                3566,
                3566,
            ],
            [
                "HaloDeviceOperation",
                "tt_dnn_device",
                6144,
                0,
                "{}",
                "",
                64,
                "",
                7238100960,
                7238207950,
                106990,
                8190174492644,
                8190174494891,
                15523063,
                15523063,
                2247,
                1297,
                1297,
            ],
            [
                "MoveDeviceOperation",
                "tt_dnn_device",
                7168,
                0,
                "{}",
                "",
                64,
                "",
                7238703199,
                7238805048,
                101849,
                8190175084777,
                8190175086513,
                590758,
                590758,
                1736,
                862,
                862,
            ],
            [
                "Pool2D",
                "tt_dnn_device",
                8192,
                0,
                "{}",
                "HiFi4",
                64,
                "",
                7250969212,
                7251081692,
                112480,
                8190187175812,
                8190187180171,
                12089978,
                12090239,
                4359,
                3658,
                3658,
            ],
        ]

        csv_file = self.create_mock_csv(data_rows)

        try:
            operations_data = parse_ops_perf_csv(csv_file)

            # Should have 4 distinct operation types (using OP CODE, not OP TYPE)
            assert len(operations_data) == 4
            assert "InterleavedToShardedDeviceOperation" in operations_data
            assert "HaloDeviceOperation" in operations_data
            assert "MoveDeviceOperation" in operations_data
            assert "Pool2D" in operations_data

            # Check entry counts
            assert operations_data["InterleavedToShardedDeviceOperation"].get_entry_count() == 2
            assert operations_data["HaloDeviceOperation"].get_entry_count() == 2
            assert operations_data["MoveDeviceOperation"].get_entry_count() == 2
            assert operations_data["Pool2D"].get_entry_count() == 2

            # Validate kernel durations are different for each operation type
            interleaved_durations = operations_data["InterleavedToShardedDeviceOperation"].get_kernel_durations()
            halo_durations = operations_data["HaloDeviceOperation"].get_kernel_durations()
            move_durations = operations_data["MoveDeviceOperation"].get_kernel_durations()
            pool_durations = operations_data["Pool2D"].get_kernel_durations()

            assert interleaved_durations == [3417.0, 3566.0]
            assert halo_durations == [941.0, 1297.0]
            assert move_durations == [645.0, 862.0]
            assert pool_durations == [2811.0, 3658.0]

        finally:
            os.unlink(csv_file)

    def create_mock_csv(self, data_rows: list) -> str:
        """Helper to create mock CSV files for testing"""
        headers = [
            "OP CODE",
            "OP TYPE",
            "GLOBAL CALL COUNT",
            "DEVICE ID",
            "ATTRIBUTES",
            "MATH FIDELITY",
            "CORE COUNT",
            "PARALLELIZATION STRATEGY",
            "HOST START TS",
            "HOST END TS",
            "HOST DURATION [ns]",
            "DEVICE FW START CYCLE",
            "DEVICE FW END CYCLE",
            "OP TO OP LATENCY [ns]",
            "OP TO OP LATENCY BR/NRISC START [ns]",
            "DEVICE FW DURATION [ns]",
            "DEVICE KERNEL DURATION [ns]",  # This is the key column we extract
        ]

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        writer = csv.writer(temp_file)
        writer.writerow(headers)

        for row in data_rows:
            writer.writerow(row)

        temp_file.close()
        return temp_file.name


class TestEndToEndOperationSeparation:
    """Test complete end-to-end operation separation workflow"""

    def test_pool2d_with_data_movement_ops(self):
        """Test Pool2D operations mixed with data movement operations"""
        # Create a realistic CSV with mixed operation types
        data_rows = [
            # Data movement operation
            [
                "InterleavedToShardedDeviceOperation",
                "tt_dnn_device",
                1024,
                0,
                "{}",
                "",
                64,
                "",
                5086794404,
                5086935254,
                140850,
                3588291991445,
                3588291995746,
                0,
                0,
                4301,
                3372,
            ],
            # First Pool2D operation
            [
                "Pool2D",
                "tt_dnn_device",
                4096,
                0,
                "{}",
                "HiFi4",
                64,
                "",
                5116326497,
                5116448086,
                121589,
                3588321096872,
                3588321100366,
                13870942,
                13871225,
                3494,
                2812,
            ],
            # Another data movement operation
            [
                "MoveDeviceOperation",
                "tt_dnn_device",
                7168,
                0,
                "{}",
                "",
                64,
                "",
                5869473273,
                5869603192,
                129919,
                3589063315424,
                3589063317181,
                684223,
                684223,
                1757,
                890,
            ],
            # Second Pool2D operation
            [
                "Pool2D",
                "tt_dnn_device",
                8192,
                0,
                "{}",
                "HiFi4",
                64,
                "",
                5882185713,
                5882319613,
                133900,
                3589075849177,
                3589075853540,
                12532674,
                12532934,
                4363,
                3664,
            ],
        ]

        csv_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        writer = csv.writer(csv_file)

        # Write headers
        headers = [
            "OP CODE",
            "OP TYPE",
            "GLOBAL CALL COUNT",
            "DEVICE ID",
            "ATTRIBUTES",
            "MATH FIDELITY",
            "CORE COUNT",
            "PARALLELIZATION STRATEGY",
            "HOST START TS",
            "HOST END TS",
            "HOST DURATION [ns]",
            "DEVICE FW START CYCLE",
            "DEVICE FW END CYCLE",
            "OP TO OP LATENCY [ns]",
            "OP TO OP LATENCY BR/NRISC START [ns]",
            "DEVICE FW DURATION [ns]",
            "DEVICE KERNEL DURATION [ns]",
        ]
        writer.writerow(headers)

        for row in data_rows:
            writer.writerow(row)

        csv_file.close()

        try:
            # Parse all operations
            operations_data = parse_ops_perf_csv(csv_file.name)

            # Should have distinct operation types (using OP CODE)
            assert len(operations_data) == 3
            assert "InterleavedToShardedDeviceOperation" in operations_data
            assert "Pool2D" in operations_data
            assert "MoveDeviceOperation" in operations_data

            # Check entry counts and durations
            assert operations_data["InterleavedToShardedDeviceOperation"].get_entry_count() == 1
            assert operations_data["Pool2D"].get_entry_count() == 2
            assert operations_data["MoveDeviceOperation"].get_entry_count() == 1

            # Verify durations for Pool2D operations
            pool_durations = operations_data["Pool2D"].get_kernel_durations()
            assert pool_durations == [2812.0, 3664.0]

        finally:
            os.unlink(csv_file.name)


def run_phase2_tests():
    """Run all Phase 2 tests and report results"""
    print("=" * 80)
    print("PHASE 2 MANDATORY TESTING: Operation Separation and Data Extraction")
    print("=" * 80)

    # Run pytest on this file
    exit_code = pytest.main([__file__, "-v"])

    if exit_code == 0:
        print("\n" + "=" * 80)
        print("✅ PHASE 2 TESTS PASSED - Operation Separation is Ready")
        print("✅ ALL MANDATORY REQUIREMENTS MET")
        print("✅ PHASE 3 ADVANCEMENT APPROVED")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ PHASE 2 TESTS FAILED")
        print("❌ PHASE ADVANCEMENT BLOCKED")
        print("❌ FIX ALL ISSUES BEFORE PROCEEDING")
        print("=" * 80)

    return exit_code == 0


if __name__ == "__main__":
    success = run_phase2_tests()
    exit(0 if success else 1)
