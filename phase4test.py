#!/usr/bin/env python3
"""
Phase 4 Testing: Visualization Generation

MANDATORY TEST FILE for Phase 4 - Plot Generation and Reporting.
"""

import pytest
import tempfile
import os
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch
import numpy as np

from visualizer import (
    create_performance_plots,
    create_summary_report,
    generate_output_folder_name,
    generate_all_visualizations,
    VisualizationConfig,
)
from parameter_detector import ParameterInfo


class TestVisualizationConfig:
    """Test visualization configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = VisualizationConfig()

        assert config.figure_size == (12, 8)
        assert config.dpi == 300
        assert config.format == "svg"
        assert len(config.colors) == 10
        assert len(config.markers) == 10


class TestSingleParameterPlots:
    """Test visualization generation for single parameter scenarios"""

    def create_single_param_viz_data(self):
        """Helper to create single parameter visualization data"""
        return {
            "Pool2D": {
                "x_values": [32, 64, 96, 128, 160],
                "y_data": [2000, 2800, 3200, 4100, 4800],
                "x_label": "input_shape dim: 1",
                "legend_label": None,
            }
        }

    def test_single_parameter_plot_creation(self):
        """Test creation of single parameter plot"""
        viz_data = self.create_single_param_viz_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = VisualizationConfig()
            plot_files = create_performance_plots(viz_data, temp_dir, config, save_plots=True)

            assert len(plot_files) == 1
            assert plot_files[0].endswith("Pool2D_performance.svg")
            assert os.path.exists(plot_files[0])

    def test_single_parameter_plot_no_save(self):
        """Test plot creation without saving"""
        viz_data = self.create_single_param_viz_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            plot_files = create_performance_plots(viz_data, temp_dir, save_plots=False)

            assert len(plot_files) == 0


class TestTwoParameterPlots:
    """Test visualization generation for two parameter scenarios"""

    def create_two_param_viz_data(self):
        """Helper to create two parameter visualization data"""
        return {
            "Pool2D": {
                "grouped_data": {
                    "32": {"x_values": [2, 3, 4], "y_values": [1800, 2200, 2600]},
                    "64": {"x_values": [2, 3, 4], "y_values": [2400, 2800, 3200]},
                    "128": {"x_values": [2, 3, 4], "y_values": [3600, 4200, 4800]},
                },
                "x_label": "kernel_size dim: 1",
                "legend_label": "input_shape dim: 1",
            }
        }

    def test_two_parameter_plot_creation(self):
        """Test creation of two parameter plot"""
        viz_data = self.create_two_param_viz_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = VisualizationConfig()
            plot_files = create_performance_plots(viz_data, temp_dir, config, save_plots=True)

            assert len(plot_files) == 1
            assert plot_files[0].endswith("Pool2D_performance.svg")
            assert os.path.exists(plot_files[0])


class TestSummaryReports:
    """Test summary report generation"""

    def create_mock_param_info(self):
        """Create mock parameter info for testing"""
        return [
            ParameterInfo(
                name="input_shape",
                values=[(1, 32, 64, 64), (1, 64, 64, 64), (1, 128, 64, 64)],
                is_numeric=True,
                display_name="input_shape dim: 1",
                extracted_values=[32, 64, 128],
                changing_dimension=1,
            )
        ]

    def create_mock_performance_data(self):
        """Create mock performance data"""
        return {
            "Pool2D": [
                ({"input_shape": (1, 32, 64, 64)}, 2000.0),
                ({"input_shape": (1, 64, 64, 64)}, 3000.0),
                ({"input_shape": (1, 128, 64, 64)}, 4000.0),
            ]
        }

    def test_summary_report_creation(self):
        """Test creation of summary report"""
        param_info = self.create_mock_param_info()
        performance_data = self.create_mock_performance_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = create_summary_report(param_info, performance_data, temp_dir)

            assert os.path.exists(report_path)
            assert report_path.endswith("profiling_summary.txt")

            with open(report_path, "r") as f:
                content = f.read()
                assert "UNIVERSAL TTNN OPERATION PROFILING REPORT" in content
                assert "PARAMETER ANALYSIS" in content
                assert "PERFORMANCE SUMMARY" in content


class TestOutputFolderNaming:
    """Test output folder name generation"""

    def test_default_folder_naming(self):
        """Test default folder name generation"""
        folder_name = generate_output_folder_name()

        parts = folder_name.split("_")
        assert len(parts) >= 5  # TTNN_Op becomes TTNN + Op when split
        assert parts[0] == "TTNN"  # First part of default operation name
        assert parts[1] == "Op"  # Second part of default operation name
        assert parts[2] in ["WH", "BH"]  # Hardware architecture
        assert len(parts[3]) == 8  # Date in YYYYMMDD format
        assert parts[4].isdigit()  # Ordinal


class TestComprehensiveVisualization:
    """Test the complete visualization generation workflow"""

    def test_generate_all_visualizations(self):
        """Test generation of all visualization types"""
        param_info = [
            ParameterInfo(
                name="input_shape",
                values=[(1, 32, 64, 64), (1, 64, 64, 64)],
                is_numeric=True,
                display_name="input_shape dim: 1",
                extracted_values=[32, 64],
                changing_dimension=1,
            )
        ]

        viz_data = {
            "Pool2D": {
                "x_values": [32, 64],
                "y_data": [2000, 3000],
                "x_label": "input_shape dim: 1",
                "legend_label": None,
            }
        }

        performance_data = {
            "Pool2D": [
                ({"input_shape": (1, 32, 64, 64)}, 2000.0),
                ({"input_shape": (1, 64, 64, 64)}, 3000.0),
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            results = generate_all_visualizations(viz_data, param_info, performance_data, temp_dir)

            # Should have static plots
            assert "static_plots" in results
            assert len(results["static_plots"]) == 1

            # Should have summary report
            assert "summary_report" in results
            assert len(results["summary_report"]) == 1

            # Should NOT have interactive plot (removed per user request)
            assert "interactive_plot" not in results

            # Verify all files exist
            for file_list in results.values():
                for file_path in file_list:
                    assert os.path.exists(file_path)


def run_phase4_tests():
    """Run all Phase 4 tests and report results"""
    print("=" * 80)
    print("PHASE 4 MANDATORY TESTING: Visualization Generation")
    print("=" * 80)

    # Run pytest on this file
    exit_code = pytest.main([__file__, "-v"])

    if exit_code == 0:
        print("\n" + "=" * 80)
        print("✅ PHASE 4 TESTS PASSED - Visualization Generation is Ready")
        print("✅ ALL MANDATORY REQUIREMENTS MET")
        print("✅ PHASE 5 ADVANCEMENT APPROVED")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ PHASE 4 TESTS FAILED")
        print("❌ PHASE ADVANCEMENT BLOCKED")
        print("❌ FIX ALL ISSUES BEFORE PROCEEDING")
        print("=" * 80)

    return exit_code == 0


if __name__ == "__main__":
    success = run_phase4_tests()
    exit(0 if success else 1)
