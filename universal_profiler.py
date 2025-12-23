#!/usr/bin/env python3
"""
Universal TTNN Operation Profiling Script

This script provides a complete automated profiling solution for any TTNN operation,
automatically detecting parameters, running Tracy profiling, and generating visualizations.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from parameter_detector import detect_and_analyze_parameters
from test_executor import run_profiling_test, create_execution_summary
from data_extractor import parse_ops_perf_csv, align_performance_with_parameters, extract_visualization_data
from visualizer import generate_all_visualizations, VisualizationConfig


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup CLI argument parser with all required options"""
    parser = argparse.ArgumentParser(
        description="Universal TTNN Operation Profiling Script", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required positional argument
    parser.add_argument("test_file", help="Path to the TTNN test file (e.g., test_maxpool_simple.py)")

    # Output options
    parser.add_argument(
        "--output", "-o", help="Custom output directory (default: auto-generated with operation_name_HW_date_ordinal)"
    )

    # Plot saving options
    parser.add_argument(
        "--pltsave", action="store_true", default=True, help="Enable saving plots as SVG files (default: enabled)"
    )
    parser.add_argument("--no-pltsave", action="store_false", dest="pltsave", help="Disable saving plots as SVG files")

    # Manual parameter specification (fallback)
    parser.add_argument("--x-param", help="Manual X-axis parameter specification for complex cases")
    parser.add_argument("--legend-param", help="Manual legend parameter specification for complex cases")

    # Configuration
    parser.add_argument("--extract-config", help="Custom parameter extraction configuration file")

    # Tracy options
    parser.add_argument("--timeout", type=int, default=600, help="Tracy execution timeout in seconds")

    # Verbose output
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    return parser


def validate_arguments(args) -> None:
    """Validate command line arguments"""
    # Check test file exists
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")

    # Check test file is a Python file
    if not args.test_file.endswith(".py"):
        raise ValueError(f"Test file must be a Python file: {args.test_file}")

    # Check extract config file exists if provided
    if args.extract_config and not os.path.exists(args.extract_config):
        raise FileNotFoundError(f"Extract config file not found: {args.extract_config}")

    # Validate timeout
    if args.timeout <= 0:
        raise ValueError(f"Timeout must be positive: {args.timeout}")


def detect_operation_name(test_file_path: str) -> str:
    """Detect operation name from test file"""
    # Simple heuristic: look for common operation patterns in filename
    filename = Path(test_file_path).stem

    # Remove common test prefixes/suffixes
    name = filename.replace("test_", "").replace("_test", "")

    return name.title()


def main() -> int:
    """Main CLI entry point"""
    parser = setup_argument_parser()
    args = parser.parse_args()

    try:
        # Validate arguments
        validate_arguments(args)

        if args.verbose:
            print("=" * 80)
            print("UNIVERSAL TTNN OPERATION PROFILING")
            print("=" * 80)
            print(f"📁 Test file: {args.test_file}")

        # Phase 1: Parameter Detection
        if args.verbose:
            print("\n🔍 Phase 1: Parameter Detection")

        try:
            param_info_list, x_axis_param, legend_param, param_combinations, static_params = detect_and_analyze_parameters(
                args.test_file
            )
        except Exception as e:
            print(f"❌ Parameter detection failed: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            return 1

        if args.verbose:
            print(f"✅ Detected {len(param_info_list)} varying parameters")
            print(f"📊 X-axis parameter: {x_axis_param.display_name}")
            if legend_param:
                print(f"📈 Legend parameter: {legend_param.display_name}")
            print(f"🔢 Total parameter combinations: {len(param_combinations)}")
            if static_params:
                print(f"📌 Static parameters: {len(static_params)}")
                for name, value in static_params.items():
                    print(f"   - {name}: {value}")

        # Manual parameter override if specified
        if args.x_param:
            # Find matching parameter
            matching_x = None
            for param in param_info_list:
                if param.name == args.x_param:
                    matching_x = param
                    break
            if not matching_x:
                print(f"❌ X-axis parameter '{args.x_param}' not found in detected parameters")
                return 1
            x_axis_param = matching_x

        if args.legend_param:
            # Find matching parameter
            matching_legend = None
            for param in param_info_list:
                if param.name == args.legend_param:
                    matching_legend = param
                    break
            if not matching_legend:
                print(f"❌ Legend parameter '{args.legend_param}' not found in detected parameters")
                return 1
            legend_param = matching_legend

        # Phase 2: Tracy Profiling
        if args.verbose:
            print(f"\n⚡ Phase 2: Tracy Profiling ({len(param_combinations)} combinations)")
            print("Starting pytest execution with Tracy profiling...\n")

        try:
            exec_result = run_profiling_test(args.test_file, verbose=args.verbose, timeout=args.timeout)
        except Exception as e:
            print(f"❌ Tracy profiling failed: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            return 1

        if not exec_result.success:
            print(f"❌ Tracy execution failed (exit code: {exec_result.exit_code})")
            return 1

        if args.verbose:
            print(f"📂 Tracy output folder: {exec_result.tracy_folder}")
            print(f"⏱️  Execution time: {exec_result.execution_time:.2f}s")

        # Phase 3: Data Extraction
        if args.verbose:
            print("\n📊 Phase 3: Data Extraction")

        try:
            # Find the CSV file in tracy folder
            csv_files = [
                f
                for f in os.listdir(exec_result.tracy_folder)
                if f.startswith("ops_perf_results") and f.endswith(".csv")
            ]
            if not csv_files:
                raise FileNotFoundError("No ops_perf_results CSV file found in Tracy output")

            csv_path = os.path.join(exec_result.tracy_folder, csv_files[0])

            if args.verbose:
                print(f"📄 Reading CSV file: {csv_path}")

            operations_data = parse_ops_perf_csv(csv_path)

            if args.verbose:
                print(f"✅ Parsed operations data: {len(operations_data)} operation types found")
                for op_type, perf_data in operations_data.items():
                    print(f"   - {op_type}: {perf_data.get_entry_count()} entries")

            # Align performance data with parameters
            aligned_data = align_performance_with_parameters(operations_data, param_combinations)

            if args.verbose:
                print("✅ Data alignment completed")
                for op_type, data_points in aligned_data.items():
                    print(f"   - {op_type}: {len(data_points)} aligned data points")

        except Exception as e:
            print(f"❌ Data extraction failed: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            return 1

        # Phase 4: Visualization Generation
        if args.verbose:
            print("\n🎨 Phase 4: Visualization Generation")

        try:
            # Extract visualization data
            viz_data = extract_visualization_data(aligned_data, x_axis_param, legend_param)

            # Determine output directory
            if args.output:
                output_dir = args.output
            else:
                # Auto-generate output directory name
                operation_name = detect_operation_name(args.test_file)
                from visualizer import generate_output_folder_name
                from data_extractor import get_hardware_architecture

                base_output_dir = "profiling_results"
                hw_arch = get_hardware_architecture()
                folder_name = generate_output_folder_name(operation_name, hw_arch=hw_arch, output_base_dir=base_output_dir)
                output_dir = os.path.join(base_output_dir, folder_name)

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Generate visualizations
            config = VisualizationConfig()
            config.format = "svg"  # Default to SVG for CLI

            results = generate_all_visualizations(viz_data, param_info_list, aligned_data, output_dir, config, csv_path, static_params)

            if args.verbose:
                print(f"✅ Visualizations generated in: {output_dir}")
                if "static_plots" in results:
                    print(f"   📊 Static plots: {len(results['static_plots'])} files")
                if "summary_report" in results:
                    print(f"   📄 Summary report: {len(results['summary_report'])} files")
                if "original_csv" in results:
                    print(f"   📄 Original CSV: {len(results['original_csv'])} files")

        except Exception as e:
            print(f"❌ Visualization generation failed: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            return 1

        # Success summary
        print("\n" + "=" * 80)
        print("✅ PROFILING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"📂 Output directory: {output_dir}")
        print(f"📊 Operations profiled: {', '.join(aligned_data.keys())}")
        print(f"🔢 Parameter combinations: {len(param_combinations)}")

        # List generated files
        all_files = []
        for file_list in results.values():
            all_files.extend(file_list)

        print(f"📁 Generated files:")
        for file_path in all_files:
            print(f"   - {os.path.relpath(file_path)}")

        return 0

    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
