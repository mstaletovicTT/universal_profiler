# Universal TTNN Operation Profiler

A comprehensive automated profiling tool for analyzing performance of TTNN (TT Neural Network) operations across different parameter configurations. This tool automatically detects test parameters, runs Tracy profiling, and generates visual performance analysis.

## Quick Start with Example

This repository includes a complete working example:

- **`ConvTestExample.py`**: A convolution input channels sweep test
- **`example_output/`**: Pre-generated output showing what to expect

Run the example immediately:
```bash
python universal_profiler.py ConvTestExample.py
```

## Overview

The Universal TTNN Operation Profiler streamlines the entire profiling workflow from parameter detection to visualization generation. It eliminates manual intervention by automatically:

1. **Detecting parameters** from pytest test files
2. **Running Tracy profiling** with real-time test execution
3. **Extracting performance data** from Tracy output
4. **Generating visualizations** with comprehensive analysis

## Key Features

- **Automatic Parameter Detection**: Intelligently identifies varying and static parameters from pytest decorators
- **Tracy Integration**: Seamlessly integrates with Tracy profiling for accurate performance measurements
- **Multi-dimensional Analysis**: Supports single and multi-parameter sweeps with automatic legend generation
- **Performance Visualization**: Creates publication-ready SVG plots with performance metrics
- **Core Count Annotations**: Displays device core utilization directly on performance graphs
- **Static Parameter Tracking**: Preserves and displays static configuration parameters
- **Comprehensive Reporting**: Generates detailed summary reports with statistical analysis

## Installation

### Prerequisites

This tool requires the **standard TT Metal Python environment** with TTNN installed. Ensure you have:

1. **TT Metal Environment**: Activate your TT Metal Python virtual environment
   ```bash
   source /path/to/tt-metal/python_env/bin/activate
   ```

2. **Required Python packages** (should already be in TT Metal environment):
   ```bash
   # Verify dependencies are available
   python -c "import matplotlib, numpy, ttnn"
   
   # Verify Tracy is available
   python -m tracy --version
   ```

3. **Device Setup**: Ensure TT Metal device is properly configured and accessible

If any dependencies are missing, install them:
```bash
pip install matplotlib numpy
```

## Usage

### Basic Usage

Profile any TTNN test file with a single command:

```bash
python universal_profiler.py /path/to/test_file.py
```

### Example with Real Test

A complete example test file is included: **`ConvTestExample.py`**

Run the example:

```bash
python universal_profiler.py ConvTestExample.py
```

This test file demonstrates:
1. **Input channels parameter sweep**: 32, 64, 96, 128, 160, 192, 256, 288, 320, 352, 384
2. **Math fidelity variations**: LoFi, HiFi2, HiFi4
3. **33 parameter combinations** (11 input channels × 3 fidelities)
4. **Static parameters**: batch_size=1, output_channels=128, kernel=3×3, etc.

The profiler will:
- Detect all varying and static parameters automatically
- Run Tracy profiling for all combinations
- Generate performance visualization plots
- Create a summary report with statistics

See the **`example_output/`** directory for the expected output from this test.

### Output Structure

The tool generates an output directory with the following structure:

```
profiling_results/
└── Conv_Input_Channels_Sweep_BH_20251224_3/
    ├── Conv2dDeviceOperation_performance.svg
    ├── HaloDeviceOperation_performance.svg
    ├── MoveDeviceOperation_performance.svg
    ├── profiling_summary.txt
    └── original_ops_perf_results.csv
```

**See `example_output/` directory** in this repository for a complete example of generated output files.

### Generated Files

#### Performance Plot (SVG)

The main visualization shows:
- **X-axis**: The primary varying parameter (e.g., `input_channels`)
- **Y-axis**: Device kernel duration in nanoseconds
- **Legend**: Secondary parameter variations (e.g., `math_fidelity`)
- **Annotations**: Core count labels on data points (showing "66" cores in the example)
- **Static Parameters Box**: Lists all non-varying parameters and their values

**Example output**: See `example_output/Conv2dDeviceOperation_performance.svg` for a complete visualization

#### Summary Report (TXT)

Contains:
- Parameter analysis with types and value counts
- Static parameter listing
- Performance statistics (min, max, avg duration)
- Timestamp and execution metadata

#### Original CSV

Preserved copy of the Tracy profiling results for reference and further analysis.

## Command Line Options

```
positional arguments:
  test_file            Path to the TTNN test file (e.g., test_maxpool_simple.py)

optional arguments:
  -h, --help           Show help message and exit
  --output, -o         Custom output directory (default: auto-generated)
  --pltsave            Enable saving plots as SVG files (default: enabled)
  --no-pltsave         Disable saving plots as SVG files
  --x-param            Manual X-axis parameter specification
  --legend-param       Manual legend parameter specification
  --extract-config     Custom parameter extraction configuration file
  --timeout            Tracy execution timeout in seconds (default: 600)
  --verbose, -v        Enable verbose output
```

## How It Works

### Phase 1: Parameter Detection

The tool analyzes the pytest test file to identify:
- **Varying Parameters**: Parameters with multiple values (e.g., `input_channels=[32, 64, 96, ...]`)
- **Static Parameters**: Parameters with single values (e.g., `batch_size=1`)
- **Parameter Types**: Numeric vs categorical classification
- **Dimensions**: Which dimension of tuple parameters varies

### Phase 2: Tracy Profiling

Executes the test file with Tracy profiling:
```bash
python -m tracy -r -m pytest test_file.py -v
```

- Streams real-time test execution output
- Captures Tracy profiling data
- Handles timeout and error conditions
- Validates Tracy output artifacts

### Phase 3: Data Extraction

Processes Tracy profiling results:
- Parses `ops_perf_results_*.csv` files
- Aligns performance data with parameter combinations
- Extracts device kernel durations and core counts
- Organizes data by operation type

### Phase 4: Visualization

Generates comprehensive visualizations:
- Creates line plots with markers for each parameter combination
- Adds core count annotations to data points
- Displays static parameters in an overlay box
- Generates high-resolution SVG output

## Example Output

### Test Configuration

```python
@pytest.mark.parametrize("input_channels", [32, 64, 96, 128, 160, 192, 256, 288, 320, 352, 384])
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("output_channels", [128])
# ... other static parameters
```

### Visualization

The generated plot displays:
- Three lines representing each `math_fidelity` value
- X-axis showing `input_channels` from 32 to 384
- Y-axis showing device kernel duration
- Annotations showing core count (e.g., "66" cores)
- Static parameters box showing all constant values

## Usefulness

### Performance Analysis
- **Compare configurations**: Quickly identify which parameter values provide optimal performance
- **Identify trends**: Visualize how performance scales with parameter changes
- **Spot anomalies**: Detect unexpected performance degradations or improvements
- **Core utilization**: Understand resource allocation across different configurations

### Development Workflow
- **Rapid iteration**: Test multiple configurations without manual intervention
- **Reproducibility**: Consistent profiling methodology across tests
- **Documentation**: Automatic generation of performance documentation
- **Regression detection**: Track performance changes over time

### Research & Optimization
- **Parameter tuning**: Systematically explore parameter space
- **Architecture comparison**: Compare different math fidelities, data types, layouts
- **Bottleneck identification**: Pinpoint performance-limiting configurations
- **Resource optimization**: Balance performance vs resource utilization

## Advanced Features

### Custom Parameter Selection

Override automatic parameter detection:

```bash
python universal_profiler.py test_file.py --x-param kernel_size --legend-param stride
```

### Custom Output Location

Specify output directory:

```bash
python universal_profiler.py test_file.py --output ./my_results
```

### Verbose Mode

Enable detailed execution information:

```bash
python universal_profiler.py test_file.py -v
```

## Architecture Components

The profiler consists of modular components:

1. **`universal_profiler.py`**: Main CLI interface and workflow orchestration
2. **`parameter_detector.py`**: AST-based parameter extraction from pytest files
3. **`test_executor.py`**: Tracy profiling execution and output management
4. **`data_extractor.py`**: Performance data parsing and alignment
5. **`visualizer.py`**: Plot generation and report creation

## Limitations & Considerations

- Requires valid pytest test files with `@pytest.mark.parametrize` decorators
- Tracy profiling must be properly configured in the environment
- Large parameter sweeps may require extended execution time
- Assumes TTNN operations emit Tracy profiling events

## Troubleshooting

### No Tracy output detected
- Verify Tracy is installed: `python -m tracy --version`
- Check test file executes successfully: `pytest test_file.py -v`
- Ensure TTNN operations include Tracy instrumentation

### Parameter detection fails
- Verify test file uses pytest parametrize decorators
- Check for syntax errors in test file
- Use `--verbose` flag for detailed error messages

### Timeout during execution
- Increase timeout: `--timeout 1200`
- Check for infinite loops or blocking operations in tests
- Verify device availability and responsiveness

## Contributing

For issues, improvements, or questions, please refer to the project repository or contact the development team.

## License

This tool is part of the TT-Metal project and follows the project's licensing terms.
