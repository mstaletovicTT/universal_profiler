#!/usr/bin/env python3
"""
Phase 1 Testing: Parameter Extraction and Classification

MANDATORY TEST FILE for Phase 1 - Parameter Detection.
This file tests ALL parameter detection scenarios outlined in the plan.

NO PHASE ADVANCEMENT WITHOUT 100% TEST COMPLETION
"""

import pytest
import tempfile
import os
from parameter_detector import (
    parse_test_script,
    filter_and_classify_parameters,
    assign_visualization_roles,
    detect_and_analyze_parameters,
    ParameterInfo,
)


class TestASTParameterParsing:
    """Test AST parsing of @pytest.mark.parametrize decorators"""

    def test_single_parametrize_basic(self):
        """Test parsing single parametrize decorator with simple values"""
        test_content = """
import pytest

@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5])
def test_func(batch_size):
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_content)
            f.flush()

            try:
                result = parse_test_script(f.name)
                assert "batch_size" in result
                assert result["batch_size"] == [1, 2, 3, 4, 5]
            finally:
                os.unlink(f.name)

    def test_single_parametrize_tuples(self):
        """Test parsing parametrize with tuple values"""
        test_content = """
import pytest

@pytest.mark.parametrize("input_shape", [(1, 32, 64, 64), (1, 64, 64, 64), (1, 128, 64, 64)])
def test_func(input_shape):
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_content)
            f.flush()

            try:
                result = parse_test_script(f.name)
                assert "input_shape" in result
                expected = [(1, 32, 64, 64), (1, 64, 64, 64), (1, 128, 64, 64)]
                assert result["input_shape"] == expected
            finally:
                os.unlink(f.name)

    def test_multiple_parametrize_decorators(self):
        """Test parsing multiple parametrize decorators"""
        test_content = """
import pytest

@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("input_shape", [(1, 32, 64, 64), (1, 64, 64, 64)])
@pytest.mark.parametrize("kernel_size", [(2, 2), (3, 3)])
def test_func(input_shape, kernel_size, device_params):
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_content)
            f.flush()

            try:
                result = parse_test_script(f.name)
                assert "input_shape" in result
                assert "kernel_size" in result
                assert "device_params" in result
                assert result["input_shape"] == [(1, 32, 64, 64), (1, 64, 64, 64)]
                assert result["kernel_size"] == [(2, 2), (3, 3)]
                assert result["device_params"] == [{"l1_small_size": 16384}]
            finally:
                os.unlink(f.name)

    def test_string_parameters(self):
        """Test parsing string categorical parameters"""
        test_content = """
import pytest

@pytest.mark.parametrize("memory_layout", ["INTERLEAVED", "HEIGHT_SHARDED", "WIDTH_SHARDED"])
def test_func(memory_layout):
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_content)
            f.flush()

            try:
                result = parse_test_script(f.name)
                assert "memory_layout" in result
                assert result["memory_layout"] == ["INTERLEAVED", "HEIGHT_SHARDED", "WIDTH_SHARDED"]
            finally:
                os.unlink(f.name)


class TestSingleValueFiltering:
    """Test filtering of parameters with only one value"""

    def test_filter_single_values(self):
        """Test that single-value parameters are correctly filtered out"""
        parametrizations = {
            "batch_size": [1, 2, 3, 4],  # Varying - should be kept
            "kernel_size": [(2, 2)],  # Single value - should be filtered
            "channels": [32, 64, 128],  # Varying - should be kept
            "device": ["cuda"],  # Single value - should be filtered
        }

        result = filter_and_classify_parameters(parametrizations)

        # Should have only 2 parameters (batch_size and channels)
        assert len(result) == 2
        param_names = [p.name for p in result]
        assert "batch_size" in param_names
        assert "channels" in param_names
        assert "kernel_size" not in param_names
        assert "device" not in param_names

    def test_no_varying_parameters_error(self):
        """Test error when all parameters have single values"""
        parametrizations = {"kernel_size": [(2, 2)], "device": ["cuda"], "batch_size": [1]}

        with pytest.raises(ValueError, match="No varying parameters found"):
            filter_and_classify_parameters(parametrizations)

    def test_too_many_varying_parameters_error(self):
        """Test error when more than 2 parameters vary"""
        parametrizations = {
            "batch_size": [1, 2, 3],
            "channels": [32, 64],
            "height": [64, 128],
            "width": [64, 128, 256],  # 4 varying parameters - should error
        }

        with pytest.raises(ValueError, match="Too many varying parameters.*Expected 1-2"):
            filter_and_classify_parameters(parametrizations)


class TestTupleListAnalysis:
    """Test dimension change detection across various parameter structures"""

    def test_single_changing_dimension_numeric(self):
        """Test tuple parameter with single changing numeric dimension"""
        parametrizations = {"input_shape": [(1, 32, 64, 64), (1, 64, 64, 64), (1, 128, 64, 64)]}

        result = filter_and_classify_parameters(parametrizations)
        assert len(result) == 1

        param = result[0]
        assert param.name == "input_shape"
        assert param.is_numeric == True
        assert param.changing_dimension == 1  # Second dimension changes
        assert param.display_name == "input_shape dim: 1"
        assert param.extracted_values == [32, 64, 128]
        assert param.is_categorical == False

    def test_multiple_changing_dimensions_categorical(self):
        """Test tuple parameter with multiple changing dimensions (categorical)"""
        parametrizations = {"shape_config": [(1, 256, 64, 64), (1, 128, 128, 128)]}  # Multiple dims change

        result = filter_and_classify_parameters(parametrizations)
        assert len(result) == 1

        param = result[0]
        assert param.name == "shape_config"
        assert param.is_numeric == False
        assert param.changing_dimension is None
        assert param.display_name == "shape_config"
        assert param.is_categorical == True
        # Should convert to string representations
        assert "(1, 256, 64, 64)" in param.extracted_values
        assert "(1, 128, 128, 128)" in param.extracted_values

    def test_no_changing_dimensions(self):
        """Test tuple parameter where all values are identical"""
        parametrizations = {
            "static_shape": [(1, 64, 64, 64), (1, 64, 64, 64), (1, 64, 64, 64)],
            "batch_size": [1, 2, 3],  # Add varying parameter to avoid error
        }

        result = filter_and_classify_parameters(parametrizations)
        # static_shape should be filtered out due to no variation, only batch_size remains
        assert len(result) == 1
        assert result[0].name == "batch_size"


class TestNumericVsCategoricalClassification:
    """Test classification logic with edge cases"""

    def test_simple_numeric_list(self):
        """Test simple numeric parameter list"""
        parametrizations = {"learning_rate": [0.01, 0.001, 0.0001]}

        result = filter_and_classify_parameters(parametrizations)
        assert len(result) == 1

        param = result[0]
        assert param.is_numeric == True
        assert param.is_categorical == False
        assert param.display_name == "learning_rate"
        assert param.extracted_values == [0.01, 0.001, 0.0001]

    def test_simple_string_list(self):
        """Test simple string categorical parameter"""
        parametrizations = {"optimizer": ["adam", "sgd", "rmsprop"]}

        result = filter_and_classify_parameters(parametrizations)
        assert len(result) == 1

        param = result[0]
        assert param.is_numeric == False
        assert param.is_categorical == True
        assert param.display_name == "optimizer"
        assert param.extracted_values == ["adam", "sgd", "rmsprop"]

    def test_mixed_type_list(self):
        """Test parameter list with mixed numeric/string types (should be categorical)"""
        parametrizations = {"mixed_param": [1, "auto", 3]}

        result = filter_and_classify_parameters(parametrizations)
        assert len(result) == 1

        param = result[0]
        assert param.is_numeric == False
        assert param.is_categorical == True
        assert param.extracted_values == ["1", "auto", "3"]


class TestVisualizationRoleAssignment:
    """Test assignment of X-axis and Legend roles"""

    def test_single_parameter_assignment(self):
        """Test role assignment with single varying parameter"""
        parametrizations = {"batch_size": [1, 2, 4, 8, 16], "device": ["cuda"]}  # Single value - filtered out

        param_info_list = filter_and_classify_parameters(parametrizations)
        x_axis, legend = assign_visualization_roles(param_info_list)

        assert x_axis.name == "batch_size"
        assert legend is None

    def test_two_numeric_parameters_more_values_wins(self):
        """Test that parameter with more values becomes X-axis when both numeric"""
        parametrizations = {"batch_size": [1, 2, 4, 8, 16], "channels": [32, 64]}  # 5 values  # 2 values

        param_info_list = filter_and_classify_parameters(parametrizations)
        x_axis, legend = assign_visualization_roles(param_info_list)

        assert x_axis.name == "batch_size"  # More values
        assert legend.name == "channels"  # Fewer values

    def test_mixed_numeric_categorical_numeric_wins_x_axis(self):
        """Test that numeric parameter becomes X-axis when mixed with categorical"""
        parametrizations = {
            "input_shape": [(1, 32, 64, 64), (1, 64, 64, 64), (1, 128, 64, 64)],  # Numeric (single changing dim)
            "memory_layout": ["HEIGHT_SHARDED", "INTERLEAVED"],  # Categorical
        }

        param_info_list = filter_and_classify_parameters(parametrizations)
        x_axis, legend = assign_visualization_roles(param_info_list)

        assert x_axis.name == "input_shape"  # Numeric parameter
        assert x_axis.is_numeric == True
        assert legend.name == "memory_layout"  # Categorical parameter
        assert legend.is_categorical == True

    def test_two_categorical_parameters_error(self):
        """Test error when both parameters are categorical"""
        parametrizations = {"optimizer": ["adam", "sgd"], "scheduler": ["cosine", "linear", "step"]}

        param_info_list = filter_and_classify_parameters(parametrizations)

        with pytest.raises(ValueError, match="At least one varying parameter must be numeric"):
            assign_visualization_roles(param_info_list)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_parameter_lists(self):
        """Test handling of empty parameter lists"""
        test_content = """
import pytest

def test_func_no_params():
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_content)
            f.flush()

            try:
                result = parse_test_script(f.name)
                assert len(result) == 0

                # This should raise an error when trying to classify
                with pytest.raises(ValueError, match="No varying parameters found"):
                    filter_and_classify_parameters(result)
            finally:
                os.unlink(f.name)

    def test_complex_nested_tuples(self):
        """Test complex nested tuple structures"""
        parametrizations = {
            "nested_config": [((1, 2), (3, 4)), ((1, 2), (5, 6))],
            "simple_param": [1, 2],  # Add simple varying parameter
        }

        result = filter_and_classify_parameters(parametrizations)

        # nested_config should be treated as categorical due to complexity
        nested_param = next(p for p in result if p.name == "nested_config")
        assert nested_param.is_categorical == True
        assert nested_param.changing_dimension is None

    def test_non_parametrize_decorators(self):
        """Test that non-parametrize decorators are ignored"""
        test_content = """
import pytest

@pytest.fixture
def some_fixture():
    return 42

@pytest.mark.slow
@pytest.mark.parametrize("value", [1, 2, 3])
def test_func(value, some_fixture):
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_content)
            f.flush()

            try:
                result = parse_test_script(f.name)
                # Should only find the parametrize decorator, not fixture or mark.slow
                assert len(result) == 1
                assert "value" in result
                assert result["value"] == [1, 2, 3]
            finally:
                os.unlink(f.name)

    def test_multiple_test_functions_same_parameters(self):
        """Test multiple test functions with same parameter names"""
        test_content = """
import pytest

@pytest.mark.parametrize("size", [1, 2])
def test_func1(size):
    pass

@pytest.mark.parametrize("size", [3, 4, 5])
def test_func2(size):
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_content)
            f.flush()

            try:
                result = parse_test_script(f.name)
                # Should find both parameter lists (last one wins in current implementation)
                assert "size" in result
                # Current implementation takes the last occurrence
                assert result["size"] == [3, 4, 5]
            finally:
                os.unlink(f.name)


class TestEndToEndWorkflow:
    """Test complete end-to-end parameter detection workflow"""

    def test_maxpool_simple_scenario(self):
        """Test with real maxpool scenario from the plan"""
        test_content = """
import pytest

@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("input_shape", [
    (1, 32, 64, 64),
    (1, 64, 64, 64),
    (1, 96, 64, 64),
    (1, 128, 64, 64),
    (1, 160, 64, 64),
    (1, 192, 64, 64),
    (1, 224, 64, 64),
    (1, 256, 64, 64),
    (1, 257, 64, 64)
])
@pytest.mark.parametrize("kernel_size", [(2, 2)])
def test_max_pool2d_height_sharded(input_shape, kernel_size, device_params):
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_content)
            f.flush()

            try:
                param_info_list, x_axis, legend, combinations = detect_and_analyze_parameters(f.name)

                # Should have only input_shape as varying (others filtered out)
                assert len(param_info_list) == 1
                assert x_axis.name == "input_shape"
                assert x_axis.is_numeric == True
                assert x_axis.changing_dimension == 1  # Channel dimension changes
                assert x_axis.display_name == "input_shape dim: 1"
                assert x_axis.extracted_values == [32, 64, 96, 128, 160, 192, 224, 256, 257]
                assert legend is None  # Single parameter scenario

                # Check parameter combinations
                assert len(combinations) == 9  # 9 input shapes

            finally:
                os.unlink(f.name)

    def test_two_parameter_scenario(self):
        """Test scenario with two varying parameters"""
        test_content = """
import pytest

@pytest.mark.parametrize("input_shape", [(1, 64, 64, 64), (1, 128, 64, 64)])
@pytest.mark.parametrize("kernel_size", [(2, 2), (2, 3), (2, 4)])
def test_func(input_shape, kernel_size):
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_content)
            f.flush()

            try:
                param_info_list, x_axis, legend, combinations = detect_and_analyze_parameters(f.name)

                assert len(param_info_list) == 2

                # kernel_size should be X-axis (3 values) vs input_shape (2 values)
                assert x_axis.name == "kernel_size"
                assert x_axis.display_name == "kernel_size dim: 1"  # Width dimension changes
                assert x_axis.extracted_values == [2, 3, 4]

                assert legend.name == "input_shape"
                assert legend.display_name == "input_shape dim: 1"  # Channel dimension changes
                assert legend.extracted_values == [64, 128]

                # Check parameter combinations
                assert len(combinations) == 6  # 2 * 3 combinations

            finally:
                os.unlink(f.name)


def run_phase1_tests():
    """Run all Phase 1 tests and report results"""
    print("=" * 80)
    print("PHASE 1 MANDATORY TESTING: Parameter Extraction and Classification")
    print("=" * 80)

    # Run pytest on this file
    exit_code = pytest.main([__file__, "-v"])

    if exit_code == 0:
        print("\n" + "=" * 80)
        print("✅ PHASE 1 TESTS PASSED - Parameter Detection is Ready")
        print("✅ ALL MANDATORY REQUIREMENTS MET")
        print("✅ PHASE 2 ADVANCEMENT APPROVED")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ PHASE 1 TESTS FAILED")
        print("❌ PHASE ADVANCEMENT BLOCKED")
        print("❌ FIX ALL ISSUES BEFORE PROCEEDING")
        print("=" * 80)

    return exit_code == 0


if __name__ == "__main__":
    success = run_phase1_tests()
    exit(0 if success else 1)
