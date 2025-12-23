"""
Universal Parameter Detection Module for TTNN Operations

This module automatically detects and classifies pytest parameters from any test file,
completely operation-agnostic. It works by parsing the AST and analyzing parameter
value structures to determine visualization roles.
"""

import ast
import itertools
from typing import Dict, List, Tuple, Any, Optional, Union

# Global variable to track decorator order for pytest combination matching
_global_decorator_order = []


class ParameterInfo:
    """Container for parameter analysis results"""

    def __init__(
        self,
        name: str,
        values: List[Any],
        is_numeric: bool,
        display_name: str,
        extracted_values: List[Union[int, float, str]],
        changing_dimension: Optional[int] = None,
        is_categorical: bool = False,
    ):
        self.name = name
        self.values = values
        self.unique_values = list(set(values))
        self.count = len(self.unique_values)
        self.is_numeric = is_numeric
        self.is_categorical = is_categorical
        self.changing_dimension = changing_dimension
        self.display_name = display_name
        self.extracted_values = extracted_values


def parse_test_script(test_file_path: str) -> Dict[str, List[Any]]:
    """
    Extract parametrizations directly from the test script using AST parsing.
    Preserves decorator order to match pytest execution order.

    Args:
        test_file_path: Path to the pytest file

    Returns:
        Dictionary mapping parameter names to their value lists
    """
    with open(test_file_path, "r") as f:
        tree = ast.parse(f.read())

    parametrizations = {}
    decorator_order = []  # Track order of parameters as they appear in decorators

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Look for test functions
            if node.name.startswith("test_"):
                # Extract @pytest.mark.parametrize decorators in order
                # Note: decorators are stored in the order they appear in code
                for decorator in node.decorator_list:
                    if _is_parametrize_decorator(decorator):
                        param_name, param_values = _extract_parametrize_info(decorator)
                        if param_name not in parametrizations:  # Only add first occurrence
                            parametrizations[param_name] = param_values
                            decorator_order.append(param_name)

    # Store the decorator order globally for combination generation
    global _global_decorator_order
    _global_decorator_order = decorator_order

    return parametrizations


def _is_parametrize_decorator(decorator) -> bool:
    """Check if a decorator is @pytest.mark.parametrize"""
    if isinstance(decorator, ast.Call):
        if isinstance(decorator.func, ast.Attribute):
            if (
                isinstance(decorator.func.value, ast.Attribute)
                and isinstance(decorator.func.value.value, ast.Name)
                and decorator.func.value.value.id == "pytest"
                and decorator.func.value.attr == "mark"
                and decorator.func.attr == "parametrize"
            ):
                return True
    return False


def _extract_parametrize_info(decorator) -> Tuple[str, List[Any]]:
    """
    Extract parameter name and values from @pytest.mark.parametrize decorator.

    Args:
        decorator: AST node representing the decorator

    Returns:
        Tuple of (parameter_name, parameter_values_list)
    """
    # Handle: @pytest.mark.parametrize("param_name", [values])
    if len(decorator.args) >= 2:
        # Parameter name (first argument)
        if isinstance(decorator.args[0], ast.Constant):
            param_name = decorator.args[0].value
        elif isinstance(decorator.args[0], ast.Str):  # Python < 3.8 compatibility
            param_name = decorator.args[0].s
        else:
            raise ValueError(f"Unexpected parameter name format in decorator: {decorator.args[0]}")

        # Parameter values (second argument)
        try:
            # First try literal_eval for simple cases
            param_values = ast.literal_eval(decorator.args[1])
        except (ValueError, TypeError):
            # If literal_eval fails, try to evaluate complex expressions
            try:
                param_values = _evaluate_complex_parameter_values(decorator.args[1])
            except Exception as e:
                # If all else fails, extract string representations
                param_values = _extract_parameter_strings(decorator.args[1])
                print(f"Warning: Using string representations for {param_name} parameter values: {e}")

        return param_name, param_values
    else:
        raise ValueError(f"Invalid parametrize decorator format: insufficient arguments")


def _evaluate_complex_parameter_values(ast_node) -> List[Any]:
    """
    Evaluate complex parameter expressions that contain module imports.

    Args:
        ast_node: AST node representing parameter values

    Returns:
        List of evaluated parameter values

    Raises:
        Exception: If evaluation fails
    """
    # Convert AST back to source code string using ast.unparse (Python 3.9+) or custom method
    try:
        # Try ast.unparse first (Python 3.9+)
        import ast

        if hasattr(ast, "unparse"):
            source_code = ast.unparse(ast_node)
        else:
            # Fallback for older Python versions
            source_code = _ast_to_source(ast_node)
    except Exception:
        # Final fallback
        source_code = _ast_to_source(ast_node)

    # Create a safe evaluation context with common imports
    eval_globals = {
        "__builtins__": {"__import__": __import__},  # Allow imports
    }

    # Try to import ttnn for evaluation context
    try:
        import ttnn

        eval_globals["ttnn"] = ttnn
    except ImportError:
        pass

    try:
        # Try to import other common modules that might be needed
        import torch

        eval_globals["torch"] = torch
    except ImportError:
        pass

    # Evaluate the expression
    result = eval(source_code, eval_globals)
    return result


def _ast_to_source(node) -> str:
    """
    Convert AST node back to source code using custom logic.

    Args:
        node: AST node

    Returns:
        Source code string
    """
    if isinstance(node, ast.List):
        elements = []
        for element in node.elts:
            elements.append(_ast_to_source(element))
        return "[" + ", ".join(elements) + "]"
    elif isinstance(node, ast.Attribute):
        # Handle things like ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        if isinstance(node.value, ast.Attribute):
            return _ast_to_source(node.value) + "." + node.attr
        elif isinstance(node.value, ast.Name):
            return node.value.id + "." + node.attr
        else:
            return str(node.attr)
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, (ast.Tuple, ast.List)):
        opener, closer = ("(", ")") if isinstance(node, ast.Tuple) else ("[", "]")
        elements = [_ast_to_source(el) for el in node.elts]
        return opener + ", ".join(elements) + closer
    else:
        # Fallback to string representation
        return str(node)


def _extract_parameter_strings(ast_node) -> List[str]:
    """
    Extract string representations of parameter values when evaluation fails.

    Args:
        ast_node: AST node representing parameter values

    Returns:
        List of string representations
    """
    if isinstance(ast_node, ast.List):
        # Handle list of parameters
        string_values = []
        for element in ast_node.elts:
            element_source = _ast_to_source(element)
            string_values.append(element_source)
        return string_values
    else:
        # Single parameter value
        source_code = _ast_to_source(ast_node)
        return [source_code]


def filter_and_classify_parameters(parametrizations: Dict[str, List[Any]]) -> Tuple[List[ParameterInfo], Dict[str, Any]]:
    """
    Filter out single-value parameters and classify varying ones.

    Args:
        parametrizations: Dictionary of parameter name -> value lists

    Returns:
        Tuple of:
        - List of ParameterInfo objects for varying parameters
        - Dictionary of static (non-varying) parameters

    Raises:
        ValueError: If no varying parameters found or too many varying parameters
    """
    # Step 1: Filter out single-value parametrizations AND parameters with no actual variation
    varying_params = {}
    static_params = {}
    
    for param_name, param_values in parametrizations.items():
        if len(param_values) > 1:
            # Check if values are actually different (not just multiple identical values)
            unique_values = list(set(param_values))
            if len(unique_values) > 1:
                varying_params[param_name] = param_values
            else:
                print(f"Ignoring parameter with identical values: {param_name} = {param_values[0]}")
                static_params[param_name] = param_values[0]
        else:
            print(f"Ignoring single-value parameter: {param_name} = {param_values[0]}")
            static_params[param_name] = param_values[0]

    # Step 2: Validate parameter count
    if len(varying_params) == 0:
        raise ValueError("No varying parameters found - all parameters have single values")
    elif len(varying_params) > 2:
        raise ValueError(
            f"Too many varying parameters ({len(varying_params)}). Expected 1-2, got: {list(varying_params.keys())}"
        )

    # Step 3: Classify each varying parameter
    classified_params = []
    for param_name, values in varying_params.items():
        param_info = _analyze_parameter(param_name, values)
        classified_params.append(param_info)

    return classified_params, static_params


def _analyze_parameter(param_name: str, values: List[Any]) -> ParameterInfo:
    """
    Analyze a single parameter to determine its type and characteristics.

    Args:
        param_name: Name of the parameter
        values: List of parameter values

    Returns:
        ParameterInfo object with analysis results
    """
    # Handle tuple/list parameters (e.g., input_shape, kernel_size)
    if isinstance(values[0], (tuple, list)):
        analysis = _analyze_parameter_structure(values)
        
        # Create a better display name by extracting the specific parameter name
        if analysis["changing_dim"] is not None and "," in param_name:
            # Multi-parameter parametrize - extract the specific parameter name
            param_names = [p.strip() for p in param_name.split(",")]
            if analysis["changing_dim"] < len(param_names):
                display_name = param_names[analysis["changing_dim"]]
            else:
                display_name = f"{param_name} dim: {analysis['changing_dim']}"
        else:
            display_name = f"{param_name} dim: {analysis['changing_dim']}" if analysis["changing_dim"] is not None else param_name
        
        return ParameterInfo(
            name=param_name,
            values=values,
            is_numeric=analysis["is_numeric"],
            display_name=display_name,
            extracted_values=analysis["extracted_values"],
            changing_dimension=analysis["changing_dim"],
            is_categorical=analysis["is_categorical"],
        )
    else:
        # Simple parameter analysis (numeric lists, strings, etc.)
        is_numeric = _detect_simple_numeric(values)
        return ParameterInfo(
            name=param_name,
            values=values,
            is_numeric=is_numeric,
            display_name=param_name,
            extracted_values=values if is_numeric else [str(v) for v in values],
            is_categorical=not is_numeric,
        )


def _analyze_parameter_structure(param_values: List[Union[tuple, list]]) -> Dict[str, Any]:
    """
    Analyze tuple/list parameters to find changing dimensions.
    
    Intelligently handles nested structures by analyzing each dimension independently.
    Constant nested dimensions (like padding=(1,1,1,1)) are ignored.

    Args:
        param_values: List of tuples/lists to analyze

    Returns:
        Dictionary with analysis results including:
        - is_numeric: Whether the parameter has numeric changing dimensions
        - changing_dim: Index of the single changing dimension (if any)
        - extracted_values: List of values from the changing dimension
        - is_categorical: Whether parameter should be treated as categorical
    """
    if not param_values or not isinstance(param_values[0], (tuple, list)):
        return {"is_numeric": False, "changing_dim": None, "extracted_values": param_values, "is_categorical": True}

    # Convert all to lists for uniform handling
    param_arrays = [list(p) for p in param_values]
    changing_dims = []
    
    # Analyze each dimension to find which ones change
    for dim_idx in range(len(param_arrays[0])):
        dim_values = [p[dim_idx] for p in param_arrays]
        
        # Check if this dimension varies
        # For nested structures, we need to compare them properly
        try:
            unique_values = set(tuple(v) if isinstance(v, list) else v if not isinstance(v, (tuple, list)) or isinstance(v, tuple) else tuple(v) for v in dim_values)
        except TypeError:
            unique_values = {str(v) for v in dim_values}
        
        if len(unique_values) > 1:
            # This dimension varies - but only count it if it's not a nested structure
            # or if the nested structure itself is changing
            first_val = dim_values[0]
            
            if isinstance(first_val, (tuple, list)):
                # This is a nested dimension that's changing - treat as categorical
                changing_dims.append((dim_idx, 'nested'))
            else:
                # Simple changing dimension
                changing_dims.append((dim_idx, 'simple'))

    # Filter to only simple (non-nested) changing dimensions
    simple_changing_dims = [(idx, type_) for idx, type_ in changing_dims if type_ == 'simple']
    
    if len(simple_changing_dims) == 0:
        # Check if we have any changing dimensions at all
        if len(changing_dims) > 0:
            # Only nested dimensions are changing - treat as categorical
            return {
                "is_numeric": False,
                "changing_dim": None,
                "extracted_values": [str(p) for p in param_values],
                "is_categorical": True,
            }
        else:
            # No changing dimensions at all
            return {
                "is_numeric": False,
                "changing_dim": None,
                "extracted_values": [str(p) for p in param_values],
                "is_categorical": True,
            }
    elif len(simple_changing_dims) == 1:
        # Single simple changing dimension - extract those values
        dim_idx, _ = simple_changing_dims[0]
        extracted_values = [p[dim_idx] for p in param_arrays]

        # Check if extracted values are numeric
        is_numeric = all(isinstance(v, (int, float)) for v in extracted_values)

        return {
            "is_numeric": is_numeric,
            "changing_dim": dim_idx,
            "extracted_values": extracted_values,
            "is_categorical": not is_numeric,
        }
    else:
        # Multiple simple changing dimensions - treat as categorical
        return {
            "is_numeric": False,
            "changing_dim": None,
            "extracted_values": [str(p) for p in param_values],
            "is_categorical": True,
        }


def _detect_simple_numeric(values: List[Any]) -> bool:
    """
    Detect if a list of simple values (not tuples/lists) are all numeric.

    Args:
        values: List of values to check

    Returns:
        True if all values are numeric (int or float)
    """
    return all(isinstance(v, (int, float)) for v in values)


def assign_visualization_roles(param_info_list: List[ParameterInfo]) -> Tuple[ParameterInfo, Optional[ParameterInfo]]:
    """
    Assign X-axis and Legend roles based on parameter characteristics.

    Args:
        param_info_list: List of ParameterInfo objects

    Returns:
        Tuple of (x_axis_param, legend_param). legend_param is None for single parameter.

    Raises:
        ValueError: If no numeric parameters or invalid configuration
    """
    varying_params = [p for p in param_info_list if p.count > 1]

    # Validate that at least one parameter is numeric
    numeric_params = [p for p in varying_params if p.is_numeric]
    categorical_params = [p for p in varying_params if p.is_categorical]

    if not numeric_params:
        raise ValueError("At least one varying parameter must be numeric")

    if len(varying_params) == 1:
        # Single varying parameter (must be numeric due to validation above)
        return varying_params[0], None

    elif len(varying_params) == 2:
        p1, p2 = varying_params

        # Rule 1: Both numeric - X-axis gets parameter with MORE values
        if p1.is_numeric and p2.is_numeric:
            if p1.count >= p2.count:
                return p1, p2  # x_axis, legend
            else:
                return p2, p1

        # Rule 2: Mixed types - Numeric parameter becomes X-axis
        elif p1.is_numeric and not p2.is_numeric:
            return p1, p2
        elif p2.is_numeric and not p1.is_numeric:
            return p2, p1
        else:
            # Both categorical - should have been caught by numeric validation
            raise ValueError("At least one parameter must be numeric")

    else:
        raise ValueError(f"Expected 1-2 varying parameters, found {len(varying_params)}")


def generate_parameter_combinations(parametrizations: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations from parametrizations in pytest execution order.

    Pytest executes combinations by applying decorators in reverse order:
    - The last (bottom) decorator becomes the outer loop
    - The first (top) decorator becomes the inner loop

    Args:
        parametrizations: Dictionary of parameter name -> value lists

    Returns:
        List of dictionaries, each representing one parameter combination in pytest order
    """
    global _global_decorator_order

    if not _global_decorator_order:
        # Fallback to original behavior if order not detected
        param_names = list(parametrizations.keys())
    else:
        # Use detected decorator order, but reverse it for pytest behavior
        # Pytest applies decorators as a stack - last decorator is outermost loop
        param_names = list(reversed(_global_decorator_order))

    param_values = [parametrizations[name] for name in param_names]
    all_combinations = list(itertools.product(*param_values))

    return [dict(zip(param_names, combination)) for combination in all_combinations]


# Main entry point for parameter detection
def detect_and_analyze_parameters(
    test_file_path: str,
) -> Tuple[List[ParameterInfo], ParameterInfo, Optional[ParameterInfo], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Complete parameter detection and analysis pipeline.

    Args:
        test_file_path: Path to the pytest file

    Returns:
        Tuple of:
        - List of all varying parameter info objects
        - X-axis parameter info
        - Legend parameter info (None if single parameter)
        - List of all parameter combinations
        - Dictionary of static (non-varying) parameters
    """
    # Step 1: Parse test script
    parametrizations = parse_test_script(test_file_path)

    # Step 2: Filter and classify parameters
    param_info_list, static_params = filter_and_classify_parameters(parametrizations)

    # Step 3: Assign visualization roles
    x_axis_param, legend_param = assign_visualization_roles(param_info_list)

    # Step 4: Generate all parameter combinations for data alignment
    param_combinations = generate_parameter_combinations(parametrizations)

    return param_info_list, x_axis_param, legend_param, param_combinations, static_params
