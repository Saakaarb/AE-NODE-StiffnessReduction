import yaml
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any,Callable
import json
from pathlib import Path

def get_activation_function(activation_function: str):
    """
    Get the JAX activation function corresponding to the given name.
    
    Args:
        activation_function (str): Name of the activation function ('relu', 'gelu', or 'tanh')
        
    Returns:
        Callable: JAX activation function
        
    Raises:
        ValueError: If the activation function name is not supported
    """
    if activation_function=='relu':
        return jax.nn.relu
    elif activation_function=='gelu':
        return jax.nn.gelu
    elif activation_function=='tanh':
        return jax.nn.tanh
    else:
        raise ValueError(f"Activation function {activation_function} not supported")


class ConfigReader():
    """
    Configuration reader class for parsing and accessing YAML configuration files.
    
    This class provides methods to read, access, and modify configuration parameters
    stored in YAML format. It supports nested configuration access using dot notation.
    """
    
    def __init__(self, config_filename: str) -> None:
        """
        Initialize the ConfigReader with a configuration file.
        
        Args:
            config_filename (str): Path to the YAML configuration file
            
        Returns:
            None: Initializes the ConfigReader object
        """
        self.config_filename = config_filename
        self.config_status = {}

        self.read_config()

    def read_config(self) -> dict:
        """
        Read and parse the YAML configuration file.
        
        Loads the configuration file and stores the parsed contents in
        self.config_status for later access.
        
        Returns:
            dict: The parsed configuration dictionary
        """
        with open(self.config_filename,'r') as f:
            self.config_status=yaml.safe_load(f)

    def get_config_status(self, key: str) -> str:
        """
        Get a configuration value using dot notation for nested keys.
        
        Args:
            key (str): Configuration key using dot notation (e.g., "encoder_decoder.architecture.network_width")
            
        Returns:
            str: The configuration value
            
        Raises:
            ValueError: If the key is not found in the configuration
        """
        curr_layer = self.config_status
        nested_keys = key.split(".")
        for i in range(len(nested_keys) - 1):
            curr_layer = curr_layer[nested_keys[i]]

        if nested_keys[-1] not in curr_layer:
            raise ValueError(f"Key {key} not found in config file. Please check the config file.")

        return curr_layer[nested_keys[-1]]

    def set_config_status(self, key: str, value: str) -> None:
        """
        Set a configuration value using dot notation for nested keys.
        
        Args:
            key (str): Configuration key using dot notation (e.g., "encoder_decoder.architecture.network_width")
            value (str): Value to set for the configuration key
            
        Returns:
            None: Modifies the configuration in-place
        """
        curr_layer = self.config_status
        nested_keys = key.split(".")
        for i in range(len(nested_keys) - 1):
            curr_layer = curr_layer[nested_keys[i]]
        curr_layer[nested_keys[-1]] = value

    def path_exists(self, key: str) -> bool:
        """
        Check if a particular path exists in the configuration.
        
        Args:
            key (str): The configuration path to check, using dot notation (e.g., "encoder_decoder.architecture.network_width")
            
        Returns:
            bool: True if the path exists, False otherwise
        """
        try:
            curr_layer = self.config_status
            nested_keys = key.split(".")
            
            for nested_key in nested_keys:
                if nested_key not in curr_layer:
                    return False
                curr_layer = curr_layer[nested_key]
            
            return True
        except (KeyError, TypeError, AttributeError):
            return False

    def get_config_status_safe(self, key: str, default: Any = None) -> Any:
        """
        Safely get a configuration value with a default fallback.
        
        Args:
            key (str): The configuration path to retrieve
            default (Any): Default value to return if the path doesn't exist
            
        Returns:
            Any: The configuration value if it exists, otherwise the default value
        """
        if self.path_exists(key):
            return self.get_config_status(key)
        return default


class VMapMLP(eqx.Module):
    """
    A wrapper around eqx.nn.MLP that applies vmap on the batch dimension.
    
    This class ensures the model is a valid JAX pytree by inheriting from eqx.Module
    and properly handling the MLP and all its parameters as a field.
    """
    
    mlp: eqx.nn.MLP
    output_scale: float
    in_size: int
    out_size: int
    width_size: int
    depth: int
    activation_name: str
    model_name: str

    def __init__(self, in_size: int, width_size: int, out_size: int, depth: int, key: jax.random.PRNGKey,activation_function:Callable= jax.nn.tanh, activation_name:str='tanh', output_scale:float=1.0):
        """
        Initialize the VMapMLP wrapper.
        
        Args:
            in_size (int): Size of input features
            width_size (int): Size of hidden layers
            out_size (int): Size of output features
            depth (int): Number of hidden layers
            key (jax.random.PRNGKey): JAX random key for weight initialization
            activation_function (Callable, optional): Activation function to use. Defaults to jax.nn.tanh.
            activation_name (str, optional): Name of the activation function. Defaults to 'tanh'.
            output_scale (float, optional): Scaling factor for the output. Defaults to 1.0.
        """
        self.output_scale=output_scale
        self.in_size=in_size
        self.out_size=out_size
        self.width_size=width_size
        self.depth=depth
        self.activation_name=activation_name
        self.model_name="mlp"

        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            width_size=width_size,
            out_size=out_size,
            depth=depth,
            key=key,
            activation=activation_function
        )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass with vmap applied to batch dimension.
        
        Args:
            x: Input tensor of shape [batch, time, features] assumed default
            
        Returns:
            Output tensor with same batch and time dimensions
        """
        
        f_time  = jax.vmap(self.mlp, in_axes=0, out_axes=0)
        f_batch = jax.vmap(f_time, in_axes=0, out_axes=0)

        result=f_batch(x)*self.output_scale
        
        return result


    def __repr__(self):
        return f"VMapMLP(mlp={self.mlp})"


class LoggingManager:
    """
    Simple logging manager that creates a log file and logs statements.
    """
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """
        Initialize the logging manager.
        
        Args:
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper())
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(self.log_dir, f"neural_ode_{timestamp}.log")
        self.log_filename=log_filename
        # Set up logging
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()  # Also print to console
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def log(self, message: str, level: str = "INFO") -> None:
        """
        Log a message with specified level.
        
        Args:
            message: Message to log
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_func = getattr(self.logger, level.lower())
        log_func(message)

class ModelSaver():
    """
    Model saving and loading utility class.
    
    This class provides functionality to save and load Equinox models along with
    their hyperparameters. It supports both model serialization and reconstruction.
    """

    def __init__(self, config_handler: ConfigReader, logging_manager: LoggingManager):
        """
        Initialize the ModelSaver.
        
        Args:
            config_handler (ConfigReader): Configuration handler for model parameters
            logging_manager (LoggingManager): Manager for logging operations
        """
        self.config_handler = config_handler
        self.logging_manager = logging_manager

    def save_model(self, model: eqx.Module, filename: str | Path):
        """
        Save a model along with its hyperparameters.
        
        Extracts hyperparameters from the model and saves both the model weights
        and hyperparameters to the specified file.
        
        Args:
            model (eqx.Module): The Equinox model to save
            filename (str | Path): Path where the model should be saved
        """
        self.model = model
        # extract hyperparams
        hyperparams = {}
        hyperparams["output_scale"] = float(self.model.output_scale)
        hyperparams["in_size"] = int(self.model.in_size)
        hyperparams["out_size"] = int(self.model.out_size)
        hyperparams["width_size"] = int(self.model.width_size)
        hyperparams["depth"] = int(self.model.depth)
        hyperparams["activation_name"] = str(self.model.activation_name)
        hyperparams["model_name"] = str(self.model.model_name)
        # store weight dtype so load_model can reconstruct the skeleton with matching dtype
        leaves = jax.tree_util.tree_leaves(eqx.filter(self.model, eqx.is_inexact_array))
        hyperparams["dtype"] = str(leaves[0].dtype) if leaves else "float32"
        self.write_model(filename, hyperparams, self.model)

    def write_model(self, filename: str | Path, hyperparams: dict, model: eqx.Module):
        """
        Write model and hyperparameters to a file.
        
        Saves the hyperparameters as JSON on the first line, followed by the
        serialized model weights using Equinox's serialization.
        
        Args:
            filename (str | Path): Path where the model should be saved
            hyperparams (dict): Dictionary containing model hyperparameters
            model (eqx.Module): The Equinox model to serialize
        """
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, model)

    def make_model(self, hyperparams: dict) -> eqx.Module:
        """
        Create a model instance from hyperparameters.
        
        Reconstructs a model with the specified architecture and parameters.
        Currently supports MLP models only.
        
        Args:
            hyperparams (dict): Dictionary containing model hyperparameters including
                              in_size, out_size, width_size, depth, activation_name,
                              output_scale, and model_name
                              
        Returns:
            eqx.Module: A new model instance with the specified parameters
            
        Raises:
            ValueError: If the model name is not supported
        """
        in_size = hyperparams["in_size"]
        out_size = hyperparams["out_size"]
        width_size = hyperparams["width_size"]
        depth = hyperparams["depth"]
        activation_name = hyperparams["activation_name"]
        output_scale = hyperparams["output_scale"]
        activation_function = get_activation_function(activation_name)
        model_name = hyperparams["model_name"]

        if model_name == "mlp":
            model = VMapMLP(in_size=in_size, out_size=out_size, width_size=width_size, depth=depth, key=jax.random.PRNGKey(0), activation_function=activation_function, activation_name=activation_name, output_scale=output_scale)
            # cast skeleton to saved dtype so tree_deserialise_leaves succeeds regardless of global x64 state
            dtype = hyperparams.get("dtype", "float32")
            model = jax.tree_util.tree_map(lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, model)
            return model
        else:
            raise ValueError(f"Model name {model_name} not supported")
   
    def load_model(self, filename: str | Path) -> eqx.Module:
        """
        Load a model from a saved file.
        
        Reads the hyperparameters from the first line of the file and reconstructs
        the model, then loads the saved weights.
        
        Args:
            filename (str | Path): Path to the saved model file
            
        Returns:
            eqx.Module: The loaded model with restored weights
        """
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = self.make_model(hyperparams)
            return eqx.tree_deserialise_leaves(f, model)
