import yaml
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any,Callable

class ConfigReader():
    def __init__(self,config_filename:str)->None:
        self.config_filename=config_filename
        self.config_status={}

        self.read_config()

    def read_config(self)->dict:
        with open(self.config_filename,'r') as f:
            self.config_status=yaml.safe_load(f)

    def get_config_status(self,key:str)->str:
        curr_layer = self.config_status
        nested_keys = key.split(".")
        for i in range(len(nested_keys) - 1):
            curr_layer = curr_layer[nested_keys[i]]

        if nested_keys[-1] not in curr_layer:
            raise ValueError(f"Key {key} not found in config file. Please check the config file.")

        return curr_layer[nested_keys[-1]]

    def set_config_status(self,key:str,value:str)->None:
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
    and properly handling the MLP as a field.
    """
    
    mlp: eqx.nn.MLP
    output_scale: float
    
    def __init__(self, in_size: int, width_size: int, out_size: int, depth: int, key: jax.random.PRNGKey,activation_function:Callable= jax.nn.tanh, output_scale:float=1.0):
        """
        Initialize the VMapMLP wrapper.
        batch_axis: axis to apply vmap on
        Args:
            in_size: Size of input features
            width_size: Size of hidden layers
            out_size: Size of output features
            depth: Number of hidden layers
            key: JAX random key for weight initialization
        """
        self.output_scale=output_scale
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            width_size=width_size,
            out_size=out_size,
            depth=depth,
            key=key,
            activation=activation_function
        )
    #@partial(jax.jit,static_argnums=(1,))
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
'''
#TODO probably needs to be created using equinox

class MLP():

    def __init__(self,network_sizes:list,config_handler:ConfigReader):

        self.config_handler=config_handler
        self.network_sizes=network_sizes

    def initialize_network(self):

        self.weights=self._init_xavier()

    """
    Xavier initialization technique for trainable weights: avoid
    vanishing/exploding gradients during backprop

    Arguments:
    size: List that contains the size information of the Variable 
    """
    
    def _init_xavier(self):

        weights_list=[]
        biases_list=[]
        for i_layer in range(len(self.network_sizes)):

            in_dim = self.network_sizes[i_layer][0]
            out_dim = self.network_sizes[i_layer][1]
            xavier_stddev = np.sqrt(2./(in_dim + out_dim))

            # create weight matrix for layer
            weight=jnp.array(np.random.normal(0,xavier_stddev,(in_dim,out_dim)))

            bias=jnp.array(np.zeros([1,out_dim]))

            weights_list.append(weight)
            biases_list.append(bias)
                                   
        return [weights_list,biases_list]

'''

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

