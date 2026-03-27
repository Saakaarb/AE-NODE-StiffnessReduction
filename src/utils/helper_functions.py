import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import random
import mlflow
import equinox
from src.utils.classes import ConfigReader, VMapMLP,LoggingManager,get_activation_function


def create_network_instance(network_sizes: list, config_handler: ConfigReader, logging_manager: LoggingManager, model_string: str, constants: dict) -> equinox.Module:
    """
    Create a neural network instance based on configuration and model type.
    
    This function creates a VMapMLP network with the specified architecture. For neural ODE models,
    it automatically determines the output scaling factor based on time scale configuration.
    
    Args:
        network_sizes (list): List containing [input_size, hidden_size, output_size]
        config_handler (ConfigReader): Configuration handler for model parameters
        logging_manager (LoggingManager): Manager for logging operations
        model_string (str): Model type identifier (e.g., 'neural_ode', 'encoder_decoder')
        constants (dict): Dictionary containing training constants including 'end_time_scale'
        
    Returns:
        equinox.Module: A VMapMLP network instance with the specified architecture
        
    Raises:
        ValueError: If the network type is not supported
    """
    if config_handler.get_config_status(f'{model_string}.architecture.network_type')=='mlp':

        # the neural ODE needs output to be scaled by end time
        if model_string=='neural_ode':
            # if time scale is provided, use it to scale the output
            if config_handler.get_config_status('neural_ode.training.time_scale') is not None:
                output_scale=1.0/float(config_handler.get_config_status('neural_ode.training.time_scale'))
            else:
                # if time scale is not provided, it is automatically inferred from the data
                output_scale=1.0/constants['end_time_scale']
        else:
            output_scale=1.0

        input_size=network_sizes[0]
        output_size=network_sizes[-1]
        hidden_size=network_sizes[1]
        num_layers=config_handler.get_config_status(f'{model_string}.architecture.num_layers')
        key = jr.PRNGKey(5678)

        # get activation function callable
        if config_handler.path_exists(f'{model_string}.architecture.activation_function'):
            activation_name=config_handler.get_config_status(f'{model_string}.architecture.activation_function')
            activation_function=get_activation_function(activation_name)
        else:
            logging_manager.log(f"Activation function not specified in config file. Using default: relu")
            activation_name='relu'
            activation_function=jax.nn.relu
        print("Output scale: ",output_scale)
        return VMapMLP(in_size=input_size,out_size=output_size,width_size=hidden_size,depth=num_layers,key=key,activation_function=activation_function,activation_name=activation_name,output_scale=output_scale)

    else:
        raise ValueError(f"Network type {config_handler.get_config_status(f'{model_string}.architecture.network_type')} not supported")

def standard_score_norm(feature_data):
    """
    Compute standard score normalization parameters for input and output features.
    
    Calculates mean and standard deviation for each feature across all samples.
    Since the mapping space is the same for inputs and outputs, the normalization
    parameters are identical for both.
    
    Args:
        feature_data (np.ndarray): Feature data of shape [n_features, n_samples]
        
    Returns:
        tuple: (mean_vals_inp, std_vals_inp, mean_vals_out, std_vals_out) where:
            - mean_vals_inp (np.ndarray): Mean values for input features
            - std_vals_inp (np.ndarray): Standard deviation values for input features
            - mean_vals_out (np.ndarray): Mean values for output features (same as input)
            - std_vals_out (np.ndarray): Standard deviation values for output features (same as input)
    """
    # use true ODE function to get normalizations for both inputs and outputs
    
    #inputs:
    mean_vals_inp=np.zeros(feature_data.shape[0])
    std_vals_inp=np.zeros(feature_data.shape[0])

    for i in range(feature_data.shape[0]):

        mean_vals_inp[i]=np.mean(feature_data[i,:])
        std_vals_inp[i]=np.std(feature_data[i,:])

    #outputs: same as inputs since the mapping space is the same
    mean_vals_out=mean_vals_inp
    std_vals_out=std_vals_inp

    return mean_vals_inp,std_vals_inp,mean_vals_out,std_vals_out

def process_raw_data(data, config_handler):
    """
    Process raw data based on the specified arrangement mode.
    
    Routes data processing to the appropriate extraction function based on
    whether the data is arranged in row-major or column-major format.
    
    Args:
        data: Raw data to be processed
        config_handler (ConfigReader): Configuration handler containing data arrangement settings
        
    Returns:
        tuple: (time_data, feature_data) where:
            - time_data: Extracted time data
            - feature_data: Extracted feature data
    """
    if config_handler.get_config_status('data_processing.data_arrange_mode')=='row_major':

        time_data,feature_data=extract_row_major_data(data,config_handler)
    elif config_handler.get_config_status('data_processing.data_arrange_mode')=='column_major':

        time_data,feature_data=extract_column_major_data(data,config_handler)

    return time_data,feature_data


def extract_row_major_data(data, config_handler):
    """
    Extract time and feature data from row-major formatted data.
    
    In row-major format, each row represents a feature and each column represents a time step.
    The first row (index 0) is assumed to be the time data.
    
    Args:
        data: Data array in row-major format where shape is [n_features+1, n_time_steps]
        config_handler (ConfigReader): Configuration handler containing feature extraction settings
        
    Returns:
        tuple: (time_data, feature_data) where:
            - time_data: Time data from the first row
            - feature_data: Feature data from the specified feature rows
            
    Raises:
        ValueError: If data dimensions don't match expected feature count or if invalid feature indices are specified
    """
    # get indices to extract (user specified)

    # assert that user provided total  matches number of rows in data
    total_feats=config_handler.get_config_status('data_processing.total_available_features')

    if data.shape[0]!=total_feats+1:
        raise ValueError(f"Number of rows in data ({data.shape[0]}) - 1 does not match the total number of features ({total_feats})")

    # get indices to extract (user specified)
    feature_indices=config_handler.get_config_status('data_processing.feature_train_index')

    if isinstance(feature_indices,str):
        if feature_indices=='all':
            feature_indices=list(range(1,total_feats))
        else:
            raise ValueError(f"Invalid feature indices: {feature_indices}. Current options are 'all' or a list of indices.")

    elif isinstance(feature_indices,list):
        if 0 in feature_indices:
            raise ValueError(f"0 is not a valid training feature index. It MUST correspond to the time column of every data file.\
                 Check the config file for the feature_train_index.")

    else:
        raise ValueError(f"Invalid feature indices: {feature_indices}. Current options are 'all' or a list of indices.")

    time_data=data[0,:]
    feature_data=data[feature_indices,:] 
    
    return time_data,feature_data

def extract_column_major_data(data, config_handler):
    """
    Extract time and feature data from column-major formatted data.
    
    In column-major format, each column represents a feature and each row represents a time step.
    The first column (index 0) is assumed to be the time data.
    
    Args:
        data: Data array in column-major format where shape is [n_time_steps, n_features+1]
        config_handler (ConfigReader): Configuration handler containing feature extraction settings
        
    Returns:
        tuple: (time_data, feature_data) where:
            - time_data: Time data from the first column
            - feature_data: Feature data from the specified feature columns (transposed for consistency)
            
    Raises:
        ValueError: If data dimensions don't match expected feature count or if invalid feature indices are specified
    """
    # get indices to extract (user specified)

    # assert that user provided total  matches number of rows in data
    total_feats=config_handler.get_config_status('data_processing.total_available_features')

    if data.shape[1]!=total_feats+1:
        raise ValueError(f"Number of columns in data ({data.shape[1]}) -1 does not match the total number of features ({total_feats})")

    # get indices to extract (user specified)
    feature_indices=config_handler.get_config_status('data_processing.feature_train_index')

    if isinstance(feature_indices,str):
        if feature_indices=='all':
            feature_indices=list(range(1,total_feats))
        else:
            raise ValueError(f"Invalid feature indices: {feature_indices}. Current options are 'all' or a list of indices.")

    elif isinstance(feature_indices,list):
        if 0 in feature_indices:
            raise ValueError(f"0 is not a valid training feature index. It MUST correspond to the time column of every data file.\
                 Check the config file for the feature_train_index.")

    else:
        raise ValueError(f"Invalid feature indices: {feature_indices}. Current options are 'all' or a list of indices.")

    time_data=data[:,0]
    feature_data=data[:,feature_indices] 
    # transpose feature data to keep consistency
    return time_data,feature_data.T
    

def divide_range_random(start, end, group_size, seed=None):
    """
    Randomly divide a range [start, end) into groups of fixed size.
    The last group will have the remaining elements if not divisible.

    Args:
        start (int): Start of the range (inclusive).
        end (int): End of the range (exclusive).
        group_size (int): Size of each group.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        list[list[int]]: A list of randomly shuffled groups.
    """
    numbers = list(range(start, end))
    if seed is not None:
        random.seed(seed)
    random.shuffle(numbers)
    return [numbers[i:i+group_size] for i in range(0, len(numbers), group_size)]

def log_to_mlflow(config_status, config_filename):
        """
        Log all configuration parameters to MLflow.
        
        Flattens nested configuration dictionaries and logs each parameter individually,
        then logs the configuration file as an artifact.
        
        Args:
            config_status (dict): Configuration dictionary to log
            config_filename (str): Path to the configuration file to log as artifact
        """
        
        def flatten_dict(d, parent_key='', sep='.'):
            """
            Flatten nested dictionary with dot notation.
            
            Args:
                d (dict): Dictionary to flatten
                parent_key (str): Parent key for nested dictionaries
                sep (str): Separator for dot notation
                
            Returns:
                dict: Flattened dictionary with dot notation keys
            """
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        # Flatten the config and log each parameter
        flat_config = flatten_dict(config_status)
        for key, value in flat_config.items():
            mlflow.log_param(key, value)
        
        # Log the config file as an artifact
        mlflow.log_artifact(config_filename, "config")
def log_to_mlflow_metrics(metrics_dict, step):
    """
    Log metrics to MLflow with step information.
    
    Iterates through a dictionary of metrics and logs each one to MLflow
    with the specified step number.
    
    Args:
        metrics_dict (dict): Dictionary containing metric names and values
        step (int): Step number for the metrics
    """
    for key, value in metrics_dict.items():
        mlflow.log_metric(key, value, step=step)

def log_to_mlflow_artifacts(artifact_path, artifact_name):
    """
    Log artifacts to MLflow.
    
    Logs a file or directory as an artifact in MLflow with the specified name.
    
    Args:
        artifact_path (str): Path to the file or directory to log
        artifact_name (str): Name to assign to the artifact in MLflow
    """
    mlflow.log_artifact(artifact_path, artifact_name)
