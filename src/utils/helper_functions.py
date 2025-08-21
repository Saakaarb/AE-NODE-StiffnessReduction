import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import random
import mlflow
import equinox
from src.utils.classes import ConfigReader, VMapMLP

# for the system created by the script 2_eq_system.py

# @jax.jit
# def _forward_pass(network_input,network):

#     interm_comp=network_input
#     for i_layer in range(len(network[0])):
        
#         if i_layer!=len(network[0])-1:
#             interm_comp=jax.nn.tanh(interm_comp@network[0][i_layer]+network[1][i_layer])
#         else:
#             interm_comp=interm_comp@network[0][i_layer]+network[1][i_layer]

#     return interm_comp

def create_network_instance(network_sizes:list,config_handler:ConfigReader)->equinox.Module:

    if config_handler.get_config_status('encoder_decoder.architecture.network_type')=='mlp':

        input_size=network_sizes[0]
        output_size=network_sizes[-1]
        hidden_size=network_sizes[1]
        num_layers=config_handler.get_config_status('encoder_decoder.architecture.num_layers')
        key = jr.PRNGKey(5678)
        return VMapMLP(in_size=input_size,out_size=output_size,width_size=hidden_size,depth=num_layers,key=key)

    else:
        raise ValueError(f"Network type {config_handler.get_config_status('encoder_decoder.architecture.network_type')} not supported")

def standard_score_norm(feature_data):

    # use true ODE function to get normalizations for both inputs and outputs
    
    #inputs:
    mean_vals_inp=np.zeros(feature_data.shape[0])
    std_vals_inp=np.zeros(feature_data.shape[0])

    for i in range(feature_data.shape[0]):

        mean_vals_inp[i]=np.mean(feature_data[i,:])
        std_vals_inp[i]=np.std(feature_data[i,:])

    #outputs
    # these are same as inp since the mapping space is the same
    #mean_vals_out=np.zeros(feature_data.shape[0])
    #std_vals_out=np.zeros(feature_data.shape[0])
    mean_vals_out=mean_vals_inp
    std_vals_out=std_vals_inp

    return mean_vals_inp,std_vals_inp,mean_vals_out,std_vals_out

def process_raw_data(data,config_handler):

    if config_handler.get_config_status('data_processing.data_arrange_mode')=='row_major':

        time_data,feature_data=extract_row_major_data(data,config_handler)
    elif config_handler.get_config_status('data_processing.data_arrange_mode')=='column_major':

        time_data,feature_data=extract_column_major_data(data,config_handler)

    return time_data,feature_data


def extract_row_major_data(data,config_handler):

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

def extract_column_major_data(data,config_handler):

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

def log_to_mlflow(config_status,config_filename):
        """Log all configuration parameters to MLflow"""
        
        def flatten_dict(d, parent_key='', sep='.'):
            """Flatten nested dictionary with dot notation"""
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
def log_to_mlflow_metrics(metrics_dict,step):
    for key, value in metrics_dict.items():
        mlflow.log_metric(key, value, step=step)

def log_to_mlflow_artifacts(artifact_path,artifact_name):
    mlflow.log_artifact(artifact_path,artifact_name)
