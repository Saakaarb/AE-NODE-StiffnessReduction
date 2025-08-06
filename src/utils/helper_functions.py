import numpy as np
import jax
import jax.numpy as jnp
import random
import mlflow

# for the system created by the script 2_eq_system.py

@jax.jit
def _forward_pass(network_input,network):

    interm_comp=network_input
    for i_layer in range(len(network[0])):
        
        if i_layer!=len(network[0])-1:
            interm_comp=jax.nn.tanh(interm_comp@network[0][i_layer]+network[1][i_layer])
        else:
            interm_comp=interm_comp@network[0][i_layer]+network[1][i_layer]

    return interm_comp

def standard_score_norm(time_data,specie_data,Temp_data,):

    # use true ODE function to get normalizations for both inputs and outputs
    
    #inputs:
    mean_vals_inp=np.zeros(specie_data.shape[0]+1)
    std_vals_inp=np.zeros(specie_data.shape[0]+1)

    for i in range(specie_data.shape[0]+1):

        if i==specie_data.shape[0]:

            mean_vals_inp[i]=np.mean(Temp_data)
            std_vals_inp[i]=np.std(Temp_data)
        else:    
            mean_vals_inp[i]=np.mean(specie_data[i,:])
            if np.std(specie_data[i,:])==0.0:
                std_vals_inp[i]=1.0
            else:
                std_vals_inp[i]=np.std(specie_data[i,:])
    #outputs
    
    mean_vals_out=np.zeros(specie_data.shape[0])
    std_vals_out=np.zeros(specie_data.shape[0])

    return mean_vals_inp,std_vals_inp,mean_vals_out,std_vals_out

def preprocess_data(data):

    time_data=data[0,:]
    specie_data=data[1:-2,:] # skip N2
    Temp_data=data[-1,:]

    Temp_data=np.expand_dims(Temp_data,axis=0)
    
    network_input=np.concatenate([specie_data,Temp_data],axis=0)

    return time_data,specie_data,Temp_data,network_input


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