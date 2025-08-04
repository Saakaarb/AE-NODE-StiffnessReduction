import numpy as np
import jax
import jax.numpy as jnp

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