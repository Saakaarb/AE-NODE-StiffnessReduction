import os
import pickle
import jax.numpy as jnp
import jax
import optax
import numpy as np
from src.utils.classes import ConfigReader,MLP
from src.utils.helper_functions import _forward_pass,log_to_mlflow_metrics,log_to_mlflow_artifacts
from src.lib.data_processing.classes import Data_Processing
from pathlib import Path
import time
import shutil
import matplotlib.pyplot as plt

# jit functions need to sit outside of classes

@jax.jit
def _loss_fn_autoencoder(constants,networks,data_dict):

    lam=constants['stiffness_reduction_weight'] # weight for stiffness reduction loss
    eps=1E-12 # prevents division by zero
    eps_dt=1E-30 # prevents division by zero

    # data in shape [num_traj_samples,max_steps,num_inputs]
    input_data=data_dict['input_data']
    time_data=data_dict['all_time_data_broadcasted']
    recon_mask=data_dict['recon_mask']
    
    # masks for condition number regularization
    cond_1_mask=data_dict['cond_1_mask']
    cond_2_mask=data_dict['cond_2_mask']
    

    # forward pass data through encoder
    latent_space_preds=_forward_pass(input_data,networks['encoder'])

    # forward pass latent rep through decoder
    predicted_specie=_forward_pass(latent_space_preds,networks['decoder'])
    recon_loss = jnp.sqrt(jnp.mean(jnp.square(jnp.multiply(predicted_specie,recon_mask)-input_data)))

    # TODO make this optional

    # condition number regularization
    # this section takes an approximation of the condition number as described in:
    # Stiffness-Reduced Neural ODE Models for Data-Driven Reduced-Order Modeling of Combustion Chemical Kinetics: Dikeman, Zhang and Yang (2022)
    # and introduces a multi-objective optimization term

    dt = jnp.diff(time_data, axis=1)  # Shape: [batch, time-1, features]
    dlatent = jnp.diff(latent_space_preds, axis=1)  # Shape: [batch, time-1, features]

    # Compute derivatives
    cond_1 = jnp.divide(dlatent[:,1:,:], dt[:,1:,:] + eps_dt) * cond_1_mask
    cond_2 = jnp.divide(dlatent[:,:-1,:], dt[:,:-1,:] + eps_dt) * cond_2_mask

    #cond_1=jnp.multiply(jnp.true_divide(latent_space_preds.at[:,2:,:].get()-latent_space_preds.at[:,1:-1,:].get(),time_data[:,2:,:]-time_data[:,1:-1,:]+eps_dt),cond_1_mask)
    
    #cond_2=jnp.multiply(jnp.true_divide(latent_space_preds.at[:,1:-1,:].get()-latent_space_preds.at[:,:-2,:].get(),time_data[:,1:-1,:]-time_data[:,:-2,:]+eps_dt),cond_2_mask)
    
    cond_numer= jnp.sqrt(jnp.mean(jnp.square(cond_1-cond_2)+eps,axis=1))
    #cond_3= jnp.sqrt(jnp.mean(jnp.square(jnp.multiply(latent_space_preds.at[:,2:,:].get()-latent_space_preds.at[:,:-2,:].get()+eps,cond_1_mask))))
    cond_3 = jnp.sqrt(jnp.mean(jnp.square(dlatent[:,1:,:] - dlatent[:,:-1,:]), axis=1))

    cond_loss=jnp.mean(cond_numer/(cond_3))
    
    return recon_loss + lam*cond_loss


class Encoder_Decoder():

    def __init__(self,config_handler:ConfigReader,data_processing_handler:Data_Processing)->None:

        self.config_handler=config_handler # constants that do not change during training

        self.data_processing_handler=data_processing_handler

        self.training_loss_values=[]
        self.test_loss_values=[]
        self.print_freq=self.config_handler.get_config_status("encoder_decoder.training.print_freq")

        self.training_constants=self.data_processing_handler.get_training_constants()

        # check if save and load directories exist
        self.check_save_load_dirs()

        # load model if specified
        if self.config_handler.get_config_status("encoder_decoder.loading.load_model"):
            print("Loading encoder and decoder weights")
            self._load_enc_dec()
        else:
            print("Initializing encoder and decoder weights, and training encoder and decoder")
            # initialize model
            self._init_enc_dec()

            # train model
            self._train_enc_dec()
   
            # test model using test

            if self.config_handler.get_config_status("encoder_decoder.loading.save_model"):
                print("Saving encoder and decoder weights")
                self.save_enc_dec()

        # test loaded model
        if self.config_handler.get_config_status("encoder_decoder.testing.test_model"):
            print("Testing loaded encoder and decoder weights")
            error=self.test_error_compute(self.encoder_object.weights,self.decoder_object.weights,save_results=True)
            print(f"Test error: {error}")

            if self.config_handler.get_config_status("encoder_decoder.testing.visualization.plot_results"):
                print("Visualizing results")
                self.visualize_results()

    def check_save_load_dirs(self):
        if not os.path.isdir(Path(self.config_handler.get_config_status("encoder_decoder.loading.model_output_dir"))):
            os.mkdir(Path(self.config_handler.get_config_status("encoder_decoder.loading.model_output_dir")))
        if not os.path.isdir(Path(self.config_handler.get_config_status("encoder_decoder.testing.save_dir"))):
            os.mkdir(Path(self.config_handler.get_config_status("encoder_decoder.testing.save_dir")))


    def __init_optimizer__(self):

        if self.config_handler.get_config_status("encoder_decoder.training.optimizer")=="adam":

            self.learning_rate=optax.exponential_decay(init_value=self.config_handler.get_config_status("encoder_decoder.training.start_learning_rate"),
                                                    transition_steps=self.config_handler.get_config_status("encoder_decoder.training.learning_rate_decay_steps"),
                                                    decay_rate=self.config_handler.get_config_status("encoder_decoder.training.learning_rate_decay"),
                                                    end_value=self.config_handler.get_config_status("encoder_decoder.training.end_learning_rate"))
            
            self.optimizer=optax.adam(self.learning_rate)
        elif self.config_handler.get_config_status("encoder_decoder.training.optimizer")=="l-bfgs":
            pass
        
        else:
            raise NotImplementedError(f"Optimizer {self.config_handler.get_config_status('encoder_decoder.training.optimizer')} not implemented")


    def _init_enc_dec(self):


        num_inputs=self.data_processing_handler.num_inputs
        n_latent_space=self.config_handler.get_config_status("data_processing.latent_space_dim")
        hidden_state_size=self.config_handler.get_config_status("encoder_decoder.architecture.network_width")

        # TODO add layers
     
        encoder_sizes=[[num_inputs,hidden_state_size]]
        decoder_sizes=[[n_latent_space,hidden_state_size]]

        for i_layer in range(self.config_handler.get_config_status("encoder_decoder.architecture.num_layers")):
            encoder_sizes.append([hidden_state_size,hidden_state_size])
            decoder_sizes.append([hidden_state_size,hidden_state_size])
        encoder_sizes.append([hidden_state_size,n_latent_space])
        decoder_sizes.append([hidden_state_size,num_inputs])

        self.encoder_object=MLP(encoder_sizes,self.config_handler)
        self.decoder_object=MLP(decoder_sizes,self.config_handler)

        self.encoder_object.initialize_network()
        self.decoder_object.initialize_network()


    def _train_enc_dec(self):

        # get constants required for training

        # add some additional constants
        self.training_constants['stiffness_reduction_weight']=float(self.config_handler.get_config_status("encoder_decoder.training.stiffness_reduction_weight"))

        #get some configuration parameters
        self.training_iters=self.config_handler.get_config_status("encoder_decoder.training.num_training_iters")

        # get test data
        self.test_data_dict=self.data_processing_handler.get_test_data()

        self.trainable_variables={'encoder':self.encoder_object.weights,'decoder':self.decoder_object.weights}
    
        # initialize optimizer
        self.__init_optimizer__()

        opt_state=self.optimizer.init(self.trainable_variables)
    
        # initialize training trackers
        self.best_training_loss=float('inf')
        self.best_test_loss=float('inf')

        t1=time.time()
        for i_step in range(self.training_iters):

                      
            opt_state=self._train_step(opt_state,i_step)
            
            if i_step%self.print_freq==0 and i_step>0:
                t2=time.time()
                print(f"Time taken for {self.print_freq} training steps: {t2-t1} seconds")
                t1=time.time()

        print("Training complete")
        
        print(f"Best test loss: {self.best_test_loss}")


    def _train_step(self,opt_state,train_step):

        # sample trajectories from all
        data_dict=self.data_processing_handler.sample_training_data()

        # get value and grad
        value,grad_loss=jax.value_and_grad(self.loss_fn,argnums=1,allow_int=True)(self.training_constants,self.trainable_variables,data_dict)
            
        #compute update to trainable variable
        updates,opt_state=self.optimizer.update(grad_loss,opt_state)

        # get new value for trainable variable
        results=optax.apply_updates(self.trainable_variables,updates)

        # update values
        self.trainable_variables.update(results)

        # according to update criteria, update the encoder and decoder weights
        
        if train_step%self.print_freq==0:

            # update values in object, which is used to compute test error

            error=self.test_error_compute(self.trainable_variables['encoder'],self.trainable_variables['decoder'])

            self.test_loss_values.append(error)
            self.training_loss_values.append(value)

            print(f"Iteration number: {train_step}, loss: {value}, test error: {error}, best test loss: {self.best_test_loss}")
        
            # log to mlflow
            log_to_mlflow_metrics({'enc_dec_training_loss':value,'enc_dec_test_loss':error},train_step)

            if  True:#error<self.best_test_loss:
                print(f"New best test loss: {error}, recording weights")
                self.best_test_loss=error
                self.encoder_object.weights=self.trainable_variables['encoder']
                self.decoder_object.weights=self.trainable_variables['decoder']
        
        return opt_state

    def loss_fn(self,constants,networks,data_dict):

        return _loss_fn_autoencoder(constants,networks,data_dict)

    # plot predictions for a single trajectory
    def test_error_compute(self,enc_weights,dec_weights,save_results:bool=False)->float:

        self.test_data_dict=self.data_processing_handler.get_test_data()
        self.test_constants=self.data_processing_handler.get_testing_constants()

        enc_dec_res_dir=Path(self.config_handler.get_config_status("encoder_decoder.testing.save_dir"))

        std_vals_inp=self.test_constants['std_vals_inp'].reshape(1,1,-1)
        mean_vals_inp=self.test_constants['mean_vals_inp'].reshape(1,1,-1)

        input_data=self.test_data_dict['input_data']

        latent_space_preds=_forward_pass(input_data,enc_weights)
        
        input_preds=_forward_pass(latent_space_preds,dec_weights)

        # compute error
        error=jnp.sqrt(jnp.mean(jnp.square(input_data-input_preds)))

        #Save and plot autoencoder performance
        #---------------------------------------------------------------
        if save_results:
            # get some constants for predictions
            prediction_lists={'specie_list_test':[],'temps_list_test':[],'time_data_test':[]}
            true_lists={'specie_list_test':[],'temps_list_test':[],'time_data_test':[]}
            
            num_test_traj=self.test_constants['num_test_traj']
            num_inputs=self.test_constants['num_inputs']
            num_timesteps_each_traj_test=self.test_constants['num_timesteps_each_traj_test']
        
            # un normalize
            input_data=input_data*std_vals_inp+mean_vals_inp
            input_preds=input_preds*std_vals_inp+mean_vals_inp

            print("Writing out results of autoencoder training to pickle file...")

            # save predictions and true values
            for i_traj in range(num_test_traj):
                prediction_lists['specie_list_test'].append(input_preds[i_traj,:num_timesteps_each_traj_test[i_traj],:-1])
                prediction_lists['temps_list_test'].append(input_preds[i_traj,:num_timesteps_each_traj_test[i_traj],num_inputs-1])
                prediction_lists['time_data_test'].append(self.test_data_dict['time_data'][i_traj][:num_timesteps_each_traj_test[i_traj]])
                
                true_lists['specie_list_test'].append(input_data[i_traj,:num_timesteps_each_traj_test[i_traj],:-1])
                true_lists['temps_list_test'].append(input_data[i_traj,:num_timesteps_each_traj_test[i_traj],num_inputs-1])
                true_lists['time_data_test'].append(self.test_data_dict['time_data'][i_traj][:num_timesteps_each_traj_test[i_traj]])

            with open(enc_dec_res_dir/Path('predictions.pkl'),'wb') as f:
                pickle.dump(prediction_lists,f,pickle.HIGHEST_PROTOCOL)
            with open(enc_dec_res_dir/Path('true_data.pkl'),'wb') as f:
                pickle.dump(true_lists,f,pickle.HIGHEST_PROTOCOL)

            # log to mlflow
            log_to_mlflow_artifacts(enc_dec_res_dir/Path('predictions.pkl'),"predictions_enc_dec")
            log_to_mlflow_artifacts(enc_dec_res_dir/Path('true_data.pkl'),"true_data_enc_dec")
        
        return error
    
    def save_enc_dec(self):

        save_mats=[self.encoder_object,self.decoder_object]

        with open(Path(self.config_handler.get_config_status("encoder_decoder.loading.model_output_dir"))/Path(self.config_handler.get_config_status("encoder_decoder.loading.load_path")),'wb') as f: 

            pickle.dump(save_mats,f,pickle.HIGHEST_PROTOCOL)
    def _load_enc_dec(self):

        print("TODO : replace by standarized format loading")
        with open(Path(self.config_handler.get_config_status("encoder_decoder.loading.model_output_dir"))/Path(self.config_handler.get_config_status("encoder_decoder.loading.load_path")),'rb') as f:
            self.encoder_object,self.decoder_object=pickle.load(f)

    def visualize_results(self):

        enc_dec_res_dir=Path(self.config_handler.get_config_status("encoder_decoder.testing.save_dir"))
        with open(enc_dec_res_dir/Path('predictions.pkl'),'rb') as f:
            prediction_lists=pickle.load(f)
        with open(enc_dec_res_dir/Path('true_data.pkl'),'rb') as f:
            true_lists=pickle.load(f)

        viz_dir=Path(self.config_handler.get_config_status("encoder_decoder.testing.save_dir"))/Path(self.config_handler.get_config_status("encoder_decoder.testing.visualization.save_dir"))
        if os.path.isdir(viz_dir):
            print(f"Removing existing visualization directory: {viz_dir}")
            shutil.rmtree(viz_dir)

        os.mkdir(viz_dir)

        # loop over saved trajectories and return comparison plots for each species and temperature
        for i_traj in range(len(prediction_lists['specie_list_test'])):
            i_traj_viz_dir=viz_dir/Path(f"traj_{i_traj}")
            os.mkdir(i_traj_viz_dir)
            num_inputs=self.test_constants['num_inputs']

            for i_input in range(num_inputs-1):
                plt.plot(prediction_lists['time_data_test'][i_traj],
                         prediction_lists['specie_list_test'][i_traj][:,i_input],label='predicted')
                plt.plot(true_lists['time_data_test'][i_traj],
                         true_lists['specie_list_test'][i_traj][:,i_input],label='true')
                plt.legend()
                plt.grid()
                plt.xlabel('Independent Variable')
                plt.ylabel('Dependent Variable')
                plt.title(f'Input No {i_input}')
                if 'xscale' in self.config_handler.get_config_status("encoder_decoder.testing.visualization.settings").keys():
                    plt.xscale(self.config_handler.get_config_status("encoder_decoder.testing.visualization.settings")['xscale'])
                if 'yscale' in self.config_handler.get_config_status("encoder_decoder.testing.visualization.settings").keys():
                    plt.yscale(self.config_handler.get_config_status("encoder_decoder.testing.visualization.settings")['yscale'])
                plt.savefig(i_traj_viz_dir/Path(f"input_{i_input}.png"))
                plt.close()
            
            plt.plot(prediction_lists['time_data_test'][i_traj],
                     prediction_lists['temps_list_test'][i_traj],label='predicted')
            plt.plot(true_lists['time_data_test'][i_traj],
                     true_lists['temps_list_test'][i_traj],label='true')
            
            if 'xscale' in self.config_handler.get_config_status("encoder_decoder.testing.visualization.settings").keys():
                plt.xscale(self.config_handler.get_config_status("encoder_decoder.testing.visualization.settings")['xscale'])
            if 'yscale' in self.config_handler.get_config_status("encoder_decoder.testing.visualization.settings").keys():
                plt.yscale(self.config_handler.get_config_status("encoder_decoder.testing.visualization.settings")['yscale'])
            plt.savefig(i_traj_viz_dir/Path(f"input_{num_inputs-1}.png"))
            plt.close()
        
        # log to mlflow
        log_to_mlflow_artifacts(viz_dir,"visualization_test_enc_dec")