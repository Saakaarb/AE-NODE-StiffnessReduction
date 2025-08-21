import os
import pickle
import jax.numpy as jnp
import jax
import optax
import numpy as np
from src.utils.classes import ConfigReader,LoggingManager
from src.utils.helper_functions import log_to_mlflow_metrics,log_to_mlflow_artifacts,create_network_instance
from src.lib.data_processing.classes import Data_Processing
from pathlib import Path
import time
import shutil
import matplotlib.pyplot as plt
from functools import partial
from typing import Any
import equinox as eqx
from typing import Any
from copy import deepcopy

# jit functions need to sit outside of classes

@jax.jit
def _compute_recon_loss(input_data:jax.Array,predicted_specie:jax.Array,data_dict:dict[str,jax.Array]):
    """
    Compute reconstruction loss between input data and predicted species.
    
    This function calculates the root mean square error between the input data and 
    predicted species, applying a reconstruction mask to focus on specific regions.
    
    Args:
        input_data (jax.Array): Original input data of shape [batch, time, features]
        predicted_specie (jax.Array): Predicted species data from the autoencoder
        data_dict (dict[str, jax.Array]): Dictionary containing 'recon_mask' for masking
        
    Returns:
        jax.Array: Scalar reconstruction loss value
    """
    recon_mask=data_dict['recon_mask']
    return jnp.sqrt(jnp.mean(jnp.square(jnp.multiply(predicted_specie-input_data,recon_mask))))

@jax.jit
def _compute_condition_number_norm(data_dict:dict[str,jax.Array],latent_space_preds:jax.Array):
    """
    Compute condition number normalization for stiffness reduction regularization.
    
    This function implements the stiffness reduction loss as described in the paper:
    "Stiffness-Reduced Neural ODE Models for Data-Driven Reduced-Order Modeling 
    of Combustion Chemical Kinetics" by Dikeman, Zhang and Yang (2022).
    
    The function computes derivatives of latent space predictions with respect to time
    and applies condition number regularization to improve numerical stability.
    
    Args:
        data_dict (dict[str, jax.Array]): Dictionary containing time data and condition masks
        latent_space_preds (jax.Array): Latent space predictions from encoder
        
    Returns:
        jax.Array: Scalar condition number loss value
    """
    eps=1E-12 # prevents division by zero
    eps_dt=1E-30 # prevents division by zero

    time_data=data_dict['all_time_data_broadcasted']

    # masks for condition number regularization
    cond_1_mask=data_dict['cond_1_mask']
    cond_2_mask=data_dict['cond_2_mask']

    dt = jnp.diff(time_data, axis=1)  # Shape: [batch, time-1, features]
    dlatent = jnp.diff(latent_space_preds, axis=1)  # Shape: [batch, time-1, features]

    # Compute derivatives
    cond_1 = jnp.divide(dlatent[:,1:,:], dt[:,1:,:] + eps_dt) * cond_1_mask
    cond_2 = jnp.divide(dlatent[:,:-1,:], dt[:,:-1,:] + eps_dt) * cond_2_mask

    cond_numer= jnp.sqrt(jnp.mean(jnp.square(cond_1-cond_2)+eps,axis=1))

    cond_3 = jnp.sqrt(jnp.mean(jnp.square(dlatent[:,1:,:] - dlatent[:,:-1,:]), axis=1))

    cond_loss=jnp.mean(cond_numer/(cond_3))

    return cond_loss

#@partial(jax.jit,static_argnums=(3,))
#def _loss_fn_autoencoder(networks:dict,constants:dict,data_dict:dict,stiffness_reduction:bool):
@eqx.filter_jit
def _loss_fn_autoencoder(networks:dict,constants:dict,data_dict:dict,stiffness_reduction:bool):
    #networks=eqx.combine(params,static)
    """
    Compute the total loss for the autoencoder training.
    
    This function combines reconstruction loss with optional stiffness reduction
    regularization. The stiffness reduction term helps improve numerical stability
    of the learned latent representation.
    
    Args:
        constants (dict): Dictionary containing training constants including stiffness reduction weight
        networks (dict): Dictionary containing encoder and decoder network weights
        data_dict (dict): Dictionary containing input data and masks
        stiffness_reduction (bool): Whether to include stiffness reduction regularization
        
    Returns:
        jax.Array: Total loss value combining reconstruction and condition number losses
    """


    lam=constants['stiffness_reduction_weight'] # weight for stiffness reduction loss
    
    # data in shape [num_traj_samples,max_steps,num_inputs]
    input_data=data_dict['input_data']
    
    # forward pass data through encoder
    #latent_space_preds=_forward_pass(input_data,networks['encoder'])
    latent_space_preds=networks['encoder'](input_data)

    # forward pass latent rep through decoder
    #predicted_specie=_forward_pass(latent_space_preds,networks['decoder'])
    predicted_specie=networks['decoder'](latent_space_preds)

    #jax.debug.print("input_data_shape :{x}",x=input_data.shape)
    #jax.debug.print("latent_space_preds_shape :{x}",x=latent_space_preds.shape)
    #jax.debug.print("predicted_specie_shape :{x}",x=predicted_specie.shape)

    # construct reconstruction
    recon_loss = _compute_recon_loss(input_data,predicted_specie,data_dict)

    # condition number regularization
    # this section takes an approximation of the condition number as described in:
    # Stiffness-Reduced Neural ODE Models for Data-Driven Reduced-Order Modeling of Combustion Chemical Kinetics: Dikeman, Zhang and Yang (2022)
    # and introduces a multi-objective optimization term

    if stiffness_reduction:
        #jax.debug.print("stiffness reduction")
        cond_loss=_compute_condition_number_norm(data_dict,latent_space_preds)
    else:
        cond_loss=jnp.zeros(())
    
    
    return recon_loss + lam*cond_loss


class Encoder_Decoder():
    """
    Autoencoder class for learning low-dimensional representations of high-dimensional data.
    
    This class implements an encoder-decoder architecture that can compress high-dimensional
    input data into a lower-dimensional latent space and reconstruct it back. It supports
    optional stiffness reduction regularization for improved numerical stability.
    
    Attributes:
        config_handler (ConfigReader): Configuration handler for model parameters
        logging_manager (LoggingManager): Manager for logging training progress
        data_processing_handler (Data_Processing): Handler for data processing operations
        training_loss_values (list): List to track training loss during training
        test_loss_values (list): List to track test loss during training
        test_cond_loss_values (list): List to track condition number loss during training
        print_freq (int): Frequency of printing training progress
        stiffness_reduction (bool): Whether to use stiffness reduction regularization
        training_constants (dict): Constants used during training
    """

    def __init__(self,config_handler:ConfigReader,logging_manager:LoggingManager,data_processing_handler:Data_Processing)->None:
        """
        Initialize the Encoder_Decoder class.
        
        This method sets up the autoencoder, initializes or loads model weights,
        trains the model if needed, and optionally tests the model performance.
        
        Args:
            config_handler (ConfigReader): Configuration handler for model parameters
            logging_manager (LoggingManager): Manager for logging training progress
            data_processing_handler (Data_Processing): Handler for data processing operations
        """

        self.config_handler=config_handler # constants that do not change during training
        self.logging_manager=logging_manager
        self.data_processing_handler=data_processing_handler

        self.training_loss_values=[]
        self.test_loss_values=[]
        self.test_cond_loss_values=[]
        self.print_freq=self.config_handler.get_config_status("encoder_decoder.training.print_freq")

        self.stiffness_reduction=bool(self.config_handler.get_config_status("encoder_decoder.training.stiffness_reduction"))

        self.training_constants=self.data_processing_handler.get_training_constants()

        # check if save and load directories exist
        self.check_save_load_dirs()

        # load model if specified
        if self.config_handler.get_config_status("encoder_decoder.loading.load_model"):
            self.logging_manager.log("Loading encoder and decoder weights")
            self._load_enc_dec()
        else:
            self.logging_manager.log("Initializing encoder and decoder weights, and training encoder and decoder")
            # initialize model
            self._init_enc_dec()

            # train model
            self._train_enc_dec()
   
            # test model using test

            if self.config_handler.get_config_status("encoder_decoder.loading.save_model"):
                self.logging_manager.log("Saving encoder and decoder weights")
                self.save_enc_dec()

        # test loaded model
        if self.config_handler.get_config_status("encoder_decoder.testing.test_model"):
            self.logging_manager.log("Testing loaded encoder and decoder weights")
            error,cond_loss=self.test_error_compute(self.best_trainable_models['encoder'],self.best_trainable_models['decoder'],save_results=True)
            self.logging_manager.log(f"Test error: {error}, cond loss: {cond_loss}")

            if self.config_handler.get_config_status("encoder_decoder.testing.visualization.plot_results"):
                self.logging_manager.log("Visualizing results")
                self.visualize_results()

    def check_save_load_dirs(self):
        """
        Check and create necessary directories for saving and loading models.
        
        This method ensures that the model output directory and testing save directory
        exist, creating them if they don't.
        """

        if not os.path.isdir(Path(self.config_handler.get_config_status("encoder_decoder.loading.model_output_dir"))):
            os.mkdir(Path(self.config_handler.get_config_status("encoder_decoder.loading.model_output_dir")))
        if not os.path.isdir(Path(self.config_handler.get_config_status("encoder_decoder.testing.save_dir"))):
            os.mkdir(Path(self.config_handler.get_config_status("encoder_decoder.testing.save_dir")))


    def __init_optimizer__(self):
        """
        Initialize the optimizer for training the autoencoder.
        
        This method sets up the optimizer based on configuration settings.
        Currently supports Adam with learning rate decay and L-BFGS optimizers.
        """

        if self.config_handler.get_config_status("encoder_decoder.training.optimizer")=="adam":

            self.learning_rate=optax.exponential_decay(init_value=self.config_handler.get_config_status("encoder_decoder.training.start_learning_rate"),
                                                    transition_steps=self.config_handler.get_config_status("encoder_decoder.training.learning_rate_decay_steps"),
                                                    decay_rate=self.config_handler.get_config_status("encoder_decoder.training.learning_rate_decay"),
                                                    end_value=self.config_handler.get_config_status("encoder_decoder.training.end_learning_rate"))
            
            self.optimizer=optax.adam(self.learning_rate)
        elif self.config_handler.get_config_status("encoder_decoder.training.optimizer")=="l-bfgs":
            self.optimizer=optax.lbfgs()
        
        else:
            raise NotImplementedError(f"Optimizer {self.config_handler.get_config_status('encoder_decoder.training.optimizer')} not implemented")


    def _init_enc_dec(self):
        """
        Initialize the encoder and decoder networks.
        
        This method creates MLP networks for both encoder and decoder based on
        configuration parameters. The encoder compresses input data to latent space,
        while the decoder reconstructs from latent space back to input space.
        """

        num_inputs=self.data_processing_handler.num_inputs
        n_latent_space=self.config_handler.get_config_status("data_processing.latent_space_dim")
        hidden_state_size=self.config_handler.get_config_status("encoder_decoder.architecture.network_width")
     
        encoder_sizes=[num_inputs,hidden_state_size,n_latent_space]
        decoder_sizes=[n_latent_space,hidden_state_size,num_inputs]


        self.encoder_object=create_network_instance(encoder_sizes,self.config_handler)
        self.decoder_object=create_network_instance(decoder_sizes,self.config_handler)

        self.best_encoder_object=deepcopy(self.encoder_object)
        self.best_decoder_object=deepcopy(self.decoder_object)


    def _train_enc_dec(self):
        """
        Train the encoder and decoder networks.
        
        This method implements the main training loop for the autoencoder.
        It samples training data, computes gradients, updates network weights,
        and tracks training progress including test performance.
        """

        # get constants required for training

        # add some additional constants
        self.training_constants['stiffness_reduction_weight']=float(self.config_handler.get_config_status("encoder_decoder.training.stiffness_reduction_weight"))

        #get some configuration parameters
        self.training_iters=self.config_handler.get_config_status("encoder_decoder.training.num_training_iters")

        # get test data
        self.test_data_dict=self.data_processing_handler.get_test_data()

        #self.trainable_variables={'encoder':self.encoder_object.weights,'decoder':self.decoder_object.weights}
        self.trainable_models={'encoder':self.encoder_object,'decoder':self.decoder_object}
        self.best_trainable_models={'encoder':self.best_encoder_object,'decoder':self.best_decoder_object}
        # initialize optimizer
        self.__init_optimizer__()

        #opt_state=self.optimizer.init(self.trainable_variables)
        
        opt_state=self.optimizer.init(eqx.filter(self.trainable_models,eqx.is_inexact_array))

        # initialize training trackers
        self.best_training_loss=float('inf')
        self.best_test_loss=float('inf')

        # training loop
        t1=time.time()
        for i_step in range(self.training_iters):

                      
            opt_state=self._train_step(opt_state,i_step)
            
            if i_step%self.print_freq==0 and i_step>0:
                t2=time.time()
                self.logging_manager.log(f"Time taken for {self.print_freq} training steps: {t2-t1} seconds")
                t1=time.time()

        self.logging_manager.log("Training complete")
        
        self.logging_manager.log(f"Best test loss: {self.best_test_loss}")


    def _train_step(self,opt_state:optax.OptState,train_step:int):
        """
        Execute a single training step.
        
        This method performs one iteration of training: samples data, computes
        loss and gradients, updates network weights, and evaluates performance.
        
        Args:
            opt_state (optax.OptState): Current optimizer state
            train_step (int): Current training step number
            
        Returns:
            optax.OptState: Updated optimizer state
        """

        # sample trajectories from all
        data_dict=self.data_processing_handler.sample_training_data()

        # get value and grad
        
        value,grad_loss=eqx.filter_value_and_grad(_loss_fn_autoencoder,allow_int=True)(self.trainable_models,self.training_constants,data_dict,self.stiffness_reduction)
       
        #compute update to trainable variable
        if self.config_handler.get_config_status("encoder_decoder.training.optimizer")=="adam":
            updates,opt_state=self.optimizer.update(grad_loss,opt_state)
        elif self.config_handler.get_config_status("encoder_decoder.training.optimizer")=="l-bfgs":
            def loss_wrapper(trainable_models):
                return self.loss_fn(trainable_models, self.training_constants,data_dict)
            updates,opt_state=self.optimizer.update(grad_loss, opt_state,self.trainable_models,value=value,grad=grad_loss,value_fn=loss_wrapper) #self.optimizer.update(grad_loss,opt_state,self.trainable_variables_NODE)
            #updates,opt_state=self.optimizer.update(grad_loss, opt_state,self.trainable_variables,value=value,grad=grad_loss,value_fn=loss_wrapper) #self.optimizer.update(grad_loss,opt_state,self.trainable_variables_NODE)
        # get new value for trainable variable
        
        results=eqx.apply_updates(self.trainable_models,updates)

        # update values
        #self.trainable_variables.update(results)
        self.trainable_models.update(results)

        # according to update criteria, update the encoder and decoder weights
        
        if train_step%self.print_freq==0:

            # update values in object, which is used to compute test error

            #error,cond_loss=self.test_error_compute(self.trainable_variables['encoder'],self.trainable_variables['decoder'])
            error,cond_loss=self.test_error_compute(self.trainable_models['encoder'],self.trainable_models['decoder'])

            self.test_loss_values.append(error)
            self.test_cond_loss_values.append(cond_loss)
            self.training_loss_values.append(value)

            self.logging_manager.log(f"Iteration number: {train_step}, loss: {value}, test error: {error}, test cond loss: {cond_loss:.2e}, best test loss: {self.best_test_loss}")
        
            # log to mlflow
            log_to_mlflow_metrics({'enc_dec_training_loss':value,'enc_dec_test_loss':error,'enc_dec_test_cond_loss':cond_loss},train_step)

            if  error<self.best_test_loss:
                self.logging_manager.log(f"New best test loss: {error}, recording weights")
                self.best_test_loss=error
                self.best_trainable_models['encoder']=deepcopy(self.trainable_models['encoder'])
                self.best_trainable_models['decoder']=deepcopy(self.trainable_models['decoder'])
        
        return opt_state

    def loss_fn(self,networks:dict[str,eqx.Module],constants:dict[str,Any],data_dict:dict[str,jax.Array]):
        """
        Compute the loss function for the autoencoder.
        
        This is a wrapper method that calls the JIT-compiled loss function
        with the current stiffness reduction setting.
        
        Args:
            constants (dict[str, Any]): Training constants
            networks (dict[str, jax.Array]): Network weights
            data_dict (dict[str, jax.Array]): Input data and masks
            
        Returns:
            jax.Array: Computed loss value
        """

        return _loss_fn_autoencoder(networks,constants,data_dict,self.stiffness_reduction)

    # plot predictions for a single trajectory
    def test_error_compute(self,enc_object:eqx.Module,dec_object:eqx.Module,save_results:bool=False)->float:
        """
        Compute test error and optionally save results.
        
        This method evaluates the autoencoder performance on test data,
        computing reconstruction error and condition number loss. It can
        optionally save predictions and true values for later analysis.
        
        Args:
            enc_object: Encoder network object
            dec_object: Decoder network object
            save_results (bool): Whether to save results to files
            
        Returns:
            tuple: (reconstruction_error, condition_number_loss)
        """

        self.test_data_dict=self.data_processing_handler.get_test_data()
        self.test_constants=self.data_processing_handler.get_testing_constants()

        enc_dec_res_dir=Path(self.config_handler.get_config_status("encoder_decoder.testing.save_dir"))

        std_vals_inp=self.test_constants['std_vals_inp'].reshape(1,1,-1)
        mean_vals_inp=self.test_constants['mean_vals_inp'].reshape(1,1,-1)

        input_data=self.test_data_dict['input_data']

        #latent_space_preds=_forward_pass(input_data,enc_weights)
        latent_space_preds=enc_object(input_data)
        #input_preds=_forward_pass(latent_space_preds,dec_weights)
        input_preds=dec_object(latent_space_preds)

        # compute error
        # TODO: use a mask to ignore the extra time steps
        #error=jnp.sqrt(jnp.mean(jnp.square(input_data-input_preds)))
        error=_compute_recon_loss(input_data,input_preds,self.test_data_dict)

        #if self.stiffness_reduction:
        # track stiffness loss
        cond_loss=_compute_condition_number_norm(self.test_data_dict,latent_space_preds)
        

        #Save and plot autoencoder performance
        #---------------------------------------------------------------
        if save_results:
            # get some constants for predictions
            prediction_lists={'feature_list_test':[],'time_data_test':[],'latent_space_test':[]}
            true_lists={'feature_list_test':[],'time_data_test':[]}
            
            num_test_traj=self.test_constants['num_test_traj']
            num_inputs=self.test_constants['num_inputs']
            num_timesteps_each_traj_test=self.test_constants['num_timesteps_each_traj_test']
        
            # un normalize
            input_data=input_data*std_vals_inp+mean_vals_inp
            input_preds=input_preds*std_vals_inp+mean_vals_inp

            self.logging_manager.log("Writing out results of autoencoder training to pickle file...")

            # save predictions and true values
            for i_traj in range(num_test_traj):
                prediction_lists['feature_list_test'].append(input_preds[i_traj,:num_timesteps_each_traj_test[i_traj],:])
                prediction_lists['time_data_test'].append(self.test_data_dict['time_data'][i_traj][:num_timesteps_each_traj_test[i_traj]])
                prediction_lists['latent_space_test'].append(latent_space_preds[i_traj,:num_timesteps_each_traj_test[i_traj],:])

                true_lists['feature_list_test'].append(input_data[i_traj,:num_timesteps_each_traj_test[i_traj],:])
                true_lists['time_data_test'].append(self.test_data_dict['time_data'][i_traj][:num_timesteps_each_traj_test[i_traj]])

            with open(enc_dec_res_dir/Path('predictions.pkl'),'wb') as f:
                pickle.dump(prediction_lists,f,pickle.HIGHEST_PROTOCOL)
            with open(enc_dec_res_dir/Path('true_data.pkl'),'wb') as f:
                pickle.dump(true_lists,f,pickle.HIGHEST_PROTOCOL)

            # log to mlflow
            log_to_mlflow_artifacts(enc_dec_res_dir/Path('predictions.pkl'),"predictions_enc_dec")
            log_to_mlflow_artifacts(enc_dec_res_dir/Path('true_data.pkl'),"true_data_enc_dec")
        
        return error,cond_loss
    
    def save_enc_dec(self):
        """
        Save the trained encoder and decoder models to disk.
        
        This method serializes the encoder and decoder objects 
        and saves them to the configured model output directory.
        """

        eqx.tree_serialise_leaves(self.trainable_models,Path(self.config_handler.get_config_status("encoder_decoder.loading.model_output_dir"))/Path(self.config_handler.get_config_status("encoder_decoder.loading.load_path")))

        #save_tree=[self.encoder_object,self.decoder_object]

        #with open(Path(self.config_handler.get_config_status("encoder_decoder.loading.model_output_dir"))/Path(self.config_handler.get_config_status("encoder_decoder.loading.load_path")),'wb') as f: 

        #    pickle.dump(save_mats,f,pickle.HIGHEST_PROTOCOL)
    def _load_enc_dec(self):
        """
        Load pre-trained encoder and decoder models from disk.
        
        This method deserializes the encoder and decoder objects
        files. 
        """

        num_inputs=self.data_processing_handler.num_inputs
        n_latent_space=self.config_handler.get_config_status("data_processing.latent_space_dim")
        hidden_state_size=self.config_handler.get_config_status("encoder_decoder.architecture.network_width")
     
        encoder_sizes=[num_inputs,hidden_state_size,n_latent_space]
        decoder_sizes=[n_latent_space,hidden_state_size,num_inputs]

        encoder_object=create_network_instance(encoder_sizes,self.config_handler)
        decoder_object=create_network_instance(decoder_sizes,self.config_handler)

        trainable_models_init={'encoder':encoder_object,'decoder':decoder_object}

        load_path=Path(self.config_handler.get_config_status("encoder_decoder.loading.model_output_dir"))/Path(self.config_handler.get_config_status("encoder_decoder.loading.load_path"))
        self.best_trainable_models=eqx.tree_deserialise_leaves(load_path,trainable_models_init)

    def visualize_results(self):
        """
        Create visualization plots for test results.
        
        This method generates comparison plots between predicted and true values
        for each test trajectory, input feature, and latent space dimension.
        Plots are saved to the configured visualization directory.
        """

        enc_dec_res_dir=Path(self.config_handler.get_config_status("encoder_decoder.testing.save_dir"))
        with open(enc_dec_res_dir/Path('predictions.pkl'),'rb') as f:
            prediction_lists=pickle.load(f)
        with open(enc_dec_res_dir/Path('true_data.pkl'),'rb') as f:
            true_lists=pickle.load(f)

        viz_dir=Path(self.config_handler.get_config_status("encoder_decoder.testing.save_dir"))/Path(self.config_handler.get_config_status("encoder_decoder.testing.visualization.save_dir"))
        if os.path.isdir(viz_dir):
            self.logging_manager.log(f"Removing existing visualization directory: {viz_dir}")
            shutil.rmtree(viz_dir)

        os.mkdir(viz_dir)

        # loop over saved trajectories and return comparison plots for each species and temperature
        for i_traj in range(len(prediction_lists['feature_list_test'])):
            i_traj_viz_dir=viz_dir/Path(f"traj_{i_traj}")
            os.mkdir(i_traj_viz_dir)
            num_inputs=self.test_constants['num_inputs']

            for i_input in range(num_inputs):
                plt.plot(prediction_lists['time_data_test'][i_traj],
                         prediction_lists['feature_list_test'][i_traj][:,i_input],label='predicted')
                plt.plot(true_lists['time_data_test'][i_traj],
                         true_lists['feature_list_test'][i_traj][:,i_input],label='true')
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
        
            if self.config_handler.get_config_status("encoder_decoder.testing.visualization.plot_latent_space"):
                for i_dim in range(self.config_handler.get_config_status("data_processing.latent_space_dim")):
                    plt.plot(prediction_lists['time_data_test'][i_traj],
                            prediction_lists['latent_space_test'][i_traj][:,i_dim],label='predicted')
                    plt.legend()
                    plt.grid()
                    plt.xlabel('Independent Variable')
                    plt.ylabel('Latent Space')
                    plt.title(f'Latent Space Dimension {i_dim}')
                    if 'xscale' in self.config_handler.get_config_status("encoder_decoder.testing.visualization.settings").keys():
                        plt.xscale(self.config_handler.get_config_status("encoder_decoder.testing.visualization.settings")['xscale'])
                    if 'yscale' in self.config_handler.get_config_status("encoder_decoder.testing.visualization.settings").keys():
                        plt.yscale(self.config_handler.get_config_status("encoder_decoder.testing.visualization.settings")['yscale'])
                    plt.savefig(i_traj_viz_dir/Path(f"latent_space_{i_dim}.png"))
                    plt.close()

        # log to mlflow
        log_to_mlflow_artifacts(viz_dir,"visualization_test_enc_dec")