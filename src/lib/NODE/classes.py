import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import optax
from functools import partial
import pickle
import os
from src.utils.helper_functions import log_to_mlflow_artifacts,log_to_mlflow_metrics,create_network_instance
from src.utils.classes import ConfigReader,LoggingManager
from src.lib.data_processing.classes import Data_Processing
from src.lib.autoencoder.classes import Encoder_Decoder
from diffrax import RESULTS
import matplotlib.pyplot as plt
from pathlib import Path
import time
import shutil
from typing import Any


##########################################################

@jax.jit
def _ode_fn(t:float,state:jax.Array,other_args:dict[str,dict[str,Any]]):
    """
    Compute the right-hand side of the Neural ODE system.
    
    This function defines the dynamics of the system by computing derivatives
    of the latent state variables. It applies scaling to improve numerical
    stability as described in "Stiff Neural Ordinary Differential Equations"
    by Kim, Ji et al.
    
    Args:
        t (float): Current time point
        state (jax.Array): Current state vector in latent space
        other_args (dict[str, dict[str, Any]]): Dictionary containing constants,
                                               trainable variables, and trajectory index
                                               
    Returns:
        jax.Array: Derivatives of the state variables
    """

    constants=other_args['constants']
    #scaling=constants['latent_scaling']
    trainable_variables_NODE=other_args['trainable_variables_NODE']
    i_traj=other_args['i_traj']
    state=jnp.expand_dims(state,axis=0)

    #TODO change scaling for input and output
    # hardcoded for now
    scaling=jnp.array([1.0,1.0])


    # scaling trick derived from 
    #Stiff Neural Ordinary Differential Equations, Kim, Ji et al
    #https://arxiv.org/pdf/2103.15341
    #jax.debug.print("input: {x}, scaled input: {y}",x=state,y=jnp.divide(state,scaling))
    derivatives=jnp.squeeze(_forward_pass(jnp.divide(state,scaling),trainable_variables_NODE['NODE']))*(1.0/constants['end_time']) # scaling included
    #jax.debug.print("i_traj: {z}, t:{y} derivatives:{x}",z=i_traj,y=t,x=derivatives)
    return jnp.squeeze(derivatives)

@partial(jax.jit,static_argnums=(5,))
def _integrate_NODE(constants: dict[str,Any],trainable_variables_NODE: dict[str,jax.Array],enc_dec_weights:dict[str,jax.Array],data_dict:dict[str,jax.Array],i_traj:int,max_traj_size:int):
    """
    Integrate the Neural ODE system for a single trajectory.
    
    This function solves the initial value problem for the Neural ODE using
    the Diffrax library. It handles the conversion between physical and
    latent spaces, sets up the ODE solver with specified tolerances and
    step size controls, and returns the solution trajectory.
    
    Args:
        constants (dict[str, Any]): Dictionary containing ODE solver parameters
                                   (pcoeff, icoeff, rtol, atol, init_dt, dtmin)
        trainable_variables_NODE (dict[str, jax.Array]): Neural ODE network weights
        enc_dec_weights (dict[str, jax.Array]): Encoder and decoder network weights
        data_dict (dict[str, jax.Array]): Dictionary containing time data and initial conditions
        i_traj (int): Index of the trajectory to integrate
        max_traj_size (int): Maximum number of time steps for integration
        
    Returns:
        diffrax.Solution: Solution object containing the integrated trajectory
    """

    start_end_time_data=data_dict['start_end_time_data']
    initial_condition_data=data_dict['initial_condition_data']
    time_data=data_dict['time_data']
    
    # get ode solver specs
    pcoeff=constants['pcoeff']
    icoeff=constants['icoeff']
    rtol=constants['rtol']
    atol=constants['atol']
    init_dt=constants['init_dt']
    dtmin=constants['dtmin']


    t_init= start_end_time_data[i_traj,0] #constants['start_time']
    t_final= start_end_time_data[i_traj,1]#constants['end_time']

    
    phys_space_init=jnp.expand_dims(initial_condition_data[i_traj,:],axis=0)#constants['y_init']

    # convert initial cond to latent space
    y_latent_init = jnp.squeeze(_forward_pass(phys_space_init,enc_dec_weights['encoder']))

    curr_time_data=time_data[i_traj,:]
    #jax.debug.print("curr_time_data: {x}",x=curr_time_data)
    #jax.debug.print("curr time data shape: {x}",x=curr_time_data.shape)

    #saveat = diffrax.SaveAt(ts=curr_time_data)
    saveat = diffrax.SaveAt(t0=True,t1=True,steps=True)
    stepsize_controller=diffrax.StepTo(ts=curr_time_data)

    term = diffrax.ODETerm(_ode_fn)
    #rtol=jnp.array([1E-2,1E-3])

    # TODO try batched euler within diffrax diffeqsolve
    #solution = diffrax.diffeqsolve(term,diffrax.Dopri5(),t0=t_init,t1=t_final,dt0 = init_dt,y0=y_latent_init,
    #                                saveat=saveat,args={'constants':constants,'trainable_variables_NODE':trainable_variables_NODE,'i_traj':i_traj},throw=False,
    #                                max_steps=100000,stepsize_controller=diffrax.PIDController(pcoeff=pcoeff,icoeff=icoeff,rtol=rtol, atol=atol,dtmin=dtmin))
    
    #solution = diffrax.diffeqsolve(term,diffrax.Euler(),t0=t_init,t1=t_final,dt0 = init_dt,y0=y_latent_init,
    #                                saveat=saveat,args={'constants':constants,'trainable_variables_NODE':trainable_variables_NODE,'i_traj':i_traj},throw=False,
    #                                max_steps=16384)

    #solution = diffrax.diffeqsolve(term,diffrax.Tsit5(),t0=t_init,t1=t_final,dt0=init_dt,y0=y_latent_init,
    #                                saveat=saveat,args={'constants':constants,'trainable_variables_NODE':trainable_variables_NODE,'i_traj':i_traj},throw=True,
    #                                max_steps=600)

    solution = diffrax.diffeqsolve(term,diffrax.Heun(),t0=t_init,t1=t_final,dt0=None,y0=y_latent_init,
                                    saveat=saveat,args={'constants':constants,'trainable_variables_NODE':trainable_variables_NODE,'i_traj':i_traj},throw=False,
                                    max_steps=max_traj_size-1,stepsize_controller=stepsize_controller)
    #jax.debug.print("solution.ts: {x}, curr_time_data: {y}",x=solution.ts,y=curr_time_data)

    #solution = diffrax.diffeqsolve(term,diffrax.Kvaerno5(),t0=t_init,t1=t_final,dt0 = 1e-11,y0=y_latent_init,
    #                                saveat=saveat,args={'constants':constants,'trainable_variables_NODE':trainable_variables_NODE,'i_traj':i_traj},throw=False,
    #                                max_steps=100000,stepsize_controller=diffrax.PIDController(pcoeff=0.3,icoeff=0.4,rtol=1e-6, atol=1e-8,dtmin=None))
    
    return solution


@partial(jax.jit,static_argnums=(4,5,))
def _loss_fn_NODE(constants:dict[str,Any],trainable_variables_NODE:dict[str,jax.Array],enc_dec_weights:dict[str,jax.Array],data_dict:dict[str,jax.Array],num_traj:int,max_traj_size:int):
    """
    Compute the loss function for Neural ODE training.
    
    This function computes a multi-objective loss that combines:
    1. Reconstruction loss in physical space (L1)
    2. Latent space consistency loss (L3)
    
    The function handles integration failures gracefully by penalizing failed
    trajectories with a high loss value. It uses vectorized operations over
    multiple trajectories for efficient computation.
    
    Args:
        constants (dict[str, Any]): Training constants and parameters
        trainable_variables_NODE (dict[str, jax.Array]): Neural ODE network weights
        enc_dec_weights (dict[str, jax.Array]): Encoder and decoder network weights
        data_dict (dict[str, jax.Array]): Training data including masks and input data
        num_traj (int): Number of trajectories to process
        max_traj_size (int): Maximum trajectory size for integration
        
    Returns:
        jax.Array: Combined loss value normalized by successful trajectory count
    """
    
    # masks
    recon_mask=data_dict['recon_mask']
    latent_space_mask=data_dict['latent_space_mask']
    input_data=data_dict['input_data']
    
    
    # Create a vectorized version of the single-trajectory loss computation
    def single_trajectory_loss(i_traj:int):
        recon_mask_curr=recon_mask[i_traj,:,:]
        latent_space_mask_curr=latent_space_mask[i_traj,:,:]
        phys_data=input_data[i_traj,:,:]

        solution=_integrate_NODE(constants,trainable_variables_NODE,enc_dec_weights,data_dict,i_traj,max_traj_size)

        failed = jnp.logical_or(solution.result == RESULTS.max_steps_reached, solution.result==RESULTS.singular)

        # predicted output in latent space
        latent_space_pred=jnp.squeeze(solution.ys)

        #jax.debug.print("solution time: {x}",x=solution.ts)
        # prediction after integration
        phys_space_pred_int=_forward_pass(latent_space_pred,enc_dec_weights['decoder'])
        #jax.debug.print("phys_space_pred_int: {x}",x=phys_space_pred_int)
        #jax.debug.print("phys_data: {x}",x=phys_data)
        #jax.debug.print("recon_mask_curr: {x}",x=recon_mask_curr)

        #jax.debug.print("loss_L1: {x}",x=jnp.sqrt(jnp.mean(jnp.square(jnp.multiply(phys_space_pred_int,recon_mask_curr)-jnp.multiply(phys_data,recon_mask_curr)))))
        loss_L1=jnp.where(failed,1E5,jnp.sqrt(jnp.mean(jnp.square(jnp.multiply(phys_space_pred_int,recon_mask_curr)-jnp.multiply(phys_data,recon_mask_curr)))))
        #jax.debug.print("phys_space_pred_int: {x}, phys_data: {y}, recon_mask_curr: {z}",x=phys_space_pred_int,y=phys_data,z=recon_mask_curr)
        #loss_L1=jnp.sqrt(jnp.mean(jnp.square(jnp.multiply(phys_space_pred_int,recon_mask_curr)-jnp.multiply(phys_data,recon_mask_curr))))

        # latent space truth
        latent_space_truth=_forward_pass(phys_data,enc_dec_weights['encoder'])
        #jax.debug.print("latent_space_pred: {x}",x=latent_space_pred)
        #jax.debug.print("latent_space_truth: {x}",x=latent_space_truth)
        #jax.debug.print("latent_space_mask_curr: {x}",x=latent_space_mask_curr)

        #jax.debug.print("loss_L3: {x}",x=jnp.sqrt(jnp.mean(jnp.square(jnp.multiply(latent_space_pred,latent_space_mask_curr)-jnp.multiply(latent_space_truth,latent_space_mask_curr)))))
        loss_L3=jnp.where(failed,1E5,jnp.sqrt(jnp.mean(jnp.square(jnp.multiply(latent_space_pred,latent_space_mask_curr)-jnp.multiply(latent_space_truth,latent_space_mask_curr)))))
        #loss_L3=jnp.sqrt(jnp.mean(jnp.square(jnp.multiply(latent_space_pred,latent_space_mask_curr)-jnp.multiply(latent_space_truth,latent_space_mask_curr))))

        return loss_L1, loss_L3, jnp.where(failed,0.0,1.0)

    # Vectorize over all trajectories
    losses_L1, losses_L3, loss_comp_success = jax.vmap(single_trajectory_loss)(jnp.arange(num_traj))
    
    # Sum up the losses
    loss_l1 = jnp.sum(losses_L1)
    loss_l3 = jnp.sum(losses_L3)
    
    total_success = jnp.sum(loss_comp_success)
    #jax.debug.print("loss_L1: {x}, loss_L3: {y}",x=loss_l1/total_success,y=loss_l3/total_success)
    return (loss_l1 + loss_l3) / (total_success)

class Neural_ODE():
    """
    Neural Ordinary Differential Equation (NODE) class for learning dynamical systems.
    
    This class implements a Neural ODE that learns the dynamics of a system in
    a learned latent space. It combines an encoder-decoder architecture with
    a neural network that defines the right-hand side of the ODE system.
    
    The class supports both standalone NODE training and simultaneous training
    with the encoder-decoder. It includes comprehensive testing, visualization,
    and model persistence capabilities.
    
    Attributes:
        config_handler (ConfigReader): Configuration handler for model parameters
        data_processing_handler (Data_Processing): Handler for data processing operations
        encoder_decoder_handler (Encoder_Decoder): Pre-trained encoder-decoder
        logging_manager (LoggingManager): Manager for logging training progress
        constants (dict): Training constants and ODE solver parameters
        test_data_dict (dict): Test data for evaluation
        test_constants (dict): Constants for testing
        trainable_enc_dec (bool): Whether to train encoder-decoder simultaneously
        NODE_object (VMapMLP): Neural network defining the ODE dynamics
    """

    def __init__(self,config_handler:ConfigReader,logging_manager:LoggingManager,data_processing_handler:Data_Processing,encoder_decoder_handler:Encoder_Decoder):
        """
        Initialize the Neural_ODE class.
        
        This method sets up the Neural ODE, configures ODE solver parameters,
        initializes or loads model weights, trains the model if needed, and
        optionally tests the model performance.
        
        Args:
            config_handler (ConfigReader): Configuration handler for model parameters
            logging_manager (LoggingManager): Manager for logging training progress
            data_processing_handler (Data_Processing): Handler for data processing operations
            encoder_decoder_handler (Encoder_Decoder): Pre-trained encoder-decoder
        """

        # init
        self.config_handler=config_handler
        self.data_processing_handler=data_processing_handler
        self.encoder_decoder_handler=encoder_decoder_handler
        self.logging_manager=logging_manager
        # get training constants
        self.constants=self.data_processing_handler.get_training_constants()

        # get testing dict and constants
        self.test_data_dict=self.data_processing_handler.get_test_data()
        self.test_constants=self.data_processing_handler.get_testing_constants()

        # update neural ODE specs in dict
        node_spec_dict={'pcoeff':float(self.config_handler.get_config_status("neural_ode.ode_solver.pcoeff")),
                        'icoeff':float(self.config_handler.get_config_status("neural_ode.ode_solver.icoeff")),
                        }
        rtol_vals=self.config_handler.get_config_status("neural_ode.ode_solver.rtol")
        atol_vals=self.config_handler.get_config_status("neural_ode.ode_solver.atol")
        if isinstance(rtol_vals,list):
            assert len(rtol_vals)==self.config_handler.get_config_status("data_processing.latent_space_dim"),"rtol list must be same length as latent space dim"
            node_spec_dict['rtol']=rtol_vals
        else:
            node_spec_dict['rtol']=float(rtol_vals)

        if isinstance(atol_vals,list):
            assert len(atol_vals)==self.config_handler.get_config_status("data_processing.latent_space_dim"),"atol list must be same length as latent space dim"
            node_spec_dict['atol']=atol_vals
        else:
            node_spec_dict['atol']=float(atol_vals)

        #'rtol':float(self.config_handler.get_config_status("neural_ode.ode_solver.rtol"))
        #'atol':float(self.config_handler.get_config_status("neural_ode.ode_solver.atol"))
        # update self.constant with ode solver specs

        if self.config_handler.get_config_status("neural_ode.ode_solver.init_dt") == "None":    
            node_spec_dict['init_dt']=None
        else:
            node_spec_dict['init_dt']=float(self.config_handler.get_config_status("neural_ode.ode_solver.init_dt"))
        if self.config_handler.get_config_status("neural_ode.ode_solver.dtmin") == "None":    
            node_spec_dict['dtmin']=None
        else:
            node_spec_dict['dtmin']=float(self.config_handler.get_config_status("neural_ode.ode_solver.dtmin"))

        self.constants.update(node_spec_dict)
        self.test_constants.update(node_spec_dict)
        
        # get encoder and decoder weights

        #self.enc_dec_weights={'encoder':self.encoder_decoder_handler.encoder_object.weights,
        #                     'decoder':self.encoder_decoder_handler.decoder_object.weights}

        self.trainable_enc_dec=self.config_handler.get_config_status("encoder_decoder.training.simultaneous_training")

        # check if save and load directories exist
        self.check_save_load_dirs()

        # load model if specified
        if self.config_handler.get_config_status("neural_ode.saving.load_model"):
            self.load_NODE_weights()
        else:

            # initialize network
            self._init_network()

            # train neural ODE
            self._train_NODE()
        
            if self.config_handler.get_config_status("neural_ode.saving.save_model"):
                self.save_NODE_weights()

        # test loaded model
        if self.config_handler.get_config_status("neural_ode.testing.test_model"):
            self.logging_manager.log("Getting predictions for test data")
            self.test_NODE_model(self.encoder_decoder_handler.encoder_object.weights,self.encoder_decoder_handler.decoder_object.weights,self.NODE_object.weights)

            # visualize results
            if self.config_handler.get_config_status("neural_ode.testing.visualization.plot_results"):
                self.logging_manager.log("Visualizing results")
                self.visualize_results()

    def check_save_load_dirs(self):
        """
        Check and create necessary directories for saving and loading models.
        
        This method ensures that the model output directory and testing save directory
        exist, creating them if they don't.
        """

        if not os.path.isdir(Path(self.config_handler.get_config_status("neural_ode.saving.model_output_dir"))):
            os.mkdir(Path(self.config_handler.get_config_status("neural_ode.saving.model_output_dir")))
        if not os.path.isdir(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))):
            os.mkdir(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir")))

    def _init_optimizer(self):
        """
        Initialize the optimizer for training the Neural ODE.
        
        This method sets up the optimizer based on configuration settings.
        Currently supports Adam with learning rate decay and L-BFGS optimizers.
        """

        if self.config_handler.get_config_status("neural_ode.training.optimizer")=="adam":

            self.init_lr_value=self.config_handler.get_config_status("neural_ode.training.start_learning_rate")
            self.lr_decay_rate=self.config_handler.get_config_status("neural_ode.training.learning_rate_decay")
            self.lr_end_value=self.config_handler.get_config_status("neural_ode.training.end_learning_rate")
            self.transition_steps=self.config_handler.get_config_status("neural_ode.training.learning_rate_decay_steps")

            # optimizer setting: Adam
            self.learning_rate=optax.exponential_decay(init_value=self.init_lr_value,
                                                    transition_steps=self.transition_steps,
                                                    decay_rate=self.lr_decay_rate,end_value=self.lr_end_value)
            self.optimizer=optax.adam(self.learning_rate)

        elif self.config_handler.get_config_status("neural_ode.training.optimizer")=="l-bfgs":
            self.optimizer=optax.lbfgs()    

    def _init_network(self):
        """
        Initialize the Neural ODE network.
        
        This method creates an MLP network that defines the right-hand side
        of the ODE system in the latent space. The network architecture is
        configured based on the specified parameters.
        """

        n_latent_space=self.config_handler.get_config_status("data_processing.latent_space_dim")
        hidden_size_NODE=self.config_handler.get_config_status("neural_ode.architecture.network_width")

        NODE_sizes=[n_latent_space,hidden_size_NODE,n_latent_space]
        #for i_layer in range(self.config_handler.get_config_status("neural_ode.architecture.num_layers")):
        #    NODE_sizes.append([hidden_size_NODE,hidden_size_NODE])
        #NODE_sizes.append([hidden_size_NODE,n_latent_space])

        self.NODE_object=create_network_instance(NODE_sizes,self.config_handler) #MLP(NODE_sizes,self.config_handler)
        #self.NODE_object.initialize_network()
        
        self.trainable_models_NODE={'NODE':self.NODE_object.weights}
        self.enc_dec_weights={'encoder':self.encoder_decoder_handler.encoder_object.weights,
                              'decoder':self.encoder_decoder_handler.decoder_object.weights}

        #if self.trainable_enc_dec:

        #self.trainable_variables_NODE.update({'encoder':self.encoder_decoder_handler.encoder_object.weights,
        #                                          'decoder':self.encoder_decoder_handler.decoder_object.weights})

    def _train_NODE(self):
        """
        Train the Neural ODE network.
        
        This method implements the main training loop for the Neural ODE.
        It samples training data, computes gradients, updates network weights,
        and tracks training progress including test performance. Training stops
        if integration failures occur.
        """

        self.logging_manager.log("Training NODE...")
        self.training_loss_values=[]
        self.test_loss_values=[]

        self.training_iters=self.config_handler.get_config_status("neural_ode.training.num_training_iters")
        self.print_freq=self.config_handler.get_config_status("neural_ode.training.print_freq")

        # run NeuralODE training
        self._init_optimizer()

        # initialize training trackers
        self.best_training_loss=float('inf')
        self.best_test_loss=float('inf')

        opt_state=self.optimizer.init(self.trainable_variables_NODE)
        success=1

        # test node model before training
        #self.test_NODE_model(self.encoder_decoder_handler.encoder_object.weights,self.encoder_decoder_handler.decoder_object.weights,self.NODE_object.weights)

        t1=time.time()
        # run training
        for i_step in range(self.training_iters):

            #rand_index=randrange(num_traj)
            opt_state,success=self._train_step(opt_state,i_step)
            if success==0:
                self.logging_manager.log("Integration failed, stopping training.")
                break
            if i_step % self.print_freq == 0:
                t2=time.time()
                self.logging_manager.log(f"Training time for {self.print_freq} steps: {t2-t1} seconds")
                t1=time.time()

    def _train_step(self,opt_state:optax.OptState,train_step:int):
        """
        Execute a single training step.
        
        This method performs one iteration of training: samples data, computes
        loss and gradients, updates network weights, and evaluates performance.
        It handles integration failures and updates the best model weights.
        
        Args:
            opt_state (optax.OptState): Current optimizer state
            train_step (int): Current training step number
            
        Returns:
            tuple: (updated_opt_state, success_flag) where success_flag indicates
                   whether the integration succeeded (1) or failed (0)
        """

        success=1
        # sample data
        data_dict=self.data_processing_handler.sample_training_data()

        num_traj=data_dict['input_data'].shape[0]
        

        # get value and grad
        #if self.trainable_enc_dec:
        #    value,grad_loss=jax.value_and_grad(self.loss_fn,argnums=(1,2),allow_int=True)(self.constants,self.trainable_variables_NODE,self.enc_dec_weights)

        #else:
        value,grad_loss=jax.value_and_grad(self.loss_fn,argnums=1,allow_int=True)(self.constants,self.trainable_variables_NODE,self.enc_dec_weights,data_dict,num_traj,self.constants['max_train_traj_size'])

        
        #if self.trainable_enc_dec:
        #    grad_loss={'NODE':grad_loss[0]['NODE'],'encoder':grad_loss[1]['encoder'],'decoder':grad_loss[1]['decoder']}


        #compute update to trainable variable
        if self.config_handler.get_config_status("neural_ode.training.optimizer")=="adam":
            updates,opt_state=self.optimizer.update(grad_loss,opt_state)
        elif self.config_handler.get_config_status("neural_ode.training.optimizer")=="l-bfgs":
            def loss_wrapper(trainable_vars):
                return self.loss_fn(self.constants, trainable_vars, self.enc_dec_weights, data_dict, num_traj,self.constants['max_train_traj_size'])
            updates,opt_state=self.optimizer.update(grad_loss, opt_state,self.trainable_variables_NODE,value=value,grad=grad_loss,value_fn=loss_wrapper) #self.optimizer.update(grad_loss,opt_state,self.trainable_variables_NODE)

        # get new value for trainable variable
        results=optax.apply_updates(self.trainable_variables_NODE,updates)

        # update values
        self.trainable_variables_NODE.update({'NODE':results['NODE']})

        if self.trainable_enc_dec:
            self.enc_dec_weights.update({'encoder':results['encoder'],'decoder':results['decoder']})

        
        if train_step % self.print_freq==0:
            #self.test_NODE_model(self.encoder_decoder_handler.encoder_object.weights,self.encoder_decoder_handler.decoder_object.weights,self.NODE_object.weights)
            if value>1E5/num_traj:
                self.logging_manager.log("Loss is too high, skipping update. This indicates failure to integrate.")
                success=0
                return opt_state,success


            test_loss=self.loss_fn(self.test_constants,self.trainable_variables_NODE,self.enc_dec_weights,self.test_data_dict,self.test_constants['num_test_traj'],self.test_constants['max_test_traj_size'])

            self.training_loss_values.append(value)
            self.test_loss_values.append(test_loss)
            if value < self.best_training_loss:
                self.best_training_loss=value

            self.logging_manager.log(f"Step: {train_step}, training loss: {value}, test loss: {test_loss}, best training loss: {self.best_training_loss}, best test loss: {self.best_test_loss}")

            if test_loss < self.best_test_loss:
                self.best_test_loss=test_loss
                self.logging_manager.log(f"New best test loss: {test_loss}, recording weights")
                self.NODE_object.weights=self.trainable_variables_NODE['NODE']
                #self.encoder_decoder_handler.encoder_object.weights=self.trainable_variables_NODE['encoder']
                #self.encoder_decoder_handler.decoder_object.weights=self.trainable_variables_NODE['decoder']
        
            # log to mlflow
            log_to_mlflow_metrics({'node_training_loss':value,'node_test_loss':test_loss},train_step)
            
        #print(f"Step: {train_step}, training loss: {value}")

        #self.NODE_object.weights=self.trainable_variables_NODE['NODE']
        return opt_state,success


    def loss_fn(self,constants:dict[str,Any],trainable_variables_NODE:dict[str,jax.Array],enc_dec_weights:dict[str,jax.Array],data_dict:dict[str,jax.Array],num_traj:int,max_traj_size:int):
        """
        Compute the loss function for Neural ODE training.
        
        This is a wrapper method that calls the JIT-compiled loss function
        with the current parameters. It ensures max_traj_size is converted
        to a Python int for compatibility.
        
        Args:
            constants (dict[str, Any]): Training constants and parameters
            trainable_variables_NODE (dict[str, jax.Array]): Neural ODE network weights
            enc_dec_weights (dict[str, jax.Array]): Encoder and decoder network weights
            data_dict (dict[str, jax.Array]): Training data including masks and input data
            num_traj (int): Number of trajectories to process
            max_traj_size(int): Maximum trajectory size for integration
            
        Returns:
            jax.Array: Computed loss value
        """

        max_traj_size=int(max_traj_size)
        return _loss_fn_NODE(constants,trainable_variables_NODE,enc_dec_weights,data_dict,num_traj,max_traj_size)

    # predict solution trajectories
    def test_NODE_model(self,enc_weights:jax.Array,dec_weights:jax.Array,NODE_weights:jax.Array):
        """
        Test the Neural ODE model on test data.
        
        This method evaluates the trained Neural ODE by integrating trajectories
        for all test data and comparing predictions with ground truth. It handles
        integration failures gracefully and saves predictions for later analysis.
        
        Args:
            enc_weights (jax.Array): Encoder network weights
            dec_weights (jax.Array): Decoder network weights
            NODE_weights (jax.Array): Neural ODE network weights
        """

        node_weights_dict={'NODE':NODE_weights}
        enc_dec_weights={'encoder':enc_weights,'decoder':dec_weights}

        # load test data

        
        num_timesteps_each_traj_test=self.test_constants['num_timesteps_each_traj_test']
        # for every test trajectory, store prediction

        num_test_traj=self.test_constants['num_test_traj']
        max_test_traj_size=self.test_constants['max_test_traj_size']
        num_inputs=self.test_constants['num_inputs']
        
        # store predicted ys
        pred_ys=np.zeros((num_test_traj,max_test_traj_size,num_inputs))
        pred_ts=np.zeros((num_test_traj,max_test_traj_size))
        
        # get ground truth labels
        raw_testing_data=self.data_processing_handler.get_raw_testing_data().copy()
        testing_predictions_list={'times_list_test':[],'feature_list_test':[]}
        testing_true_list={'times_list_test':[],'feature_list_test':[]}

        # get mean and std of inputs
        # make shape (1,num_inputs) since each trajectory is computed separately
        mean_vals_inp=self.test_constants['mean_vals_inp'].reshape(1,-1)
        std_vals_inp=self.test_constants['std_vals_inp'].reshape(1,-1)
        max_traj_size=int(self.test_constants['max_test_traj_size']) # convert to python int

        for i_traj in range(num_test_traj):
            self.logging_manager.log(f"Predicting trajectory {i_traj+1} of {num_test_traj}")
            solution=_integrate_NODE(self.test_constants,node_weights_dict,enc_dec_weights,self.test_data_dict,i_traj,max_traj_size) 

            # check if integration failed
            if solution.result==RESULTS.max_steps_reached or solution.result==RESULTS.singular:
                self.logging_manager.log(f"Integration failed for trajectory {i_traj+1} of {num_test_traj}")
                
            latent_space_pred=jnp.squeeze(solution.ys)
            phys_space_pred_int=_forward_pass(latent_space_pred,enc_dec_weights['decoder'])

            # unscale predicted ys
            # std_vals and mean_vals are of shape (1,num_inputs)
            #phys_space_pred_int=phys_space_pred_int*std_vals_inp+mean_vals_inp

            pred_ys[i_traj,:,:]=phys_space_pred_int*std_vals_inp+mean_vals_inp
            pred_ts[i_traj,:]=solution.ts


            #true soln
            true_ys=self.test_data_dict['input_data'][i_traj,:,:]*std_vals_inp+mean_vals_inp

            # append to testing lists
            testing_predictions_list['times_list_test'].append(pred_ts[i_traj,:num_timesteps_each_traj_test[i_traj]])
            testing_predictions_list['feature_list_test'].append(pred_ys[i_traj,:num_timesteps_each_traj_test[i_traj],:])

            # append to true lists
            testing_true_list['times_list_test'].append(self.test_data_dict['time_data'][i_traj][:num_timesteps_each_traj_test[i_traj]])
            testing_true_list['feature_list_test'].append(true_ys[:num_timesteps_each_traj_test[i_traj],:])

        # save predictions and true values
        with open(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))/Path("predictions.pkl"),'wb') as f:
            pickle.dump(testing_predictions_list,f,pickle.HIGHEST_PROTOCOL)
        with open(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))/Path("true_list.pkl"),'wb') as f:
            pickle.dump(testing_true_list,f,pickle.HIGHEST_PROTOCOL)

        # log to mlflow
        log_to_mlflow_artifacts(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))/Path("predictions.pkl"),"predictions_node")
        log_to_mlflow_artifacts(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))/Path("true_list.pkl"),"true_list_node")
        
    # save neural ODE weights out
    def save_NODE_weights(self):
        """
        Save the trained Neural ODE model to disk.
        
        This method serializes the Neural ODE object using pickle
        and saves it to the configured model output directory.
        """

        with open(Path(self.config_handler.get_config_status("neural_ode.saving.model_output_dir"))/Path(self.config_handler.get_config_status("neural_ode.saving.load_path")),'wb') as f:

            pickle.dump(self.NODE_object,f,pickle.HIGHEST_PROTOCOL)

    def load_NODE_weights(self):
        """
        Load pre-trained Neural ODE model from disk.
        
        This method deserializes the Neural ODE object from pickle
        files and loads it into the current instance.
        """
        
        with open(Path(self.config_handler.get_config_status("neural_ode.saving.model_output_dir"))/Path(self.config_handler.get_config_status("neural_ode.saving.load_path")),'rb') as f:

            self.NODE_object=pickle.load(f)

    def save_predictions(self,predictions_list:dict[str,list[np.ndarray]],true_list:dict[str,list[np.ndarray]]):
        """
        Save prediction results to disk.
        
        This method saves both predicted and true values to pickle files
        for later analysis and visualization.
        
        Args:
            predictions_list (dict[str, list[np.ndarray]]): Dictionary containing predicted values
            true_list (dict[str, list[np.ndarray]]): Dictionary containing true values
        """

        with open(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))/Path("predictions.pkl"),'wb') as f:

            pickle.dump(predictions_list,f,pickle.HIGHEST_PROTOCOL)
        
        with open(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))/Path("true_list.pkl"),'wb') as f:

            pickle.dump(true_list,f,pickle.HIGHEST_PROTOCOL)

    def load_predictions(self):
        """
        Load previously saved prediction results from disk.
        
        This method loads prediction results from pickle files
        for analysis or visualization.
        
        Returns:
            dict: Loaded prediction results
        """

        with open(Path(self.config_handler.get_config_status("neural_ode.testing.predictions_output_dir"))/Path(self.config_handler.get_config_status("neural_ode.testing.predictions_output_path")),'rb') as f:

            return pickle.load(f)

    def visualize_results(self):
        """
        Create visualization plots for test results.
        
        This method generates comparison plots between predicted and true values
        for each test trajectory and input feature. Plots are saved to the
        configured visualization directory.
        """

        viz_dir=Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))/Path(self.config_handler.get_config_status("neural_ode.testing.visualization.save_dir"))
        if os.path.isdir(viz_dir):
            self.logging_manager.log(f"Removing existing visualization directory: {viz_dir}")
            shutil.rmtree(viz_dir)

        os.mkdir(viz_dir)

        with open(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))/Path("predictions.pkl"),'rb') as f:
            prediction_lists=pickle.load(f)
        with open(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))/Path("true_list.pkl"),'rb') as f:
            true_lists=pickle.load(f)

        # loop over saved trajectories and return comparison plots for each species and temperature
        for i_traj in range(len(prediction_lists['feature_list_test'])):
            i_traj_viz_dir=viz_dir/Path(f"traj_{i_traj}")
            os.mkdir(i_traj_viz_dir)

            num_inputs=self.test_constants['num_inputs']

            for i_input in range(num_inputs):
                plt.plot(prediction_lists['times_list_test'][i_traj],
                         prediction_lists['feature_list_test'][i_traj][:,i_input],label='predicted')
                plt.plot(true_lists['times_list_test'][i_traj],
                         true_lists['feature_list_test'][i_traj][:,i_input],label='true')
                plt.legend()
                plt.grid()
                plt.xlabel('Independent Variable')
                plt.ylabel('Dependent Variable')
                plt.title(f'Input No {i_input}')
                if 'xscale' in self.config_handler.get_config_status("neural_ode.testing.visualization.settings").keys():
                    plt.xscale(self.config_handler.get_config_status("neural_ode.testing.visualization.settings")['xscale'])
                if 'yscale' in self.config_handler.get_config_status("neural_ode.testing.visualization.settings").keys():
                    plt.yscale(self.config_handler.get_config_status("neural_ode.testing.visualization.settings")['yscale'])
                plt.savefig(i_traj_viz_dir/Path(f"input_{i_input}.png"))
                plt.close()

        # log to mlflow
        log_to_mlflow_artifacts(viz_dir,"visualization_test_node")