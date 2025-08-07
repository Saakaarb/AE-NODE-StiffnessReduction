import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import optax
from functools import partial
import pickle
import os
from src.utils.helper_functions import _forward_pass,log_to_mlflow_artifacts,log_to_mlflow_metrics
from src.utils.classes import ConfigReader,MLP
from src.lib.data_processing.classes import Data_Processing
from src.lib.autoencoder.classes import Encoder_Decoder
from diffrax import RESULTS
import matplotlib.pyplot as plt
from pathlib import Path
import time
import shutil


jax.config.update("jax_enable_x64", True)


##########################################################

@jax.jit
def _ode_fn(t,state,other_args):

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
    derivatives=jnp.squeeze(_forward_pass(jnp.divide(state,scaling),trainable_variables_NODE['NODE']))*(1.0/constants['end_time']) # scaling included
    #jax.debug.print("i_traj: {z}, t:{y} derivatives:{x}",z=i_traj,y=t,x=derivatives)
    return jnp.squeeze(derivatives)

@jax.jit
def _integrate_NODE(constants,trainable_variables_NODE,enc_dec_weights,data_dict,i_traj):

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

    saveat = diffrax.SaveAt(ts=curr_time_data)

    term = diffrax.ODETerm(_ode_fn)
    #rtol=jnp.array([1E-2,1E-3])

    # TODO try batched euler within diffrax diffeqsolve
    solution = diffrax.diffeqsolve(term,diffrax.Dopri5(),t0=t_init,t1=t_final,dt0 = init_dt,y0=y_latent_init,
                                    saveat=saveat,args={'constants':constants,'trainable_variables_NODE':trainable_variables_NODE,'i_traj':i_traj},throw=False,
                                    max_steps=100000,stepsize_controller=diffrax.PIDController(pcoeff=pcoeff,icoeff=icoeff,rtol=rtol, atol=atol,dtmin=dtmin))
    
    #solution = diffrax.diffeqsolve(term,diffrax.Kvaerno5(),t0=t_init,t1=t_final,dt0 = 1e-11,y0=y_latent_init,
    #                                saveat=saveat,args={'constants':constants,'trainable_variables_NODE':trainable_variables_NODE,'i_traj':i_traj},throw=False,
    #                                max_steps=100000,stepsize_controller=diffrax.PIDController(pcoeff=0.3,icoeff=0.4,rtol=1e-6, atol=1e-8,dtmin=None))
    
    return solution


@partial(jax.jit,static_argnums=(4,))
def _loss_fn_NODE(constants,trainable_variables_NODE,enc_dec_weights,data_dict,num_traj):
  

        loss_l1=0.0
        loss_l3=0.0
       
        # masks
        recon_mask=data_dict['recon_mask']
        latent_space_mask=data_dict['latent_space_mask']
        input_data=data_dict['input_data']
        
        loss_comp_success=jnp.ones(num_traj)
        #if True:
        def loss_fn(carry,i_traj):
            
            loss_l1,loss_l3,loss_comp_success_itraj=carry

            recon_mask_curr=recon_mask[i_traj,:,:]
            latent_space_mask_curr=latent_space_mask[i_traj,:,:]
            phys_data=input_data[i_traj,:,:]

            solution=_integrate_NODE(constants,trainable_variables_NODE,enc_dec_weights,data_dict,i_traj)

            failed = jnp.logical_or(solution.result == RESULTS.max_steps_reached, solution.result==RESULTS.singular)

            # if integration failed, set loss_comp_success_itraj to 0
            loss_comp_success_itraj=loss_comp_success_itraj.at[i_traj].set(jnp.where(failed,0.0,1.0))

            #SECTION L1 (ENC+NODE+DEC)
            #---------------------------------------------------------------------------------
           
            # predicted output in latent space
            latent_space_pred=jnp.squeeze(solution.ys)

            # prediction after integration
            phys_space_pred_int=_forward_pass(latent_space_pred,enc_dec_weights['decoder'])

            
            loss_L1=jnp.where(failed,1E5,jnp.sqrt(jnp.mean(jnp.square(jnp.multiply(phys_space_pred_int,recon_mask_curr)-jnp.multiply(phys_data,recon_mask_curr)))))


            #---------------------------------------------------------------------------------
            #SECTION L2 (ENC+DEC)
            #---------------------------------------------------------------------------------
            # TODO only keep for simultaneous training
            #y_latent_all = _forward_pass(phys_data,enc_dec_weights['encoder'])    
            #phys_space_pred_data = _forward_pass(y_latent_all,enc_dec_weights['decoder'])
            
            # TODO employ filters
            #loss_L2= jnp.sqrt(jnp.mean(jnp.square(phys_data-phys_space_pred_data)))

            #SECTION L3 (ENC+NODE)
            #---------------------------------------------------------------------------------
            # TODO can be removed outside
            latent_space_truth=_forward_pass(phys_data,enc_dec_weights['encoder'])

            #TODO apply filters
            loss_L3=jnp.where(failed,1E5,jnp.sqrt(jnp.mean(jnp.square(jnp.multiply(latent_space_pred,latent_space_mask_curr)-jnp.multiply(latent_space_truth,latent_space_mask_curr)))))
            loss_l1+=loss_L1
            loss_l3+=loss_L3
            return (loss_l1,loss_l3,loss_comp_success_itraj),None

        losses,_=jax.lax.scan(loss_fn,(loss_l1,loss_l3,loss_comp_success),jnp.arange(num_traj))

        loss_l1=losses[0]
        loss_l3=losses[1]
        loss_comp_success=losses[2]

        return (loss_l1+loss_l3)/jnp.sum(loss_comp_success)

class Neural_ODE():

    def __init__(self,config_handler:ConfigReader,data_processing_handler:Data_Processing,encoder_decoder_handler:Encoder_Decoder):

        # init
        self.config_handler=config_handler
        self.data_processing_handler=data_processing_handler
        self.encoder_decoder_handler=encoder_decoder_handler

        # get training constants
        self.constants=self.data_processing_handler.get_training_constants()

        # get testing dict and constants
        self.test_data_dict=self.data_processing_handler.get_test_data()
        self.test_constants=self.data_processing_handler.get_testing_constants()

        # update neural ODE specs in dict
        node_spec_dict={'pcoeff':float(self.config_handler.get_config_status("neural_ode.ode_solver.pcoeff")),
                        'icoeff':float(self.config_handler.get_config_status("neural_ode.ode_solver.icoeff")),
                        'rtol':float(self.config_handler.get_config_status("neural_ode.ode_solver.rtol")),
                        'atol':float(self.config_handler.get_config_status("neural_ode.ode_solver.atol"))}
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
            print("Getting predictions for test data")
            self.test_NODE_model(self.encoder_decoder_handler.encoder_object.weights,self.encoder_decoder_handler.decoder_object.weights,self.NODE_object.weights)

            # visualize results
            if self.config_handler.get_config_status("neural_ode.testing.visualization.plot_results"):
                print("Visualizing results")
                self.visualize_results()

    def check_save_load_dirs(self):
        if not os.path.isdir(Path(self.config_handler.get_config_status("neural_ode.saving.model_output_dir"))):
            os.mkdir(Path(self.config_handler.get_config_status("neural_ode.saving.model_output_dir")))
        if not os.path.isdir(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))):
            os.mkdir(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir")))

    def _init_optimizer(self):

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

        # TODO add BFGS

    def _init_network(self):

        n_latent_space=self.config_handler.get_config_status("data_processing.latent_space_dim")
        hidden_size_NODE=self.config_handler.get_config_status("neural_ode.architecture.network_width")

        NODE_sizes=[[n_latent_space,hidden_size_NODE]]
        for i_layer in range(self.config_handler.get_config_status("neural_ode.architecture.num_layers")):
            NODE_sizes.append([hidden_size_NODE,hidden_size_NODE])
        NODE_sizes.append([hidden_size_NODE,n_latent_space])

        self.NODE_object=MLP(NODE_sizes,self.config_handler)
        self.NODE_object.initialize_network()
        
        self.trainable_variables_NODE={'NODE':self.NODE_object.weights}
        self.enc_dec_weights={'encoder':self.encoder_decoder_handler.encoder_object.weights,
                              'decoder':self.encoder_decoder_handler.decoder_object.weights}

        #if self.trainable_enc_dec:

        #self.trainable_variables_NODE.update({'encoder':self.encoder_decoder_handler.encoder_object.weights,
        #                                          'decoder':self.encoder_decoder_handler.decoder_object.weights})

    def _train_NODE(self):

        print("Training NODE...")
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
                print("Integration failed, stopping training.")
                break
            if i_step % self.print_freq == 0:
                t2=time.time()
                print(f"Training time for {self.print_freq} steps: {t2-t1} seconds")
                t1=time.time()

    def _train_step(self,opt_state,train_step):

        success=1
        # sample data
        data_dict=self.data_processing_handler.sample_training_data()

        num_traj=data_dict['input_data'].shape[0]
        

        # get value and grad
        #if self.trainable_enc_dec:
        #    value,grad_loss=jax.value_and_grad(self.loss_fn,argnums=(1,2),allow_int=True)(self.constants,self.trainable_variables_NODE,self.enc_dec_weights)

        #else:
        value,grad_loss=jax.value_and_grad(self.loss_fn,argnums=1,allow_int=True)(self.constants,self.trainable_variables_NODE,self.enc_dec_weights,data_dict,num_traj)

        if value>1E5/num_traj:
            print("Loss is too high, skipping update. This indicates failure to integrate.")
            success=0
            return opt_state,success
        
        #if self.trainable_enc_dec:
        #    grad_loss={'NODE':grad_loss[0]['NODE'],'encoder':grad_loss[1]['encoder'],'decoder':grad_loss[1]['decoder']}


        #compute update to trainable variable
        updates,opt_state=self.optimizer.update(grad_loss,opt_state)

        # get new value for trainable variable
        results=optax.apply_updates(self.trainable_variables_NODE,updates)

        # update values
        self.trainable_variables_NODE.update({'NODE':results['NODE']})

        if self.trainable_enc_dec:
            self.enc_dec_weights.update({'encoder':results['encoder'],'decoder':results['decoder']})

        
        if train_step % self.print_freq==0:
            #self.test_NODE_model(self.encoder_decoder_handler.encoder_object.weights,self.encoder_decoder_handler.decoder_object.weights,self.NODE_object.weights)
            
            test_loss=self.loss_fn(self.test_constants,self.trainable_variables_NODE,self.enc_dec_weights,self.test_data_dict,self.test_constants['num_test_traj'])

            self.training_loss_values.append(value)
            self.test_loss_values.append(test_loss)
            if value < self.best_training_loss:
                self.best_training_loss=value

            print(f"Step: {train_step}, training loss: {value}, test loss: {test_loss}, best training loss: {self.best_training_loss}, best test loss: {self.best_test_loss}")

            if test_loss < self.best_test_loss:
                self.best_test_loss=test_loss
                print(f"New best test loss: {test_loss}, recording weights")
                self.NODE_object.weights=self.trainable_variables_NODE['NODE']
                #self.encoder_decoder_handler.encoder_object.weights=self.trainable_variables_NODE['encoder']
                #self.encoder_decoder_handler.decoder_object.weights=self.trainable_variables_NODE['decoder']
        
            # log to mlflow
            log_to_mlflow_metrics({'node_training_loss':value,'node_test_loss':test_loss},train_step)
        #print(f"Step: {train_step}, training loss: {value}")

        #self.NODE_object.weights=self.trainable_variables_NODE['NODE']
        return opt_state,success


    def loss_fn(self,constants,trainable_variables_NODE,enc_dec_weights,data_dict,num_traj):

        return _loss_fn_NODE(constants,trainable_variables_NODE,enc_dec_weights,data_dict,num_traj)

    # predict solution trajectories
    def test_NODE_model(self,enc_weights,dec_weights,NODE_weights):


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
        testing_predictions_list={'times_list_test':[],'specie_list_test':[],'temps_list_test':[]}
        testing_true_list={'times_list_test':[],'specie_list_test':[],'temps_list_test':[]}

        # get mean and std of inputs
        # make shape (1,num_inputs) since each trajectory is computed separately
        mean_vals_inp=self.test_constants['mean_vals_inp'].reshape(1,-1)
        std_vals_inp=self.test_constants['std_vals_inp'].reshape(1,-1)

        for i_traj in range(num_test_traj):
            print(f"Predicting trajectory {i_traj+1} of {num_test_traj}")
            solution=_integrate_NODE(self.test_constants,node_weights_dict,enc_dec_weights,self.test_data_dict,i_traj) 

            # check if integration failed
            if solution.result==RESULTS.max_steps_reached or solution.result==RESULTS.singular:
                print(f"Integration failed for trajectory {i_traj+1} of {num_test_traj}")
                
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
            testing_predictions_list['specie_list_test'].append(pred_ys[i_traj,:num_timesteps_each_traj_test[i_traj],:-1])
            testing_predictions_list['temps_list_test'].append(pred_ys[i_traj,:num_timesteps_each_traj_test[i_traj],num_inputs-1])

            # append to true lists
            testing_true_list['times_list_test'].append(self.test_data_dict['time_data'][i_traj][:num_timesteps_each_traj_test[i_traj]])
            testing_true_list['specie_list_test'].append(true_ys[:num_timesteps_each_traj_test[i_traj],:-1])
            testing_true_list['temps_list_test'].append(true_ys[:num_timesteps_each_traj_test[i_traj],num_inputs-1])

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


        with open(Path(self.config_handler.get_config_status("neural_ode.saving.model_output_dir"))/Path(self.config_handler.get_config_status("neural_ode.saving.load_path")),'wb') as f:

            pickle.dump(self.NODE_object,f,pickle.HIGHEST_PROTOCOL)

    def load_NODE_weights(self):
        
        with open(Path(self.config_handler.get_config_status("neural_ode.saving.model_output_dir"))/Path(self.config_handler.get_config_status("neural_ode.saving.load_path")),'rb') as f:

            self.NODE_object=pickle.load(f)

    def save_predictions(self,predictions_list,true_list):

        with open(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))/Path("predictions.pkl"),'wb') as f:

            pickle.dump(predictions_list,f,pickle.HIGHEST_PROTOCOL)
        
        with open(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))/Path("true_list.pkl"),'wb') as f:

            pickle.dump(true_list,f,pickle.HIGHEST_PROTOCOL)

    def load_predictions(self):

        with open(Path(self.config_handler.get_config_status("neural_ode.testing.predictions_output_dir"))/Path(self.config_handler.get_config_status("neural_ode.testing.predictions_output_path")),'rb') as f:

            return pickle.load(f)

    def visualize_results(self):

        viz_dir=Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))/Path(self.config_handler.get_config_status("neural_ode.testing.visualization.save_dir"))
        if os.path.isdir(viz_dir):
            print(f"Removing existing visualization directory: {viz_dir}")
            shutil.rmtree(viz_dir)

        os.mkdir(viz_dir)

        with open(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))/Path("predictions.pkl"),'rb') as f:
            prediction_lists=pickle.load(f)
        with open(Path(self.config_handler.get_config_status("neural_ode.testing.save_dir"))/Path("true_list.pkl"),'rb') as f:
            true_lists=pickle.load(f)

        # loop over saved trajectories and return comparison plots for each species and temperature
        for i_traj in range(len(prediction_lists['specie_list_test'])):
            i_traj_viz_dir=viz_dir/Path(f"traj_{i_traj}")
            os.mkdir(i_traj_viz_dir)

            num_inputs=self.test_constants['num_inputs']

            for i_input in range(num_inputs-1):
                plt.plot(prediction_lists['times_list_test'][i_traj],
                         prediction_lists['specie_list_test'][i_traj][:,i_input],label='predicted')
                plt.plot(true_lists['times_list_test'][i_traj],
                         true_lists['specie_list_test'][i_traj][:,i_input],label='true')
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
            
            plt.plot(prediction_lists['times_list_test'][i_traj],
                     prediction_lists['temps_list_test'][i_traj],label='predicted')
            plt.plot(true_lists['times_list_test'][i_traj],
                     true_lists['temps_list_test'][i_traj],label='true')
            
            if 'xscale' in self.config_handler.get_config_status("neural_ode.testing.visualization.settings").keys():
                plt.xscale(self.config_handler.get_config_status("neural_ode.testing.visualization.settings")['xscale'])
            if 'yscale' in self.config_handler.get_config_status("neural_ode.testing.visualization.settings").keys():
                plt.yscale(self.config_handler.get_config_status("neural_ode.testing.visualization.settings")['yscale'])
            plt.savefig(i_traj_viz_dir/Path(f"input_{num_inputs-1}.png"))
            plt.close()

        # log to mlflow
        log_to_mlflow_artifacts(viz_dir,"visualization_test_node")