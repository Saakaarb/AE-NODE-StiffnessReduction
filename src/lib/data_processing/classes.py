import numpy as np
import os
import h5py
import random
from src.utils.helper_functions import process_raw_data,standard_score_norm,divide_range_random
from src.utils.classes import ConfigReader
import mlflow
import jax

# This class loads and pre processes data to run the experiments

class Data_Processing():

    def __init__(self, config_handler:ConfigReader)->None:
        """
        Initialize the Data_Processing class with configuration settings.
        
        Args:
            config_handler (ConfigReader): Configuration reader object containing data processing parameters
            
        Returns:
            None: Initializes the Data_Processing object
        """
        self.config_handler=config_handler

        # list objects that are created during pre processing
        self.times_list_train=[]
        self.feature_list_train=[]
        self.inputs_list_train=[]

        self.times_list_test=[]
        self.feature_list_test=[]
        self.inputs_list_test=[]

        # max trajectory size
        self.max_train_traj_size=0 # largest trajectory size (number of entries) for training data
        self.max_test_traj_size=0 # largest trajectory size (number of entries) for testing data

        # number of timesteps of each trajectory
        self.num_timesteps_each_traj_train=None # number of timesteps for each trajectory in training data
        self.num_timesteps_each_traj_test=None # number of timesteps for each trajectory in testing data

        # number of trajectories
        self.num_train_traj=None # number of trajectories in training data
        self.num_test_traj=None # number of trajectories in testing data

        # input and output statistics for norm
        self.mean_vals_inp=None # mean of inputs (Nsp+1) [c1,c2...,Temp]
        self.std_vals_inp=None # standard deviation of inputs (Nsp+1)
        self.mean_vals_out=None # mean of outputs (Nsp)
        self.std_vals_out=None # standard deviation of outputs (Nsp)

        # assimilated arrays
        self.all_input_data_train=None #Accumulated, normalized input training data of shape: [num_train_traj,max_train_traj_size,num_inputs]
        self.all_time_data_train=None #Accumulated time training data of shape: [num_train_traj,max_train_traj_size]
        self.start_end_time_data_train=None # start and end time of each trajectory in training data of shape: [num_train_traj,2]
        self.initial_condition_data_train=None # initial condition of each trajectory in training data of shape: [num_train_traj,num_inputs]

        self.all_input_data_test=None #Accumulated, normalized input testing data of shape: [num_test_traj,max_test_traj_size,num_inputs]
        self.all_time_data_test=None #Accumulated time testing data of shape: [num_test_traj,max_test_traj_size]
        self.start_end_time_data_test=None # start and end time of each trajectory in testing data of shape: [num_test_traj,2]
        self.initial_condition_data_test=None # initial condition of each trajectory in testing data of shape: [num_test_traj,num_inputs]

        # masks used during training
        self.recon_mask_train=None # mask for encoder-decoder reconstruction loss of shape: [num_train_traj,max_train_traj_size,num_inputs]
        self.latent_space_mask_train=None # mask for latent space loss of shape: [num_train_traj,max_train_traj_size,n_latent_space]

        self.cond_1_mask_train=None # mask for condition number regularization used for stiffness reduction of latent space of shape: [num_train_traj,max_train_traj_size-2,n_latent_space]
        self.cond_2_mask_train=None # mask for condition number regularization used for stiffness reduction of latent space of shape: [num_train_traj,max_train_traj_size-1,n_latent_space]
        self.all_time_data_broadcasted_train=None # mask for all time data of shape: [num_train_traj,max_train_traj_size,n_latent_space]

        # load data file lists
        self.load_data_files()

        self.preprocess_data()

    def preprocess_data(self)->None:
        """
        Main data preprocessing pipeline.
        
        Either loads preprocessed data from disk or performs full preprocessing including 
        training/testing data preparation, data assimilation, mask construction, and optional data saving.
        
        Args:
            None: Uses instance variables
            
        Returns:
            None: Modifies instance variables
        """
        

        # data preprocess
        #------------------------------------------------------------------
        # lists for important information
        self.num_timesteps_each_traj_train=np.zeros(len(self.train_data_files_list),dtype=int)
        self.num_timesteps_each_traj_test=np.zeros(len(self.test_data_files_list),dtype=int)

        # detect devices
        self.device = jax.devices()[0] if jax.device_count() > 0 else jax.devices()[0]

        # load saved data
        if self.config_handler.get_config_status("data_processing.saving_loading.load_data"):
            print("Loading data...")
            self.load_data()

        # prepare data and save it
        else:
            print("Preprocessing data...")
            # prepare training data
            self.prepare_training_data_lists()
            
            # prepare testing data
            self.prepare_testing_data_lists()
            
            # assimilate data array using obtained data
            self.assimilate_training_data_arrays()
            self.assimilate_testing_data_arrays()

            # construct masks that will be used for training
            # this is necessary because the training data is of different trajectory lengths
            # but has to be assimilated into a single array for training
            self.construct_masks_training_data()
            self.construct_masks_testing_data()
            print("Data preprocessing complete")
            # save data
            if self.config_handler.get_config_status("data_processing.saving_loading.save_data"):
                self.save_data()
                print("Data saved to disk")
        
        # divide loaded data into batches
        self.divide_data_into_batches()

    def prepare_training_data_lists(self)->None:
        """
        Process training data files.
        
        Loads raw data, applies preprocessing, computes normalization parameters from the first trajectory,
        standardizes inputs, and stores processed data in lists.
        
        Args:
            None: Uses instance variables
            
        Returns:
            None: Modifies instance variables
        """
        
        for i_traj,data_file in enumerate(self.train_data_files_list):
            data=np.genfromtxt(data_file,delimiter=',')

            #TODO this is a custom preprocessing function for the data
            # outputs: time_data(Nts,),feature_data(Nfeat,Nts)
            time_data,feature_data=process_raw_data(data,self.config_handler)

            if self.config_handler.get_config_status("data_processing.check_raw_data_shape"):
                pass

            # record time steps present for each trajectory
            # used to construct masks later
            self.num_timesteps_each_traj_train[i_traj]=time_data.shape[0]

            # update largest trajectory size
            if self.max_train_traj_size<self.num_timesteps_each_traj_train[i_traj]:
                self.max_train_traj_size=int(self.num_timesteps_each_traj_train[i_traj])


            # get (approx) standard score normalization parameters
            if i_traj==0:
                mean_vals_inp,std_vals_inp,mean_vals_out,std_vals_out=standard_score_norm(time_data,feature_data)
                self.mean_vals_inp=mean_vals_inp
                self.std_vals_inp=std_vals_inp
                self.mean_vals_out=mean_vals_out
                self.std_vals_out=std_vals_out
                print("Warning: using only the first trajectory to get standard score normalization parameters")
                self.end_time=time_data[-1]
                self.num_inputs=feature_data.shape[0]
                self.latent_scaling=np.ones(int(self.config_handler.get_config_status("data_processing.latent_space_dim")))
                # inferred from data
                self.config_handler.set_config_status("data_processing.num_inputs",self.num_inputs)
            else:
                if time_data[-1]!=self.end_time:
                    raise ValueError("End time is not the same for all trajectories!")
                if feature_data.shape[0]!=self.num_inputs:
                    raise ValueError("Number of inputs is not the same for all trajectories!")

            network_input=(feature_data-np.expand_dims(self.mean_vals_inp,axis=1))/np.expand_dims(self.std_vals_inp,axis=1)            


            self.times_list_train.append(time_data)
            self.feature_list_train.append(feature_data)
            self.inputs_list_train.append(network_input)

        self.num_train_traj=len(self.train_data_files_list)


    def prepare_testing_data_lists(self)->None:
        """
        Process testing data files.
        
        Loads raw data, applies preprocessing, standardizes inputs using training normalization parameters,
        and stores processed data in lists.
        
        Args:
            None: Uses instance variables
            
        Returns:
            None: Modifies instance variables
        """
        
    # testing data preparation

        for i_traj,data_file in enumerate(self.test_data_files_list):
            data=np.genfromtxt(data_file,delimiter=',')

            #TODO this is a custom preprocessing function for the data
            time_data,feature_data=process_raw_data(data,self.config_handler)
            self.num_timesteps_each_traj_test[i_traj]=time_data.shape[0]

            if self.max_test_traj_size<self.num_timesteps_each_traj_test[i_traj]:
                self.max_test_traj_size=int(self.num_timesteps_each_traj_test[i_traj])      

            
            network_input=(feature_data-np.expand_dims(self.mean_vals_inp,axis=1))/np.expand_dims(self.std_vals_inp,axis=1)            


            self.times_list_test.append(time_data)
            self.feature_list_test.append(feature_data)
            self.inputs_list_test.append(network_input)

        self.num_test_traj=len(self.test_data_files_list)

    def assimilate_training_data_arrays(self)->None:
        """
        Convert training data lists into 3D numpy arrays for efficient batch processing.
        
        Handles variable trajectory lengths by padding with end time values.
        
        Args:
            None: Uses instance variables
            
        Returns:
            None: Modifies instance variables
        """
        
        all_input_data=np.zeros([self.num_train_traj,self.max_train_traj_size,self.num_inputs])
        all_time_data=np.zeros([self.num_train_traj,self.max_train_traj_size])
        start_end_time_data=np.zeros([self.num_train_traj,2])
        initial_condition_data=np.zeros([self.num_train_traj,self.num_inputs])


        for i_traj in range(self.num_train_traj):

            all_input_data[i_traj,:int(self.num_timesteps_each_traj_train[i_traj]),:]=self.inputs_list_train[i_traj].T
            all_time_data[i_traj,:int(self.num_timesteps_each_traj_train[i_traj])]=self.times_list_train[i_traj]
            all_time_data[i_traj,int(self.num_timesteps_each_traj_train[i_traj]):]=self.end_time#times_list[i_traj]

            start_end_time_data[i_traj,0]=self.times_list_train[i_traj][0]
            start_end_time_data[i_traj,1]=self.times_list_train[i_traj][-1]
            initial_condition_data[i_traj,:]=self.inputs_list_train[i_traj][:,0]

        self.all_input_data_train=all_input_data
        self.all_time_data_train=all_time_data
        self.start_end_time_data_train=start_end_time_data
        self.initial_condition_data_train=initial_condition_data


    def assimilate_testing_data_arrays(self)->None:
        """
        Convert testing data lists into 3D numpy arrays for efficient batch processing.
        
        Handles variable trajectory lengths by padding with end time values.
        
        Args:
            None: Uses instance variables
            
        Returns:
            None: Modifies instance variables
        """

        all_input_data=np.zeros([self.num_test_traj,self.max_test_traj_size,self.num_inputs])
        all_time_data=np.zeros([self.num_test_traj,self.max_test_traj_size])
        start_end_time_data=np.zeros([self.num_test_traj,2])
        initial_condition_data=np.zeros([self.num_test_traj,self.num_inputs])
        
        for i_traj in range(self.num_test_traj):

            all_input_data[i_traj,:int(self.num_timesteps_each_traj_test[i_traj]),:]=self.inputs_list_test[i_traj].T
            all_time_data[i_traj,:int(self.num_timesteps_each_traj_test[i_traj])]=self.times_list_test[i_traj]
            all_time_data[i_traj,int(self.num_timesteps_each_traj_test[i_traj]):]=self.end_time#times_list[i_traj]

            start_end_time_data[i_traj,0]=self.times_list_test[i_traj][0]
            start_end_time_data[i_traj,1]=self.times_list_test[i_traj][-1]
            initial_condition_data[i_traj,:]=self.inputs_list_test[i_traj][:,0]

        self.all_input_data_test=all_input_data
        self.all_time_data_test=all_time_data
        self.start_end_time_data_test=start_end_time_data
        self.initial_condition_data_test=initial_condition_data
        

    

    def load_data_files(self):
        """
        Load raw data file paths and split into training/testing sets.
        
        Loads raw data file paths from the configured directory, splits them into training 
        and testing sets based on the train split ratio, and stores the file lists.
        
        Args:
            None: Uses config_handler
            
        Returns:
            None: Modifies instance variables
        """
        file_path=self.config_handler.get_config_status("data_processing.saving_loading.raw_data_path")
        data_files_list=[os.path.join(file_path, f) for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]

        # train_test_split
        train_split_ratio=self.config_handler.get_config_status("data_processing.train_split_ratio")

        # select x% randomly
        self.train_data_files_list=np.random.choice(data_files_list,size=int(len(data_files_list)*train_split_ratio),replace=False)
        self.test_data_files_list=np.setdiff1d(data_files_list,self.train_data_files_list)


    def construct_masks_training_data(self)->None:
        """
        Create binary masks for training data to handle variable trajectory lengths.
        
        Creates reconstruction masks, latent space masks, condition number regularization masks,
        and broadcasted time data for JAX compatibility.
        
        Args:
            None: Uses instance variables
            
        Returns:
            None: Modifies instance variables
        """
        
        n_latent_space=self.config_handler.get_config_status("data_processing.latent_space_dim")

        # construct masks for training data
        # since the trajectories are of variable length, we discard the predictions beyond the end timestep
        # this is done by multiplying with a zero mask in the loss calculation

        # mask for encoder-decoder reconstruction loss
        self.recon_mask_train=np.ones_like(self.all_input_data_train)

        # mask for latent space loss
        self.latent_space_mask_train=np.ones([self.num_train_traj,self.max_train_traj_size,n_latent_space])

        # mask for condition number regularization used for stiffness reduction of latent space
        # this will be calculated as per the reference:
        #Stiffness-Reduced Neural ODE Models for Data-Driven Reduced-Order Modeling of Combustion Chemical Kinetics: Dikeman, Zhang and Yang (2022)
        # -2 is since the computations are derivatives
        self.cond_1_mask_train=np.ones([self.num_train_traj,self.max_train_traj_size-2,n_latent_space])
        self.cond_2_mask_train=np.ones([self.num_train_traj,self.max_train_traj_size-2,n_latent_space])

        # required so that lax.scan works
        self.all_time_data_broadcasted_train=np.ones([self.num_train_traj,self.max_train_traj_size,n_latent_space])

        for i_traj in range(self.num_train_traj):

            self.recon_mask_train[i_traj,int(self.num_timesteps_each_traj_train[i_traj]):,:]=0.0
            self.latent_space_mask_train[i_traj,int(self.num_timesteps_each_traj_train[i_traj]):,:]=0.0
            self.cond_1_mask_train[i_traj,(int(self.num_timesteps_each_traj_train[i_traj])-2):,:]=0.0
            self.cond_2_mask_train[i_traj,(int(self.num_timesteps_each_traj_train[i_traj])-1):,:]=0.0

        for i_latent in range(n_latent_space):

            self.all_time_data_broadcasted_train[:,:,i_latent]=self.all_time_data_train

    def construct_masks_testing_data(self)->None:
        """
        Create binary masks for testing data to handle variable trajectory lengths.
        
        Creates reconstruction masks and latent space masks for testing data.
        
        Args:
            None: Uses instance variables
            
        Returns:
            None: Modifies instance variables
        """

        n_latent_space=self.config_handler.get_config_status("data_processing.latent_space_dim")

        # mask for encoder-decoder reconstruction loss
        self.recon_mask_test=np.ones_like(self.all_input_data_test)

        # mask for latent space loss
        self.latent_space_mask_test=np.ones([self.num_test_traj,self.max_test_traj_size,n_latent_space])

        for i_traj in range(self.num_test_traj):

            self.recon_mask_test[i_traj,int(self.num_timesteps_each_traj_test[i_traj]):,:]=0.0
            self.latent_space_mask_test[i_traj,int(self.num_timesteps_each_traj_test[i_traj]):,:]=0.0


    def get_raw_training_data(self)->dict[str,list]:
        """
        Get raw preprocessed training data in list format.
        
        Returns the raw preprocessed training data in list format before assimilation into arrays.
        
        Args:
            None
            
        Returns:
            dict[str, list]: Dictionary containing raw training data lists (times, species, temperatures, inputs)
        """
        
        return {
            "times_list_train":self.times_list_train,
            "feature_list_train":self.feature_list_train,
            "inputs_list_train":self.inputs_list_train,
        }

    def get_raw_testing_data(self)->dict[str,list]:
        """
        Get raw preprocessed testing data in list format.
        
        Returns the raw preprocessed testing data in list format before assimilation into arrays.
        
        Args:
            None
            
        Returns:
            dict[str, list]: Dictionary containing raw testing data lists (times, species, temperatures, inputs)
        """

        return {
            "times_list_test":self.times_list_test,
            "feature_list_test":self.feature_list_test,
            "inputs_list_test":self.inputs_list_test,
        }

    def get_training_constants(self)->dict:
        """
        Get training data constants and metadata.
        
        Returns training data constants including trajectory counts, dimensions, 
        normalization parameters, and configuration values.
        
        Args:
            None
            
        Returns:
            dict: Dictionary containing training data constants and metadata
        """

        constants_dict={
            "num_train_traj":self.num_train_traj,
            "max_train_traj_size":self.max_train_traj_size,
            "num_inputs":self.num_inputs,
            "n_latent_space":self.config_handler.get_config_status("data_processing.latent_space_dim"),
            "num_timesteps_each_traj_train":self.num_timesteps_each_traj_train,
            "mean_vals_inp":self.mean_vals_inp, # mean of inputs (Nsp+1) [c1,c2...,Temp]
            "std_vals_inp":self.std_vals_inp, # standard deviation of inputs (Nsp+1)
            "mean_vals_out":self.mean_vals_out, # mean of outputs (Nsp)
            "std_vals_out":self.std_vals_out, # standard deviation of outputs (Nsp)
            "end_time":self.end_time,
            "latent_scaling":self.latent_scaling,
        }
        return constants_dict

    def get_testing_constants(self)->dict:
        """
        Get testing data constants and metadata.
        
        Returns testing data constants including trajectory counts, dimensions, 
        normalization parameters, and configuration values.
        
        Args:
            None
            
        Returns:
            dict: Dictionary containing testing data constants and metadata
        """
        constants_dict={
            "num_test_traj":self.num_test_traj,
            "max_test_traj_size":self.max_test_traj_size,
            "num_inputs":self.num_inputs,
            "n_latent_space":self.config_handler.get_config_status("data_processing.latent_space_dim"),
            "num_timesteps_each_traj_test":self.num_timesteps_each_traj_test,
            "mean_vals_inp":self.mean_vals_inp, # mean of inputs (Nsp+1) [c1,c2...,Temp]
            "std_vals_inp":self.std_vals_inp, # standard deviation of inputs (Nsp+1)
            "mean_vals_out":self.mean_vals_out, # mean of outputs (Nsp)
            "std_vals_out":self.std_vals_out, # standard deviation of outputs (Nsp)
            "end_time":self.end_time,
            "latent_scaling":self.latent_scaling,
        }
        return constants_dict

    def divide_data_into_batches(self,)->None:
        """
        Randomly sample a subset of training trajectories.
        
        Randomly samples a subset of training trajectories and returns the corresponding 
        data arrays and masks for batch training.
        
        Args:
            num_samples (int): Number of trajectories to sample
            
        Returns:
            None: Modifies instance variables
        """

        self.data_samples_train=[]
        

        num_samples_per_batch=min(self.config_handler.get_config_status("data_processing.num_samples_per_batch"),self.num_train_traj)


        #generate num_samples_per_batch sample index groups and sample data, without replacament
        sample_indices_groups=divide_range_random(0,self.num_train_traj,num_samples_per_batch)
         
        for sample_indices in sample_indices_groups:

            # get all relevant training data subsets and masks
            input_data_subset=self.all_input_data_train[sample_indices]
            time_data_subset=self.all_time_data_train[sample_indices]
            start_end_time_data_subset=self.start_end_time_data_train[sample_indices]
            initial_condition_data_subset=self.initial_condition_data_train[sample_indices]
            recon_mask_subset=self.recon_mask_train[sample_indices]

            latent_space_mask_subset=self.latent_space_mask_train[sample_indices]
            cond_1_mask_subset=self.cond_1_mask_train[sample_indices]
            cond_2_mask_subset=self.cond_2_mask_train[sample_indices]
            all_time_data_broadcasted_subset=self.all_time_data_broadcasted_train[sample_indices]

            
            subset_dict={
                "input_data":self._place_on_device(input_data_subset),
                "time_data":self._place_on_device(time_data_subset),
                "start_end_time_data":self._place_on_device(start_end_time_data_subset),
                "initial_condition_data":self._place_on_device(initial_condition_data_subset),
                "recon_mask":self._place_on_device(recon_mask_subset),
                "latent_space_mask":self._place_on_device(latent_space_mask_subset),
                "cond_1_mask":self._place_on_device(cond_1_mask_subset),
                "cond_2_mask":self._place_on_device(cond_2_mask_subset),
                "all_time_data_broadcasted":self._place_on_device(all_time_data_broadcasted_subset)
            }

            self.data_samples_train.append(subset_dict)
        
        

    def get_test_data(self)->dict[str,np.ndarray]:
        """
        Get all testing data in the same format as training data.
        
        Returns all testing data in the same format as training data for evaluation purposes.
        
        Args:
            None
            
        Returns:
            dict[str, np.ndarray]: Dictionary containing all testing data arrays and masks
        """

        test_data_dict={
            "input_data":self.all_input_data_test,
            "time_data":self.all_time_data_test,
            "start_end_time_data":self.start_end_time_data_test,
            "initial_condition_data":self.initial_condition_data_test,
            "recon_mask":self.recon_mask_test,
            "latent_space_mask":self.latent_space_mask_test,
            
        }
        return test_data_dict

    def save_data(self):
        """
        Save all training and testing data to HDF5 files for DVC version control.
        
        Saves all processed training and testing data, masks, and metadata to HDF5 files 
        with compression for efficient storage and DVC version control.
        
        Args:
            None: Uses instance variables
            
        Returns:
            None: Saves files to disk
        """
        
        # Create data directory if it doesn't exist
        data_dir = self.config_handler.get_config_status("data_processing.saving_loading.processed_data_path")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save training data
        with h5py.File(os.path.join(data_dir, 'training_data.h5'), 'w') as f:
            # Main data arrays
            f.create_dataset('all_input_data_train', data=self.all_input_data_train, 
                            compression='gzip', compression_opts=9)
            f.create_dataset('all_time_data_train', data=self.all_time_data_train, 
                            compression='gzip', compression_opts=9)
            f.create_dataset('start_end_time_data_train', data=self.start_end_time_data_train)
            f.create_dataset('initial_condition_data_train', data=self.initial_condition_data_train)
            
            # Masks
            f.create_dataset('recon_mask_train', data=self.recon_mask_train, 
                            compression='gzip', compression_opts=9)
            f.create_dataset('latent_space_mask_train', data=self.latent_space_mask_train, 
                            compression='gzip', compression_opts=9)
            f.create_dataset('cond_1_mask_train', data=self.cond_1_mask_train, 
                            compression='gzip', compression_opts=9)
            f.create_dataset('cond_2_mask_train', data=self.cond_2_mask_train, 
                            compression='gzip', compression_opts=9)
            f.create_dataset('all_time_data_broadcasted_train', data=self.all_time_data_broadcasted_train, 
                            compression='gzip', compression_opts=9)
            
            # Metadata
            f.attrs['num_train_traj'] = self.num_train_traj
            f.attrs['max_train_traj_size'] = self.max_train_traj_size
            f.attrs['num_inputs'] = self.num_inputs
            f.attrs['end_time'] = self.end_time
            
            # Normalization parameters
            f.create_dataset('mean_vals_inp', data=self.mean_vals_inp)
            f.create_dataset('std_vals_inp', data=self.std_vals_inp)
            f.create_dataset('mean_vals_out', data=self.mean_vals_out)
            f.create_dataset('std_vals_out', data=self.std_vals_out)
            f.create_dataset('latent_scaling', data=self.latent_scaling)
            
            # Trajectory info
            f.create_dataset('num_timesteps_each_traj_train', data=self.num_timesteps_each_traj_train)
        
        # Save testing data
        with h5py.File(os.path.join(data_dir, 'testing_data.h5'), 'w') as f:
            f.create_dataset('all_input_data_test', data=self.all_input_data_test, 
                            compression='gzip', compression_opts=9)
            f.create_dataset('all_time_data_test', data=self.all_time_data_test, 
                            compression='gzip', compression_opts=9)
            f.create_dataset('start_end_time_data_test', data=self.start_end_time_data_test)
            f.create_dataset('initial_condition_data_test', data=self.initial_condition_data_test)
            
            # Testing metadata
            f.attrs['num_test_traj'] = self.num_test_traj
            f.attrs['max_test_traj_size'] = self.max_test_traj_size
            f.attrs['num_inputs'] = self.num_inputs
            f.attrs['end_time'] = self.end_time
            
            # Same normalization parameters (shared between train/test)
            f.create_dataset('mean_vals_inp', data=self.mean_vals_inp)
            f.create_dataset('std_vals_inp', data=self.std_vals_inp)
            f.create_dataset('mean_vals_out', data=self.mean_vals_out)
            f.create_dataset('std_vals_out', data=self.std_vals_out)
            f.create_dataset('latent_scaling', data=self.latent_scaling)
            
            f.create_dataset('num_timesteps_each_traj_test', data=self.num_timesteps_each_traj_test)
        
            # save masks
            f.create_dataset('recon_mask_test', data=self.recon_mask_test, 
                            compression='gzip', compression_opts=9)
            f.create_dataset('latent_space_mask_test', data=self.latent_space_mask_test, 
                            compression='gzip', compression_opts=9)

        # log to mlflow
        mlflow.log_artifact(os.path.join(data_dir, 'training_data.h5'), "training_data")
        mlflow.log_artifact(os.path.join(data_dir, 'testing_data.h5'), "testing_data")

        print(f"Data saved to {data_dir}/")

    def sample_training_data(self,)->dict[str,np.ndarray]:
        """
        Divide data into batches.
        
        Args:
            
        """
        return random.choice(self.data_samples_train)

    def load_data(self):
        """
        Load training and testing data from HDF5 files.
        
        Loads preprocessed training and testing data, masks, and metadata from HDF5 files 
        into instance variables.
        
        Args:
            None: Uses config_handler
            
        Returns:
            None: Modifies instance variables
        """
        
        data_dir = self.config_handler.get_config_status("data_processing.saving_loading.processed_data_path")
        
        # Load training data
        with h5py.File(os.path.join(data_dir, 'training_data.h5'), 'r') as f:
            self.all_input_data_train = f['all_input_data_train'][:]
            self.all_time_data_train = f['all_time_data_train'][:]
            self.start_end_time_data_train = f['start_end_time_data_train'][:]
            self.initial_condition_data_train = f['initial_condition_data_train'][:]
            
            self.recon_mask_train = f['recon_mask_train'][:]
            self.latent_space_mask_train = f['latent_space_mask_train'][:]
            self.cond_1_mask_train = f['cond_1_mask_train'][:]
            self.cond_2_mask_train = f['cond_2_mask_train'][:]
            self.all_time_data_broadcasted_train = f['all_time_data_broadcasted_train'][:]
            
            # Load metadata
            self.num_train_traj = f.attrs['num_train_traj']
            self.max_train_traj_size = f.attrs['max_train_traj_size']
            self.num_inputs = f.attrs['num_inputs']
            self.end_time = f.attrs['end_time']
            
            # Load normalization parameters
            self.mean_vals_inp = f['mean_vals_inp'][:]
            self.std_vals_inp = f['std_vals_inp'][:]
            self.mean_vals_out = f['mean_vals_out'][:]
            self.std_vals_out = f['std_vals_out'][:]
            self.latent_scaling = f['latent_scaling'][:]
            
            self.num_timesteps_each_traj_train = f['num_timesteps_each_traj_train'][:]
        
        # Load testing data
        with h5py.File(os.path.join(data_dir, 'testing_data.h5'), 'r') as f:
            self.all_input_data_test = f['all_input_data_test'][:]
            self.all_time_data_test = f['all_time_data_test'][:]
            self.start_end_time_data_test = f['start_end_time_data_test'][:]
            self.initial_condition_data_test = f['initial_condition_data_test'][:]

            self.recon_mask_test = f['recon_mask_test'][:]
            self.latent_space_mask_test = f['latent_space_mask_test'][:]
            
            self.num_test_traj = f.attrs['num_test_traj']
            self.max_test_traj_size = f.attrs['max_test_traj_size']
            self.num_timesteps_each_traj_test = f['num_timesteps_each_traj_test'][:]
        
        # log to mlflow
        mlflow.log_artifact(os.path.join(data_dir, 'training_data.h5'), "training_data")
        mlflow.log_artifact(os.path.join(data_dir, 'testing_data.h5'), "testing_data")

        print(f"Data loaded from {data_dir}/")
    
    def _place_on_device(self,data):
        """Place numpy array on GPU device and convert to JAX array"""
        if isinstance(data, np.ndarray):
            return jax.device_put(data, self.device)
        return data 