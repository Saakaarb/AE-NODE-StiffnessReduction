# from src.lib.data_processing.classes import Data_Processing
# from src.lib.autoencoder.classes import Encoder_Decoder
# from src.lib.NODE.classes import Neural_ODE
from src.utils.classes import ConfigReader,LoggingManager
from src.utils.helper_functions import log_to_mlflow,log_to_mlflow_artifacts
import os
import jax
from pathlib import Path
import mlflow

if __name__=="__main__":



    print(jax.devices())
    # roll all information into a constants dictionary
    constants={}
    data_dict={}

    config_path=Path("config/config.yml")

    config_handler=ConfigReader(str(config_path))

    logging_manager=LoggingManager()
    logging_manager.log("Config file read")

    if config_handler.get_config_status("neural_ode.training.precision")=='float64':
        jax.config.update("jax_enable_x64", True)
        logging_manager.log("Using float64 precision")
        assert jax.numpy.array(0.0).dtype=='float64', "Precision set to float64 but dtype is not float64"
        #logging_manager.log("Verify dtype:", jax.numpy.array(0.0).dtype)

    # import classes that use jax after setting precision
    from src.lib.data_processing.classes import Data_Processing
    from src.lib.autoencoder.classes import Encoder_Decoder
    from src.lib.NODE.classes import Neural_ODE

    # set up mlflow

    with mlflow.start_run():
        mlflow.set_tracking_uri("./mlruns")
        print("mlflow tracking uri set")
        print("mlflow artifact uri:",mlflow.get_artifact_uri())
        mlflow.set_experiment("AE_NODE")
        
        # log config to mlflow as a flattened dict
        log_to_mlflow(config_handler.config_status,str(config_path))

        # set up data processing
        data_processing_handler=Data_Processing(config_handler,logging_manager)

        # encoder-decoder
        encoder_decoder_handler=Encoder_Decoder(config_handler,logging_manager,data_processing_handler)

        # neural ode
        neural_ode_handler=Neural_ODE(config_handler,logging_manager,data_processing_handler,encoder_decoder_handler)

        logging_manager.log("Training complete")
        log_to_mlflow_artifacts(logging_manager.log_filename,"log_filename")
        mlflow.end_run()