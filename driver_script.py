from src.lib.data_processing.classes import Data_Processing
from src.lib.autoencoder.classes import Encoder_Decoder
from src.lib.NODE.classes import Neural_ODE
from src.utils.classes import ConfigReader
import os
import jax
#jax.config.update('jax_default_device', jax.devices('cpu')[0])
from pathlib import Path

if __name__=="__main__":

    print(jax.devices())
    # roll all information into a constants dictionary
    constants={}
    data_dict={}

    config_path=Path("config/config.yml")

    config_handler=ConfigReader(str(config_path))

    # set up data processing
    data_processing_handler=Data_Processing(config_handler)

    # encoder-decoder
    encoder_decoder_handler=Encoder_Decoder(config_handler,data_processing_handler)

    # neural ode
    neural_ode_handler=Neural_ODE(config_handler,data_processing_handler,encoder_decoder_handler)

