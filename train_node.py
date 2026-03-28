from src.utils.classes import ConfigReader, LoggingManager
from src.utils.helper_functions import log_to_mlflow, log_to_mlflow_artifacts
import os
import jax
from pathlib import Path
import mlflow
import argparse
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Neural ODE (stage 2 of AE-NODE pipeline). Requires saved AE weights.')
    parser.add_argument('--config', '-c', type=str, default='config/config.yml',
                        help='Path to configuration file (default: config/config.yml)')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file '{config_path}' not found!")
        sys.exit(1)

    print(f"Using config file: {config_path}")
    print(jax.devices())

    config_handler = ConfigReader(str(config_path))
    logging_manager = LoggingManager()
    logging_manager.log("Config file read")

    # NODE-specific precision
    node_precision = config_handler.get_config_status_safe("neural_ode.training.precision", default='float32')
    if node_precision == 'float64':
        jax.config.update("jax_enable_x64", True)
        logging_manager.log("NODE training: using float64 precision")
        assert jax.numpy.array(0.0).dtype == 'float64', "Precision set to float64 but dtype is not float64"
    else:
        logging_manager.log("NODE training: using float32 precision")

    from src.lib.data_processing.classes import Data_Processing
    from src.lib.autoencoder.classes import Encoder_Decoder
    from src.lib.NODE.classes import Neural_ODE

    # Force AE to load from disk — this script does not train the AE
    ae_load_model_key = ['encoder_decoder', 'loading', 'load_model']
    if not config_handler.config_status['encoder_decoder']['loading']['load_model']:
        logging_manager.log("train_node.py: overriding encoder_decoder.loading.load_model to True (AE must be pre-trained)")
        config_handler.config_status['encoder_decoder']['loading']['load_model'] = True

    with mlflow.start_run():
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("AE_NODE")
        log_to_mlflow(config_handler.config_status, str(config_path))

        data_processing_handler = Data_Processing(config_handler, logging_manager)

        # Loads AE weights from disk; does not retrain
        encoder_decoder_handler = Encoder_Decoder(config_handler, logging_manager, data_processing_handler)

        neural_ode_handler = Neural_ODE(config_handler, logging_manager, data_processing_handler, encoder_decoder_handler)

        logging_manager.log("NODE training complete")
        log_to_mlflow_artifacts(logging_manager.log_filename, "log_filename")
        mlflow.end_run()
