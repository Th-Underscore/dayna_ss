import copy
import traceback

from modules import shared
from modules.models import load_model
from modules.models_settings import get_model_metadata, update_model_parameters


def load_secondary_model(model_name: str, params: dict):
    """
    Loads a secondary model without interfering with the main model loaded in the UI.

    Args:
        model_name (str): The filename of the model to load.
        params (dict, optional): A dictionary of parameters to use for loading the model.
                                 These will temporarily override shared.args.
                                 Example: {'n_gpu_layers': 20, 'n_ctx': 4096}

    Returns:
        (model, tokenizer): A tuple containing the loaded model and tokenizer, or (None, None) on failure.
    """
    print(f"Attempting to load secondary model: {model_name}")

    # Save the state of the main model
    original_model_name = shared.model_name
    original_args = copy.deepcopy(shared.args)

    model_2, tokenizer_2 = None, None

    try:
        # Set up the environment for the secondary model
        for k, v in params.items():
            setattr(shared.args, k, v)

        model_settings = get_model_metadata(model_name)
        update_model_parameters(model_settings)

        model_2, tokenizer_2 = load_model(model_name)
        print(f"Successfully loaded secondary model: {model_name}")
    except Exception:
        print(f"Failed to load secondary model '{model_name}'.")
        traceback.print_exc()
        model_2, tokenizer_2 = None, None
    finally:
        print("Restoring main model state...")
        shared.args = original_args
        shared.model_name = original_model_name

        if original_model_name and original_model_name != "None":
            original_model_settings = get_model_metadata(original_model_name)
            update_model_parameters(original_model_settings, initial=True)

        print("Main model state restored.")

    return model_2, tokenizer_2
