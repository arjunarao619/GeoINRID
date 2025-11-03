from sc_main import *

def get_satclip(ckpt_path, device, return_all=False):
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Remove Lightning-specific keys that shouldn't be passed to __init__
    hyper_params = ckpt['hyper_parameters'].copy()
    
    # Remove any keys that start with underscore (Lightning internal keys)
    keys_to_remove = [k for k in hyper_params.keys() if k.startswith('_')]
    for key in keys_to_remove:
        hyper_params.pop(key, None)
    
    # Also remove any other non-init parameters if they exist
    hyper_params.pop('eval_downstream', None)  # Remove before adding it back
    hyper_params.pop('air_temp_data_path')
    hyper_params.pop('election_data_path')
    
    # # Add/modify parameters as needed
    # hyper_params['eval_downstream'] = False
    
    # Create the model with cleaned hyperparameters
    lightning_model = SatCLIPLightningModule(**hyper_params).to(device)
    
    lightning_model.load_state_dict(ckpt['state_dict'])
    lightning_model.eval()
    
    geo_model = lightning_model.model
    
    if return_all:
        return geo_model
    else:
        return geo_model.location
