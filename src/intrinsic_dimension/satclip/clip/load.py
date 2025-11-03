from main import *

def get_geoclip(ckpt_path, device, return_all=False):
    ckpt = torch.load(ckpt_path,map_location=device)

    # ckpt['hyper_parameters']['sh_embedding_dims']=32 #Manual change of hyperparams needed as the default values in  GeoCLIPLightningModule changed
    # ckpt['hyper_parameters']['legendre_polys']=20
    ckpt['hyper_parameters']['eval_downstream'] = False
    lightning_model = GeoCLIPLightningModule(**ckpt['hyper_parameters']).to(device)

    lightning_model.load_state_dict(ckpt['state_dict'])
    lightning_model.eval()

    geo_model = lightning_model.model
    #vis_model = geo_model.visual
    if return_all:
        return geo_model
    else:
        return geo_model.location