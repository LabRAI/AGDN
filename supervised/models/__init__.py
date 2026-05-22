from models.agd import Encoder_AGD
from models.gcn import Encoder_GatedGCN


def create_model(cfg):
    model_name = cfg.model_name
    if model_name == 'agd':
        model = Encoder_AGD(**vars(cfg))
    elif model_name == 'gcn':
        model = Encoder_GatedGCN(**vars(cfg))
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    return model
