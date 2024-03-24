import monai.networks.nets as nets

def build_model(model_name):

    # TODO: add models here
    if model_name == "densenet":
        model = nets.DenseNet(spatial_dims=3, in_channels=1, out_channels=7)

    assert model_name != None, "model is not supported. Add models in models.py yourself."

    return model