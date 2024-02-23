def build_model(model_name):
    model_name = None

    # TODO: add models here
    if model_name == "resnet":
        model = 1
    elif model_name == "vit":
        model = 1
    elif model_name == "swin":
        model = 1

    assert model_name != None, "model is not supported. Add models in models.py yourself."

    return model