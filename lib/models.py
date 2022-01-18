from .model_example import wide_resnet, easyCNN

def get_model(model_name, num_classes=4):
    if model_name in ['wideresnet-28-10', 'wrn-28-10']:
        model = wide_resnet.WideResNet(28, 10, 0, num_classes)

    elif model_name in ['wideresnet-40-2', 'wrn-40-2']:
        model = wide_resnet.WideResNet(40, 2, 0, num_classes)

    elif model_name in ['easycnn']:
        model = easyCNN.EasyCNN(num_classes)

    return model