
def get_model(name, max_iter, num_classes=2, pretrained=True, normed_fc=True, use_bias=False, simsiam=False):
    if name == "ResNet18_lgt":
        from .resnet_lgt import resnet18
        model = resnet18(max_iter=max_iter, num_classes=num_classes, pretrained=pretrained, normalized=normed_fc, use_bias=use_bias, simsiam=simsiam)
        return model