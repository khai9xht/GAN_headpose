import torch
import torch.nn as nn


def base_layer(filter_in, filter_out, relu_alpha=None, bn_momentum=None):
    base_dict = OrderedDict([
        ('linear', nn.Linear(filter_in, filter_out))
    ])
    if relu_alpha is not None:
        base_dict.update({
            'relu': nn.LeakyReLU(alpha=relu_alpha)
        })
    if bn_momentum is not None:
        base_dict.update({
            'bn': nn.BatchNorm2d(momentum=bn_momentum)
        })
    return nn.Sequential(base_dict)

class Generator(nn.modules):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        flatten_img = self.img_shape[0] * self.img_shape[1] * self.img_shape[2]
        self.layers = nn.ModuleList([
            base_layer(self.latent_dim, 256),
            base_layer(256, 512, 0.2, 0.8),
            base_layer(512, 1024, 0.2, 0.8),
            base_layer(1024, 2048, 0.2, 0.8),
            nn.Linear(2048, flatten_img)
        ])
    
    def forward(x):
        def _run(sub_net, input):
            for layer in sub_net:
                input = layer(input)
            return input
        
        x = _run(x)
        return x

class Discriminator(nn.modules):
    def __init__(self, img_shape):
        super(self, Discriminator).__init__()
        self.img_shape = img_shape
        flatten_img = self.img_shape[0] * self.img_shape[1] * self.img_shape[2]
        self.layers = nn.ModuleList([
            base_layer(flatten_img, 1024, 0.2),
            base_layer(1024, 512, 0.2),
            base_layer(512, 256, 0.2),
            nn.Linear(256, 1)
        ])

    def forward(x):
        def _run(sub_net, input):
            for layer in sub_net:
                input = layer(input)
            return input
        
        x = _run(x)
        return x


if __name__ == "__main__":
    gen = Generator(1, (64, 64, 3))
    dis = Discriminator((64, 64, 3))
    print("generator: \n", gen)
    print("discriminator: \n", dis)

