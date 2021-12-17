import torch.nn as nn
import torch
import timm

from config import CONFIG


"""

Efficient Net as embedder
"""
class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, patch_size=1, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = (feature_size, feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



"""
Model:  Swin Transformer
Input:  (224, 224) image, 12 Pawpularity features
Method: Combine image feature and pawpularity feature by one FC layer.
Results:
Best Train RMSE: Around 19, Appear at the fifth epoch
Best Valid RMSE: 19.2688, Appear at the fifth epoch 

"""
class PawpularityModelV1(nn.Module):
    def __init__(self, backbone, embedder, pretrained=True):
        super(PawpularityModelV1, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)   # img size 224
        self.n_features = self.backbone.head.in_features
        self.backbone.reset_classifier(0)
        # (N, 1024)
        self.fc = nn.Linear(self.n_features+12, CONFIG['num_classes'])

    def forward(self, images, denseFeatures):
        features = self.backbone(images)              # features = (bs, embedding_size)
        allFeatures = torch.cat((features, denseFeatures), 1)
        output = self.fc(allFeatures)                    # outputs  = (bs, num_classes)

        return output


"""
Model:  Swin Transformer
Input:  (224, 224) image, 12 Pawpularity features
Method: Combine image feature and pawpularity feature by two FC layer.
 
num_neurons=1024 Results: 
Overfitting
Training RMSE continue decreasing to around 16
Best Valid RMSE: 18.6053 Appears at the third epoch, later rise to 19.23


"""
class PawpularityModelV2(nn.Module):
    def __init__(self, backbone, embedder, pretrained=True, num_neurons=512):
        super(PawpularityModelV2, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)  # img size 224
        self.n_features = self.backbone.head.in_features
        self.backbone.reset_classifier(0)
        
        # (N, 1024)
        self.fc1 = nn.Linear(self.n_features+12, num_neurons)
        self.fc2 = nn.Linear(num_neurons, CONFIG['num_classes'])

    def forward(self, images, denseFeatures):
        features = self.backbone(images)              # features = (bs, embedding_size)
        allFeatures = torch.cat((features, denseFeatures), 1)
        output = self.fc1(allFeatures)                    # outputs  = (bs, num_classes)
        output = self.fc2(output)

        return output

# reduce neurons in FC layers
# add dropout layer to prevent overfitting
class PawpularityModelV3(nn.Module):
    def __init__(self, backbone, embedder, pretrained=True):
        super(PawpularityModelV3, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)  # img size 224
        self.n_features = self.backbone.head.in_features
        self.backbone.reset_classifier(0)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.n_features, 128)
        self.fc2 = nn.Linear(140, 128)
        self.fc3 = nn.Linear(128, CONFIG['num_classes'])

    def forward(self, images, denseFeatures):
        features = self.backbone(images)    
        features = self.dropout(features)          
        h = self.fc1(features)                    
        h = torch.cat((h, denseFeatures), 1)
        h = self.fc2(h)
        output = self.fc3(h)

        return output

if __name__ == "__main__":

    model = PawpularityModel(CONFIG['backbone'], CONFIG['embedder'])

    if CONFIG['use_cuda']:
        model = model.to(CONFIG['device'])

    if torch.cuda.device_count() > 1 and CONFIG['use_cuda']:   
        print("[*] GPU #", torch.cuda.device_count())
        model = nn.DataParallel(model)

    img = torch.randn(1, 3, CONFIG['img_size'], CONFIG['img_size']).to(CONFIG['device'])
    dense = torch.randn(1, 12).to(CONFIG['device'])
    print(model(img, dense))
