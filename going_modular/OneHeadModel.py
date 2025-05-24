import torch
import torchvision
from torch import nn

class OneHeadModel(nn.Module):
    def __init__(self, device, p_dropout):
        super(OneHeadModel, self).__init__()

        self.device = device
        self.p_dropout = p_dropout

        # weights = torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT
        # model = torchvision.models.resnext50_32x4d(weights=weights)
        # model = torch.nn.Sequential(*(list(model.children())[:-2])) # remove last two layers
        # self.encoder = model

        # Load EfficientNet encoder
        weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
        efficientNet = torchvision.models.efficientnet_b1(weights=weights)
        self.encoder = efficientNet.features

        # Pooling layers
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.batch_norm_1= nn.BatchNorm1d(1280) 
        self.batch_norm_2= nn.BatchNorm1d(1280)

        self.dense1 = nn.Sequential(
            nn.Linear(1280 * 2, 512),
            nn.ReLU(),
            nn.Dropout(p=self.p_dropout)
        )

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(p=self.p_dropout),
            nn.Linear(32, 5) # 5 output nodes for classification
            )
        
        # Apply He initialization to classification_head
        self._initialize_weights()
        
    def _initialize_weights(self):
        
        for module in self.dense1:
            if isinstance(module, nn.Linear):
                # Apply He initialization to weights
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # Initialize biases to zero (optional, common practice)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in self.classification_head:
            if isinstance(module, nn.Linear):
                # Apply He initialization to weights
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # Initialize biases to zero (optional, common practice)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.encoder(x) # Extract features

        # Apply pooling layers
        max_pooled = self.global_max_pool(x).view(x.size(0), -1)
        avg_pooled = self.global_avg_pool(x).view(x.size(0), -1)

        # Concatenate
        x1 = self.batch_norm_1(max_pooled)
        x2 = self.batch_norm_2(avg_pooled)

        # enc_out for visualizing data with t-SNE
        enc_out = torch.concat([x1, x2], dim=1)
        x = self.dense1(enc_out)

        # Classification branch
        class_out = self.classification_head(x).float()

        return class_out, enc_out

    