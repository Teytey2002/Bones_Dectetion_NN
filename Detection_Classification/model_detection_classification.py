import torch
import torch.nn as nn
import torch.nn.functional as F

class BoneFractureMultiTaskDeepNet(nn.Module):
    def __init__(self, num_classes=7, num_points=4):
        super(BoneFractureMultiTaskDeepNet, self).__init__()
        self.num_points = num_points

        # --- Backbone convolutionnel profond ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Flatten size après convolutions
        self.flatten_size = 512 * 7 * 7

        # --- Tête de classification ---
        self.fc_class1 = nn.Linear(self.flatten_size, 256)
        self.class_out = nn.Linear(256, num_classes)

        # --- Tête de détection ---
        self.fc_bbox1 = nn.Linear(self.flatten_size, 256)
        self.bbox_out = nn.Linear(256, num_points * 2)

        # --- Dropout ---
        self.dropout = nn.Dropout(0.6)


    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.adaptive_pool(x)

        x = torch.flatten(x, 1)

        # Tête classification
        class_feat = self.dropout(F.relu(self.fc_class1(x)))
        class_out = self.class_out(class_feat)

        # Tête détection
        bbox_feat = self.dropout(F.relu(self.fc_bbox1(x)))
        bbox_out = self.bbox_out(bbox_feat)

        return class_out, bbox_out
