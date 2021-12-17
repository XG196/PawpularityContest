import cv2
from torch.utils.data import Dataset

class PawpularityDataset(Dataset):
    def __init__(self, root_dir, df, feature_cols, transforms=None, is_train=True):
        self.root_dir = root_dir
        self.df = df
        self.file_names = df['file_path'].values
        self.denseFeatures = df[feature_cols].values
        self.is_train = is_train
        if self.is_train:
            self.targets = df['Pawpularity'].values
            
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        denseFeature = self.denseFeatures[index, :]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.is_train:
            target = self.targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
        if self.is_train:
            return img, denseFeature, target
        else:
            return img, denseFeature