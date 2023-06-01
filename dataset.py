from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import cv2
class SegDataset(Dataset):
    #=================================================
    def __init__(self, is_train, root_dir):
        self.root_dir = root_dir
        images_list=[]
        for path_img in list(Path(self.root_dir).glob('*.png')):
            img=cv2.imread(str(path_img))
            images_list.append(img)
        self.images=images_list
        self.transform=transforms.Compose(
        [
            transforms.ToTensor(),
        ])
    #===============================================
    def __len__(self):
        return len(self.images)
    #=============================================
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample=self.images[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
#===============================================
if __name__=="__main__":
    seg_dataset=SegDataset(is_train=None,root_dir="test_imgs")
    print(seg_dataset[0])
