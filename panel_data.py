import torch.utils.data as data

from PIL import Image

class PanelDataLoader(data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_path = self.file_list[idx]
        img = Image.open(image_path)
        transformed_img = self.transform(img)

        label = 0

        return transformed_img, label