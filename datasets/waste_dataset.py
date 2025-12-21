# datasets/waste_dataset.py
from torch.utils.data import Dataset
from PIL import Image

class WasteDataset(Dataset):
    def __init__(self, dataframe, transforms=None):
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["full_path"]).convert("RGB")
        label = row["waste"]

        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(label, dtype=torch.long)
