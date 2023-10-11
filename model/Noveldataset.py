from torch.utils.data import Dataset
from PIL import ImageFile                                                      
ImageFile.LOAD_TRUNCATED_IMAGES = True 
import numpy as np 

class ImageCaptioningDataset(Dataset):
    def __init__(self,dataset,image_processor,tokenizer):
        self.dataset = dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        #seperate and preprocessing image/text 
        image=item['image'].resize((224,224))
        image=image.convert('RGB')
        image_tensor = np.array(image)
        text = item['text']
        #processing of img,text 
        pixel_values = self.image_processor(image_tensor, return_tensors="pt").pixel_values
        labels = self.tokenizer(text,return_tensors="pt",padding="max_length", max_length = 327,truncation=True).input_ids
        sample = {"pixel_values": pixel_values[0], "labels": labels[0]}
        return sample