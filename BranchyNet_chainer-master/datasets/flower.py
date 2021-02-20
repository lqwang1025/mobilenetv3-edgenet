
from torchvision import transforms, datasets
import torch
import os
import json

class flower :
    def __init__(self, root):
        self.root = root
        self.data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }
        self.val_path   = os.path.join(root, "flower_data", "val")
        assert os.path.exists(self.val_path), "{} path does not exist.".format(image_path)

        self.train_path = os.path.join(root, "flower_data", "train")
        assert os.path.exists(self.val_path), "{} path does not exist.".format(image_path)

    def get_train_loader(self, batch_size):
        train_dataset = datasets.ImageFolder(self.train_path,
                                            transform=self.data_transform["train"])
        train_num = len(train_dataset)
        print("using {} images for training.".format(train_num))
        flower_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in flower_list.items())
        json_str = json.dumps(cla_dict, indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)
        return train_loader

    def get_val_loader(self, batch_size):
        validate_dataset = datasets.ImageFolder(self.train_path,
                                            transform=self.data_transform["val"])
        val_num = len(validate_dataset)
        print("using {} images for val.".format(val_num))
        flower_list = validate_dataset.class_to_idx
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)
        return validate_loader

    

