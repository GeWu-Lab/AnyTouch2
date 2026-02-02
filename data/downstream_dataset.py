import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader
import json

max_force_abs_xyz = {
    'digit': [5.25, 10.61, 14.14],
    'biotip': [2.98, 3.91, 5.68],
    'gelsight': [6.84, 9.87, 8.52],
    'duragel': [3.92, 3.64, 7.89],
}

force_sensor_id_to_name = {
    1: 'digit',
    3: 'gelsight',
    4: 'duragel',
}

class TAGDataset_video(Dataset):
    def __init__(self, args, mode='train'):
        TAG_dir = 'datasets/TAG/touch_and_go/'

        self.datalist = []
        self.labels = []
        self.sensor_type = []
        self.bg_list = []
        self.offset = 0.0
        self.model_type = args.model

        if mode == 'train':
            if args.dataset == 'rough':
                self.txt = 'data/train_rough_bg.txt'
            elif args.dataset == 'material':
                self.txt = 'data/train_bg.txt'
        else:
            if args.dataset == 'rough':
                self.txt = 'data/test_rough_bg.txt'
            elif args.dataset == 'material':
                self.txt = 'data/test_bg.txt'
        
        for line in open(self.txt):
            item = line.split(',')[0]
            label = int(line.split(',')[1])
            bg_name = line.split(',')[2].strip()
            if label == -1:
                continue
            
            folder = item.split('/')[0]
            image = item.split('/')[1]
            image_id = int(image.split('.')[0])
            image_list = []
            bg_image = TAG_dir + folder + '/gelsight_frame/' + bg_name
            for i in range(args.num_frames):
                now_image_id = max(0, image_id - args.num_frames + 1 + i)
                now_image = f'{now_image_id:010d}.jpg'
                image_list.append(TAG_dir + folder +'/gelsight_frame/'+ now_image)

            self.datalist.append(image_list)
            self.bg_list.append(bg_image)
            self.labels.append(label)
            self.sensor_type.append(0)
        

        self.offset = 130.0 / 255.0
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        
        if mode == 'train':
                self.transform = transforms.Compose([
                        transforms.Resize(size=(224, 224), antialias=True),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.3),
                        # transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
            
        else:
                self.transform = transforms.Compose([
                        transforms.Resize(size=(224, 224), antialias=True),
                        # transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
        self.to_tensor = transforms.ToTensor()
        print(f'Number of samples: {len(self.datalist)}')

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        img_list = self.datalist[index]
        # print(img_list)
        img = []
        for image_path in img_list:
            img.append(self.to_tensor(Image.open(image_path).convert('RGB')))

        if self.model_type == 'anytouch':
            bg_image = self.to_tensor(Image.open(self.bg_list[index]).convert('RGB'))
            img.append(bg_image)
        
        img = torch.stack(img, dim=0)  # Stack along the time dimension
        # print(img.shape)  # (num_frames, 3, H, W)

        bg_img = img[-1:]
        img = img[:-1]
        img = img - bg_img + self.offset
        img = torch.clamp(img, 0.0, 1.0)

        img = self.transform(img)
        
        return img, self.sensor_type[index], self.labels[index]

class ClothDataset_video(Dataset):
    def __init__(self, args, mode='train'):
        Cloth_dir = 'datasets/yuan18/Data_ICRA18/Data/'

        self.datalist = []
        self.labels = []
        self.sensor_type = []
        self.bg_list = []
        self.offset = 0.0
        self.model_type = args.model
        self.load_from_clip = args.load_from_clip

        self.labels_map = {}

        if mode == 'train':
            self.txt = 'data/train_data_new.json'
        else:
            self.txt = 'data/test_data_new.json'
        
        self.label_json = 'data/cloth_metadata.json'

        with open(self.label_json, 'r') as file:
            temp_data = json.load(file)
            for k,v in temp_data.items():
                cloth_index = k
                label = v[-1]
                self.labels_map[cloth_index] = label
        
        with open(self.txt, 'r') as file:
            data = json.load(file)
            for k, v in data.items():
                cloth_index = v['cloth_index']
                trail_index = v['trial_index']
                folder = Cloth_dir + str(cloth_index) + '/' + str(trail_index) + '/gelsight_frame/'
                start_index = v['count']
                bg_img = folder + '0001.png'

                for tt in range(2):
                    start_index = start_index - args.num_frames * 2 + 1
                    now_image_list = []
                    for i in range(start_index, start_index + args.num_frames * 2, 2):
                        image = folder + str(i).zfill(4) + '.png'
                        now_image_list.append(image)
                    
                    self.datalist.append(now_image_list)
                    self.labels.append(self.labels_map[str(cloth_index)])
                    self.sensor_type.append(0)
                    self.bg_list.append(bg_img)

                    # print(now_image_list, bg_img, v['count'])

        self.offset = 130.0 / 255.0
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

       
        if mode == 'train':
                self.transform = transforms.Compose([
                        transforms.Resize(size=(224, 224), antialias=True),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.3),
                        # transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
           
        else:
                self.transform = transforms.Compose([
                        transforms.Resize(size=(224, 224), antialias=True),
                        # transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])


        self.to_tensor = transforms.ToTensor()
        print(f'Number of samples: {len(self.datalist)}')

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        img_list = self.datalist[index]
        # print(img_list)
        img = []
        for image_path in img_list:
            # img.append(Image.open(image_path).convert('RGB'))
            img.append(self.to_tensor(Image.open(image_path).convert('RGB')))


        bg_image = self.to_tensor(Image.open(self.bg_list[index]).convert('RGB'))
        img.append(bg_image)
        
        img = torch.stack(img, dim=0)  # Stack along the time dimension
        # print(img.shape)  # (num_frames, 3, H, W)
            
        bg_img = img[-1:]
        img = img[:-1]
        img = img - bg_img + self.offset
        img = torch.clamp(img, 0.0, 1.0)

        img = self.transform(img)
        
        return img, self.sensor_type[index], self.labels[index]

class MyForceDataset_video(Dataset):
    def __init__(self, args, mode='train'):
        force_dir = 'datasets/ToucHD-Force/'
        force_file = 'datasets/ToucHD-Force/all_data_direction.json'

        self.datalist = []
        self.force_labels = []
        self.force_scales = []
        self.sensor_type = []
        self.bg_list = []
        self.offset = 0.0
        self.model_type = args.model
        self.load_from_clip = args.load_from_clip

        self.labels_map = {}

        if mode == 'train':
            self.obj_list = [6,41,52,53,59,69,70]
        else:
            self.obj_list = [18,22,61]
        

        json_data = json.load(open(force_file, 'r'))

        for obj_speed_name in json_data.keys():
            obj_speed_data = json_data[obj_speed_name]
            obj_name = obj_speed_name.split('_speed')[0]
            obj_name_id = int(obj_name.split('obj')[-1])
            if obj_name_id not in self.obj_list:
                continue

            for sensor_name in obj_speed_data.keys():
                if args.num_frames == 4:
                    this_stride = 2
                elif args.num_frames == 2:
                    this_stride = 6
                else:
                    print('Error num_frames')
                    exit(0)
                if 'digit' in sensor_name:
                    sensor_type = 1
                elif 'gelsight' in sensor_name:
                    sensor_type = 3
                    this_stride = this_stride // 2
                else:
                    continue

                if sensor_name != args.data_sensor:
                    continue

                sensor_data = obj_speed_data[sensor_name]

                for i in range(args.num_frames*this_stride-1, len(sensor_data)):
                    now_image_list = []
                    for j in range(args.num_frames):
                        now_index = i - (args.num_frames - 1) * this_stride + j * this_stride
                        # now_image_id = temp_data[now_index][0]
                        now_image_id = sensor_data[now_index][0]
                        now_touch = force_dir + obj_speed_name + '/' + sensor_name + '/image_' + str(now_image_id) +'.png'
                        now_image_list.append(now_touch)

                        if j == args.num_frames - 1:
                            force_xyz = [sensor_data[now_index][1], sensor_data[now_index][2], -sensor_data[now_index][3]]
                            force_tensor = torch.tensor(force_xyz).float()
                            force_tensor[0] = force_tensor[0] / max_force_abs_xyz[force_sensor_id_to_name[sensor_type]][0]
                            force_tensor[1] = force_tensor[1] / max_force_abs_xyz[force_sensor_id_to_name[sensor_type]][1]
                            force_tensor[2] = torch.clip(force_tensor[2] / max_force_abs_xyz[force_sensor_id_to_name[sensor_type]][2], 0.0, 1.0)
                            self.force_labels.append(force_tensor)
                            self.force_scales.append(torch.tensor(max_force_abs_xyz[force_sensor_id_to_name[sensor_type]]))
                            self.datalist.append(now_image_list)
                            self.sensor_type.append(sensor_type)
                            self.bg_list.append(force_dir + obj_speed_name + '/' + sensor_name + '/image_1.png')
                            break

        self.offset = 130.0 / 255.0
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        
        
        if mode == 'train':
                self.transform = transforms.Compose([
                        transforms.Resize(size=(224, 224), antialias=True),
                        transforms.Normalize(mean, std)
                    ])
            
        else:
                self.transform = transforms.Compose([
                        transforms.Resize(size=(224, 224), antialias=True),
                        # transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])

        self.to_tensor = transforms.ToTensor()
        print(f'Number of samples: {len(self.datalist)}')

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        img_list = self.datalist[index]
        # print(img_list)
        img = []
        for image_path in img_list:
            # img.append(Image.open(image_path).convert('RGB'))
            img.append(self.to_tensor(Image.open(image_path).convert('RGB')))


        bg_image = self.to_tensor(Image.open(self.bg_list[index]).convert('RGB'))
        img.append(bg_image)
        
        img = torch.stack(img, dim=0)  # Stack along the time dimension
        # print(img.shape)  # (num_frames, 3, H, W)
            
        bg_img = img[-1:]
        img = img[:-1]
        img = img - bg_img + self.offset
        img = torch.clamp(img, 0.0, 1.0)

        img = self.transform(img)


        return img, self.sensor_type[index], self.force_labels[index], self.force_scales[index]