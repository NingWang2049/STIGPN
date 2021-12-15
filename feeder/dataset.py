import torch
import numpy as np
import joblib
from PIL import Image
import feeder.gtransforms as gtransforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self,args,is_val=False,isSegment=False,isAnticipation=True):
        self.is_val = is_val
        self.num_boxes = args.nr_boxes
        self.coord_nr_frames = args.nr_frames
        self.pre_resize_shape = (640, 480)
        self.if_augment = True
        self.segment = isSegment

        if self.if_augment:
            if not self.is_val:  # train, multi scale cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(output_size=(640,480),
                                                                   scales=[1, .875, .75])
            else:  # val, only center cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(output_size=(640,480),
                                                                   scales=[1],
                                                                   max_distort=0,
                                                                   center_crop_only=True)
        else:
            self.random_crop = None

        if not is_val:
            load_dir = 'datasets/cad_train_data_with_appearence_features.p'
        else:
            load_dir = 'datasets/cad_test_data_with_appearence_features.p'
        with open(load_dir,'rb') as f:
            data = joblib.load(f)
        self.sub_activity_list = data['sub_activity_list']
        self.affordence_list = data['affordence_list']
        self.classes_list = data['classes_list']
        self.classes_list.insert(0,'person')
        self.classes_list.insert(0,'_background_')
        if not is_val:
            self.load_data = data['train_data']
        else:
            self.load_data = data['test_data']
        if isAnticipation:
            video_length = len(self.load_data)
            load_data = []
            for i in range(video_length-1):
                if self.load_data[i]['video_id'] == self.load_data[i+1]['video_id']:
                    self.load_data[i]['label'] = self.load_data[i+1]['label']
                    load_data.append(self.load_data[i])
            self.load_data = load_data

    def __len__(self):
        return len(self.load_data)

    def _sample_indices(self, nr_video_frames):
        average_duration = nr_video_frames * 1.0 / self.coord_nr_frames
        if average_duration > 0:
            offsets = np.multiply(list(range(self.coord_nr_frames)), average_duration) \
                      + np.random.uniform(0, average_duration, size=self.coord_nr_frames)
            offsets = np.floor(offsets)
        elif nr_video_frames > self.coord_nr_frames:
            offsets = np.sort(np.random.randint(nr_video_frames, size=self.coord_nr_frames))
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def _get_val_indices(self, nr_video_frames):
        if nr_video_frames > self.coord_nr_frames:
            tick = nr_video_frames * 1.0 / self.coord_nr_frames
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.coord_nr_frames)])
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets
    
    def _cn_dataset(self):
        pre_video_id = -1
        num_objs_list,appearance_feats_list,box_tensors_list,box_categories_list,sub_activity_label_list,affordence_label_list = \
            [],[],[],[],[],[]
        for i in range(0,self.__len__()):
            video_id,seg_frames,num_objs,appearance_feats,box_tensors,box_categories,sub_activity_label,affordence_label = \
                self.__getitem__(i)
            if pre_video_id == -1:
                pre_video_id = video_id
            if pre_video_id != video_id or i == self.__len__() - 1:
                num_objs_ = torch.tensor(num_objs_list)
                appearance_feats_ = torch.cat(appearance_feats_list,dim=0)
                box_tensors_ = torch.cat(box_tensors_list,dim=0)
                box_categories_ = torch.cat(box_categories_list,dim=0)
                sub_activity_label_ = torch.cat(sub_activity_label_list,dim=0)
                affordence_label_ = torch.cat(affordence_label_list,dim=0)
                yield num_objs_,appearance_feats_,box_tensors_,box_categories_,sub_activity_label_,affordence_label_
                pre_video_id = video_id
                num_objs_list,appearance_feats_list,box_tensors_list,box_categories_list,sub_activity_label_list,affordence_label_list = \
                    [],[],[],[],[],[]
            num_objs_list.append(num_objs)
            appearance_feats_list.append(appearance_feats.unsqueeze(0))
            box_tensors_list.append(box_tensors.unsqueeze(0))
            box_categories_list.append(box_categories.unsqueeze(0))
            sub_activity_label_list.append(sub_activity_label.unsqueeze(0))
            affordence_label_list.append(affordence_label.unsqueeze(0))
    
    def __getitem__(self, i):
        sub_activity_feature = self.load_data[i]
        video_id = sub_activity_feature['video_id']
        seg_frames = sub_activity_feature['seg_frames']
        label = sub_activity_feature['label']
        bboxes = sub_activity_feature['bboxes']
        apperence_features = sub_activity_feature['apperence_features']
        
        object_set = [x[::-1] for x in label.keys()]
        object_set = sorted(object_set)
        object_set = [x[::-1] for x in object_set]
        object_set.remove('person')
        sub_activity_label = torch.tensor(self.sub_activity_list.index(label['person'])).float()
        affordence_label = torch.tensor([self.affordence_list.index(label[x]) for x in object_set]).float()
        affordence_label = torch.cat([affordence_label,torch.zeros((5-affordence_label.shape[0])).float()],dim=0)
        object_set.insert(0,'person')

        n_frame = len(bboxes)

        if not self.is_val:  # train
            coord_frame_list = self._sample_indices(n_frame)
        else:  # val
            coord_frame_list = self._get_val_indices(n_frame)

        frames = []
        frames.append(Image.new('RGB',(640,480)))
        height, width = frames[0].height, frames[0].width

        if self.random_crop is not None:
            frames, (offset_h, offset_w, crop_h, crop_w) = self.random_crop(frames)
        else:
            offset_h, offset_w, (crop_h, crop_w) = 0, 0, self.pre_resize_shape
        #print(crop_h, crop_w)
        scale_resize_w, scale_resize_h = self.pre_resize_shape[1] / float(width), self.pre_resize_shape[0] / float(height)
        scale_crop_w, scale_crop_h = 640 / float(crop_w), 480 / float(crop_h)
        
        box_categories = torch.zeros((self.num_boxes))
        box_tensors = torch.zeros((self.coord_nr_frames, self.num_boxes, 4), dtype=torch.float32)
        appearance_feats = torch.zeros([self.num_boxes, self.coord_nr_frames, 2048])*1.0
        for frame_index, frame_id in enumerate(coord_frame_list):
            try:
                frame_bboxes_data = bboxes[frame_id]
                appearance_feats[:,frame_index,:] = torch.tensor(apperence_features[:,frame_id,:]).float()
            except:
                frame_bboxes_data = {}
            for box_data_key in frame_bboxes_data.keys():
                global_box_id = object_set.index(box_data_key)
                try:
                    x0, y0, x1, y1 = frame_bboxes_data[box_data_key]
                except:
                    x0, y0, x1, y1 = 0,0,0,0
                
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                # scaling due to initial resize
                x0, x1 = x0 * scale_resize_w, x1 * scale_resize_w
                y0, y1 = y0 * scale_resize_h, y1 * scale_resize_h

                # shift
                x0, x1 = x0 - offset_w, x1 - offset_w
                y0, y1 = y0 - offset_h, y1 - offset_h

                x0, x1 = np.clip([x0, x1], a_min=0, a_max=crop_w-1)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=crop_h-1)

                # scaling due to crop
                x0, x1 = x0 * scale_crop_w, x1 * scale_crop_w
                y0, y1 = y0 * scale_crop_h, y1 * scale_crop_h

                # precaution
                x0, x1 = np.clip([x0, x1], a_min=0, a_max=640)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=480)

                # (cx, cy, w, h)
                gt_box = np.array([(x0 + x1) / 2., (y0 + y1) / 2., x1 - x0, y1 - y0], dtype=np.float32)

                # normalize gt_box into [0, 1]
                gt_box[::2] = gt_box[::2]/640.0
                gt_box[1::2] = gt_box[1::2]/480.0

                # load box into tensor
                try:
                    box_tensors[frame_index, global_box_id] = torch.tensor(gt_box).float()
                except:
                    pass
        for idx,o in enumerate(object_set):
            box_categories[idx] = self.classes_list.index(o.split('_')[0])
        box_categories = torch.cat([box_categories.unsqueeze(0) for x in range(self.coord_nr_frames)],dim=0)
        num_objs = torch.tensor(len(object_set)-1)
        if self.segment:
            return video_id,seg_frames,num_objs,appearance_feats,box_tensors,box_categories,sub_activity_label,affordence_label
        return num_objs,appearance_feats,box_tensors,box_categories,sub_activity_label,affordence_label

if __name__ == "__main__":
    a = Dataset(is_val=False)
    for i in range(a.__len__()):
        a.__getitem__(i)
