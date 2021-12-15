from models.STIGPN import VisualModelV
from models.STIGPN import SemanticModelV
from torch.utils.data import DataLoader
from feeder.dataset import Dataset
import os
import torch
import numpy as np
import sklearn.metrics
from tqdm import tqdm
import argparse

def run_model(args,model_file1,model_file2,isAnticipation=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model1 = VisualModelV(args)
    model1.load_state_dict(torch.load(model_file1))

    model2 = SemanticModelV(args)
    model2.load_state_dict(torch.load(model_file2))

    model1.eval()
    model2.eval()
    model1.float().cuda()
    model2.float().cuda()
    val_dataset = Dataset(args,is_val=True,isAnticipation=isAnticipation)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
    
    H_preds, H_gts, O_preds, O_gts = [], [], [], []
    for num_objs,appearance_feats,box_tensors,box_categories,sub_activity_label,affordence_label in tqdm(val_dataloader):
        batchSize = len(num_objs)
        appearance_feats = appearance_feats.cuda()
        box_tensors = box_tensors.cuda()
        box_categories = box_categories.cuda()
        
        valid_labels = []
        for b in range(batchSize):
            for n in range(0, num_objs[b]):
                valid_labels.append(affordence_label[b][n])
        affordence_label = torch.Tensor(valid_labels)
        sub_activity_label,affordence_label = sub_activity_label.cuda(),affordence_label.cuda()
        with torch.no_grad():
            vis_subact_cls_scores, vis_afford_cls_scores = model1(num_objs,appearance_feats,box_tensors,box_categories)
            sem_subact_cls_scores, sem_afford_cls_scores = model2(num_objs,appearance_feats,box_tensors,box_categories)
            subact_cls_scores = vis_subact_cls_scores + sem_subact_cls_scores
            afford_cls_scores = vis_afford_cls_scores + sem_afford_cls_scores
        subact_cls_scores = subact_cls_scores.cpu().detach().numpy()
        afford_cls_scores = afford_cls_scores.cpu().detach().numpy()

        h_preds = []
        h_gts = []
        for b in range(batchSize):
            H_pred = np.argmax(subact_cls_scores[b])
            h_preds.append(H_pred)
            h_gts.append(sub_activity_label.cpu().numpy()[b])

        o_preds = []
        o_gts = []
        for b in range(affordence_label.shape[0]):
            O_pred = np.argmax(afford_cls_scores[b])
            o_preds.append(O_pred)
            o_gts.append(affordence_label.cpu().numpy()[b].item())
        
        H_preds += h_preds
        O_preds += o_preds
        H_gts += h_gts
        O_gts += o_gts
    H_gts = list(map(int, H_gts)) 
    O_gts = list(map(int, O_gts))
    subact_f1_score = sklearn.metrics.f1_score( H_gts, H_preds, labels=range(10), average='macro')*100
    afford_f1_score = sklearn.metrics.f1_score( O_gts, O_preds, labels=range(12), average='macro')*100
    print(subact_f1_score,afford_f1_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="You Can Do It!")
    parser.add_argument('--model', default='VisualModelV',help='VisualModelV,SemanticModelV') 
    parser.add_argument('--task', default='Detection')
    parser.add_argument('--batch_size', '--b_s', type=int, default=3, help='batch size: 1')
    parser.add_argument('--start_epoch', type=int, default=0,help='number of beginning epochs : 0')
    parser.add_argument('--epoch', type=int, default=300,help='number of epochs to train: 300')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate: 0.0001')#2e-5
    parser.add_argument('--weight_decay', type=float, default=0.8, help='learning rate: 0.0001')
    parser.add_argument('--nr_boxes', type=int, default=6,help='number of bbox : 6')
    parser.add_argument('--nr_frames', type=int, default=10,help='number of frames : 10')
    parser.add_argument('--subact_classes', type=int, default=10,help='number of subact_classes : 10')
    parser.add_argument('--afford_classes', type=int, default=12,help='number of afford_classes : 12')
    parser.add_argument('--feat_drop', type=float, default=0,help='dropout parameter: 0')
    parser.add_argument('--attn_drop', type=float, default=0,help='dropout parameter: 0')
    parser.add_argument('--cls_dropout', type=float, default=0,help='dropout parameter: 0')
    parser.add_argument('--step_size', type=int, default=50,help='number of steps for validation loss: 10') 
    parser.add_argument('--eval_every', type=int, default=1,help='number of steps for validation loss: 10') 
    parser.add_argument('--obj_scal', type=int, default=1,help='number of steps for validation loss: 10') 
    args = parser.parse_args()
    
    if args.task == 'Detection':
        model_file1 = './checkpoints/VisualModelV_max_scores_model.pkl'
        model_file2 = './checkpoints/SemanticModelV_max_scores_model.pkl'
        run_model(args,model_file1,model_file2,isAnticipation=False)
    else:
        model_file1 = './checkpoints/VisualModelVAnti_max_scores_model.pkl'
        model_file2 = './checkpoints/SemanticModelVAnti_max_scores_model.pkl'
        run_model(args,model_file1,model_file2,isAnticipation=True)
    
    
    
