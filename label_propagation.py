import torch
from torch import Tensor
import os
from PIL import Image
from torchvision import transforms as pth_transforms
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import pandas as pd
import random
import argparse
import torch.nn.functional as F
import time
import sklearn
import json



start_time = time.time()

from numpy import dot
from numpy.linalg import norm
def cos_sim(A, B):
  A = A.squeeze()
  B = B.squeeze()
  ret = dot(A, B)/(norm(A)*norm(B))
  return ret.item()

def parse_args():
    """
        Parse input arguments
    """
    parser = argparse.ArgumentParser(description='label propagation')
    parser.add_argument('--model',
                        help='vanilla / linear_probing / full_tuning',
                        default='linear_probing', type=str)
    parser.add_argument('--thresh', default=0.7, type=float)
    parser.add_argument('--iter', default=3, type=int)
    parser.add_argument('--save_nodes', default=False)
    parser.add_argument('--nshot', default=1, type=int)
    parser.add_argument('--sample_n', default=400, type=int)
    parser.add_argument('--ignore_interaction', default=False, type=bool)
    parser.add_argument('--seed', default=777, type=int)
    args = parser.parse_args()
    return args


class Linear(nn.Module):
    def __init__(self, in_feat_dim=768, out_feat_dim=512):
        super(Linear, self).__init__()
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.fc1 = nn.Linear(self.in_feat_dim, out_feat_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    
def LabelPropagation(V,E,max_iter, model):
    V = V.copy()
    for i in range(max_iter):
        print("Processing {} iteration ...".format(i))
        changed_node_cnt = 0
        for vi, v in enumerate(V):
            if v["known"] is True:
                continue
            # Gather labels from neighbors
            label_counts = {}
            for ui,u in enumerate(V):
                if u['labelled'] is True:
                    if u['label'] in label_counts.keys():
                        label_counts[u['label']].append(E[ui,vi]) # label_counts[u['label']] += E(u,v) # avg.
                    else:
                        label_counts[u['label']] = [E[ui,vi]]
                    if u['known'] is True:
                        if '_' not in u['img_id']: # first observation
                            for labelled_weight in range(10):
                                label_counts[u['label']].append(E[ui,vi])
            # Update node label
            if len(label_counts.keys()) != 0:
                for ii in label_counts.keys():
                    label_counts[ii] = np.array(label_counts[ii]).mean()
                if model in ['vanilla', 'full_tuning', 'linear_probing']:
                    if max(label_counts.values()) > args.thresh:
                        max_key = max(label_counts, key=label_counts.get)
                        v['label'] = max_key
                        v['labelled'] = True
                        changed_node_cnt += 1
                # elif model in []: # for euclidien metric
                #     if min(label_counts.values()) < args.thresh:
                #         min_key = min(label_counts, key=label_counts.get)
                #         v['label'] = min_key
                #         v['labelled'] = True
                #         changed_node_cnt += 1
        if changed_node_cnt/len(V) < 0.05:
            break
    return V
    
if __name__ == '__main__':
    print('1')
    args = parse_args()

    random.seed(args.seed)

    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #################################################
    ################ Load model #####################

    if args.model == 'linear_probing':
        vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        vitb16.eval()
        vitb16 = vitb16.to(device)

        model_path = '/data/jhkim/icra24/dino_model/linear_probing_model.pth'
        model = Linear()

        # Load the state dict normally
        state_dict = torch.load(model_path)

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:] # remove `module.` from the key
            new_state_dict[name] = v

        # Load the parameters
        model.load_state_dict(new_state_dict)

        model.eval()
        model = model.to(device)

    elif args.model == 'vanilla':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        model = model.to(device)
        print('loaded vanilla DINO')

    elif args.model == 'full_tuning':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        # model.fc = nn.Linear(768,512)

        model_path = '/data/jhkim/icra24/dino_model/full_tuning.pth'

        state_dict = torch.load(model_path)

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:] # remove `module.` from the key
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

        model.eval()
        model = model.to(device)
    else:
        print("no compatible model!!!!!")
        exit()
    
    #################################################
    ################ Read Excel #####################
    print('start reading excel')
    xcel_path = '/home/jhkim/icra24/test.xlsx'
    xl = pd.ExcelFile(xcel_path)

    df = xl.parse(xl.sheet_names[0])

    df = df.fillna("")

    data_dict = {}

    for index, row in df.iterrows():
        if type(row[1]) == float: #confirm that this is a valid object
            key = row[3].strip()  # NL query
            values = list(row[i] for i in range(3, len(row)))
            data_dict[key] = values
        else:
            if row[2] == 'line':
                key = 'line'
                values = list(row[i] for i in range(4, len(row)))
                data_dict[key] = values
            else: # vague/unable to distinguish
                key = 'vague'
                values = list(row[i] for i in range(4, len(row)))
                data_dict[key] = values
    print('done reading excel')

    #################################################
    ########### Make initial Graoh node V ###########

    image_path='/data/jhkim/icra24/cropped_objects'

    V = []
    categories_list = []
    selected_img_ids = random.sample(range(400), args.sample_n)

    data_dict_len = len(data_dict.keys())
    cnt = 0
    for label, list_ids in data_dict.items():
        for id in list_ids:
            if '_' in id:
                tmp = id.split('_')
                try:
                    img_id = str(int(tmp[0])) + '.png'
                    object_id = 'object_' + str(int(tmp[1])) + '.png'
                except:
                    print("erorr on excel ... {}".format(id))
                if int(img_id.split('.')[0]) in selected_img_ids:
                    data = {} 
                    data['interaction'] = False
                    data['img_id'] = img_id          # 0.png,1.png,...  
                    data['object_id'] = object_id   # object_0.png, ...
                    data['category'] = label
                    categories_list.append(label)

                    image = Image.open(os.path.join(image_path, os.path.join(img_id, object_id)))
                    image = transform(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        if args.model == 'linear_probing':
                            feat = vitb16(image)
                            feat = model(feat)
                        else:
                            feat = model(image)
                    data['visual feature'] = feat
                    data['known'] = None
                    data['labelled'] = None
                    data['label'] = ""
                    V.append(data)
        cnt += 1
        if cnt%2 == 0:
            print("Generating V ... {} percent done. ".format(int(100*cnt/data_dict_len)))

    raw_scene_len = len(V)
    print("{} number of nodes constructed from raw scene".format(len(V)))


    # Get interactive annotations
    inter_crop_path = '/data/jhkim/icra24/cropped_interaction/'
    inter_annot_path = '/home/jhkim/icra24/interactive/interactive_perQ_bbox.json'  #{'0.png':[perQ, bbox], ...}

    with open(inter_annot_path, 'r') as f:
        inter_annotation = json.load(f)

    inter_ignore = 0
    for inter_obj_key, inter_obj_val in inter_annotation.items():     #{'0.png':[perQ, bbox], ...}
        personal_q = inter_obj_val[0].strip()
        bbox = inter_obj_val[1]

        if personal_q not in categories_list:
            print('"{}" not in category list !!!'.format(personal_q))

        found = False
        if args.ignore_interaction:
            for ig in range(2,10):
                if '_{}'.format(ig) in inter_obj_key:
                    found = True
        if found:
            continue
        data = {}
        data['interaction'] = True
        data['img_id'] = inter_obj_key   # 0_0.png, ...       
        data['object_id'] = inter_obj_key     # 0_0.png, ...
        data['category'] = personal_q
        try:
            image = Image.open(os.path.join(inter_crop_path, inter_obj_key))
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                if args.model == 'linear_probing':
                    feat = vitb16(image)
                    feat = model(feat)
                else:
                    feat = model(image)
            data['visual feature'] = feat
            data['known'] = True
            data['labelled'] = True
            data['label'] = personal_q
            V.append(data)
        except:
            inter_ignore += 1
    print("ignored {} number of data in interaction".format(inter_ignore))

    print("{} number of additional nodes constructed from interaction".format(len(V)-raw_scene_len))


    # n-shot labelling

    # labelled_category = {}
    # for i in V:
    #     labelled_category[i['category']] = 0
    # labelled_category['line'] = args.nshot 
    # labelled_category['vague'] = args.nshot

    # random.shuffle(V)
    # for i, instance in enumerate(V):
    #     if labelled_category[instance['category']] != args.nshot:
    #         V[i]['label'] = instance['category']
    #         V[i]['labelled'] = True
    #         V[i]['known'] = True
    #         labelled_category[instance['category']] += 1

    
    count_labelled = 0
    count_unlabelled = 0
    for k in range(len(V)):
        if V[k]['labelled'] is True:
            count_labelled += 1
        # if k%500 == 0:
        #     print('making Edges... {}/{}'.format(k,len(V)))
    print("(Init V) total object: ", len(V))
    print("(Init V) count labelled: ", count_labelled)

    #################################################
    ########### Make initial Graoh edge E ###########

    # E = torch.empty((len(V),len(V)))

    # for i in range(E.shape[0]):
    #     for j in range(E.shape[1]):
    #         if j>i:
    #             continue
    #         # euclidien distance
    #         # if args.model in []:
    #         #     E[i][j] = F.pairwise_distance(V[i]["visual feature"].cpu(), V[j]["visual feature"].cpu())
    #         # cosine similarity
    #         E[i][j] = cos_sim(V[i]["visual feature"].cpu(), V[j]["visual feature"].cpu())
    #         E[j][i] = E[i][j]

    features = torch.tensor([fi['visual feature'].squeeze().tolist() for fi in V]).cuda()
    norm_features = F.normalize(features, p=2, dim=1)
    E = torch.mm(norm_features, norm_features.t()).cpu()

    #################################################
    ########### Label Propagation ###########

    max_iter = args.iter
    V_propagated = LabelPropagation(V,E,max_iter, args.model)

    #################################################
    ########### Evaluate Results ###########

    labelled = 0
    correct = 0
    valid_num = 0

    for i in range(len(V_propagated)):
        if not V_propagated[i]['interaction']:
            if V_propagated[i]['category'] not in ['line', 'vague']:
                valid_num += 1  # valid objects in raw iamges
                if V_propagated[i]['labelled'] is True:
                    labelled += 1
                    if V_propagated[i]['label'] == V_propagated[i]['category']:
                        correct += 1
                    
    shldnt_label = 0
    line = 0
    line_wrong = 0
    vague = 0
    vague_wrong = 0

    for v in V_propagated:
        if v['category'] == 'line':
            line += 1
            if v['labelled'] == True:
                line_wrong += 1
        if v['category'] == 'vague':
            vague += 1
            if v['labelled'] == True:
                vague_wrong += 1

    print("propagated: {}/{} = ".format(labelled, valid_num), labelled/valid_num)
    # print("\n")
    print("correct answers among labelled images: {}/{} = ".format(correct, labelled), correct/labelled)
    # print("\n")
    print("labelled while shouldn't be (line): {}/{} = ".format(line_wrong, line), line_wrong/line)
    print("labelled while shouldn't be (vague): {}/{} = ".format(vague_wrong, vague), vague_wrong/vague)
    print("labelled while shouldn't be (total): {}/{} = ".format(line_wrong+vague_wrong, line+vague), (line_wrong+vague_wrong)/(line+vague))

    #################################################
    ########### Save propagated Nodes ###########
    if args.save_nodes:
        if args.ignore_interaction:
            pth_path = '/data/jhkim/icra24/ofa_vg_data/lp_{}_{}_{}sampled_{}_ignore_from2.pth'.format(args.model, args.thresh, args.sample_n, args.seed)
        else:
            pth_path = '/data/jhkim/icra24/ofa_vg_data/lp_{}_{}_{}sampled.pth'.format(args.model, args.thresh, args.sample_n)
        torch.save(V_propagated, pth_path)

    # V structure
        # data['interaction'] = True
        # data['img_id'] = inter_obj_key   # 0, 1, 2, ...       
        # data['object_id'] = inter_obj_key     # 0_0.png, ...
        # data['category'] = personal_q
        # data['visual feature'] = feat
        # data['known'] = True
        # data['labelled'] = True
        # data['label'] = personal_q

    end_time = time.time()

    tot_time = end_time - start_time
    print(f"total {tot_time/60} min spent ...")