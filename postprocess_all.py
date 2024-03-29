import csv        
import torch
from PIL import Image
from io import BytesIO
import base64
import json
import os


# .pth to .tsv

# ### .pth
# - [obj1_dict, obj2_dict, ...]
# - objn_dict =  {
#           'visual feature': datas,
# 		    'category' : category,
# 		    'label' : "",
# 		    'known' : None,
# 		    'labelled' : None,
# 			'img_id' : str
# 			'obj_id': str
#            }


# Node structure
#     data['interaction'] = True
#     data['img_id'] = inter_obj_key   # 0, 1, 2, ...       
#     data['object_id'] = inter_obj_key     # 0_0.png, ...
#     data['category'] = personal_q
#     data['visual feature'] = feat
#     data['known'] = True
#     data['labelled'] = True
#     data['label'] = personal_q




## Check .pth file

pth_path = '' #INSERT YOUR .pth file
data = torch.load(pth_path)
tot = 0 #from raw images
cnt = 0
cnt_labelled = 0
for d in data:
    if d['category'] not in ['line','vague']:
        if d['interaction'] == False:
            tot += 1
            if d['label'].strip() == d['category'].strip():
                cnt += 1
            if d['labelled']:
                cnt_labelled += 1
print(cnt)
print(cnt/cnt_labelled)
print(tot)




json_path = '' # path to R_object_features.json
interact_json_path = '' # json file provided in OIA, {img_id: [personal indicator, bounding box coordinates], ...}


raw_image_path = ''  # path to images in Reminiscence.zip
raw_inter_image_path = '' # path to images in HRI.zip
curious_tsv_path = '' # path to save data
passive_tsv_path = '' # path to save data
propagation_tsv_path = '' # path to save data
gt_tsv_path = '' # path to save data

data = torch.load(pth_path)
with open(json_path, 'r') as f:
    json_data = json.load(f)
with open(interact_json_path, 'r') as f:
    interact_json_data = json.load(f)

print('number of data = {}'.format(len(data)))

curious_cnt = 0 # propagation - interaction
passive_cnt = 0
propagation_cnt = 0
gt_cnt = 0
with open(gt_tsv_path, 'w', newline='') as f_gt:
    gt_tsv_output = csv.writer(f_gt, delimiter='\t')
    with open(passive_tsv_path, 'w', newline='') as f_passive:
        passive_tsv_output = csv.writer(f_passive, delimiter='\t')
        with open(curious_tsv_path, 'w', newline='') as f_curious:
            curious_tsv_output = csv.writer(f_curious, delimiter='\t')
            with open(propagation_tsv_path, 'w', newline='') as f_output:
                propagation_tsv_output = csv.writer(f_output, delimiter='\t')
                for indx, sample in enumerate(data):
                    img_id = sample['img_id']
                    object_id = sample['object_id']
                    use = []
                    use.append(propagation_cnt)
                    use.append(img_id)
                    use.append(sample['label']) #Q
                    if sample['interaction'] == False:
                        bbox = json_data[img_id][object_id][1]
                        use.append("{},{},{},{}".format(bbox[0],bbox[1],bbox[2],bbox[3])) #bbox
                        img = Image.open(os.path.join(raw_image_path, img_id))
                    else:
                        bbox = interact_json_data[object_id][1]
                        use.append("{},{},{},{}".format(bbox[0],bbox[1],bbox[2],bbox[3])) #bbox
                        img = Image.open(os.path.join(raw_inter_image_path, img_id))
                    img_buffer = BytesIO()
                    img.save(img_buffer, format=img.format)
                    byte_data = img_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data) # bytes
                    base64_str = base64_str.decode("utf-8") # str
                    use.append(base64_str)
                    gtuse = use.copy()
                    gtuse[2] = sample['category']
                    gtuse[0] = gt_cnt
                    gt_tsv_output.writerow(gtuse)
                    gt_cnt += 1
                    if sample['labelled'] == True:
                        propagation_tsv_output.writerow(use)
                        propagation_cnt += 1
                        if '_' not in img_id:
                            curioususe = use.copy()
                            curioususe[0] = curious_cnt
                            curious_tsv_output.writerow(curioususe)
                            curious_cnt += 1                                
                        if sample['known'] == True:
                            if '_' not in img_id:
                                passiveuse = use.copy()
                                passiveuse[0] = passive_cnt
                                passive_tsv_output.writerow(passiveuse)
                                passive_cnt += 1
                    if indx%500 == 0:
                        print("{} processing ... {}%".format(pth_path, int(100*indx/len(data))))

print("number of utilized triplets")
print(f'passive: {passive_cnt}')
print(f'curious: {curious_cnt}')
print(f'propagation: {propagation_cnt}')
print(f'gt: {gt_cnt}')