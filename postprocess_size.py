import csv        
import torch
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
import json
import os


samples = [25, 100, 400]
# seeds = [666,777,888]

for sample_n in samples:
    print(sample_n)
    pth_path = '' #INSERT YOUR .pth file
    json_path = '' # path to R_object_features.json
    interact_json_path = '' # json file provided in OIA, {img_id: [personal indicator, bounding box coordinates], ...}
    raw_image_path = ''  # path to images in Reminiscence.zip
    raw_inter_image_path = '' # path to images in HRI.zip
    propagation_tsv_path = '' # path to save data

    data = torch.load(pth_path)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    with open(interact_json_path, 'r') as f:
        interact_json_data = json.load(f)

    print('number of data = {}'.format(len(data)))

    propagation_cnt = 0
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
            if sample['labelled'] == True:
                propagation_tsv_output.writerow(use)
                propagation_cnt += 1                           

        print("number of utilized triplets")
        print(f'propagation: {propagation_cnt}')

