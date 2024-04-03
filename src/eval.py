from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import argparse

import numpy as np
from tqdm import tqdm
from metric import TEDS
import json
from lib.utils.eval_utils import coco_into_labels, Table, pairTab

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type = str)
    parser.add_argument('--predict_dir', type = str)
    parser.add_argument('--markup_dir', type = str)
    args = parser.parse_args()

    coco_into_labels(args.dataset_dir, args.predict_dir)

    gt_bbox_path = os.path.join(args.predict_dir, 'gt_center')
    gt_logi_path = os.path.join(args.predict_dir, 'gt_logi')
    bbox_path = os.path.join(args.predict_dir, 'center')
    logi_path = os.path.join(args.predict_dir, 'logi')
    html_path = os.path.join(args.predict_dir, 'html')

    
    table_dict = []
    for file_name in tqdm(os.listdir(os.path.join(args.predict_dir, 'gt_center'))):
        if 'txt' in file_name:
            pred_table = Table(bbox_path, logi_path, file_name)
            gt_table = Table(gt_bbox_path, gt_logi_path, file_name)
            table_dict.append({'file_name': file_name, 'pred_table': pred_table, 'gt_table': gt_table})
    a, b = {}, {}
    acs = []
    bbox_recalls = []
    bbox_precisions = []
    TEDS_scores = []
    teds = TEDS()
    for i in tqdm(range(len(table_dict))):
        filename = table_dict[i]['file_name']
        with open(os.path.join(args.markup_dir, filename[:-4]+'.txt')) as f:
            gt_html = f.readline()
            gt_html = '<html><body><table>'+gt_html+'</table></body></html>' 
        with open(os.path.join(html_path, filename[:-4]+'.txt')) as f:
            pred_html = f.readline()
            pred_html = '<html><body><table>'+pred_html+'</table></body></html>'
        TEDS_score = teds.evaluate(pred_html, gt_html)
        TEDS_scores.append(TEDS_score)
        pair = pairTab(table_dict[i]['pred_table'], table_dict[i]['gt_table'])
        #Acc of Logical Locations
        ac = pair.evalAxis()
        if ac != 'null':
            acs.append(ac)
        a[filename] = TEDS_score
        b[filename] = ac
    with open('eval_result_TEDS.json', 'w') as f:
        json.dump(a, f)
    with open('eval_result_log.json', 'w') as f:
        json.dump(b, f)
    

        #Recall of Cell Detection 
        # recall = pair.evalBbox('recall')
        # bbox_recalls.append(recall)
        
        # #Precision of Cell Detection 
        # precision = pair.evalBbox('precision')
        # bbox_precisions.append(precision)
    
    # det_precision =  np.array(bbox_precisions).mean()
    # det_recall =  np.array(bbox_recalls).mean()
    # f = 2 * det_precision * det_recall / (det_precision + det_recall)

    print('Evaluation Results | Accuracy of Logical Location: {:.2f}.'.format(np.array(acs).mean()))
    print('Evaluation Results | TEDS score: {:.2f}.'.format(np.array(TEDS_scores).mean()))

    