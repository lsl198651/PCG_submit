#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for running models for the 2022 Challenge. You can run it as follows:
#
#   python run_model.py model data outputs
#
# where 'model' is a folder containing the your trained model, 'data' is a folder containing the Challenge data, and 'outputs' is a
# folder for saving your model's outputs.

from torcheval.metrics.functional import binary_precision,binary_recall,binary_auprc, binary_auroc,binary_accuracy,binary_f1_score,binary_confusion_matrix
import torch
import numpy as np, os, sys
from helper_code import *
from tqdm import tqdm
from team_code import load_challenge_model, run_challenge_model
import csv
# Run model.
def run_model(model_folder, data_folder, verbose):
    # Load models.
    if verbose >= 1:
        print('Loading Challenge model...')
    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')
    for f in range(5):
        print(f'valid Fold {f}')
        model = load_challenge_model(model_folder, verbose,f) ### Teams: Implement this function!!!
        with open(f'test_fold{f}.csv', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            valid_id = [row[0] for row in reader]
        targeta_all=[]
        label_all=[]
    # Iterate over the patient files.
        for i in tqdm(range(num_patient_files)):
            # print(f'Running model on file {i+1} of {num_patient_files}...')
            patient_data = load_patient_data(patient_files[i])
            pid=get_patient_id(patient_data)  
            if pid in valid_id:
                recordings = load_recordings(data_folder, patient_data)
                murmur = get_murmur(patient_data)        
                target=1 if murmur=='Present' else 0
                targeta_all.append(target)
                label = run_challenge_model(model, patient_data, recordings, f) ### Teams: Implement this function!!!
                label_all.append(label)
            else:
                continue
                # t.update()
        labels_patients,target_patients=torch.tensor(label_all),torch.tensor(targeta_all),
        prc_seg=binary_auprc(labels_patients,target_patients)
        roc_seg=binary_auroc(labels_patients,target_patients)
        acc_seg=binary_accuracy(labels_patients,target_patients)
        f1_seg=binary_f1_score(labels_patients,target_patients)
        ppv_seg=binary_precision(labels_patients,target_patients)
        trv_seg=binary_recall(labels_patients,target_patients)
        cm_seg=binary_confusion_matrix(labels_patients,target_patients)
        print(f'----patient_wise---- \n acc={acc_seg:.3%}\n roc:{roc_seg:.3f}\n prc:{prc_seg:.3f}\n f1:{f1_seg:.3f}\n ppv:{ppv_seg:.3f}\n recall:{trv_seg:.3f}\n cm:')
        print(cm_seg)
        print('Done.')

if __name__ == '__main__':
    # # Parse the arguments.
    # if not (len(sys.argv) == 4 or len(sys.argv) == 5):
    #     raise Exception('Include the model, data, and output folders as arguments, e.g., python run_model.py model data outputs.')

    # # Define the model, data, and output folders.
    # model_folder = sys.argv[1]
    # data_folder = sys.argv[2]
    # output_folder = sys.argv[3]

    # # Allow or disallow the model to fail on parts of the data; helpful for debugging.
    # allow_failures = False

    # # Change the level of verbosity; helpful for debugging.
    # if len(sys.argv)==5 and is_integer(sys.argv[4]):
    #     verbose = int(sys.argv[4])
    # else:
    #     verbose = 1
    data_folder=r'D:\Shilong\murmur\Dataset\PCGdataset\training_data'
    model_folder=r'D:\Shilong\murmur\00_Code\LM\PCG_submit\model'

    run_model(model_folder, data_folder,verbose=2)
