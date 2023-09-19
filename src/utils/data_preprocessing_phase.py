# basic modules
import os
import glob 
import random
import copy
from tqdm import tqdm 
from typing import Dict, Tuple
import six 

# for data processing
import numpy as np
import pandas as pd


def get_imgFile_list(nifti_Dir, mask_Dir, CT_phase='P') -> pd.DataFrame:
    """
    getting image files' directories and segmentation mask files' directories according to CT phase
    """
    nifti_subject_list = [] 
    nifti_img_list = []
    mask_subject_list = []
    mask_img_list = []

    nifti_subject_dir_list = glob.glob(os.path.join(nifti_Dir,'*'))
    for i, nifti_subject_dir in enumerate(nifti_subject_dir_list):
        subj_nifti_list = glob.glob(os.path.join(nifti_subject_dir, "*"))
        for subj_nifti in subj_nifti_list: 
            if os.path.split(subj_nifti)[-1].find('_'+CT_phase) != -1: 
                # getting subject name 
                nifti_subject_list.append(os.path.split(os.path.split(subj_nifti)[0])[-1])
                # getting specific CT phase image from subject folder 
                nifti_img_list.append(subj_nifti)
    nifti_df = pd.DataFrame({'subjectkey': nifti_subject_list, 'nifti_dir': nifti_img_list})

    mask_subject_dir_list = glob.glob(os.path.join(mask_Dir,'*'))
    for i, mask_subject_dir in enumerate(mask_subject_dir_list):
        subj_mask_list = glob.glob(os.path.join(mask_subject_dir, "*"))
        for subj_mask in subj_mask_list: 
            if os.path.split(subj_mask)[-1].find('_'+CT_phase) != -1: 
                # getting subject name 
                mask_subject_list.append(os.path.split(os.path.split(subj_mask)[0])[-1])
                # getting specific CT phase image from subject folder 
                mask_img_list.append(subj_mask)
    mask_df = pd.DataFrame({'subjectkey': mask_subject_list, 'mask_dir': mask_img_list})

    return pd.merge(nifti_df, mask_df, how='inner', on='subjectkey') 




def loading_datasets_OnePhase(save_Dir, CT_phase: str) -> Dict[str, pd.DataFrame]: 
    dataset = {}
    dataset_tmp = pd.read_csv(os.path.join(save_Dir, 'radiomics_features_CTphase_'+ CT_phase + '.csv'))
    dataset[CT_phase] = dataset_tmp
    return dataset


def assign_train_test_BinaryClassification(meta_data: pd.DataFrame, train_ratio=0.7):
    num_total_control = len(meta_data[meta_data['Pathology_binary'] == 0])
    num_total_case = len(meta_data[meta_data['Pathology_binary'] == 1])
    num_train_control = int(num_total_control * train_ratio)
    num_test_control = num_total_control - num_train_control

    num_train_case = int(num_total_case * train_ratio)
    num_test_case = num_total_case - num_train_case

    train_subject = []
    test_subject = []

    train_control_count = 0 
    train_case_count = 0 
    test_control_count = 0
    test_case_count = 0  
    
    #assert len(meta_data[meta_data['CT phase'] >= 3]) > len(meta_data[meta_data['CT phase'] < 3])
    #print("Train subjects (case/control): {}/{}. Test subjects (case/control): {}/{}.".format(num_train_case, num_train_control, num_test_case, num_test_control))
    for i in range(len(meta_data)):
        if meta_data.iloc[i]['Pathology_binary'] == 0: 
            if test_control_count <= num_test_control: 
                test_subject.append(meta_data.iloc[i]['subjectkey'])
                test_control_count += 1
            else:  
                train_subject.append(meta_data.iloc[i]['subjectkey'])
                train_control_count += 1
        elif meta_data.iloc[i]['Pathology_binary'] == 1: 
            if test_case_count <= num_test_case: 
                test_subject.append(meta_data.iloc[i]['subjectkey'])
                test_case_count += 1
            else:  
                train_subject.append(meta_data.iloc[i]['subjectkey'])
                train_case_count += 1
    """
    for i in range(len(meta_data)):
        phase_count = meta_data.iloc[i]['CT phase']
        # assign subjects having single or 2 phase CT scans only to test set 
        if phase_count == 3: 
            if meta_data.iloc[i]['Pathology_binary'] == 0: 
                if test_control_count <= num_test_control: 
                    test_subject.append(meta_data.iloc[i]['subjectkey'])
                    test_control_count += 1
                else:  
                    train_subject.append(meta_data.iloc[i]['subjectkey'])
                    train_control_count += 1
            elif meta_data.iloc[i]['Pathology_binary'] == 1: 
                if test_case_count <= num_test_case: 
                    test_subject.append(meta_data.iloc[i]['subjectkey'])
                    test_case_count += 1
                else:  
                    train_subject.append(meta_data.iloc[i]['subjectkey'])
                    train_case_count += 1
        elif phase_count >= 3:
            if  meta_data.iloc[i]['Pathology_binary'] == 0:
                if train_control_count <= num_train_control: 
                    train_subject.append(meta_data.iloc[i]['subjectkey'])
                    train_control_count += 1
                else: 
                    test_subject.append(meta_data.iloc[i]['subjectkey'])
                    test_control_count += 1                     
            elif meta_data.iloc[i]['Pathology_binary'] == 1: 
                if train_case_count <= num_train_case: 
                    train_subject.append(meta_data.iloc[i]['subjectkey'])
                    train_case_count += 1 
                else: 
                    test_subject.append(meta_data.iloc[i]['subjectkey'])
                    test_case_count += 1                  
    """

    return train_subject, test_subject

"""
def assign_train_test_BinaryClassification(meta_data: pd.DataFrame, train_ratio=0.7, sampling_method=None):
    num_total_control = len(meta_data[meta_data['Pathology_binary'] == 0])
    num_total_case = len(meta_data[meta_data['Pathology_binary'] == 1])
    num_train_control = int(num_total_control * train_ratio)
    num_test_control = num_total_control - num_train_control
    if sampling_method == 'undersampling':
        assert int(num_total_case * train_ratio) > num_train_control        # only if (the number of case > the number of train) 
        assert num_total_case - int(num_total_case * train_ratio) > num_test_control     # only if (the number of case > the number of train) 
        num_train_case = num_train_control
        num_test_case = num_test_control
    else:
        num_train_case = int(num_total_case * train_ratio)
        num_test_case = num_total_case - num_train_case
    
    print("Train subjects (case/control): {}/{}. Test subjects (case/control): {}/{}.".format(num_train_case, num_train_control, num_test_case, num_test_control))

    train_subject = []
    test_subject = []

    train_control_count = 0 
    train_case_count = 0 
    test_control_count = 0
    test_case_count = 0  
    
    assert len(meta_data[meta_data['CT phase'] >= 3]) > len(meta_data[meta_data['CT phase'] < 3])

    
    for i in range(len(meta_data)):
        phase_count = meta_data.iloc[i]['CT phase']
        # assign subjects having 3 or 4 phase CT scans only to test set 
        if phase_count >= 3: 
            if meta_data.iloc[i]['Pathology_binary'] == 0: 
                if test_control_count <= num_test_control: 
                    test_subject.append(meta_data.iloc[i]['subjectkey'])
                    test_control_count += 1
                else:  
                    train_subject.append(meta_data.iloc[i]['subjectkey'])
                    train_control_count += 1
            elif meta_data.iloc[i]['Pathology_binary'] == 1: 
                if test_case_count <= num_test_case: 
                    test_subject.append(meta_data.iloc[i]['subjectkey'])
                    test_case_count += 1
                else:  
                    train_subject.append(meta_data.iloc[i]['subjectkey'])
                    train_case_count += 1
        elif phase_count < 3:
            if  meta_data.iloc[i]['Pathology_binary'] == 0:
                if train_control_count <= num_train_control: 
                    train_subject.append(meta_data.iloc[i]['subjectkey'])
                    train_control_count += 1
                else: 
                    test_subject.append(meta_data.iloc[i]['subjectkey'])
                    test_control_count += 1                     
            elif meta_data.iloc[i]['Pathology_binary'] == 1: 
                if train_case_count <= num_train_case: 
                    train_subject.append(meta_data.iloc[i]['subjectkey'])
                    train_case_count += 1 
                else: 
                    test_subject.append(meta_data.iloc[i]['subjectkey'])
                    test_case_count += 1                  


    return train_subject, test_subject
"""



def preparing_dataset(meta_data, dataset, train_subject_list, test_subject_list): 
    meta_data = meta_data[['subjectkey', 'Pathology_binary']]

    train_X, train_y = [], []
    test_X, test_y = [], []
    for phase in dataset.keys(): 
        #print(dataset[phase])
        dataset_tmp = pd.merge(meta_data, dataset[phase], on='subjectkey', how='inner')

        dataset_train = dataset_tmp.loc[dataset_tmp['subjectkey'].isin(train_subject_list)]     # fet radiomics features of train subjects
        features_train = dataset_train.drop(['subjectkey', 'Pathology_binary'], axis=1).values       # get radiomics features of train subjects as numpy array 
        train_X.append(features_train)
        labels_train = dataset_train['Pathology_binary'].values       # get labels of train subjects as numpy array 
        labels_train = np.where(labels_train == 0, 0, 1)       # change label "benign" to 0 and "malignancy" to 1 
        train_y.append(np.expand_dims(labels_train, axis=-1))
        
        dataset_test = dataset_tmp.loc[dataset_tmp['subjectkey'].isin(test_subject_list)]     # fet radiomics features of test subjects
        features_test = dataset_test.drop(['subjectkey', 'Pathology_binary'], axis=1).values       # get radiomics features of test subjects as numpy array
        test_X.append(features_test)
        labels_test = dataset_test['Pathology_binary'].values       # get labels of train subjects as numpy array 
        labels_test = np.where(labels_test == 0, 0, 1)       # change label "benign" to 0 and "malignancy" to 1  
        test_y.append(np.expand_dims(labels_test, axis=-1))
    
    train_X, train_y = np.vstack(train_X), np.vstack(train_y)
    test_X, test_y = np.vstack(test_X), np.vstack(test_y)

    return train_X, train_y, test_X, test_y 





