import argparse
import glob 
import os

import pandas as pd
import numpy as np 

from tqdm import tqdm 

from multiprocessing import Pool
from functools import partial

import os
import numpy as np
import SimpleITK as sitk
import six

from radiomics import featureextractor, getTestCase


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



def make_result_table(subjectkey, result) -> pd.DataFrame:
    """
    summarizing the results from feature extractor as pandas dataframe
    """
    column_names = ['subjectkey']
    values = [subjectkey]
    for key, val in six.iteritems(result):
        if key.find('diagnostics') != -1: 
            pass
        else:
            column_names.append(key)
            values.append(val)
    df = pd.DataFrame([values], columns=column_names)
    return df 



def _feature_extracting(param_config_Dir, nifti_Dir, mask_Dir, CT_phase, n_process=1): 
    """
    A modlue consists of 
    1)getting image and mask files directories, 
    2)feature extracting, 
    3)summarizing results from feature extracting
    """

    # Instantiate feature extractor class with parameters cofiguration file
    extractor = featureextractor.RadiomicsFeatureExtractor(param_config_Dir)

    # getting subjects' CT image and mask image if only both type of images are exist
    nifti_mask_df: pd.DataFrame = get_imgFile_list(nifti_Dir, mask_Dir, CT_phase=CT_phase)
    subjects_keys: list = list(nifti_mask_df['subjectkey'].values)
    subjects_nifti_dir: list = list(nifti_mask_df['nifti_dir'].values)
    subjects_mask_dir: list = list(nifti_mask_df['mask_dir'].values)
    


    # running feature extractor by multiprocessing
    result = []
    with Pool(processes=n_process) as pool: 
        with tqdm(total = len(subjects_nifti_dir)) as pbar:
            for result_tmp in pool.imap_unordered(partial(_feature_extract_engine, extractor=extractor), zip(subjects_keys, subjects_nifti_dir, subjects_mask_dir)):
                result.append(result_tmp)
                pbar.update()
    result = pd.concat(result, axis=0)
    result = result.sort_values(by='subjectkey') # sorting by 'subjectkey' because feature extracting done by unordered multiprocessing 
    result = result.reset_index(drop=True)
    return result


## for multiprocessing function
def _feature_extract_engine(subj_images_dir, extractor):
    """
    An engine modlue consists of 
    1)getting image and mask files directories, 
    2)feature extracting, 
    3)summarizing results from feature extracting
    """
    subjectkey, subj_nifti_dir, subj_mask_dir = subj_images_dir
        
    # extracting features
    result_tmp = extractor.execute(subj_nifti_dir, subj_mask_dir)

    # summarize result
    result_tmp = make_result_table(subjectkey=subjectkey, result=result_tmp)
    return result_tmp


def make_dir(dir): 
    if os.path.exists(dir) == False: 
        os.mkdir(dir)


def argument_setting(): 
    parser = argparse.ArgumentParser() 

    parser.add_argument("--n_process", default=8, type=int, required=False, help="Increasing the number of processes speed up radiomics feature extracting")
    parser.add_argument("--param_config", default='/Users/wangheehwan/Desktop/kidney_radiomics/pipeline/params/Params2.yaml', type=str, required=True, help="Directory of configuration file for radiomics feature extraction")
    parser.add_argument("--nifti_Dir", default="/Users/wangheehwan/Desktop/kidney_radiomics/NifTi", type=str, required=True, help="Directory of Nifti images")
    parser.add_argument("--mask_Dir", default="/Users/wangheehwan/Desktop/kidney_radiomics/Segmentation", type=str, required=True, help="Directory of segmentation mask Nifti images")
    parser.add_argument("--save_Dir", default='/Users/wangheehwan/Desktop/kidney_radiomics/pipeline/result', type=str, required=True, help="Directory to save the result from radiomics feature extraction")
    parser.add_argument("--CT_phase", default=['A', 'N', 'D', 'P'], type=int,required=False, nargs="*", help="CT phases to use", choices=["A", "D", "N", "P"])
    
    args = parser.parse_args()
    return args  


if __name__ == '__main__':
    # setting arguments
    args = argument_setting()
    # make directories if not exist 
    make_dir(args.save_Dir)
    # running radiomics feature extraction
    for phase in args.CT_phase: 
        print('=> Starting Feature Extraction Stage')
        result_df: pd.DataFrame = _feature_extracting(param_config_Dir=args.param_config, nifti_Dir=args.nifti_Dir, mask_Dir=args.mask_Dir, CT_phase=phase, n_process=args.n_process)
        print('=> Done Feature Extraction Stage')
        save_result_dir = os.path.join(args.save_Dir, 'radiomics_features_CTphase_'+phase+'.csv')
        result_df.to_csv(save_result_dir, index=False)
        print("=> The Result from Feature Extraction Stage is Saved as '{}' ".format(save_result_dir)) 
        
