# Cause 
There are some reasons that you should perform **Feature Extraction** stage with a python script named "feature_extraction.py".  
  
## 1. Incompatible Dependency 
If you running iPython notebook (**"kidney_radiomics.ipynb"**) with only python version 3.7 and install pyradiomics, you could perform **Feature Extraction** stage with iPython notebook.  
  
However, sklearn version 1.1+, which could be only compatible with only python 3.8+, only support ```n_features_to_select='auto'``` option in sklearn.feature_selection.SequentialFeatureSelector.  
  
In other words, **Feature Extraction** stage and **Feature Selection 2 & Model Selection** stage could not be run in a single iPython notebook with a single environment.  
  
  
## 2. Multiprocessing 
It's because iPython notebook do not support multiprocess which could dramatically speed up feature extraction process.  
  
Performing **Feature Extraction** stage with a single process require too long time to extract radiomics features. 
  
Thus, it is highly recommend to perform **Feature Extraction** with a python script named "feature_extraction.py" before running code blocks in iPython notebook (**"kidney_radiomics.ipynb"**).  


# Environment (anaconda)  
For **Feature Extraction** stage, use **"feature_extracion.yml"** which based on ```python == 3.7``` and contain ***pyradiomics*** python module.  
  
For running iPython notebook (**"kidney_radiomics.ipynb"**), use **"kidney_radiomics.yml"** which based on ```python >= 3.8``` and contain ```scikit-learn >= 1.1.0```.
  

# Usage 
## 1. Performing Feature Extraction with a stand alone python script 
```
$python3 feature_extraction.py --n_process 4 --param_config /Users/wangheehwan/Desktop/kidney_radiomics/pipeline/params/Params2.yaml --nifti_Dir /Users/wangheehwan/Desktop/kidney_radiomics/NifTi --mask_Dir /Users/wangheehwan/Desktop/kidney_radiomics/Segmentation --save_Dir /Users/wangheehwan/Desktop/kidney_radiomics/pipeline/result
```  
  
## 2. Running code blocks in iPython notebook while skipping Feature Extraction stage
running code blocks in the following procedure:   
- import modules and functions 
- perform **Split Dataset** stage. In this stage, loading csv files resulted from **Feature Extraction** stage and meta data files
- perform machine learning experiments 
- visualize the result



