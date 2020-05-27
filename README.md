# Pneumonia_detector

Takes chest xray images (dicom)  for input and outputs the same image with a bounding box of where model detect pneumonia with confidence of 50% +
Please take a look at the notebook to see that. 

## to install for inference:

  #### $cd Pneumonia_detector
  #### $conda create -n pneumonia python=3.7
  #### $conda activate pneumonia
  #### $pip install -r requirements.txt

## to train:

 #### $mkdir data
 
 #### $mkdir model
 
 #### $cd data
 
 #### $wget https://pneumoniadata.s3.ca-central-1.amazonaws.com/rsna-pneumonia-detection-challenge1.zip
 
 #### $unzip rsna-pneumonia-detection-challenge1.zip
 
 #### $cd ..
 
 #### $conda create -n tf python=3.7
 
 #### $conda activate tf
 
 #### $pip install -r requirements_train.txt
 
 #### $python train.py