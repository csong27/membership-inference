# Membership Inference Attack against Machine Learning Models
This repository contains example of experiments for the paper Membership Inference Attack against Machine Learning Models (http://ieeexplore.ieee.org/document/7958568/). 

### Attack Experiment
python attack.py train_feat_file train_label_file

train_feat_file should be a text file where each line is a feature vector with floating point values separated by comma. 
train_label_file should be a text file where each line is a label with integer value. 

Once data is loaded, we will split the data for training target model and shadow models and save the split to disk. Then we will train the target model as well as shadow models. Finally attack model can be trained with predictions from shadow models and test on the target model. 
