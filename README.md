# Tuberculosis-Classification
Apply deep learning techniques to classify tuberculosis from chest x-ray images


## Object
To implement a image classication software to predict from chest x-ray whether there is a tubercolosis or not using deep learning technique.

## Requirement

## Preprocessing Data
Create binary classes from Pandas dataframe:

Connect dataframe with images in specific directory with flow_from_dataframe command:

### Augmentation
To generate more data to be trained in the model using this command:

## Model Implementation

1) [Sequential](https://github.com/Suchawit/Tuberculosis-Classification/blob/main/Model/Xray_Seq.py)
2) [ResNet](https://github.com/Suchawit/Tuberculosis-Classification/blob/main/Model/Xray_ResNet.py)
3) [Transfer learning, MobileNetV2](https://github.com/Suchawit/Tuberculosis-Classification/blob/main/Model/Xray_Transfer_Learning.py) 



## Result

These three models were trained for 100. From these results, it shows that Transfer Learning model obtained highest accuracy, but validation and training accuracies of other models are still slowly increasing.

### Sequential model

### ResNet model

### Transfer learning model



