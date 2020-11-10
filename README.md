# Tuberculosis-Classification
Apply deep learning techniques to classify tuberculosis from chest x-ray images

<img src="https://github.com/Suchawit/Tuberculosis-Classification/blob/main/Img/Chest_X.PNG" width="500px"/>

## Object
To implement a image classication software to predict from chest x-ray whether there is a tubercolosis or not using deep learning technique.

## Dataset


## Requirement

## Preprocessing Data
Create binary classes from Pandas dataframe:

<img src="https://github.com/Suchawit/Tuberculosis-Classification/blob/main/Img/Create_binaryclasses.PNG" width="600px"/>

Connect dataframe with images in specific directory with flow_from_dataframe command:

<img src="https://github.com/Suchawit/Tuberculosis-Classification/blob/main/Img/Flow_from_dataframe.PNG" width="600px"/>

### Augmentation
To generate more data to be trained in the model using this command:

<img src="https://github.com/Suchawit/Tuberculosis-Classification/blob/main/Img/Augmentation.PNG" width="600px"/>

From augmenting data

<img src="https://github.com/Suchawit/Tuberculosis-Classification/blob/main/Img/Augmentation_pic.PNG" width="600px"/>

## Model Implementation

1) [Sequential](https://github.com/Suchawit/Tuberculosis-Classification/blob/main/Model/Xray_Seq.py)
2) [ResNet](https://github.com/Suchawit/Tuberculosis-Classification/blob/main/Model/Xray_ResNet.py)
3) [Transfer learning, MobileNetV2](https://github.com/Suchawit/Tuberculosis-Classification/blob/main/Model/Xray_Transfer_Learning.py) 



## Result

These three models were trained for 100. From these results, it shows that Transfer Learning model obtained highest accuracy, but validation and training accuracies of other models are still slowly increasing.

### Sequential model

<img src="https://github.com/Suchawit/Tuberculosis-Classification/blob/main/Img/Squential_pic.png" width="600px"/>

### ResNet model

<img src="https://github.com/Suchawit/Tuberculosis-Classification/blob/main/Img/ResNet_pic.png" width="600px"/>

### Transfer learning model

<img src="https://github.com/Suchawit/Tuberculosis-Classification/blob/main/Img/Transfer_Learning_pic.png" width="600px"/>

