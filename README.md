# Lane Detection using CNN 
This project emphasises on detecting road lanes for autonomous vehicles.
This uses U-NET CNN model for road lane detection 

<img src="images/u-net-architecture.png" alt="U-NET Model" width="700" height="600" />

# Clone the project
```shell
git clone https://github.com/Sachdevabhavya/Lane-Detection-.git
```

# Setup of the project
```shell
pip install -r requirements.txt
```

# Flow of the project
1. Load the dataset in the data/processed folder 
2. There are two sets of datasets : train and val
3. Train folder contains images and its corresponding masks , stored in images and masks folder respectively
4. Val folder contains images and its corresponding masks , stored in images and masks folder respectively
5. Now execute the train.py script , which will load the dataset and load the model and start training the U-NET CNN model
```shell
python train.py
```



