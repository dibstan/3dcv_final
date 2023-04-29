# Final project for the lecture "Computer Vision: 3D Reconstruction"
 Winter term 2022/23
 by Konstantin Dibbern, Robin Jansen and Stefan Wahl

The scope of this project is to train a Unet for semantic segmentation of images from the Aerial Semantic Segmentation Drone dataset. for furter information about this data set, please refer to https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset and https://www.tugraz.at/index.php?id=22387. 


# Getting started:

To run this project, please create a new anaconda environment:

```bash
conda create -n FinalProjectEnv python=3.10 anaconda
```

Change to the directory where yu want to store the project and clone the repository and change to the root folder of the repository.

```bash
git clone https://github.com/dibstan/3dcv_final.git
cd ./3dcv_final
```

After that, activate the environmen and install the required dependencies:

```
conda activate FinalProjectEnv
pip install -r ./requirements.txt
```

# Run the Example Notebook

We provide a jupyter notebook which allows you to train and evaluate a UNet for the segmentation of drone images. To run this notebook, start the jupyter notebook server and open the notebook "main.ipynb". Please follow the instructions in the notebook to prepare the data set and to train the model.

```bash
jupyter notebook
```