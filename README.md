### Image Classification with Convolutional Neural Network

### Conda Environment Setup
`conda create --name <yourenvname> python=3.7.4` <br>
`conda activate <yourenvname>`

#### Conda will setup your gpu and install tensorflow-gpu and other packages
`conda install --yes --file requirements.txt`

#### kaggle package is not available in conda so you need to install it with pip, however conda env supports pip command
`pip install kaggle==1.5.6`

#### Note
For serving with environment with GPU set environment variable
> `$ export TF_FORCE_GPU_ALLOW_GROWTH=true`


### Data Download
#### Note: Download your api keys (kaggle.json) from kaggle and paste it inside `~/.kaggle/`
> `$ sh data_download.sh`


# Backend

### Model Training
`cd backend/src` <br>
`python model.py`
#### Note: Everytime you run model.py starts, it starts training from latest checkpoint. Only the latest model weights will be saved.

### Monitor with tensorboard (optional)
> `$ tensorboard --logdir=temp`

### Server
`cd backend` <br>
`python app.py`

## Frontend
`cd frontend` <br>
`npm i && npm start`