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


## Model Architecture
#### Convolutional Layers
|   	Input| Input Size 	|Kernel   	|Stride   	| Num Kernels  	| Output Size  	|
|---	|---	|---	|---	|---	|---	|
|   Image	|   	150* 150 *3|  9 * 9 * 3 	|3   	| 64  	| 48 *48 *64  	|
|  Max Pool  	| 48* 48 *64  	|   2 * 2	| 2  	|   	|   	|   	|  	|   	|   	|
|---	|---	|---	|---	|---	|---	|
|   	|   	|  	|   	|   	|   	|
|   	|   	|   	|   	|   	|   	|
|   	|   	|   	|   	|   	|   	|
|   	|   	|   	|   	|   	|   	|
|   	|   	|   	|   	|   	|   	|
|   	|   	|   	|   	|   	|   	|
|   	|   	|   	|   	|   	|   	|
|   	|   	|   	|   	|   	|   	|
|   	|   	|   	|   	|   	|   	|
|   	|   	|   	|   	|   	|   	|
|   	|   	|   	|   	|   	|   	|
| 64  	| 24 * 24 * 64  	|
| Conv 1  	| 24 * 24 *64  	| 5 * 5 * 64  	| 1  	| 32  	| 20 * 20 *32  	|
|  Max Pool 1|20* 20* 32   	|3 * 3   	| 1  	| 32  	| 18 * 18 * 32  	|
|  Conv 2 	| 18 * 18 * 32  	| 3 * 3 * 32  	| 1  	| 16  	| 16 * 16 * 16  	|
|   Max Pool 2| 16 * 16 *16  	| 3 * 3  	| 1  	|  16 	| 14 * 14 * 16  	|

#### Dense layers
|   Input Layer	| Input Shape  	|   Output Shape	|   Output Layer 	|
|---	|---	|---	|---	|
|   Max Pool 2	|   14 * 14 * 16	|  3136  	|   Flatten	|
|  Flatten 	|  3136 	|   512	|   Dense 1	| 
|   Dense 1	|  512 	| 256  	| Dense 2  	|
|   Dense 2	| 256  	| 64  	| Dense 3  	|
|  Dense 3 	| 64  	| 6  	|   Final Output	|