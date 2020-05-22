[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transformer-networks-for-trajectory/trajectory-prediction-on-ethucy)](https://paperswithcode.com/sota/trajectory-prediction-on-ethucy?p=transformer-networks-for-trajectory)
# Transformer Networks for Trajectory Forecasting
This is the code for the paper **<a href="https://arxiv.org/abs/2003.08111">Transformer Networks for Trajectory Forecasting</a>**




## Requirements
  - Pytorch 1.0+
  - Numpy
  - Scipy
  - Pandas
  - Tensorboard
  - <a href="https://github.com/overshiki/kmeans_pytorch">kmeans_pytorch</a> (included in the project is a modified version)
#
## Usage

### Data setup
The dataset folder must have the following structure:

    - dataset
      - dataset_name
        - train_folder
        - test_folder
        - validation_folder (optional) 
        - clusters.mat (For quantizedTF)
### Individual Transformer
To train just run the *train_individual.py* with different parameters

example: to train on the data for eth
```
CUDA_VISIBLE_DEVICES=0 python train_individualTF.py --dataset_name eth --name eth --max_epoch 240 --batch_size 100 --name eth_train --factor 1
```

### QuantizedTF
#### Step1: Create the clusters
```
NOTE: We used a pytorch based method that use GPUs to lower the computational time, but it requires both a GPU and a high amount of RAM (25 GB).
Since clusters do not change over time they can be created with any code, you just need to create a file with the centroids inside the dataset/dataset_name folder

For ease of use the cluster informations are already upladed for eth+ucy
```

To create the cluster_mat file run *kmeans.py*
```
CUDA_VISIBLE_DEVICES=0 python kmeans.py --dataset_name eth
```
After that put the clusters.mat inside the appropriate dataset folder.

### Step 2: Train the quantized
Run *ClassifyTF.py*

```
CUDA_VISIBLE_DEVICES=0 python train_quantizedTF.py --dataset_name zara1 --name zara1 --batch_size 1024
```

### Step 3: Evaluate Best-of-N
Run *test_class.py* with the parameters for the dataset_name, the name of the trained model, the epoch to test and the number of samples


```
CUDA_VISIBLE_DEVICES=0 python test_quantizedTF.py --dataset_name eth --name eth --batch_size 1024 --epoch 00030 --num_samples 20
```


## Visualization
The training loss, validation loss, mad and fad for the test can be seen for each epoch by running tensorboard
```
tensorboard --logdir logs
```
#


## Citation
If you use the code please cite our paper.
```
@misc{giuliari2020transformer,
    title={Transformer Networks for Trajectory Forecasting},
    author={Francesco Giuliari and Irtiza Hasan and Marco Cristani and Fabio Galasso},
    year={2020},
    eprint={2003.08111},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

# Thanks




## TODO
- [x] Add BERT
- [x] Add QuantizedBert
- [ ] Upload Pretrained-Models

## Changelog
 - 14/05
   - Added Quantized Bert
 - 27/04
   - Added Bert
   - Renamed the training files to make more sense
   - fixed some issues with the individualTF
 - 10/04 
   - Uploaded the code for the Individual and QuantizedTF
  

