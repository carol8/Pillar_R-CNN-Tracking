## Getting Started with CenterPoint on Waymo

### Prerequisite 

- Follow [INSTALL.md](INSTALL.md) to install all required libraries. 
- Waymo-open-dataset devkit

```bash
conda activate pillarnet 
pip install waymo-open-dataset-tf-2-4-0
```
# use tensorflow 2.4.1 to avoid core dumped issues
### Prepare data

#### Download data and organise as follows

```
# For Waymo Dataset         
└── WAYMO_DATASET_ROOT
       ├── tfrecord_training       
       ├── tfrecord_validation   
       ├── tfrecord_testing 
```

Convert the tfrecord data to pickle files.

```bash
# train set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path '/mnt/Database/datasets/waymo140/training/*.tfrecord'  --root_path 'data/Waymo/train/'

CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path '/mnt/Database/datasets/waymo140/training/*.tfrecord'  --root_path 'data/Waymo/train/'

# validation set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path '/mnt/Database/datasets/waymo140/tfrecord_validation/*.tfrecord'  --root_path 'data/Waymo/val/'

CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path '/mnt/Database/datasets/waymo140/validation/*.tfrecord'  --root_path 'data/Waymo/val/'

# testing set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path 'WAYMO_DATASET_ROOT/tfrecord_testing/*.tfrecord'  --root_path 'WAYMO_DATASET_ROOT/test/'

# my commands
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path '/home/cristian/dpercept/waymo_1.4.2/waymo_og_format_unpacked/validation/*.tfrecord'  --root_path '/home/cristian/dpercept/waymo_1.4.2/waymo_pickle_pillar_r-cnn/val' |& tee -a logs/preprocessing_val_output.txt
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path '/home/cristian/dpercept/waymo_1.4.2/waymo_og_format_unpacked/training/*.tfrecord'  --root_path '/home/cristian/dpercept/waymo_1.4.2/waymo_pickle_pillar_r-cnn/train' |& tee -a logs/preprocessing_train_output.txt
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path '/home/cristian/dpercept/waymo_1.4.2/waymo_og_format_unpacked/testing/*.tfrecord'  --root_path '/home/cristian/dpercept/waymo_1.4.2/waymo_pickle_pillar_r-cnn/test' |& tee -a logs/preprocessing_test_output.txt
```

Create a symlink to the dataset root 
```bash
mkdir data && cd data
ln -s WAYMO_DATASET_ROOT Waymo
```
Remember to change the WAYMO_DATASET_ROOT to the actual path in your system. 


#### Create info files

```bash
# Separating some of the dataset
python smart_copy.py /home/cristian/HDD/LocalDatasets/Waymo/test/annos/ /home/cristian/HDD/LocalDatasets/Waymo_0.05_sampled/test/annos/ --percentage 5
python smart_copy.py /home/cristian/HDD/LocalDatasets/Waymo/test/lidar/ /home/cristian/HDD/LocalDatasets/Waymo_0.05_sampled/test/lidar/ --percentage 5

python smart_copy.py /home/cristian/HDD/LocalDatasets/Waymo/val/annos/ /home/cristian/HDD/LocalDatasets/Waymo_0.05_sampled/val/annos/ --percentage 5
python smart_copy.py /home/cristian/HDD/LocalDatasets/Waymo/val/lidar/ /home/cristian/HDD/LocalDatasets/Waymo_0.05_sampled/val/lidar/ --percentage 5

python smart_copy.py /home/cristian/HDD/LocalDatasets/Waymo/train/annos/ /home/cristian/HDD/LocalDatasets/Waymo_0.05_sampled/train/annos/ --percentage 5
python smart_copy.py /home/cristian/HDD/LocalDatasets/Waymo/train/lidar/ /home/cristian/HDD/LocalDatasets/Waymo_0.05_sampled/train/lidar/ --percentage 5

# One Sweep Infos 
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --nsweeps=1 |& tee -a logs/infos_1sweep_train_preprocessing_output.txt

python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split val --nsweeps=1 |& tee -a logs/infos_1sweep_val_preprocessing_output.txt

python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split test --nsweeps=1 |& tee -a logs/infos_1sweep_test_preprocessing_output.txt

# Tracking Infos
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split test --nsweeps=1 |& tee logs/infos_tracking_log.txt

# Two Sweep Infos (for two sweep detection and tracking models)
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --nsweeps=2 |& tee -a logs/infos_2sweep_train_preprocessing_output.txt

python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split val --nsweeps=2 |& tee -a logs/infos_2sweep_val_preprocessing_output.txt

python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split test --nsweeps=2 |& tee -a logs/infos_2sweep_test_preprocessing_output.txt
```

In the end, the data and info files should be organized as follows

```
└── CenterPoint
       └── data    
              └── Waymo 
                     ├── tfrecord_training       
                     ├── tfrecord_validation
                     ├── train <-- all training frames and annotations 
                     ├── val   <-- all validation frames and annotations 
                     ├── test   <-- all testing frames and annotations 
                     ├── infos_train_01sweeps_filter_zero_gt.pkl
                     ├── infos_train_02sweeps_filter_zero_gt.pkl
                     ├── infos_val_01sweeps_filter_zero_gt.pkl
                     ├── infos_val_02sweeps_filter_zero_gt.pkl
                     ├── infos_test_01sweeps_filter_zero_gt.pkl
                     ├── infos_test_02sweeps_filter_zero_gt.pkl
```

### Train & Evaluate in Command Line

Use the following command to start a distributed training using 4 GPUs. The models and logs will be saved to ```work_dirs/CONFIG_NAME```. 

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py CONFIG_PATH
my command:
python -m torch.distributed.launch --nproc_per_node=1 ./tools/train.py configs/pillarrcnn/pillarrcnn_fpn_centerhead_waymo.py |& tee -a logs/train_output.txt
```

For distributed testing with 4 gpus,

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth 
my command
python -m torch.distributed.launch --nproc_per_node=1 ./tools/dist_test.py configs/pillarrcnn/pillarrcnn_fpn_centerhead_waymo.py --work_dir work_dirs/pillarrcnn_fpn_centerhead_waymo --checkpoint work_dirs/pillarrcnn_fpn_centerhead_waymo/latest.pth 
```

For testing with one gpu and see the inference time,

```bash
python ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth --speed_test 
my command
python ./tools/dist_test.py configs/pillarrcnn/pillarrcnn_fpn_centerhead_waymo.py --work_dir work_dirs/pillarrcnn_fpn_centerhead_waymo --checkpoint work_dirs/pillarrcnn_fpn_centerhead_waymo/latest.pth --speed_test |& tee -a logs/dist_test.txt 
python ./tools/dist_test.py configs/pillarrcnn/pillarrcnn_fpn_centerhead_waymo.py --work_dir work_dirs/pillarrcnn_fpn_centerhead_waymo --checkpoint work_dirs/pillarrcnn_fpn_centerhead_waymo/latest.pth --testset --speed_test |& tee -a logs/dist_test.txt 
```

For testing and outputting tracking info
```bash
python ./tools/dist_test.py configs/pillarrcnn/pillarrcnn_fpn_centerhead_waymo.py --work_dir work_dirs/pillarrcnn_fpn_centerhead_waymo --checkpoint work_dirs/pillarrcnn_fpn_centerhead_waymo/latest.pth --testset --speed_test |& tee -a logs/dist_test.txt
```

For visualising the predictions
```bash
python ./tools/visualise_predictions.py configs/pillarrcnn/pillarrcnn_fpn_centerhead_waymo.py --work_dir work_dirs/pillarrcnn_fpn_centerhead_waymo --checkpoint work_dirs/pillarrcnn_fpn_centerhead_waymo/latest.pth --testset |& tee logs/visualise_predictions_log.txt

```

This will generate a `my_preds.bin` file in the work_dir. You can create submission to Waymo server using waymo-open-dataset code by following the instructions [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md).  

If you want to do local evaluation (e.g. for a subset), generate the gt prediction bin files using the script below and follow the waymo instructions [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md).

```bash
python det3d/datasets/waymo/waymo_common.py --info_path data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl --result_path data/Waymo/ --gt
```

I found the instructions at https://github.com/waymo-research/waymo-open-dataset/blob/r1.3/docs/quick_start.md
Following them, I created a new repo with the waymo_open_dataset for evaluation in the same directory as this repo
I followed the instructions as indicated, using this command to build the evaluation script and use it to generate the metrics for this dataset
```bash
cd ../waymo-od/
bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main 
cd ..
waymo-od/bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main Pillar_R-CNN/work_dirs/pillarrcnn_fpn_centerhead_waymo/detection_pred.bin Pillar_R-CNN/data/Waymo/gt_preds.bin 
```
I had an error when building bazel, the following build command worked though:
bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main --copt=-Wno-error=array-bounds --copt=-Wno-error=array-parameter --verbose_failures


All pretrained models and configurations are in [MODEL ZOO](../configs/waymo/README.md).python det3d/datasets/waymo/waymo_common.py --info_path data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl --result_path data/Waymo/ --gt

### Second-stage Training 

Our final model follows a two-stage training process. For example, to train the two-stage CenterPoint-Voxel model, you first need to train the one stage model using [ONE_STAGE](../configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py) and then train the second stage module using [TWO_STAGE](../configs/waymo/voxelnet/two_stage/waymo_centerpoint_voxelnet_two_stage_bev_5point_ft_6epoch_freeze.py). You can also contact us to access the pretrained models, see details [here](../configs/waymo/README.md). 

### Tracking 

Please refer to options in [test.py](../tools/waymo_tracking/test.py). The prediction file is an intermediate file generated using [dist_test.py](../tools/dist_test.py) that stores predictions in KITTI lidar format. 

### Visualization 

Please refer to [visual.py](../tools/visual.py). It will take a prediction file generated by [simple_inference_waymo.py](../tools/simple_inference_waymo.py) and visualize the point cloud and detections.  

### Test Set 

Add the ```--testset``` flag to the end. 

```bash
python ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth --testset 
```
