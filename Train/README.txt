The procedure to install and run the training scripts is provided below.

**********************************
INSTALLATION
**********************************

1. Install Anaconda3 (A distribution of Python3, please see https://www.anaconda.com/distribution/)
2. Create a new virtual environment:
    2.1. Open terminal (Ubuntu)
    2.2. Run "conda create -n envname python=3.6"
    2.3. Install packages:
        2.3.1 lmdb          "conda install -c conda-forge python-lmdb"
        2.3.2 pillow        "conda install -c anaconda pillow"
              (1) openjpeg for JPEG2000 support            "conda install -c conda-forge openjpeg"
              (2) pillow                                   " python -m pip install --upgrade Pillow --global-option="build_ext" --global-option="--enable-jpeg2000" "     
        2.3.3 PyTorch (please install first and uninstall pillow then install as instructed)  "conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=11.0 -c pytorch"
        2.3.4 scipy         "conda install -c anaconda scipy"
        2.3.5 numpy         "conda install -c anaconda numpy" 
        2.3.6 pandas        "conda install -c anaconda pandas"
        2.3.7 tensorboardX  "conda install -c conda-forge  tensorboardx"
        2.3.8 tensorboard   "pip install tensorboard"
	   
        

**********************************
USAGE
**********************************

1. Open terminal (Ubuntu)
2. Activate the virtual environment: "conda activate envname"
3. 
   (1)run FReonss(not tried, lr is just a guess):
       python MainNoDT.py --lr 0.0001 --batch_size 50 --model FReonss --dataset_root ../waterloo2_lmdb --ckpt_path ./checkpoint_FReonss_StepLR_CRIQAV3_1e-4 --board ./board_FReonss_StepLR_CRIQAV3_1e-4
   (2) run EONSS:
       python MainNoDT.py --lr 0.00001 --batch_size 50 --model eonss --dataset_root ../waterloo2_lmdb --ckpt_path ./checkpoint_eonss_StepLR_CRIQAV3_1e-5 --board ./board_eonss_StepLR_CRIQAV3_1e-5
   (3) FR_deepIQA (have bug now):
       python MainNoDT.py --lr 0.0001 --batch_size 4 --model FR_deepIQA --dataset_root ../waterloo2_lmdb --ckpt_path ./checkpoint_deepIQA_StepLR_CRIQAV3_1e-4 --board ./board_deepIQA_StepLR_CRIQAV3_1e-4
   The "--dataset_root" should be config to point to the database containing rocksdb/lmdb packages
   The "--ckpt_path" point to your directory to store output model
   The "--board" point to your directory to store loss log 

**********************************
EXPERIMENTS
**********************************
Idea1, Resnet
(1). Resnet is used to extract high level image features
(3). Use another NN map image feature and metadata jointly to Pawpularity score 


Idea2, A Combination of Resnet and NN1
(1). Resnet is used to extract high level image features
(2). NN1 is used to map metadata to features
(3). Use another NN, NN2, map image feature and mapped metadata jointly to Pawpularity score 

Some new ideas in literature to do this?


################################
BUGS
################################



