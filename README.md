# HYKA-ML classfication model for skin lesions and eye damage
## Environment set up
0. Make a new folder on your computer and clone the HYKA-ML classfication mode repository from GitHub:

        git clone https://github.sydney.edu.au/hngu5920/usyd_cs10_2.git
        cd usyd_csv10_2
1. Download the pre-trained MViTv2-T  model from https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_T_in1k.pyth or our project pre-trained mdoels into "*./checkpoints*" folder
2. Download the ImageNet images OR our project datasets and unzip into *./mvit/DATA* folder.
3. Download and Install Anaconda Distribution from https://www.anaconda.com/products/distribution
4. Open Anaconda Prompt , with hyka_ml as the name of the new environment, type in: 

        conda create -n hyka_ml python=3.8
5. Activate the new Conda env: 

        conda activate hyka_ml
6. Install the required packages, type in: 

        pip install -r pip_modules_list.txt
7.  From the project folder, build the mvit module: 

        cd mvit
        python setup.py build develop
        cd ..

## Configure the model training or transfer learning/prediction parameters
Create a config.ini file using the following configurations - a copy of the configurations are also in config.ini.bak file:

        [MODEL]
        NUM_CLASSES = 5
        OUTPUT_DIR = "./checkpoints"
        MODEL_NAME = "MViTv2_T_in1k_skin.pyth"

        [DATA]
        DATA_TYPE = "skin"
        PATH_TO_DATA_DIR = "./mvit/DATA/01_lesions"

        [TRAIN]
        BATCH_SIZE = 65

        [TEST]
        BATCH_SIZE = 65

        [SOLVER]
        MAX_EPOCH = 100

        [FREEZE]
        BLOCKS_FROZEN = 5

        [WARMUP]
        WARMUP_EPOCHS = 0

        [CHECKPOINT]
        RESET_CHECKPOINT = 0

        [DATA_AUG]
        COLOR_JITTER = 0.4
        RAND_AUG = None
        ERASE = 0.25
        NUM_COUNT = 1
        MIXUP = False

- *NUM_CLASSES* : the number of classes which are being classified e.g. **5** for skin lesions and **7** for eye damage
- *OUTPUT_DIR*  : the output directory of the model checkpoints
- *MODEL_NAME*  : the filename of the model including its extension *pyth*
- *DATA_TYPE*   : *skin* or *eye* data to use the correct labelling
- *PATH_TO_DATA_DIR* : a data folder with the below structure
                
        /path/to/imagenet-1k/
            train/
                class1/
                        img1.jpeg
                class2/
                        img2.jpeg
            val/
                class1/
                        img3.jpeg
                class2/
                        img4.jpeg
- *BATCH_SIZE* : the number of images being loaded and trained at a time
- *MAX_EPOCH* : the number of times to go through the entire dataset to train
- *BLOCKS_FROZEN* : the number of blocks (of deep learning layers) to freeze to enable transfer learning, 0 means re-training all layers and use no transfer learning
- *WARMUP_EPOCHS* : the number of epochs the model takes to reach a fixed learning rate; currently set at 0.00025
- *RESET_CHECKPOINT*: to the epoch count of a pre-trained model for ease of performing many ablation studies without manipulating the MAX_EPOCH number
---

## Transfer learning
Use a pre-trained MViTv2 model on new datasets. 

Our project team use MViTv2 codes from https://github.com/facebookresearch/mvit which is the official PyTorch implementation of MViTv2 from [MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](https://arxiv.org/abs/2112.01526).  We then customised heavily on the final layer of the model structure to apply transfer learning on skin lesions and eye damage classification problems.

To perform transfer learning, from the root project folder, run as below (for Windows):

        conda activate hyka_ml
        python train.py  

Re-configure the config.ini for skin lesion or eye damage datasets and models appropriately

**NOTE**:  The train.py module runs best on a decent GPU with sufficient VRAM e.g. we generally use NVIDIA RTX2060+ with 8GB+ VRAM as the minimum in our tests or **Artemis HPC** which is described briefly in the below section. Any hardware configurations with lower specs will require lower *BATCH_SIZE* and a lot more time for training.

---

## Predict
### Run with test data
1. Download the "balanced" datasets from the [OneDrive Link](https://unisydneyedu-my.sharepoint.com/:f:/g/personal/mjun9806_uni_sydney_edu_au/EjxT3hVq5adPlUEWrXXHXrgBB8D1HDaf9XpS1h0rcUSQog).
2. Download the final model weights from the [OneDrive Link](https://unisydneyedu-my.sharepoint.com/:f:/g/personal/mjun9806_uni_sydney_edu_au/EjzpIVQZ--dAu4aQqSwWc_oBz5_Td9toZN_qTxTcGrNh4g). The model weights you are looking for are inside the folders with **Final** prefix and ending with `.pyth` extension.
3. Unzip them and put them into directory of your choice.
4. Overwrite `config.ini` file with either `pred_01_lesions.ini` (to classify skin lesions) or `pred_02_eyes.ini` (to classify eye damage).
5. Change the input file paths inside `config.ini` for **model weights** as well as **data** to match the location you saved the files from step 1 to 3.
6. Use the below command to run prediction:

        python predict.py
7. The results will be saved in `logs/<timestamp_of_execution>`. There will be 2 confusion matrices and a `.log` file that holds F1 score results.


### Run with unknown classes
1. Inside `test` folder of the dataset, delete all the images inside each class label folder.
2. Put your new images with unknown classes into **any** one of the empty class label folder. This is purely due to the way MViT2's data loader works.
3. Run the prediction with the following command:

        python predict.py
4. The prediction results will be saved in `logs/<timestamp_of_execution>` as `prediction.csv`.
5. The left column is irrelevant because it just shows you where it sourced the image from. If the class labels are unknown and you are using our program to classify unknown diseases, refer to the **right column only** as those are the results predicted by our model.



---

## Artemis HPC setup
### Environment setup
- Artemis does not allow direct download of latest packages using **pip** or **conda** commands but it supports a Singularity container which allows us to build any env from scratch on our local PC and run it here.
- Singularity needs permission from USyd ICT to be added
- Once the permission is granted:

        module load singularity
- The instruction on how to build the compatible Singularity container to run our MViTv2 codes are described in the below section.
- A copy of the working container is included in the **Project Artifacts** folder/zip file. 
- A QSub job submission file, **run_model.pbs** ,is needed to execute a training session on Artemis using the settings as below or customised as needed:

        #!/bin/bash
        # Create indexes for reference sequence
        #PBS -P HYKA_ML
        #PBS -N testrun_1
        #PBS -l select=1:ncpus=1:ngpus=1:mem=16gb
        #PBS -l walltime=05:00:00
        #PBS -q defaultQ
        #PBS -M <youremail@gmail.com>
        #PBS -m abe

        # Load modules
        module load singularity

        cd $PBS_O_WORKDIR

        #Mount the current directory in the container at /project, change directories and run your script
        export hykaml_main_dir="/project/HYKA_ML"; \
        singularity exec --nv -B $PBS_O_WORKDIR:$hykaml_main_dir $hykaml_main_dir/hyka_ml.sif  \
        /bin/bash -c "cd $hykaml_main_dir/artemis_usyd_cs10_2 && PYTHONPATH=$hykaml_main_dir/artemis_usyd_cs10_2/mvit && python train_gpu.py"

- Finally, to run use:

        qsub run_model.pbs

- Monitor jobs: 
	
        jobstat

- Monitor queue:
        
        qstat
        

Source:  [Sydney Informatics Hub](https://sydney-informatics-hub.github.io/training.artemis.introhpc/04-submitting-jobs/)

### Singularity container setup

- Install Singularity:

        https://docs.sylabs.io/guides/3.7/admin-guide/installation.html

- Build Singularity container:

        https://docs.sylabs.io/guides/3.7/user-guide/build_a_container.html



---

## License

This HYKA ML project was developed by the University of Sydney CS10-02 project team of Semester 2, 2022. Its full copyright and ownership belongs to our client, Metasense.

---

## Acknowledgement

> This repository is built based on the [MViT GitHub repository](https://github.com/facebookresearch/mvit)

> The authors acknowledge the Sydney Informatics Hub and the University of Sydney’s high performance computing cluster, Artemis, for providing the computing resources that have contributed to the results reported herein.

> Big thanks to the expert advice from our supervisor Clinton Mo, constructive feedback from our client representatives Reza and Gordon and the ingenuity, resourcefulness, dedication and skills from the CS10-02 (semester 2, 2022) project team, namely, Lucas Fang (430174571), 	Haseeb Munir Malik (510274212), Minjun Jung (510668802), Yu Hao (490599846), Kenric Nguyen (490341832) and Jung Hyung Lee (510560643) who have all contributed towards the successful delivery of this project.

