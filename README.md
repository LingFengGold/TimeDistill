# TimeDistill
This repository contains the official implementation for the paper: **TimeDistill: Efficient Long-Term Time Series Forecasting with MLPs via Cross-Architecture Distillation**.

## Usage
1. **Install requirements.** ```pip install -r requirements.txt``` or ```conda env create -f environment.yml```
2. **Download data.** You can download the all datasets from [Google Driver](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download) and put them in ./dataset/. **All the datasets are well pre-processed** and can be used easily. 
3. **Train the teacher model.** To obtain well-trained teacher model, run the corresponding script: 
```bash
bash ./run_scripts/train_teacher.sh
```
**Set ```method``` in ```./run_scripts/train_teacher.sh``` to the specific teacher model name.** Supported teacher models include: ```iTransformer, ModernTCN, TimeMixer, PatchTST, MICN, Fedformer, TimesNet, Autoformer```. The trained parameters for the teacher model will be saved in the ```./checkpoints/``` folder for use in student MLP training.

4. **Train the student MLP.** Run the following scripts to train the student MLP for each dataset. **MAKE SURE YOU HAVE TRAINED THIS TEACHER MODEL BY USING ABOVE ```bash ./run_scripts/train_teacher.sh``` BEFORE RUNNING SCRIPT BELOW.**
```bash
bash ./run_scripts/train_student_iTransformer.sh # Teacher is iTransformer with tuned hyperparameter
bash ./run_scripts/train_student_ModernTCN.sh # Teacher is ModernTCN with tuned hyperparameter
bash ./run_scripts/train_student.sh # Customize Teacher. 
```
You can specific the teacher model name using ```model_t``` in ```./run_scripts/train_student.sh```. The above scripts default to running all datasets across all prediction lengths ```(96, 192, 336 ,720)```.

