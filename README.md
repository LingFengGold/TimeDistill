# TimeDistill
This repository contains the official implementation for the paper: "TimeDistill: Efficient Long-Term Time Series Forecasting with MLPs via Cross-Architecture Distillation."

## Usage
1. **Install requirements.** ```pip install -r requirements.txt``` or ```conda env create -f environment.yml```
2. **Download data.** You can download the all datasets from [Google Driver](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download), [Baidu Driver](https://pan.baidu.com/share/init?surl=r3KhGd0Q9PJIUZdfEYoymg&pwd=i9iy) or [Kaggle Datasets](https://www.kaggle.com/datasets/wentixiaogege/time-series-dataset) and put them in ./dataset/. **All the datasets are well pre-processed** and can be used easily. 
(The dataset links are sourced from other time series repositories on GitHub, which does not violate the anonymization policy.)
3. **Train the teacher model.** Experiment scripts for all benchmarks are available in the ./scripts directory. To obtain well-trained teacher model, run the corresponding script: 
```bash
bash ./run_scripts/train_teacher.sh
```
**Set "method" in "./run_scripts/train_teacher.sh" to the name of the specific teacher model.** Supported teacher models include: iTransformer, ModernTCN, TimeMixer, PatchTST, MICN, Fedformer, TimesNet, Autoformer. The trained parameters for the teacher model will be saved in the ./checkpoints/ folder for use in student MLP training.

4. **Train the student MLP.** Run the following scripts to train the student MLP for each dataset:

```bash
# MAKE SURE YOU HAVE TRAINED THIS TEACHER MODEL BY USING ABOVE "bash ./run_scripts/train_teacher.sh" BEFORE RUNNING SCRIPT BELOW.
bash ./run_scripts/train_student_iTransformer.sh # Teacher is iTransformer with tuned hyperparameter
bash ./run_scripts/train_student_ModernTCN.sh # Teacher is ModernTCN with tuned hyperparameter
bash ./run_scripts/train_student.sh # Customize Teacher. Specific the teacher model name using "model_t" in "./run_scripts/train_student.sh"
```
The above scripts default to running all datasets across all prediction lengths (96, 192, 336 ,720).

