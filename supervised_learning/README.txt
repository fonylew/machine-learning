CS7641: Supervised Learning
============================
Kamolphan Liwprasert (kliwprasert3)

kliwprasert3@gatech.edu

GTID: 903457032

Dataset Link
============================
Car Evaluation Dataset : https://archive.ics.uci.edu/ml/datasets/car+evaluation

OSMI Mental Health in Tech Survey : https://www.kaggle.com/osmi/mental-health-in-tech-survey

Git Repository (codes only) : https://github.com/fonylew/machine-learning/

Files
============================
├── kliwprasert3-analysis.pdf     *(analysis report) 
└── README.txt                    *(how to run)

### Jupyter Notebooks and Python code
├── car_evaluation.ipynb          *(main code for car_evaluation task)
    https://github.com/fonylew/machine-learning/blob/master/supervised_learning/car_evaluation.ipynb
├── external_fn.py                *(mandatory : Helper functions)
    https://github.com/fonylew/machine-learning/blob/master/supervised_learning/external_fn.py
└── mental_health_survey.ipynb    *(main code for mental_health_survey task)
    https://github.com/fonylew/machine-learning/blob/master/supervised_learning/mental_health_survey.ipynb

### Dataset files
├── car.data                      *(mandatory : Car Evaluation dataset)
└── datasets_311_673_survey.csv   *(mandatory : Mental Health dataset)

### Environment files
├── environment.yml               *(conda environment file)
└── requirements.txt              *(simplified version of requirements)

How to run
============================

`jupyter lab`

or

`jupyter notebook`


Environment Setup
============================
Conda
------
Create virtual environment using Conda, or Miniconda
```bash
conda create -n ml pandas numpy scikit-learn matplotlib scipy seaborn
```
Activate the virtual environment
```bash
conda activate ml
```
Install additional packages:
 - Jupyter or Jupyterlab, if not exist
```bash
conda install jupyter jupyterlab
```

Or using Pip
------------
```
# requirements.txt
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
jupyterlab
```
`pip3 install -r requirements.txt`


Directory Structure
============================
```
.
├── README.txt                    *(mandatory : How to run)
├── car.c45-names                  (from UCI website, not use in code)
├── car.data                      *(mandatory : Car Evaluation dataset)
├── car.names                      (from UCI website, not use in code)
├── car_evaluation.ipynb          *(mandatory : main code for car_evaluation task)
├── car_evaluation_result.json     (auto-generated from code)
├── car_plots/                     (auto-generated from code)
├── datasets_311_673_survey.csv   *(mandatory : Mental Health dataset)
├── environment.yml               *(conda environment file)
├── external_fn.py                *(mandatory : Helper functions)
├── kliwprasert3-analysis.pdf     *(mandatory : analysis report)
├── mental_health_result.json      (auto-generated from code)
├── mental_health_survey.ipynb    *(mandatory : main code for mental_health_survey task)
├── mental_plots/                  (auto-generated from code)
└── requirements.txt              *(simplified version of requirements)
```
