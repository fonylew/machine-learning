CS7641: Supervised Learning
============================
Kamolphan Liwprasert (kliwprasert3)
kliwprasert3@gatech.edu

OneDrive link: https://gtvault-my.sharepoint.com/:f:/g/personal/kliwprasert3_gatech_edu/EoKdJ7Mml-xOggJ4DBRA-JcBZ0e3RgUSMGObElfKapFUKg?e=87vcrb
(Need GT login)

Dataset Link
============================
Car Evaluation Dataset : https://archive.ics.uci.edu/ml/datasets/car+evaluation

OSMI Mental Health in Tech Survey : https://www.kaggle.com/osmi/mental-health-in-tech-survey

Code files : https://gtvault-my.sharepoint.com/:f:/g/personal/kliwprasert3_gatech_edu/EoKdJ7Mml-xOggJ4DBRA-JcBZ0e3RgUSMGObElfKapFUKg?e=87vcrb

Files
============================
├── kliwprasert3-analysis.pdf     *(analysis report) 
└── README.txt                    *(how to run)

### Jupyter Notebooks and Python code
├── car_evaluation.ipynb          *(main code for car_evaluation task)
├── external_fn.py                *(mandatory : Helper functions)
└── mental_health_survey.ipynb    *(main code for mental_health_survey task)

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
Install with this command:
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
