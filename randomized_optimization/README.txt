CS7641: Randomized Optimization
============================
Kamolphan Liwprasert (kliwprasert3)
kliwprasert3@gatech.edu

OneDrive link: https://gtvault-my.sharepoint.com/:f:/g/personal/kliwprasert3_gatech_edu/EhB_NXdSPSlGu_DW-OqBu88Bi7yIuct-OT1YQRFleRmOIQ?e=loIRoH
(Need GT login)

Dataset Link
============================
Car Evaluation Dataset : https://archive.ics.uci.edu/ml/datasets/car+evaluation


Files
============================
├── kliwprasert3-analysis.pdf     *(analysis report) 
└── README.txt                    *(how to run)

### Jupyter Notebooks and Python code
├── car_evaluation.ipynb          *(main code for car_evaluation task)
├── external_fn.py                *(mandatory : Helper functions)
└── mental_health_survey.ipynb    *(main code for mental_health_survey task)

### Dataset files
└── car.data                      *(mandatory : Car Evaluation dataset)

### Environment files
└── requirements.txt              *(simplified version of requirements)

How to run
===========
The code is using Jypyter notebook. Please install Jypyter with `pip3 install jupyterlab`

Run with command:

`jupyter lab`

or

`jupyter notebook`


Environment Setup
==================
Requirements:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
jupyterlab
mlrose-hiive
```
Install with this command:
`pip3 install -r requirements.txt`

To get the latest version of the `mlrose-hiive` use this command additionally:
```
pip3 install git+https://github.com/hiive/mlrose.git
```

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
├── external_fn.py                *(mandatory : Helper functions)
├── kliwprasert3-analysis.pdf     *(mandatory : analysis report)
└── requirements.txt              *(simplified version of requirements)
```
