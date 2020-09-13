# CS7641: Supervised Learning
Kamolphan Liwprasert (kliwprasert3)

kliwprasert3@gatech.edu

GTID: 903457032

# 0 Online Notebook
Link: (read-only)
# 1 Environment Setup
## Conda
Create virtual environment using Conda, or Miniconda
```bash
conda create -n ml pandas numpy scikit-learn matplotlib tensorflow scipy 
```
Activate the virtual environment
```
conda activate ml
```
Install additional packages:
 - CometML for experiment tracking (required different Conda Forge)
 - Jupyter or Jupyterlab, if not exist
 ```
conda install -f comet_ml comet_ml
conda install jupyter jupyterlab
```
## Or Pip
Create a file `requirements.txt`
```
# requirements.txt
pandas
numpy
scikit-learn
matplotlib
tensorflow
scipy
comet_ml
jupyter
jupyterlab
```
`pip3 install -r requirements.txt`
# 2 How to run
 