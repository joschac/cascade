# Cascade: Causal Discovery from Event Sequences by Local Cause-Effect Attribution

Implementation of publication [*Causal Discovery from Event Sequences by Local Cause-Effect Attribution*](https://eda.rg.cispa.io/prj/cascade/).

## Install

    python3 -m venv ./venv
    source venv/bin/activate
    pip install -r requirements.txt

## Usage
See `example.ipynb` notebook. 

## Run Notebook

Add Virtual Environment as Jupyter kernel:
    
    python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
Start Jupyter Notebook

    jupyter notebook example.ipynb
Select Virtual Environment:\
**Kernel** > **Change Kernel** > **Python (myenv)**.

Now the Jupyter Notebook can be run. 

## BibTeX
    @article{cueppers2024causal,
        title={Causal Discovery from Event Sequences by Local Cause-Effect Attribution},
        author={C{\"u}ppers, Joscha and Xu, Sascha and Ahmed, Musa and Vreeken, Jilles},
        journal={Advances in Neural Information Processing Systems},
        volume={37},
        year={2024}
    }