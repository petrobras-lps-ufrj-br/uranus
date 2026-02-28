

import os, json
from pprint import pprint
from maestro_lightning import Flow, Task, Dataset, Image


basepath = os.getcwd()
n_splits = 10
batch_size = 128
dataset_path = f"{os.environ['URANUS_DATA_PATH']}/compressor.csv"
input_path = f"{basepath}/jobs"
virtualenv = os.environ["VIRTUALENV_PATH"]

envs = {
    #"MLFLOW_TRACKING_URI":"http://caloba92.lps.ufrj.br:8000"
}

os.makedirs(input_path, exist_ok=True)
for i in range(10):

    with open(f"{input_path}/job_{i}.json",'w') as f:
        d={
            "csv_path": dataset_path,
            "fold": i,
            "epochs": 5000,
            "splits": n_splits,
            "batch_size": batch_size
        }
        json.dump(d,f)



with Flow(name="task_mlp_v1", path=f"{basepath}/task_mlp_v1", virtualenv=virtualenv , partition="cpu") as session:


    input_dataset   = Dataset(name="jobs", path=f"{basepath}/jobs")
    #image            = Image(name="python", path=f"{basepath}/python3.10.sif")
    pre_exec = f"source {virtualenv}/bin/activate"
    # since the output is a directory, we need to tar it
    command = f"{pre_exec} && python {basepath}/job_mlp_v1.py -j %IN -o %OUT"
    binds = {}
    task_1 = Task(name="task_mlp_v1",
                  #image=image,
                  command=command,
                  input_data=input_dataset,
                  outputs={'OUT':'output.tar'},
                  partition='cpu',
                  binds=binds,
                  envs=envs)

    session.run()

    

    
