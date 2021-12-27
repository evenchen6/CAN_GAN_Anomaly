# CAN_GAN_Anomaly
"Fine-Grained Known/Unknown CAN Bus Intrusion Detection based on Out-of-Distribution Detection for Automotive CPS Security" Code 

## Conda environment
```shell
    # create conda environment
    conda create -n ACGIDS python=3.7
    # delete conda environment
    conda remove -n ACGIDS --all
    # activate conda environment
    conda activate ACGIDS
    # install third-party libraries
    pip install --user --requirement requirements.txt
```

## Usage
```shell
    # Train
    bash experiments/run_can.sh
    
    # Test
    # NOTE: Need to be modified to your storage model parameter location
    bash experiments/run_can_val.sh
```

## Visualdl
```shell
    # you can pip visualdl lib follow as:
    pip install visualdl -i https://mirror.baidu.com/pypi/simple
    
    # start visualdl service, load model saved log
    nohup visualdl --logdir log --port 8080 &
```

if you find our model/method/dataset useful, please cite our work:
```angular2html

```