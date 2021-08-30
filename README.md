# CAN_GAN_Anomaly
Based-ACGAN CAN Anomaly Detection Method

## Conda environment
```shell
    # create conda environment
    conda create -n ACGIDS python=3.7
    # delete conda environment
    conda remove -n ACGIDS --all
    # activate conda environment
    conda activate ACGIDS
    # load third-part lib
    pip install --user --requirement requirements.txt
```

## Usage
```shell
    # CAN
    bash experiments/run_can.sh
```

## Visualdl
```shell
    # we can pip visualdl lib follow as:
    pip install visualdl -i https://mirror.baidu.com/pypi/simple
    
    # start visualdl service
    nohup visualdl --logdir log --port 8080 &
```

if you find our model/method/dataset useful, please cite our work:
```angular2html

```