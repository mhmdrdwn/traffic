# Traffic Forcasting
Using Different GNN models for traffic forcasting

## Data

[Seoul Speed Data](https://github.com/yuyolshin/SeoulSpeedData)

![alt text](https://user-images.githubusercontent.com/31876093/141141076-7d44ed1e-7868-4cf4-9e93-3597b1d97f9f.png)

REFERENCE: Shin, Y., & Yoon, Y. (2020). Incorporating dynamicity of transportation network with multi-weight traffic graph convolutional network for traffic forecasting. IEEE Transactions on Intelligent Transportation Systems.

## Methods

[Temporal Graph convolutional Network (TGCN)](https://github.com/mhmdrdwn/traffic/blob/main/notebooks/tgcn.ipynb)

[Attention Temporal Graph Convolutional Network (A3TGCN)](https://github.com/mhmdrdwn/traffic/blob/main/notebooks/a3tgcn.ipynb)

[Spatiotemporal Graph convolutional Network (STGCN)](https://github.com/mhmdrdwn/traffic/blob/main/notebooks/stgcn.ipynb)

[Diffusion Convolutional Graph Neural Network (DCRNN)](https://github.com/mhmdrdwn/traffic/blob/main/notebooks/dcrnn.ipynb)


## Results

| Model         | MAE    | RMSE   | MAPE   |
| ------------- |:------:|:------:|:------:|
| TGCN          |  1.74  |  4.33  |  3.67% |
| A3TGCN        |  2.45  |  7.97  |  5.32% |
| STGCN         |        |        |        |
| DCRNN         |  1.97  |  5.15  |  4.45% |

