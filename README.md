# Traffic Forcasting (Demo)
Using Different GNN and fusion models for traffic forcasting

## Speed Data

[Seoul Speed Data](https://github.com/yuyolshin/SeoulSpeedData)

The forcating of traffic is the predictions of traffic speed, traffic flow, and traffic density. In these notebooks, we use traffic speed (Seoul traffic speed dataset gathered by Transport Operation & Information Service (TOPIS)). the dataset contains two sets: Urban-core and Urban-mix datasets. The data was gathered from april 1st, 2018 to april 30th, 2018. The dataset is 5-minute interval speed data for road segments in the Seoul traffic. The data can be downloaded from https://github.com/yuyolshin/SeoulSpeedData

The Urban core dataset consists of 304 sensors, with 8640 observations every 5 minutes for 30 days (30 days x 24 hours x 12 intervals = 8640)

![alt text](https://user-images.githubusercontent.com/31876093/141141076-7d44ed1e-7868-4cf4-9e93-3597b1d97f9f.png)

REFERENCE: Shin, Y., & Yoon, Y. (2020). Incorporating dynamicity of transportation network with multi-weight traffic graph convolutional network for traffic forecasting. IEEE Transactions on Intelligent Transportation Systems.

### Methods

[Temporal Graph convolutional Network (TGCN)](https://github.com/mhmdrdwn/traffic/blob/main/speed/tgcn.ipynb)

[Attention Temporal Graph Convolutional Network (A3TGCN)](https://github.com/mhmdrdwn/traffic/blob/main/speed/a3tgcn.ipynb)

[Spatiotemporal Graph convolutional Network (STGCN)](https://github.com/mhmdrdwn/traffic/blob/main/speed/stgcn.ipynb)

[Diffusion Convolutional Recurrent Neural Network (DCRNN)](https://github.com/mhmdrdwn/traffic/blob/main/speed/dcrnn.ipynb)


### Results

| Model         | MAE    | RMSE   | MAPE   |
| ------------- |:------:|:------:|:------:|
| TGCN          |  2.96  |  4.31  | 12.92% |
| A3TGCN        |  5.39  |  7.12  | 23.86% |
| STGCN         |  3.49  |  4.81  | 15.78% |
| DCRNN         |  2.81  |  4.17  | 11.01% |


## Flow Data (Under Progress)
Fusion of Weather and holidays information

Here the task is to predict the flow of traffic. The data consists of weather, holidays and flow data. The flow data is a grid of 15x5 with flows (inflow and outflows). The number of time steps (time slots) in the data are 17520. Meteorology data consists of temperature, windspeed and weather. weather data is: {'Cloudy': 0, 'Cloudy / Windy': 1, 'Fair': 2......}. Holidays data include list of holidays. 

![alt text]()

### Methods

[MLP](https://github.com/mhmdrdwn/traffic/blob/main/flow/baseline_mlp.ipynb)

[LSTM](https://github.com/mhmdrdwn/traffic/blob/main/flow/baseline_lstm.ipynb)

[ST-Resnet](https://github.com/mhmdrdwn/traffic/blob/main/flow/stresnet.ipynb)





