# Side Note: What is Long Short Term Memory?

## Hochreiter, Sepp & Schmidhuber, Jürgen. (1997). Long Short-term Memory. Neural computation. 9. 1735-80. 10.1162/neco.1997.9.8.1735. 

## Summary

LSTM is one solution to the core vanishing and exploding gradient challenge in training RNN. LSTM adds/removes information from cell state (memory) using the gates as shown in the following figure. The gates are forget gate (<img src="https://latex.codecogs.com/svg.image?F_t" title="F_t" />), input gate (<img src="https://latex.codecogs.com/svg.image?I_t" title="I_t" />) and output gate (<img src="https://latex.codecogs.com/svg.image?O_t" title="O_t" />) given as in the following equations. 

<img src="https://latex.codecogs.com/svg.image?\begin{aligned}\mathbf{F}_t&space;&=&space;\sigma(\mathbf{X}_t&space;\mathbf{W}_{xf}&space;&plus;&space;\mathbf{H}_{t-1}&space;\mathbf{W}_{hf}&space;&plus;&space;\mathbf{b}_f)\\\mathbf{I}_t&space;&=&space;\sigma(\mathbf{X}_t&space;\mathbf{W}_{xi}&space;&plus;&space;\mathbf{H}_{t-1}&space;\mathbf{W}_{hi}&space;&plus;&space;\mathbf{b}_i)\\\mathbf{O}_t&space;&=&space;\sigma(\mathbf{X}_t&space;\mathbf{W}_{xo}&space;&plus;&space;\mathbf{H}_{t-1}&space;\mathbf{W}_{ho}&space;&plus;&space;\mathbf{b}_o)\end{aligned}" title="\begin{aligned}\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f)\\\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i)\\\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o)\end{aligned}" />

Where <img src="https://latex.codecogs.com/svg.image?\mathbf{W}_{xf},&space;\mathbf{W}_{xi},&space;\mathbf{W}_{xo},&space;\mathbf{W}_{hf},&space;\mathbf{W}_{hi},&space;\mathbf{W}_{ho},&space;\mathbf{b}_f,&space;\mathbf{b}_i,&space;\mathbf{b}_o" title="\mathbf{W}_{xf}, \mathbf{W}_{xi}, \mathbf{W}_{xo}, \mathbf{W}_{hf}, \mathbf{W}_{hi}, \mathbf{W}_{ho}, \mathbf{b}_f, \mathbf{b}_i, \mathbf{b}_o" /> are learnable weights and biases. <img src="https://latex.codecogs.com/svg.image?X_t" title="X_t" /> is the input item at certain time (i.e. word or time series item). <img src="https://latex.codecogs.com/svg.image?{H}_{t-1}" title="{H}_{t-1}" /> is the hidden state in the memory. We notice some degree of resembance between the three previous equations. The three equations involve sigmoid activation and learnable weights (training on sequence data through backpropagation). Hidden state is a main property of RNNs. RNN used mainly to laverage the temporal dependency between items in the sequence.


### LSTM Gates:

* The forget gate (<img src="https://latex.codecogs.com/svg.image?F_t" title="F_t" />) is responsible for what previous information (i.e. representation) to keep and delete from the cell state.  This is done using the sigmoid activation. This means that it outputs values between 0 (delete) and 1 (keep). Multiplication (element-wise multiplication) of the sigmoid value 0 by <img src="https://latex.codecogs.com/svg.image?C_{t-1}" title="C_{t-1}" /> gives 0 and this means 'forget <img src="https://latex.codecogs.com/svg.image?C_{t-1}" title="C_{t-1}" /> entirely'.  

* The input gate (<img src="https://latex.codecogs.com/svg.image?I_t" title="I_t" />) is responsible for what new information to add to the cell state. This involves a sigmoid activation to decide how much of the new information to be added to the cell state (values between 0 and 1). 

    * Now we know that the input gate involves a sigmoid activation to output values between (0, 1) to be multiplied with what we need to add in the cell. But the question here, what exactly is this the new information? Here we need a new equation. So input gate is not only a sigmoid activation but also involves candidate generation.

        <img src="https://latex.codecogs.com/svg.image?\tilde{\mathbf{C}}_t&space;=&space;\text{tanh}(\mathbf{X}_t&space;\mathbf{W}_{xc}&space;&plus;&space;\mathbf{H}_{t-1}&space;\mathbf{W}_{hc}&space;&plus;&space;\mathbf{b}_c)" title="\tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c)" />

        Where <img src="https://latex.codecogs.com/svg.image?W_{xc},&space;W_{hc},&space;b_c" title="W_{xc}, W_{hc}, b_c" /> are learnable weights and bias.

    * Now we multiply (element-wise) the sigmoid by the generated candidate to give the new information. And we add (element-wise) the new information to the cell state (memory). Now we can say that the new cell state <img src="https://latex.codecogs.com/svg.image?C_{t}" title="C_{t}" /> is sum of previous cell state <img src="https://latex.codecogs.com/svg.image?C_{t}" title="C_{t}" /> multiplied by the forget gate sigmoid and the generated candidate (<img src="https://latex.codecogs.com/svg.image?\tilde{\mathbf{C_t}" title="\tilde{\mathbf{C_t}" />) multiplied by input gate sigmoid.

        <img src="https://latex.codecogs.com/svg.image?\mathbf{C}_t&space;=&space;\mathbf{F}_t&space;\odot&space;\mathbf{C}_{t-1}&space;&plus;&space;\mathbf{I}_t&space;\odot&space;\tilde{\mathbf{C}}_t" title="\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t" />


    * Question: Why Tanh is used in candidate generation? Why not ReLU or sigmoid?
        ReLU is not a good fit for RNNs in general as ReLU outputs very large values. This means that it leads to exploding gradient. Relative to ReLU, Tanh can bound the the outputs between (-1: 1). In the same time, Tanh performs better than Sigmoid as it saturates slower than sigmoid (saturation = derivative reaches zero). while the average of sigmoid and Tanh outputs is 0.5 and 0, respectively. Convergence is usually faster when input mean values is close to zero which is the case in Tanh which can make the convergence of the next layer faster (same reason we normalize the data before training). This means that Tanh found to converge faster in practice. To deeper undertanding of that idea of convergence, read "Lecun, Yann & Bottou, Leon & Orr, Genevieve & Müller, Klaus-Robert. (2000). Efficient BackProp". 


*  The output gate (<img src="https://latex.codecogs.com/svg.image?O_t" title="O_t" />) is used to compute the hidden state (<img src="https://latex.codecogs.com/svg.image?H_t" title="H_t" />). The hidden state will be based on the cell state (<img src="https://latex.codecogs.com/svg.image?C_{t}" title="C_{t}" />) and the output gate sigmoid (0:1). So, when the sigmoid is 1, pass all memory information from the cell state through the hidden state. In this case, simply, hidden state could be seen as filtered version of the cell state as explained in the [article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).  

    <img src="https://latex.codecogs.com/svg.image?\mathbf{H}_t&space;=&space;\mathbf{O}_t&space;\odot&space;\tanh(\mathbf{C}_t)" title="\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t)" />


![lstm](https://github.com/mhmdrdwn/papers-summary/blob/main/_/lstm.png?raw=true)
    
    Figure: CC by Zhang, Aston and Lipton, Zachary C. and Li, Mu and Smola, Alexander J., Dive into Deep Learning, arXiv preprint arXiv:2106,11342, 2021.
