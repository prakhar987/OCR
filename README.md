A simple supervised, feed-forward, back-propagation network with sigmoid activation
function for the problem of optical character recognition for three digits 0,1 and 3.

The code is from scratch and does not use any standard library.

######Data:  optdigits data set that comes from the UCI Machine Learning Repository.

--Preporcessing: Digitized and down-sampled images (to 8x8 or so).

Classifier: Tried few configurations for a 3-Layer Neural Network with varying number of hidden
units and two output units.

USAGE : python3.5 nn-test.py 
Files needed : optdigits-orig.cv, optdigits-orig.tra

Analysis : 
With increase in number of layers, there was no significant improvement in acccuracy 

FOR n= 20
Correct Answers : 94

Weights of final layer :
[[ 0.26011099  0.1005338 ]
 [ 0.06418003 -0.94903212]
 [ 0.43774013  0.07448291]
 [-1.03906563 -1.45565363]
 [-0.2675988   0.35827713]
 [ 0.52842403  0.51112281]
 [-0.61921481 -0.33784368]
 [-0.75494916 -0.05249157]
 [-0.07002506 -0.50408772]
 [-0.0999042  -1.268079  ]
 [-0.01535449 -0.28472326]
 [ 0.48915249  0.84500741]
 [ 0.11291178 -0.14273624]
 [ 0.23663773  0.13529871]
 [-0.33753677 -0.02109039]
 [-0.54362438 -1.09714712]
 [-0.54539854 -0.62368434]
 [-0.43574826  0.74679339]
 [-0.61973144 -0.45160469]
 [-0.20900802 -0.67358155]]


FOR n=2 
Correct Answers : 66
[[-2.59448304 -2.78349909]
 [-0.76314892 -0.76963078]]


FOR n=8
Correct Answers:90
[[-1.50973236 -1.20175838]
 [-0.33395132 -0.39843667]
 [-0.3057592  -0.49541508]
 [ 0.05488232  0.17468914]
 [-0.00723074 -0.18034427]
 [-0.02213164 -0.02959909]
 [ 0.13769117 -0.96281759]
 [-1.24427629 -1.34203571]]

So decided to stick with 8 layer NN network architecture.
The initial weights were initialized with mean 0.
Accuracy of result improves with increasing the number of epochs.
