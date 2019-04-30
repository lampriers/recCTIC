# A Recurrent Neural Cascade-based Model for Continuous-Time Diffusion

This is the platform developped for the paper "A Recurrent Neural Cascade-based Model for Continuous-Time Diffusion - Sylvain Lamprier, LIP6, Sorbonne Universités" published at ICML 2019.

It contains the code for the proposed model and the compared baselines in PyTorch1.0 (python3.6):
- recCTIC.py: the proposed model and the embCTIC baseline
- rnnDiffusion.py: the RNN based baselines refered in the paper as RNN, CYAN and CYAN-cov
- DAN.py: the attention-based baseline refered in the paper as DAN
- CTIC.py: the classical continous-time independent cascade model

It also comes with the artificial dataset used in the paper. 

## Data

The platform uses data from text files, where each line corresponds to a specific diffusion episode. Each line is formatted as \<infections\>;\<content\>;\<infectors\>, where:
- \<infections\> is a list of pairs <node\_id>:<infection\_timestamp> separated by tabulations;
- \<content\> is a (possibly empty) list of features describing the diffused content, where each feature is a pair \<feature\_id\>:\<weight\>;   
- \<infectors\> is a (possibly empty) list of pairs giving for each infected node, the name of the node that influenced it (or world for spontaneous infection). 

Note that the content and the infectors are hidden for the models, they are only given for evaluation purposes. Actually, content features are not used at all in the current version of the paper. Infectors are used to compute the INF measure if given.  

The files arti\_episodes\_train.txt, arti\_episodes\_val.txt  and arti\_episodes\_test.txt are examples of datafiles that were used in the paper for our experiments. 


## Recurrent Neural CTIC Model

The model proposed in the paper can be run by:

`python3 recCTIC.py -o xp/recDiff/arti -itr arti_episodes_train.txt -ite arti_episodes_test.txt -c 0 --fromTests 0 2 5 10 -cl -ns 0`

where:
- -o is the output directory
- -itr is the file of training episodes
- -ite is the file of test episodes
- -c is the device (-1 for cpu)
- --fromTests is the list of conditioning times in test (for instance 2 denotes that test results are given for test episodes after time 2 given the infections before it)
- -cl is optional. If present and the infectors given in the data, it computes the INF measure
- -ns gives the number of simulations to be performed for the computation of the CE measure (very time consuming so only set with values greater than 0 on pre-trained models)

To load a pre-trained model, the --fromFile option can specify the path to the model to load. 

For using the pre-trained model with a different test file than the one loaded for training (for instance for testing on test rather than on val), pease add the option --rebuildTest True (this rebuilds useful structures for efficient computations on the test episodes)

For the embCTIC baseline, please add the option -hw 0.


## CTIC

The CTIC model can be run by (same options as recCTIC, except for the device which is always the cpu):

`python3 CTIC.py -o xp/CTIC/arti -itr arti_episodes_train.txt -ite arti_episodes_test.txt --fromTests 0 2 5 10 -cl -ns 0`

## Recurrent Baselines

The recurrent baselines can be run by (same options as recCTIC):

`python3 rnnDiffusion.py -o xp/RNN/arti -itr arti_episodes_train.txt -ite arti_episodes_test.txt -c 0 --fromTests 0 2 5 10 -cl -ns 0`

Please add -cyan for the CYAN approach and also -coverage for the CYAN-cov approach. 

## DAN

The baseline DAN can be run by (same options as recCTIC):

`python3 DAN.py -o xp/DAN/arti -itr arti_episodes_train.txt -ite arti_episodes_test.txt -c 0 --fromTests 0 2 5 10 -cl -ns 0`
