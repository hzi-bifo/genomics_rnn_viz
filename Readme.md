
# char-dna

## Usage

### predict plasmid probability

precalculated model was trained on 500 randomly selected full plasmid sequences (NCBI 2015 dump) on a Tesla K80 with RNN size 800 and sequence length of 150. Using this model you can predict the sequence similarity to the trained model using the `check.lua` script. This will ourput a .json formatted file of the probability for every character in your sequences. Because the model was trained on plasmid sequences only, this should give you a per-nucleotide probability that the input sequences follows the structure of the trained plasmid sequences. 

get the [lastest precalculated model](https://www.dropbox.com/s/fwty30auk82155b/lm_lstm_epoch19.08_1.2870.t7_cpu.t7?dl=1) and the [gpu model](https://www.dropbox.com/s/504uihdc936p09y/lm_lstm_epoch19.08_1.2870.t7?dl=1)

```bash
th check.lua model.t7 -sequence "AAACACAGTGGTGGTTACATCTATGTGATTGCCCCTAATCCATACACAAAAAGCCGTATC" > sequence.json
```

to visualize the results, open `index.html` 

![ScreenShot](sample.jpg)

to get the overall probability of the input sequence you can use

```bash
th check.lua model.t7 -sequence "AAACACAGTGGTGGTTACATCTATGTGATTGCCCCTAATCCATACACAAAAAGCCGTATC" -overall 1 -log 1
> Mean sample log probability: -3.351086258032	
```


### predict CRISPR spacer

we learned a LSTM model on 10000 syntetic generated CRISPR spacer sequences (script available at https://gist.github.com/philippmuench/07a0f52fcbb682b4ce1776c8dd9a865b) using a sequence length of 50 and a network size of 128 hidden neurones. Even after 30 minutes of training the model can succesfully find palindromic spacer sequences in new generated sequences. 

You can download the [lastest precalculated model for CRISPR prediction](https://www.dropbox.com/s/2e6px4ergyenrb7/crispr_cpu.t7?dl=1) and the [GPU model](https://www.dropbox.com/s/ikqryoreblfu6zc/crispr.t7?dl=1)

```
 th check.lua crispr_cpu.t7 -sequence "CGTGATAGCAGGACTCCTCGTTGTGTGCCTTGCTAGTCGTATAACTAAGCCAGGCCGGACCGAATCAATATGCTGATCGTTCCGTGTGTTGCTCCTCAGGACGATAGTGCCCCGTCATGGAGCTCGCCGCTCGAGGTACTGCCCTGGTATCCGTTCTTTCGCACATAATTAAAGCTCTATGAACCCGTCATGGAGCTCGCCGCTCGAGGTACTGCCCGGTCGATCCTTCCGAAGTTAAGCACATGACGGCTTCCCCGTCATGGAGCTCGCCGCTCGAGGTACTGCCCCATGGGTTGGGGCCCTAAGGAAAAGAGACGCCTATTGTCACGTACATAAGCTTGGCGCATTAGTAGAATAGCCCAGTCCTGGATGGTAGGTCCTGACCCGATAAGATGATTACGCGGTTCGAATACATGCACTGTTA" > sequence.json

```

![ScreenShot](crispr.png)

This is a synthetic CRISPR sequence with the CRISPR spacer of `CCCGTCATGGAGCTCGCCGCTCGAGGTACTGCCC`. You can see a high confidence of model at the spacer sequence


## installation

This code is written in Lua and requires [Torch](http://torch.ch/). If you're on Ubuntu, installing Torch in your home directory may look something like: 

```bash
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; 
./install.sh      # and enter "yes" at the end to modify your bashrc
source ~/.bashrc
```

See the Torch installation documentation for more details. After Torch is installed we need to get a few more packages using [LuaRocks](https://luarocks.org/) (which already came with the Torch install). In particular:

```bash
luarocks install nngraph 
luarocks install optim
luarocks install nn
luarocks install json
```

If you'd like to train on an NVIDIA GPU using CUDA (this can be to about 15x faster), you'll of course need the GPU, and you will have to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). Then get the `cutorch` and `cunn` packages:

```bash
luarocks install cutorch
luarocks install cunn
```

If you'd like to use OpenCL GPU instead (e.g. ATI cards), you will instead need to install the `cltorch` and `clnn` packages, and then use the option `-opencl 1` during training ([cltorch issues](https://github.com/hughperkins/cltorch/issues)):

```bash
luarocks install cltorch
luarocks install clnn
```


## Acknowledgements

Implementation based by karpathy. This code was originally based on Oxford University Machine Learning class [practical 6](https://github.com/oxford-cs-ml-2015/practical6), which is in turn based on [learning to execute](https://github.com/wojciechz/learning_to_execute) code from Wojciech Zaremba. Chunks of it were also developed in collaboration with my labmate [Justin Johnson](http://cs.stanford.edu/people/jcjohns/).

## License

MIT
