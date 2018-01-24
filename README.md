# Mobilenet_v2 in tensorflow
The very rough version of mobilenetv2 in tensorflow implementation

## Insertion to train from scratch
The network structure is evolved from tensorflow.models.research.slim.  
Please adopt the nets_factory.py and preprocessing_factory.py into your folder.  
Lastly, add mobilenet_v2.py into your models/research/slim/nets.

## Declaration
If any luck, you have some idea about this code. Please feel free to open an issue or PR to me :)  
I'm still immature to tensorflow, there might be some error inside the bone structure.  
Having any question, please contact me via lcycoding@gmail.com

## Issues
Per paper on arxiv, authors said that there are 19 bottleneck in the network structure.  
But with the Table-2 given information, there are only 17 bottleneck.  
And there are lots of parameters that I haven't looked into.  
Training with this network may or may not spend tons of time, please verify by yourself :-)

## References
[Mobilenet_v2 on arxiv](https://arxiv.org/pdf/1801.04381.pdf)  
[tensorflow_github](https://github.com/tensorflow)
