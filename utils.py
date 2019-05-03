from torchvision import models
from torch.autograd import Variable
from torch._thnn import type2backend
from torch.nn import Upsample
import numpy as np
        
def load_model(arch):
    '''
    Args:
        arch: (string) valid torchvision model name,
            recommendations 'vgg16' | 'googlenet' | 'resnet50'
    '''
    if arch == 'googlenet':
        from googlenet import get_googlenet
        model = get_googlenet(pretrain=True)
    elif arch == 'resnext':
	from resnext import get_resnext
	model = get_resnext(pretrain=True)
    else:
        model = models.__dict__[arch](pretrained=True)

    model.eval()
    return model


def cuda_var(tensor, requires_grad=False):
    return Variable(tensor.cuda(),requires_grad=requires_grad)


def upsample(inp, size):
    '''
    Args:
        inp: (Tensor) input
        size: (Tuple [int, int]) height x width
    '''
    #backend = type2backend[inp.type()]
    #f = getattr(backend, 'SpatialUpSamplingBilinear_updateOutput')
    #upsample_inp = inp.new()
    #f(backend.library_state, inp, upsample_inp, size[0], size[1])
    m=Upsample(scale_factor=size, mode='trilinear')
    #print np.min(inp.squeeze().cpu().numpy())    
    upsample_inp=m(inp)
    
    attr=upsample_inp 
    

    return attr


import csv


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data[0]

    return n_correct_elems / batch_size

class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

