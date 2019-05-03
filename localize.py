
import run2 as r
import matplotlib.pyplot as plt
import numpy as np
#import selectivesearch
import pickle
import sys
import torch
sys.path.append('./anet_toolkit/Evaluation/')
import subprocess
import utils
import gc
import os 
from eval_detection import compute_average_precision_detection
from utiles import segment_iou
import cv2
import math
from sklearn import svm
import pywt
from sklearn import mixture
from sliding import sliding_window
from model import (generate_model, generate_model_linear)

def vis(inputs):
	#print "++++",subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
	#pre = r.Run()
	#print inputs
	out = pre.saliency(inputs)
	#del pre
		#torch.cuda.empty_cache()
	#gc.collect()
	#print "****",subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
	return out

def list_files(dir):                                                                                                  
    r = []      
    print dir                                                                                                      
    subdirs = [x[0] for x in os.walk(dir)] 
    print subdirs                                                                           
    for subdir in subdirs:

    	#print subdir.find(".AppleDouble")
    	if subdir.find(".AppleDouble")!=-1:
    		continue

        files = os.walk(subdir).next()[2]
        
        if (len(files) > 0):                                                                                          
            #for file in files: 
        	r.append(subdir + "/" )                                                                         
    return r    

filtered = list_files("./base")
model = utils.load_model("resnext")
model.cuda()
print filtered
pre=r.Run()
#extract = vis("./images/frames4")
extracts= [[l,vis(l)] for l in filtered]

torch.cuda.empty_cache()
#print "****",subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])


with open('localization', 'wb') as f:
   pickle.dump(extracts, f)
   f.close()
