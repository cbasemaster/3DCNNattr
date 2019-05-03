import sys
sys.path.append('./')

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from create_explainer import get_explainer
import utils
import viz
import torch
import time
import os
import pylab
import numpy as np
#import matplotlib.pyplot as plt
import functools
from PIL import Image
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from mean import get_mean, get_std
import selectivesearch as ss
import operator
import glob
import subprocess
import cv2
import math
import scipy.io
import nms
import matplotlib.patches as patches
np.set_printoptions(threshold='nan')



params = {
    'font.family': 'sans-serif',
    #'axes.titlesize': 25,
    #'axes.titlepad': 10,
}
pylab.rcParams.update(params)
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Run:
    def __init__(self):


	self.model_methods = [
	    ['resnext', 'gradcam','camshow']   
	]

	self.classes=["brush_hair", "cartwheel", "catch", "chew","clap","climb","climb_stairs","dive","draw_sword", "dribble", "drink", "eat","fall_floor","fencing","flic_flac","golf", "handstand","hit","hug","jump","kick", "kick_ball", "kiss","laugh","pick", "pour", "pullup","punch","push", "pushup", "ride_bike", "ride_horse","run","shake_hands", "shoot_ball", "shoot_bow", "shoot_gun","sit","situp","smile","smoke","somersault","stand","swing_baseball","sword","sword_exercise","talk","throw","turn","walk","wave"]

	scales = [1.0]

	self.spatial_transform = Compose([
		    MultiScaleCornerCrop(scales,112),
		    ToTensor(1.0), 
		    Normalize(get_mean(1.0, dataset='activitynet'),
	            get_std(1.0))
		])
	
	self.spatial_transform2 = Compose([
		    MultiScaleCornerCrop(scales,112)

		])

	self.spatial_transform3 = Compose([
		  
		    MultiScaleCornerCrop(scales,112),
            	    ToTensor(1), Normalize([0, 0, 0], [1, 1, 1])

		])
	
	self.model = utils.load_model(self.model_methods[0][0])
        self.model.cuda()
        #self.video=[]
	#self.flows=[]
	self.bb_frames=[]
        #self.explainer= get_explainer
	method_name='gradcam'
        self.explainer = get_explainer(self.model, method_name, "conv1")
        self.explainer2 = get_explainer(self.model, method_name, "layer1")
        self.explainer3 = get_explainer(self.model, method_name, "layer2")    
        self.explainer4 = get_explainer(self.model, method_name, "layer3")
        self.explainer5 = get_explainer(self.model, method_name, "layer4")
        self.explainer6 = get_explainer(self.model, method_name, "avgpool")
        path="images/frames4"
	#print path
	self.path = path+"/"
	#dirc = os.listdir(path)
	#self.files = [ fname for fname in dirc if fname.startswith('img')]
	#self.files2 = [ fname for fname in dirc if fname.startswith('flow_x')]
	self.seq=[]
	self.kls=[]
        self.scr=[]
	self.totalhit=0
	self.totalhit2=0
	self.totalhit3=0
	self.totalhit4=0
	self.totalhit5=0
	self.totalhit6=0
	self.totalhit7=0
	self.totalframes=0

    def myRange(self,start,end,step):
    	i = start
    	while i < end:
        	yield i
        	i += step
   	yield end

    def bounding_box(points):
        x_coordinates, y_coordinates = zip(*points)
        return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]
    
    def saliency(self,path):
         
     #path = path.replace("base2","base") 
     path_gt = path.replace("base/","base2/")    
     
     path="./"+path
     path_gt="./"+path_gt
     print path_gt
     if os.path.isdir(path_gt):
	     
	     for model_name, method_name, _ in self.model_methods:
		
		dirc=os.listdir(path)
		dirc_gt=os.listdir(path_gt)
	       
		files=[fname for fname in dirc if fname.endswith('png')]
		files_gt=[fname for fname in dirc_gt if fname.endswith('mat')]	
		
	     def bounding_box(points):
		x_coordinates, y_coordinates = zip(*points)
		return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

	     for index in self.myRange(0,len(files),16):
		#print "frame ke", index, "from", len(files)        
		if index==len(files):
			continue
		video=[]
		flows=[]
		boxes=[]

		for filename in sorted(files)[index:index+16]:
			video.append(Image.open(path+filename))
		
		diff=0
		matfile=scipy.io.loadmat(path_gt+files_gt[0])
		coor= np.array(matfile["pos_img"]).transpose(2,1,0).tolist()[index:index+16]
		scale= matfile["scale"][0]	
	        
                if len(coor)==0:
			continue	
		
	        print len(video),len(coor)		
		
		if len(coor)!=16 or len(video)!=16:
			diff=len(video)
			video=video * (16/len(video)) * 2
			video=video[0:16]
			coor=coor * (16/len(coor)) * 2
			coor=coor[0:16]
		
	        print len(video),len(coor)
	        if len(coor)==0:
			continue	
		for e in range(0,len(coor),1):
		    		
		    box=bounding_box([(abs(dots[0])*112.0/320.0,abs(dots[1])*112.0/240.0) for dots in coor[e]])
		    boxes.append(box)
		
		self.spatial_transform.randomize_parameters()
		self.spatial_transform2.randomize_parameters()
		self.spatial_transform3.randomize_parameters()
		
		clip = [self.spatial_transform3(img) for img in video]
		inp = torch.stack(clip,0).permute(1, 0, 2, 3)	
		
		all_saliency_maps = []
		for model_name, method_name, _ in self.model_methods:
		    
		    if method_name == 'googlenet':  # swap channel due to caffe weights
			inp_copy = inp.clone()
			inp[0] = inp_copy[2]
			inp[2] = inp_copy[0]
		    inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)
		    
		    saliency,s,kls,scr,c = self.explainer.explain(inp)
		    saliency2, s2,kls,scr,c2 = self.explainer2.explain(inp)
		    saliency3,s3,kls,scr,c3 = self.explainer3.explain(inp)	
		    saliency4,s4,kls,scr,c4 = self.explainer4.explain(inp)
		    saliency5, s5,kls,scr,c5 = self.explainer5.explain(inp)
		    saliency6,pool,kls,scr,c6 = self.explainer6.explain(inp)
		      
		    torch.cuda.empty_cache()
		    
		    saliency=(saliency6+saliency5+saliency4+saliency3+saliency2+saliency)
		    
		    if self.classes[kls]==path.split("/")[5]:
			label=1
		    else:
			label=0
		    
		    saliency = torch.clamp(saliency, min=0)   
		       
		    temp=saliency.shape[2]
		    
		    if temp>1:   
			all_saliency_maps.append(saliency.squeeze().cpu().data.numpy())
		    else:
			all_saliency_maps.append(saliency.squeeze().unsqueeze(0).cpu().numpy())
		    		    
		    del pool,inp,saliency,saliency6 
		    torch.cuda.empty_cache()
		    
		plt.figure(figsize=(50, 5))
		      
		for k in range(len(video[0:(16-diff)])):
		    hit=0
		    hit2=0
		    hit3=0
		    hit4=0
		    hit5=0
		    hit6=0
	            hit7=0
		    plt.subplot(2, 16, k+1)
		    img=self.spatial_transform2(video[k])
		     
		    if len(boxes)>0: 
			    box=boxes[k]
			    
			    x=box[0][0]
			    y=box[0][1]
			    w = box[1][0] -x
			    h = box[1][1] -y
			   
			    if (x+w)>112:
				w=(112-x)
			    if (y+h)>112:
				h=(112-y)
			    	
			    ax=viz.plot_bbox([x,y,w,h],img)
			
		    plt.axis('off')
		    ax = plt.gca()
		    
		    ax.imshow(img)
		    sal=all_saliency_maps[0][k]
		    sal=(sal-np.mean(sal))/np.std(sal)
		    ret,thresh = cv2.threshold(sal,np.mean(sal)+((np.amax(sal)-np.mean(sal))*0.5),1,cv2.THRESH_BINARY)
		    
	            contours,hierarchy = cv2.findContours(thresh.astype(np.uint8), 1, 2)
		    
		    areas = [cv2.contourArea(c) for c in contours]
		    
		    if len(contours)>0:	
			    
			    glob=np.array([cv2.boundingRect(cnt) for cnt in contours])
			    #print glob.shape
			    x3=np.amin(glob[:,0])
		            y3=np.amin(glob[:,1])
			    x13=np.amax(glob[:,0]+glob[:,2])
			    y13=np.amax(glob[:,1]+glob[:,3])
			    
			    rect3 = patches.Rectangle((x3, y3), x13-x3 , y13-y3,
                                         linewidth=2, edgecolor='y', facecolor='none')
                            ax.add_patch(rect3)
			    
			    for cnt in contours:
				    x2,y2,w2,h2 = cv2.boundingRect(cnt) 
	 
				    rect2 = patches.Rectangle((x2, y2), w2 , h2,
						 linewidth=2, edgecolor='r', facecolor='none')
				    ax.add_patch(rect2)
				    
		            overlap=nms.get_iou([x,x+w,y,y+h],[x3,x13,y3,y13])
	                    
			    if label==1:
				    if overlap>=0.6:
					    hit=1
				    if overlap>=0.5:
					   hit2=1
				    if overlap>=0.4:
					   hit3=1
				    if overlap>=0.3:
					   hit4=1
				    if overlap>=0.2:
					   hit5=1
				    if overlap>=0.1:
					   hit6=1
				    if overlap>0.0:
					   hit7=1
			
		    self.totalhit+=hit
		    self.totalhit2+=hit2
		    self.totalhit3+=hit3
		    self.totalhit4+=hit4
		    self.totalhit5+=hit5
		    self.totalhit6+=hit6
		    self.totalhit7+=hit7
		    self.totalframes+=1
	            print "================="
		    print "accuracy0.6=",float(self.totalhit)/self.totalframes
		    print "accuracy0.5=",float(self.totalhit2)/self.totalframes
		    print "accuracy0.4=",float(self.totalhit3)/self.totalframes
		    print "accuracy0.3=",float(self.totalhit4)/self.totalframes
		    print "accuracy0.2=",float(self.totalhit5)/self.totalframes
		    print "accuracy0.1=",float(self.totalhit6)/self.totalframes
		    print "accuracy0.0=",float(self.totalhit7)/self.totalframes
		    
		    for saliency in all_saliency_maps:
			    show_style='camshow'
			    
			    plt.subplot(2, 16, k+17)	
			    if show_style == 'camshow':
				
				viz.plot_cam(np.abs(saliency[k]).squeeze(), img, 'jet', alpha=0.5)
				
				plt.axis('off')
				plt.title(float(np.average(saliency[k])))
				
				self.seq.append(np.array(np.expand_dims(saliency[k],axis=2))*np.array(img))		
				
			    else:
				if model_name == 'googlenet' or method_name == 'pattern_net':
				    saliency = saliency.squeeze()[::-1].transpose(1, 2, 0)
				else:
				    saliency = saliency.squeeze().transpose(1, 2, 0)
				saliency -= saliency.min()
				saliency /= (saliency.max() + 1e-20)
				plt.imshow(saliency, cmap='gray')
			    
			    if method_name == 'excitation_backprop':
				plt.title('Exc_bp')
			    elif method_name == 'contrastive_excitation_backprop':
				plt.title('CExc_bp')
			    else:
				plt.title('%s' % (method_name))
		
		plt.tight_layout()
		print path.split("/")
		
		plt.savefig('./embrace_%i_%s.png' % (index,path.split("/")[-2]))

	     torch.cuda.empty_cache()
	     
	     print path.split("/")
	     
	    
	     
	     return self.seq,self.kls,self.scr
	     
