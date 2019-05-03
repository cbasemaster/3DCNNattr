import sys
sys.path.append('./')

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from create_explainer import get_explainer
from preprocess import get_preprocess
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
	    #['googlenet', 'vanilla_grad',       'imshow'],
	    #['googlenet', 'grad_x_input',       'imshow'],
	    #['googlenet', 'saliency',           'imshow'],
	    #['googlenet', 'integrate_grad',     'imshow'],
	    #['googlenet', 'deconv',             'imshow'],
	    #['googlenet', 'deeplift_rescale',    'imshow'],
	    #['googlenet', 'gradcam',            'camshow'],
	    ['resnext', 'gradcam',            'camshow']
	    #['googlenet', 'excitation_backprop', 'camshow'],
	    #['googlenet', 'contrastive_excitation_backprop', 'camshow'],
	    #['vgg16',     'pattern_net',        'imshow'],
	    #['vgg16',     'pattern_lrp',        'camshow'],
	    #['resnet50',  'real_time_saliency', 'camshow']
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
	#transf = get_preprocess(self.model_methods[0][0], self.model_methods[0][1])
        #self.model = utils.load_model(self.model_methods[0][0])
        #self.model.cuda()

    def shannon(self,freqList):
        ent = 0.0
        freqList=freqList
        #print (freqList)
        freqList=cv2.normalize(np.ravel(freqList).astype(float),None,1.0,0.0,cv2.NORM_L1)
        #print (freqList)
        for freq in freqList:
                freq=freq[0]
                if freq==0:
                        freq=1.0
                #print (freq)
                ent = ent + freq * math.log(freq, 2)

        if ent<0.0:
                ent = -ent
        #print (ent)
        #print ("--------")
        return ent


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
     path_gt = path.replace("base2/","base/")    
     
     path="../visual-attribution/"+path
     path_gt="../visual-attribution/"+path_gt
     print path_gt
     if os.path.isdir(path_gt):
	     print "YOOOOOOOOOOOOO"                
     
    

	     for model_name, method_name, _ in self.model_methods:
		#path=self.path
	 

	 
		dirc=os.listdir(path)
		dirc_gt=os.listdir(path_gt)
	        print dirc_gt
	        print "NIIIIIIIIIIIIIIIII"
	        	

		files=[fname for fname in dirc if fname.endswith('png')]
		#files=[fname for fname in dirc if fname.startswith('img')]
	        #files2=[fname for fname in dirc if fname.startswith('flow_x')]
		files_gt=[fname for fname in dirc_gt if fname.endswith('mat')]
		print files,files_gt	
		#transf = get_preprocess(model_name, method_name)
		#model = utils.load_model(model_name)
		#model.cuda()

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


		#for filename in sorted(files_gt)[index:index+16]:

		
		
			
		#video.append(Image.open(path+filename))

		for filename in sorted(files)[index:index+16]:
		    
			#print filename	
			video.append(Image.open(path+filename))
			
		"""
		for filename in sorted(files2)[index:index+16]:


			flows.append((Image.open(path+filename)))
		"""
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
			#flows=flows * (16/len(flows)) * 2
			#flows=flows[0:16]
		#print np.array(flows).shape	
		#image_class = 63 # floorgymnastic
		#flows = np.array(flows)-np.mean(np.array(flows))
		
		#matfile=scipy.io.loadmat(path_gt+files_gt[0])
		#coor= np.array(matfile["pos_img"]).transpose(2,1,0).tolist()
		
	        print len(video),len(coor)
	        if len(coor)==0:
			continue	
		for e in range(0,len(coor),1):
		    		
		    box=bounding_box([(abs(dots[0])*112.0/320.0,abs(dots[1])*112.0/240.0) for dots in coor[e]])
		    boxes.append(box)
		#print boxes
		
		#flows=[Image.fromarray(np.uint8(f)) for f in flows]	
		"""
		for i in range(1, 5):
			scales.append(scales[-1] * 0.84089641525)
		"""
		self.spatial_transform.randomize_parameters()
		self.spatial_transform2.randomize_parameters()
		self.spatial_transform3.randomize_parameters()
		#spatial_transform2.crop_position=spatial_transform.crop_position
		#spatial_transform2.MultiScaleCornerCrop().scale = spatial_transform.MultiScaleCornerCrop().scale
		#spatial_transform2.MultiScaleCornerCrop().crop_position = spatial_transform.MultiScaleCornerCrop().crop_position
		#self.video = [img.paste(img2) for img,img2 in zip(self.video,self.flows)]
		
		clip = [self.spatial_transform3(img) for img in video]
		#op = [self.spatial_transform(img) for img in flows]
		#print torch.stack(clip,0).permute(1, 0, 2, 3).shape
		inp = torch.stack(clip,0).permute(1, 0, 2, 3)	
		
		
		"""      
		inp_op = torch.stack(op,0).permute(1, 0, 2, 3)
		
	       
		inp_op=inp_op/inp_op.max()
		mean=inp_op.mean()
		inp_op=(inp_op-mean)
	        inp_op=torch.clamp(inp_op, min=0)
		"""
		all_saliency_maps = []
		for model_name, method_name, _ in self.model_methods:
		    #transf = get_preprocess(model_name, method_name)
		    #model = utils.load_model(model_name)
		    #model.cuda()
		    #model=self.model
		    #explainer = get_explainer(model, method_name, "conv1")
		    """
		    explainer2 = get_explainer(model, method_name, "layer1")
		    explainer3 = get_explainer(model, method_name, "layer2")    
		    explainer4 = get_explainer(model, method_name, "layer3")
		    explainer5 = get_explainer(model, method_name, "layer4")
		    explainer6 = get_explainer(model, method_name, "avgpool")
		    """
		    #inp = transf(clip)
		    if method_name == 'googlenet':  # swap channel due to caffe weights
			inp_copy = inp.clone()
			inp[0] = inp_copy[2]
			inp[2] = inp_copy[0]
		    inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)
		    #inp.cuda.empty_cache()
		    #with torch.no_grad():
		    #target = torch.LongTensor([image_class]).cuda()
		    #print subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])    
		    
		    saliency,s,kls,scr,c = self.explainer.explain(inp)
		    saliency2, s2,kls,scr,c2 = self.explainer2.explain(inp)
		    saliency3,s3,kls,scr,c3 = self.explainer3.explain(inp)	
		    saliency4,s4,kls,scr,c4 = self.explainer4.explain(inp)
		    saliency5, s5,kls,scr,c5 = self.explainer5.explain(inp)
		    
		    saliency6,pool,kls,scr,c6 = self.explainer6.explain(inp)
		    """
		    #print subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
		    second="second"
		    _saliency,_s,_kls,_scr,_c = self.explainer.explain(inp,ind=second)
                    _saliency2, _s2,_kls,_scr,_c2 = self.explainer2.explain(inp,ind=second)
                    _saliency3,_s3,_kls,_scr,_c3 = self.explainer3.explain(inp,ind=second)
                    _saliency4,_s4,_kls,_scr,_c4 = self.explainer4.explain(inp,ind=second)
                    _saliency5,_s5,_kls,_scr,_c5 = self.explainer5.explain(inp,ind=second)
                    _saliency6,_pool,_kls,_scr,_c6 = self.explainer6.explain(inp,ind=second)
		    

		    second="third"
                    __saliency,__s,__kls,__scr,__c = self.explainer.explain(inp,ind=second)
                    __saliency2, __s2,__kls,__scr,__c2 = self.explainer2.explain(inp,ind=second)
                    __saliency3,__s3,__kls,__scr,__c3 = self.explainer3.explain(inp,ind=second)
                    __saliency4,__s4,__kls,__scr,__c4 = self.explainer4.explain(inp,ind=second)
                    __saliency5,__s5,__kls,__scr,__c5 = self.explainer5.explain(inp,ind=second)
                    __saliency6,__pool,_kls,_scr,__c6 = self.explainer6.explain(inp,ind=second)
		    print kls+1,scr,index
		    """
		    """
		    saliency=saliency.sum(1)
		    #cam=saliency.cumsum(2)
		    #cam=cam.sum(1).unsqueeze(1)
		    #mean=cam.mean()
		    #std=cam.std()
		    #saliency=(cam-mean)/std
		    saliency2_2=saliency2
			
		    cam=saliency2.sum(1).unsqueeze(1)
		    #mean=cam.mean()
		    #std=cam.std()
		    #cam=(cam-mean)/std
		    #cam /= (cam.max()+ 1e-20)	    
		    saliency2 = torch.clamp(cam, min=0)
		    
		    cam=saliency2_2.cumsum(2)
		    #cam=cam.sum(1).unsqueeze(1)
		    #mean=cam.mean()
		    #std=cam.std()
		    #cam=(cam-mean)/std
		    #cam /= (cam.max()+ 1e-20)
		    saliency2_2 = torch.clamp(cam, min=0) 
		    #print saliency2.shape
		    #quit()
		    """
		    """
		    cam=saliency3.sum(1).unsqueeze(1)
		    #mean=cam.mean()
		    #std=cam.std()
		    #cam=(cam-mean)/std
		    #cam /= (cam.max()+ 1e-20)
		    saliency3 = torch.clamp(cam, min=0)
		    
		    cam=saliency4.sum(1).unsqueeze(1)
		    #mean=cam.mean()
		    #std=cam.std()
		    #cam=(cam-mean)/std
		    #cam /= (cam.max()+ 1e-20)
		    saliency4 = torch.clamp(cam, min=0)
		    #saliency4=saliency4.sum(2).unsqueeze(2)
		    """
		    """
		    cam=saliency5.sum(1).unsqueeze(1)
		    #cam=saliency5.sum(1).unsqueeze(1)
		    #mean=cam.mean()
		    #std=cam.std()
		    #cam=(cam-mean)/(std+ 1e-20)
		    #cam /= (cam.max()+ 1e-20)
		    saliency5 = torch.clamp(cam, min=0)
		    torch.cuda.empty_cache()
		    """
		    #cam=saliency6.sum(1).unsqueeze(1)
		    #print saliency6.shape
		    #cam=saliency5.sum(1).unsqueeze(1)
		    #mean=cam.mean()
		    #std=cam.std()
		    #cam=(cam-mean)/(std+ 1e-20)
		    #cam /= (cam.max()+ 1e-20)
		    #saliency6 = torch.clamp(cam, min=0)    
		    torch.cuda.empty_cache()
		    
		    """
		    #mean=cam.mean()
		    #std=cam.std()
		    #cam=(cam-mean)/std
		    #cam /= (cam.max()+ 1e-20)
		    s = torch.clamp(cam, min=0)
		    
		    cam=s2.sum(1).unsqueeze(1)	
		    #mean=cam.mean()
		    #std=cam.std()
		    #cam=(cam-mean)/std
		    #cam /= (cam.max()+ 1e-20)

		    s2 = torch.clamp(cam, min=0)
		    """
		    """
		    cam=s3.sum(1).unsqueeze(1)
		    #mean=cam.mean()
		    #std=cam.std()
		    #cam=(cam-mean)/std
		    #cam /= (cam.max()+ 1e-20)
		    
		    s3 = torch.clamp(cam, min=0)
		    
		    cam=s4.sum(1).unsqueeze(1)
		    
		    #mean=cam.mean()
		    #std=cam.std()
		    #cam=(cam-mean)/std
		    #cam /= (cam.max()+ 1e-20)
		    
		    s4 = torch.clamp(cam, min=0)
		    """
		    """
		    cam=s5.sum(1).unsqueeze(1)
		    #mean=cam.mean()
		    #std=cam.std()
		    #cam=(cam-mean)/std
		    #cam /= (cam.max()+ 1e-20)

		    s5 = torch.clamp(cam, min=0)
		    #saliency=((saliency+saliency2+saliency3+saliency4+saliency5))    
		    #saliency=(s2*(saliency2+saliency2_2.sum(2)+(s3*saliency3)))
		     
		    torch.cuda.empty_cache()
		    #print subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
		    """
		    #saliency /= (saliency.max()+ 1e-20)
		    #saliency2 = torch.clamp(saliency2, min=0)

		    #_saliency /= (_saliency.max()+ 1e-20)
		    #saliency3 = torch.clamp(saliency3, min=0)

		    #saliency4 /= (saliency4.max()+ 1e-20)
		    #saliency4 = torch.clamp(saliency4, min=0)


		    #__saliency /= (__saliency.max()+ 1e-20)
		    #saliency6 = torch.clamp(saliency6, min=0)


		    #inp_op /= (inp_op.max()+ 1e-20)
		    #inp_op = torch.clamp(inp_op, min=0)
		    #s = torch.clamp(s, min=0)
		    
		    #saliency = inp_op*(pool*saliency6+s4*saliency4+s3*saliency3+s3*saliency2)
		    #saliency = (saliency2+saliency3+saliency4+saliency6)*(pool+s4*s3+s2)
		    
		    #------------ikilo
		    saliency=(saliency6+saliency5+saliency4+saliency3+saliency2+saliency)
		    #saliency=saliency6

		    """ 
		    _saliency=(_saliency6+_saliency5+_saliency4+_saliency3+_saliency2+_saliency)
		    __saliency=(__saliency6+__saliency5+__saliency4+__saliency3+__saliency2+__saliency)
		    #saliency=(scr*saliency)+(_scr*_saliency)+(__scr*__saliency)	
		    
		    saliency /= (saliency.max()+ 1e-20)
                    #saliency2 = torch.clamp(saliency2, min=0)

                    _saliency /= (_saliency.max()+ 1e-20)
                    #saliency3 = torch.clamp(saliency3, min=0)

                    #saliency4 /= (saliency4.max()+ 1e-20)
                    #saliency4 = torch.clamp(saliency4, min=0)


                    __saliency /= (__saliency.max()+ 1e-20)


		    count=scr*saliency
		    """
		    #print "predict-->", (scr*(c+c2+c3+c4+c5+c6)).mean()
		    
		    if self.classes[kls]==path.split("/")[5]:
			label=1
		    else:
			label=0
		    		    


		    #saliency=saliency2
		    #saliency=inp_op+(pool*saliency6+s2*saliency2+s3*saliency3+s4*saliency4+s5*saliency5)
		    
		    """
		    #saliency=((inp_op.data+saliency+saliency2+saliency5+saliency6)) 
		    #saliency=s*(inp_op.data+saliency)
		    #saliency -= saliency.min()
		    #saliency /= (saliency.max()+ 1e-20) 
		    """
		    """ 
		    mean=saliency.mean()
		    std=saliency.std()
		    cam=(saliency-mean)/std
		    
		    #saliency -= saliency.min()
		    saliency /= (saliency.max()+ 1e-20)
		    """
		    saliency = torch.clamp(saliency, min=0)   
		    #print inp_op.shape	    
		    """
		    print saliency.shape
		    """	     
		    temp=saliency.shape[2]
		    """
		    #target2 = utils.cuda_var(torch.zeros(1,1,temp, 112, 112))

		    #target2[:,:, :, 15:95,:] = saliency[:,:,:,15:95,:]
		    
		    #saliency=target2
		    """
		    """
		    mean=saliency.mean()
		    std=saliency.std()
		    saliency=(saliency-mean)/std
		    
		    saliency -= saliency.min()
		    saliency /= (saliency.max()+ 1e-20)
		    """
		    #saliency = torch.clamp(saliency, min=0)
		    if temp>1:   
			all_saliency_maps.append(saliency.squeeze().cpu().data.numpy())
		    else:
			all_saliency_maps.append(saliency.squeeze().unsqueeze(0).cpu().numpy())
		    #print subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
		    #del s,s2,s5,pool,inp,saliency,saliency2,saliency5,saliency6		    
		    del pool,inp,saliency,saliency6 
		    torch.cuda.empty_cache()
		    #print subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
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
			    #print box
			    #print np.array(img).shape   
			    #plt.imshow(img)
			    x=box[0][0]
			    y=box[0][1]
			    w = box[1][0] -x
			    h = box[1][1] -y
			    

			    
			    if (x+w)>112:
				w=(112-x)
			    if (y+h)>112:
				h=(112-y)
			    #print x,y,x+w,y+h	 
			    #preds= nms.non_max_suppression_fast(R2,0.5)	
			    ax=viz.plot_bbox([x,y,w,h],img)
		    plt.axis('off')
		    ax = plt.gca()
		    #print np.array(img).shape
		    #print inp_op.shape,img.shape
    		    #ax.imshow(inp_op.squeeze(0)[k])
		    ax.imshow(img)
		    sal=all_saliency_maps[0][k]
		    sal=(sal-np.mean(sal))/np.std(sal)
		    ret,thresh = cv2.threshold(sal,np.mean(sal)+((np.amax(sal)-np.mean(sal))*0.5),1,cv2.THRESH_BINARY)
		    
		    
		    #thresh=cv2.cvtColor(thresh.astype(np.uint8), cv2.COLOR_BGR2GRAY)
		    
	            contours,hierarchy = cv2.findContours(thresh.astype(np.uint8), 1, 2)
		    #print contours
		    areas = [cv2.contourArea(c) for c in contours]
		    #print areas
		    #quit()
		    #max_index = np.argmax(areas)
		    #cnt=contours[max_index] 
		    if len(contours)>0:	
			    #print cnt
		    	    
			    #x2,y2,w2,h2 = cv2.boundingRect(cnt)	
			    
			    """
			    img_lbl, regions = ss.selective_search(np.array(img), scale=5000, sigma=0.8, min_size=10)
			    R=regions
			    R2=[hout["rect"] for hout in R][1:-1]
			    
			    reds= nms.non_max_suppression_fast(R2,0.1)
			    #print reds
			   
			    #print preds
			    #print all_saliency_maps[0][k].shape 
			    #avglist=np.argsort([np.average(all_saliency_maps[0][k][p[1]:p[1]+p[3],p[0]:p[0]+p[2]]) for p in preds if p[2]!=0 and p[3]!=0])
						
			    #preds= [x for _,x in sorted(zip(avglist,preds))]
			    
			    
			    for bbox in reds:
				rect3 = patches.Rectangle((bbox[0], bbox[1]), bbox[2] , bbox[3],
                                         linewidth=2, edgecolor='b', facecolor='none')
                            	ax.add_patch(rect3)
			    """
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
				    #print x,w,y,h,x2,w2 
		            overlap=nms.get_iou([x,x+w,y,y+h],[x3,x13,y3,y13])
	                    #print overlap
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
		    #plt.title(k+1)
		    #plt.title('class_id=51 (violin)')
		    #plt.subplot(3, 5, i + 2 + i // 4)
		    for saliency in all_saliency_maps:
			    show_style='camshow'
			    #print saliency, img
			    plt.subplot(2, 16, k+17)	
			    if show_style == 'camshow':
				#print np.min(saliency[k])
				#print np.max(saliency[k])
				#print np.mean(saliency,axis=1).shape	
				viz.plot_cam(np.abs(saliency[k]).squeeze(), img, 'jet', alpha=0.5)
				#print "suweneee sung"
				plt.axis('off')
				plt.title(float(np.average(saliency[k])))
					
				#plt.subplot(2, 16, k+1)
				#teme=saliency / (saliency[k].max()+ 1e-20)
				#print np.amax(saliency[k])
				#teme=cv2.normalize(np.ravel(saliency[k]).astype(float),None,1.0,0.0,cv2.NORM_L1)	
				#print np.amax(teme)
				#quit()
				#teme=np.histogram(np.ravel(teme), bins=np.arange(0.0, 1.0, 0.25), density=True)
				#print teme[0]
				#print float(np.sum(saliency[k]),float(np.amax(saliency[k])),float(teme[0][-1]/ (teme[0].max()+ 1e-20)),float(np.var(saliency[k])),int(kls),float(scr) 
				#print  float(np.amax(saliency[k])),int(kls),scr
				
				#print (np.array(np.expand_dims(saliency[k],axis=2))*np.array(img)).shape			
				
					
				self.seq.append(np.array(np.expand_dims(saliency[k],axis=2))*np.array(img))		
				
				#self.scr.append(scr)
				
				#self.kls.append(int(kls))
					
				#imgl = ss._generate_segments(np.repeat(saliency[k][:, :, np.newaxis], 3, axis=2), 200,0.8,10)
				
				"""
				im_orig=np.repeat(saliency[:, :, np.newaxis], 3, axis=2)
				lb=np.zeros((112,112),dtype=np.uint8)
				
				index=0
				for i in range(0,112,int(112/8)):
					for j in range(0,112,int(112/8)):	
						lb[i:i+int(112/8),j:j+int(112/8)]=index
						index +=1
				im_orig = np.append(im_orig, np.zeros(saliency.shape[:2])[:, :, np.newaxis], axis=2)
				im_orig[:,:,3] = lb 
				R = ss._extract_regions(im_orig)
				
				
				top_avg=0
				"""
				"""
				x=value["min_x"]
				y=value["min_y"]
				x_max=value["max_x"] 
				y_max=value["max_y"] 
				w = x_max -x
				h = y_max -y
				
				
				if top_avg==0:
					top_key=key
					top_avg=value["avg"]
				elif top_avg < value["avg"]:
					top_key=key
					top_avg=value["avg"]
				#for x, y, w, h in bb_frames[k]:
				
					
				x=R[top_key]["min_x"]
				y=R[top_key]["min_y"]
				x_max=R[top_key]["max_x"]
				y_max=R[top_key]["max_y"]
				"""
				#print R[int(top_key)]	
				#plt.title(k+1)
			    else:
				if model_name == 'googlenet' or method_name == 'pattern_net':
				    saliency = saliency.squeeze()[::-1].transpose(1, 2, 0)
				else:
				    saliency = saliency.squeeze().transpose(1, 2, 0)
				saliency -= saliency.min()
				saliency /= (saliency.max() + 1e-20)
				plt.imshow(saliency, cmap='gray')
			    
			    #plt.axis('off')
			    if method_name == 'excitation_backprop':
				plt.title('Exc_bp')
			    elif method_name == 'contrastive_excitation_backprop':
				plt.title('CExc_bp')
			    else:
				plt.title('%s' % (method_name))
		#del explainer,explainer2,explainer5,explainer6
		#del explainer6	
		plt.tight_layout()
		print path.split("/")
		
		plt.savefig('./embrace_%i_%s.png' % (index,path.split("/")[-2]))

	     torch.cuda.empty_cache()
	     #plt.tight_layout()
	     print path.split("/")
	     
	     #plt.savefig('./embrace_%i_%s.png' % (index,path.split("/")[-3]))
	     
	     return self.seq,self.kls,self.scr
	     #plt.tight_layout()
	     #plt.savefig('images/embrace.png')
