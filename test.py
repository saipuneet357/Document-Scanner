import cv2
import numpy as np


image = cv2.imread('b.jpg')

(h,w,c) = image.shape


ar = h/w


image = cv2.resize(image,(300,int(300*ar)))

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray,(5,5),0)

can = cv2.Canny(blur,30,150)

_,thresh = cv2.threshold(blur,190,255,cv2.THRESH_BINARY)


thresh = cv2.dilate(thresh,None,iterations=5) 

thresh = cv2.erode(thresh,None,iterations=5) 

cnts,_ = cv2.findContours(can.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 

output = image.copy()


c = max(cnts, key = cv2.contourArea)

extL = tuple(c[c[:, :, 0].argmin()][0])
extR = tuple(c[c[:, :, 0].argmax()][0])
extT = tuple(c[c[:, :, 1].argmin()][0])
extB = tuple(c[c[:, :, 1].argmax()][0])



#roi = image[extT[1]:extB[1],extL[0]:extR[0]]



lp = [extL,extT,extR,extB]


def compare(elem):
	return [elem[0],elem[1]]



l = sorted(lp,key = compare)



topL = l[np.argmin([l[0][1],l[1][1]])]
botL = l[np.argmax([l[0][1],l[1][1]])]
topR = l[np.argmin([l[2][1],l[3][1]])+2]
botR = l[np.argmax([l[2][1],l[3][1]])+2]


points = [topL,topR,botL,botR]


h1 = np.linalg.norm(np.array(topL)-np.array(botL))
h2 = np.linalg.norm(np.array(topR)-np.array(botR))


maxh = max(int(h1),int(h2))


w1 = np.linalg.norm(np.array(topL)-np.array(topR))
w2 = np.linalg.norm(np.array(botL)-np.array(botR))

maxw = max(int(w1),int(w2))


d = [0]*4

d[0] = (0,0)
d[1] = (maxw-1,0)
d[2] = (0,maxh-1)
d[3] = (maxw-1,maxh-1)

#cv2.rectangle(output,d[0],d[3],(0,255,0),2)

#rect = [0]*4
#print(rect)
#s = np.sum(l,axis=1
#rect[0] = l[np.argmin(s)]
#rect[3] = l[np.argmax(s)]
#d = np.diff(l,axis=1)
#rect[1] = l[np.argmin(d)]
#rect[2] = l[np.argmax(d)]
#print(rect)


M = cv2.getPerspectiveTransform(np.array(points,dtype='float32'),np.array(d,dtype='float32'))
warped = cv2.warpPerspective(image,M,(maxw,maxh))


cv2.drawContours(output,[c],0,(240,0,159),3) 
cv2.imshow('contours',output)
cv2.waitKey(0)

for point in points:
	cv2.circle(output,point,10,(255,255,0),-1)
	cv2.imshow('contours',output)
	cv2.waitKey(0)

cv2.imshow('warped',warped)
cv2.waitKey(0)





