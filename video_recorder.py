
import numpy as np 
import cv2
import random


def start_on():
    cap = cv2.VideoCapture(0)
    num=int(random.random()*1000)
    
    output = "OUTPUT/video"+str(num)+".avi"
 
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter(output, fourcc, 20.0, (640, 480))
   
    while(True): 

        ret, frame = cap.read()  
        out.write(frame)  
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release() 

    out.release()  
  
  
    cv2.destroyAllWindows()


