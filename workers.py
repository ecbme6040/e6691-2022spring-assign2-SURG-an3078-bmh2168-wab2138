
import cv2
import os



def process_vid(file):
    
    path = os.fsdecode(file)
    path = os.path.join('./data', path)
    if path.endswith('.mp4') :               
        title=os.path.basename(path)[:-4]
        vidcap = cv2.VideoCapture(path)
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        success,image = vidcap.read()
        count = 0
  
        # Directory
        path = os.path.join('./data', title)
        os.mkdir(path)
        while success:
            cv2.imwrite(path+"/frame%d.jpg" % count, image)     # save frame as JPEG file      
            success,image = vidcap.read()
            if (count%500==499):
                print(title, success,count*100/length)
            count += 1
        if count!=length or length==0:
            print('pb with', title)
            return 'ERROR '+title
        
        return "SUCCESS "+ path
    return "not a vid"+path