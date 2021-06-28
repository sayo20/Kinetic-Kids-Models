import imageio
import os
import numpy 
import torch
import glob
from  tqdm import tqdm
import math
import time


def process_dataset(directory_path,dir_,fps_final):
    
    dirs = glob.glob(os.path.join(directory_path,'*'))
    for dir in tqdm(dirs):
        new_path = os.path.join(dir_,os.path.basename(dir))
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        videos = glob.glob(os.path.join(dir,'*'))
        
        for video_path in videos:
            reader = imageio.get_reader(video_path,  'ffmpeg')
            frame_count = reader.count_frames()
            fps =  reader.get_meta_data()['fps']
       

            read_path = video_path
            if not fps>= fps_final:
                mutliple = int(math.ceil(fps_final/fps)) - 1
                command= "python inference_video.py --exp=" +str(mutliple)+" --video=\""+video_path+ "\" --output movie.mp4"
                # command = f'python inference_video.py --exp={mutliple} --video="{video_path}"  --output movie.mp4'
                os.system(command)
                read_path  = 'movie.mp4'
            clean_path = video_path.replace(' ',"_")
            write_path  = os.path.join(new_path,os.path.basename(clean_path))
            if os.path.exists(write_path):
                os.remove(write_path)
            command = "ffmpeg -i \""+read_path+"\" -filter:v fps=fps="+str(fps_final)+" "+ write_path
            # command = f'ffmpeg -i "{read_path}" -filter:v fps=fps={fps_final} {write_path}'
            while not os.path.isfile('movie.mp4') and ( fps< fps_final):
                pass
            os.system(command)
            time.sleep(5)
            if os.path.exists('movie.mp4'):
                os.remove('movie.mp4')
          

   

            pass

    





if __name__ == '__main__':
    process_dataset('Kinetic-Adults','Kinetic-Adults-processed',30)
    process_dataset('Kinetic-Kids','Kinetic-Adults-processed',30)
    