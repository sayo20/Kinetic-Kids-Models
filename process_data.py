# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import subprocess
import glob
import os
from  tqdm import tqdm
import imageio
import os
import numpy 
import torch
import glob
from  tqdm import tqdm
import math
import time


# %%
def process_dataset(directory_path,dir_,fps_final,category,skip_existing=False):
    if os.path.exists('movie.mp4'):
            os.remove('movie.mp4')

    dirx = os.path.join(directory_path,category+"_kid") 
    new_path = os.path.join(dir_,os.path.basename(dirx))
    
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    videos = glob.glob(os.path.join(dirx,'*'))

    for video_path in tqdm(videos):
        output_path = os.path.join(os.path.join(dir_,os.path.basename(os.path.dirname(video_path)),os.path.basename(video_path)))
        if skip_existing and os.path.isfile(output_path):
            continue
        reader = imageio.get_reader(video_path,  'ffmpeg')

        fps =  reader.get_meta_data()['fps']


        read_path = video_path
        if not fps>= fps_final:
            mutliple = int(math.ceil(fps_final/fps)) - 1
            command= "python inference_video.py --exp=" +str(mutliple)+" --video=\""+video_path+ "\" --output movie.mp4"

            subprocess.call(command, shell=True)
            read_path  = 'movie.mp4'

        write_path  = os.path.join(new_path,os.path.basename(video_path))
        if os.path.exists(write_path):
            os.remove(write_path)

        
        command = "ffmpeg -i \""+read_path+"\" -filter:v fps=fps=30 \"" + output_path+"\""
        
        while not os.path.isfile('movie.mp4') and ( fps< fps_final):
            pass
        subprocess.call(command, shell=True)

        time.sleep(3)
        if os.path.exists('movie.mp4'):
            os.remove('movie.mp4')
          

   

# pass


# %%
categories = ['playing basketball', 'playing tennis', 'dribbling basketball', 
              'dunking basketball', 'water skiing',
              'kicking soccer ball', 'bowling', 'catching or throwing baseball', 'hitting baseball',
              'bouncing on trampoline', 'shooting basketball', 'playing cricket', 'playing badminton',
              'somersaulting', 'cartwheeling', 'catching or throwing softball', 'playing volleyball',
              'juggling soccer ball', 'archery', 'parkour','catching or throwing frisbee']



# %%
for category in categories:
    print(category)
    process_dataset('Kinetic-Kids','Kinetic-Kids-processed',30,category,skip_existing=True)
    


# %%



