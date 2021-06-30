import os
import pandas as pd

def changename(csvpath,video_path):
    """csvpath: path to the csv e.g Data_Csv/TrainSplit-kids_Half.csv
        video_path: path to video (Kinetic-Kids-processed or Kinetic-adults-processed), depends on which csv you're passing in

    """
    df = pd.read_csv(csvpath)
    old_filename = list(df["Original name"])

    new_filename = list(df["Video name"])
    video_paths = list(df["Video Path"])
    labels = list(df["Action Label"])
    for index,file in enumerate(new_filename):
        old_pth = labels[index]+"/"+old_filename[index]+".mp4"
        new_pth = labels[index]+"/"+file+".mp4"
        src = os.path.join(video_path,old_pth)
        dst = os.path.join(video_path,new_pth)

        try:
#             print(src,dst)
            os.rename(src,dst)
        except:
            pass

#rename kids
#train
changename("Data_Csv/TrainSplit-kids.csv","Kinetic-Kids-processed")
#test
changename("Data_Csv/TestSplit-kids.csv","Kinetic-Kids-processed")
#val
changename("Data_Csv/ValSplit-kids.csv","Kinetic-Kids-processed")
#half-train
changename("Data_Csv/ValSplit-kids_Half.csv","Kinetic-Kids-processed")

#rename adults
#train
changename("Data_Csv/TrainSplit-adults.csv","Kinetic-adults-processed")
#test
changename("Data_Csv/TestSplit-adults.csv","Kinetic-adults-processed")
#val
changename("Data_Csv/ValSplit-adults.csv","Kinetic-adults-processed")
#half-train
changename("Data_Csv/ValSplit-adults_Half.csv","Kinetic-adults-processed")
