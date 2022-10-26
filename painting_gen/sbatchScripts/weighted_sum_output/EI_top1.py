import numpy as np
from PIL import Image
import os
import shutil


# Using readlines()
file1 = open('./calc_top_matches/top5_photo_monet_final.txt', 'r')
Lines = file1.readlines()

path = "./painting_gen/Output/top5/Paint2image/"
save_path = "./eval_dir/top/top1_s6/"
photo_path="./painting_gen/Input/photo_jpg/"
monet_path="./model_gen/Input/monet_jpg_names/"
out_name_file="_out/start_scale=6.jpg"
if not os.path.isdir(save_path):
      os.mkdir(save_path)

for line in Lines:
    sentence = line.split()
    photo_name = os.path.splitext(sentence[0])[0]
    monet_name_1 = os.path.splitext(sentence[1])[0]

    val_1 = 1 - float(sentence[6])

    val_sum = val_1
    save_path_curr_dir = save_path + photo_name +"/"
    if not os.path.isdir(save_path_curr_dir):
      os.mkdir(save_path_curr_dir)
    shutil.copyfile(photo_path+photo_name+".jpg", save_path_curr_dir+photo_name+".jpg")
    shutil.copyfile(monet_path+monet_name_1+".jpg", save_path_curr_dir+monet_name_1+".jpg")

    img_path = path + monet_name_1 +"/" + photo_name + out_name_file
    img = np.asarray(Image.open(img_path))
    img_w_1 = (val_1 * img)/val_sum

    save_path_curr = save_path_curr_dir +"/EI_"+ photo_name + ".jpg"
    img_out = ((img_w_1)).astype(np.uint8)
    img_out_pil = Image.fromarray(img_out)
    img_out_pil.save(save_path_curr)

