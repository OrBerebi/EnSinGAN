import numpy as np
from PIL import Image
import os
import shutil


# Using readlines()
file1 = open('top5_photo_monet_final.txt', 'r')
Lines = file1.readlines()

path = "./Output/top5/Paint2image/"
save_path = "./eval_dir/top/top5_s1/"
photo_path="./Input/Paint/photo_jpg/"
monet_path="./Input/Images/monet_jpg_names/"
scale_out_name="_out/start_scale=1.jpg"
if not os.path.isdir(save_path):
      os.mkdir(save_path)

for line in Lines:
    sentence = line.split()
    photo_name = os.path.splitext(sentence[0])[0]
    monet_name_1 = os.path.splitext(sentence[1])[0]
    monet_name_2 = os.path.splitext(sentence[2])[0]
    monet_name_3 = os.path.splitext(sentence[3])[0]
    monet_name_4 = os.path.splitext(sentence[4])[0]
    monet_name_5 = os.path.splitext(sentence[5])[0]
    
    val_1 = 1 - float(sentence[6])
    val_2 = 1 - float(sentence[7])
    val_3 = 1 - float(sentence[8])
    val_4 = 1 - float(sentence[9])
    val_5 = 1 - float(sentence[10])

    val_sum = val_1 +val_2 +val_3 +val_4 +val_5 
    save_path_curr_dir = save_path + photo_name +"/"
    if not os.path.isdir(save_path_curr_dir):
      os.mkdir(save_path_curr_dir)
    shutil.copyfile(photo_path+photo_name+".jpg", save_path_curr_dir+photo_name+".jpg")
    shutil.copyfile(monet_path+monet_name_1+".jpg", save_path_curr_dir+monet_name_1+".jpg")
    shutil.copyfile(monet_path+monet_name_2+".jpg", save_path_curr_dir+monet_name_2+".jpg")
    shutil.copyfile(monet_path+monet_name_3+".jpg", save_path_curr_dir+monet_name_3+".jpg")
    shutil.copyfile(monet_path+monet_name_4+".jpg", save_path_curr_dir+monet_name_4+".jpg")
    shutil.copyfile(monet_path+monet_name_5+".jpg", save_path_curr_dir+monet_name_5+".jpg")

    img_path = path + monet_name_1 +"/" + photo_name + scale_out_name
    img = np.asarray(Image.open(img_path))
    img_w_1 = (val_1 * img)/val_sum

    img_path = path + monet_name_2 +"/" + photo_name + scale_out_name
    img = np.asarray(Image.open(img_path))
    img_w_2 = (val_2 * img)/val_sum

    img_path = path + monet_name_3 +"/" + photo_name + scale_out_name
    img = np.asarray(Image.open(img_path))
    img_w_3 = (val_3 * img)/val_sum

    img_path = path + monet_name_4 +"/" + photo_name + scale_out_name
    img = np.asarray(Image.open(img_path))
    img_w_4 = (val_4 * img)/val_sum

    img_path = path + monet_name_5 +"/" + photo_name + scale_out_name
    img = np.asarray(Image.open(img_path))
    img_w_5 = (val_5 * img)/val_sum

    save_path_curr = save_path_curr_dir +"/EI_"+ photo_name + ".jpg"
    img_out = ((img_w_1 + img_w_2 + img_w_3 + img_w_4 + img_w_5)).astype(np.uint8)
    #v_min = img_out.min(axis=(0, 1), keepdims=True)
    #v_max = img_out.max(axis=(0, 1), keepdims=True)
    #img_out_norm = 255*((img_out - v_min)/(v_max - v_min))    
    #print("out shape:\n", img_out)
    img_out_pil = Image.fromarray(img_out)
    img_out_pil.save(save_path_curr)
