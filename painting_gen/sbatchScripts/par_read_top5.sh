#!/bin/bash

input="./top5_photo_monet_final.txt"
output_path="./Output/top5/Paint2image/monet1/rest1050_out/start_scale\=9.jpg"

#input="./test.txt"

while IFS= read -r line
do
  clean_line=`echo $line | grep "photo"`
  photo_name=`awk -F' ' '{print $1}' <<< $clean_line`

  monet_name_1=`awk -F' ' '{print $2}' <<< $clean_line`
  monet_name_2=`awk -F' ' '{print $3}' <<< $clean_line`
  monet_name_3=`awk -F' ' '{print $4}' <<< $clean_line`
  monet_name_4=`awk -F' ' '{print $5}' <<< $clean_line`
  monet_name_5=`awk -F' ' '{print $6}' <<< $clean_line`

  #get current photo and monet 
  python_line=`cat sbatch_gpu_inference_injection.example | grep python`
  curr_id=`cat sbatch_gpu_inference_injection.example | grep "name of the job" | cut -d " " -f 3`
  curr_monet=`awk -F' ' '{print $5}' <<< $python_line`
  curr_photo=`awk -F' ' '{print $7}' <<< $python_line`
  sed -i "s/$curr_monet/$monet_name_1/g" sbatch_gpu_inference_injection.example
  sed -i "s/$curr_photo/$photo_name/g" sbatch_gpu_inference_injection.example
  sed -i "s/$curr_id/$photo_name/g" sbatch_gpu_inference_injection.example
  sbatch sbatch_gpu_inference_injection.example

  #get current photo and monet 
  python_line=`cat sbatch_gpu_inference_injection.example | grep python`
  curr_id=`cat sbatch_gpu_inference_injection.example | grep "name of the job" | cut -d " " -f 3`
  curr_monet=`awk -F' ' '{print $5}' <<< $python_line`
  curr_photo=`awk -F' ' '{print $7}' <<< $python_line`
  sed -i "s/$curr_monet/$monet_name_2/g" sbatch_gpu_inference_injection.example
  sed -i "s/$curr_photo/$photo_name/g" sbatch_gpu_inference_injection.example
  sed -i "s/$curr_id/$photo_name/g" sbatch_gpu_inference_injection.example
  sbatch sbatch_gpu_inference_injection.example

  #get current photo and monet 
  python_line=`cat sbatch_gpu_inference_injection.example | grep python`
  curr_id=`cat sbatch_gpu_inference_injection.example | grep "name of the job" | cut -d " " -f 3`
  curr_monet=`awk -F' ' '{print $5}' <<< $python_line`
  curr_photo=`awk -F' ' '{print $7}' <<< $python_line`
  sed -i "s/$curr_monet/$monet_name_3/g" sbatch_gpu_inference_injection.example
  sed -i "s/$curr_photo/$photo_name/g" sbatch_gpu_inference_injection.example
  sed -i "s/$curr_id/$photo_name/g" sbatch_gpu_inference_injection.example
  sbatch sbatch_gpu_inference_injection.example

  #get current photo and monet 
  python_line=`cat sbatch_gpu_inference_injection.example | grep python`
  curr_id=`cat sbatch_gpu_inference_injection.example | grep "name of the job" | cut -d " " -f 3`
  curr_monet=`awk -F' ' '{print $5}' <<< $python_line`
  curr_photo=`awk -F' ' '{print $7}' <<< $python_line`
  sed -i "s/$curr_monet/$monet_name_4/g" sbatch_gpu_inference_injection.example
  sed -i "s/$curr_photo/$photo_name/g" sbatch_gpu_inference_injection.example
  sed -i "s/$curr_id/$photo_name/g" sbatch_gpu_inference_injection.example
  sbatch sbatch_gpu_inference_injection.example

  #get current photo and monet 
  python_line=`cat sbatch_gpu_inference_injection.example | grep python`
  curr_id=`cat sbatch_gpu_inference_injection.example | grep "name of the job" | cut -d " " -f 3`
  curr_monet=`awk -F' ' '{print $5}' <<< $python_line`
  curr_photo=`awk -F' ' '{print $7}' <<< $python_line`
  sed -i "s/$curr_monet/$monet_name_5/g" sbatch_gpu_inference_injection.example
  sed -i "s/$curr_photo/$photo_name/g" sbatch_gpu_inference_injection.example
  sed -i "s/$curr_id/$photo_name/g" sbatch_gpu_inference_injection.example
  sbatch sbatch_gpu_inference_injection.example

done < "$input"

#python -u paint2image.py --input_name monet168.jpg --ref_name 23.jpg --paint_start_scale 3 --scale_factor 0.4 --input_dir ./Input/Images/monet_jpg_names/ --ref_dir ./Input/Paint/realistic




