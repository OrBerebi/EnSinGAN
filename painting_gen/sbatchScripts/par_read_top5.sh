#!/bin/bash

input="./calc_top_matches/top5_photo_monet.txt"


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
  python_line=`cat painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example | grep python`
  curr_id=`cat painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example | grep "name of the job" | cut -d " " -f 3`
  curr_monet=`awk -F' ' '{print $5}' <<< $python_line`
  curr_photo=`awk -F' ' '{print $7}' <<< $python_line`
  sed -i "s/$curr_monet/$monet_name_1/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sed -i "s/$curr_photo/$photo_name/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sed -i "s/$curr_id/$photo_name/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sbatch painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example

  #get current photo and monet 
  python_line=`cat painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example | grep python`
  curr_id=`cat painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example | grep "name of the job" | cut -d " " -f 3`
  curr_monet=`awk -F' ' '{print $5}' <<< $python_line`
  curr_photo=`awk -F' ' '{print $7}' <<< $python_line`
  sed -i "s/$curr_monet/$monet_name_2/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sed -i "s/$curr_photo/$photo_name/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sed -i "s/$curr_id/$photo_name/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sbatch painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example

  #get current photo and monet 
  python_line=`cat painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example | grep python`
  curr_id=`cat painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example | grep "name of the job" | cut -d " " -f 3`
  curr_monet=`awk -F' ' '{print $5}' <<< $python_line`
  curr_photo=`awk -F' ' '{print $7}' <<< $python_line`
  sed -i "s/$curr_monet/$monet_name_3/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sed -i "s/$curr_photo/$photo_name/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sed -i "s/$curr_id/$photo_name/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sbatch painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example

  #get current photo and monet 
  python_line=`cat painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example | grep python`
  curr_id=`cat painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example | grep "name of the job" | cut -d " " -f 3`
  curr_monet=`awk -F' ' '{print $5}' <<< $python_line`
  curr_photo=`awk -F' ' '{print $7}' <<< $python_line`
  sed -i "s/$curr_monet/$monet_name_4/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sed -i "s/$curr_photo/$photo_name/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sed -i "s/$curr_id/$photo_name/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sbatch painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example

  #get current photo and monet 
  python_line=`cat painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example | grep python`
  curr_id=`cat painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example | grep "name of the job" | cut -d " " -f 3`
  curr_monet=`awk -F' ' '{print $5}' <<< $python_line`
  curr_photo=`awk -F' ' '{print $7}' <<< $python_line`
  sed -i "s/$curr_monet/$monet_name_5/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sed -i "s/$curr_photo/$photo_name/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sed -i "s/$curr_id/$photo_name/g" painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example
  sbatch painting_gen/sbatchScripts/sbatch_gpu_inference_injection.example

done < "$input"
