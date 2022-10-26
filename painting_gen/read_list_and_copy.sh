!/bin/bash

input="./calc_top_matches/top5_photo_monet.txt"
monet_path="./model_gen/Input/monet_jpg_names//"
photo_path="./painting_gen/Input/photo_jpg/"
out_path="./painting_gen/Output/top1/Paint2image/"

file_out_path="./out_triplets_s6/"

#input="./test.txt"

while IFS= read -r line
do
  clean_line=`echo $line | grep "monet"`
  monet_name_jpg=`awk -F' ' '{print $2}' <<< $clean_line`
  photo_name_jpg=`awk -F' ' '{print $1}' <<< $clean_line`
  val_name=`awk -F' ' '{print $7}' <<< $clean_line`


  monet_name=$(echo "$monet_name_jpg" | cut -f 1 -d '.')
  photo_name=$(echo "$photo_name_jpg" | cut -f 1 -d '.')

  out_name=$monet_name"/"$photo_name"_out/start_scale=6.jpg"

  file_name=$val_name"__"$photo_name"__"$monet_name".jpg"

  python horoz_paste_images.py $monet_path$monet_name_jpg $photo_path$photo_name_jpg $out_path$out_name $file_out_path$file_name

  #echo $out_path$out_name
  #echo $monet_path$monet_name_jpg
  #echo $photo_path$photo_name_jpg

  #get current photo and monet 
  #python_line=`cat sbatch_gpu_inference_injection.example | grep python`
  #curr_id=`cat sbatch_gpu_inference_injection.example | grep "name of the job" | cut -d " " -f 3`
  #curr_monet=`awk -F' ' '{print $5}' <<< $python_line`
  #curr_photo=`awk -F' ' '{print $7}' <<< $python_line`

  #echo $monet_name
  #echo $photo_name
  #echo $curr_monet
  #echo $curr_photo
  #sed -i "s/$curr_monet/$monet_name/g" sbatch_gpu_inference_injection.example
  #sed -i "s/$curr_photo/$photo_name/g" sbatch_gpu_inference_injection.example
  #sed -i "s/$curr_id/$photo_name/g" sbatch_gpu_inference_injection.example


  #sbatch sbatch_gpu_inference_injection.example
done < "$input"




