#!/bin/bash
for i in {1..5}
do
  sbatch ./model_gen/sbatchScripts/sbatch_gpu_new.example
  tmp=$((i+1))
  curr_im="monet$i"
  next_im="monet$tmp"
  sed -i "s/$curr_im/$next_im/g" ./model_gen/sbatchScripts/sbatch_gpu_new.example
done
