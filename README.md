# EnSinGAN
Utilizing Single image GANs to capture Claude Monet's impressionistic paintings as generating new Monet-inspired ones.

# Installation
Use the provided environment.yml file to install a Anconda environment with the necessary packages to run the code. 

From the main folder run:
```
conda env create -f environment_tf.yml
```
At the end, an Anconda environment named EnSinGAN_env should be successfully installed. This environment includes:
- python=3.6
- pytorch=1.4
- tesnsoreflow-gpu=1.14

and other important packages.


# How to use
## step 1 - download the data
Download the monet_jpg pantings dataset from: https://drive.google.com/drive/folders/1Z22cJ3ll8kJE6OyVBpQoFQ6WY4f6dD1r?usp=sharing
and store its content in ./model_gen/Input/monet_jpg_names/ 

Download the photo_jpg images dataset from: https://drive.google.com/drive/folders/1Z22cJ3ll8kJE6OyVBpQoFQ6WY4f6dD1r?usp=sharing
store its content in ./painting_gen/Input/photo_jpg/

## step 2 - Train the SinGAN models
From the main folder run
```
bash model_gen/sbatchScripts/par_sbathc.sh
```
This script will initiate the 300 model training. In the end the folder ./model_gen/TrainedModels/ should be filed with 300 trained models.


## step 3 - Cosine distance file
From the main folder run
```
sbatch calc_top_matches/sbatchScripts/sbatch_gpu_get_featurs.example
```
This script will create a text file that lists each image in photo_jpg and its top5 closest paintings in monet_jpg.

## step 4 - inference: Model Selection
From the main folder run
```
bash painting_gen/sbatchScripts/par_read_top5.sh
```
This script will inject each photo image in photo_jpg into its top5 SinGAN models according to the .txt file generated in step 3. The Output images will be saved to painting_gen/Output/

## step 5 - inference: Aggregation
From the main folder run
```
python ./painting_gen/sbatchScripts/weighted_sum_output/EI_top5.py
```
This script will aggregate the generated images according to the weighted sum defined in the paper and will save it into ./eval_dir/top5/ alongside its source image and the 5 Monet paintings. It is optional to use EI_top1.py or EI_top3.py to use the top1 or top3 matches.
