from __future__ import absolute_import, division, print_function
import warnings
warnings.filterwarnings('ignore')
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import os
import gzip, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from scipy import linalg
import pathlib
import urllib
import warnings
from tqdm import tqdm
from PIL import Image

import argparse
import re



class KernelEvalException(Exception):
    pass

model_params = {
    'Inception': {
        'name': 'Inception', 
        'imsize': 256,
        'output_layer': 'Pretrained_Net/pool_3:0', 
        'input_layer': 'Pretrained_Net/ExpandDims:0',
        'output_shape': 2048,
        'cosine_distance_eps': 0.1
        }
}

def create_model_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile( pth, 'rb') as f:
        graph_def =tf.compat.v1.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='Pretrained_Net')

def _get_model_layer(sess, model_name):
    # layername = 'Pretrained_Net/final_layer/Mean:0'
    layername = model_params[model_name]['output_layer']
    layer = sess.graph.get_tensor_by_name(layername)
    ops = layer.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return layer

def get_activations(images, sess, model_name, batch_size=1, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_model_layer(sess, model_name)
    n_images = images.shape[0]
    if batch_size > n_images:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_images
    print("the number of images:",n_images)
    n_batches = n_images//batch_size + 1
    pred_arr = np.empty((n_images,model_params[model_name]['output_shape']))
    for i in tqdm(range(n_batches)):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        if start+batch_size < n_images:
            end = start+batch_size
        else:
            end = n_images
                    
        batch = images[start:end]
        pred = sess.run(inception_layer, {model_params[model_name]['input_layer']: batch})
        pred_arr[start:end] = pred.reshape(-1,model_params[model_name]['output_shape'])
    if verbose:
        print(" done")
    return pred_arr


# def calculate_memorization_distance(features1, features2):
#     neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
#     neigh.fit(features2) 
#     d, _ = neigh.kneighbors(features1, return_distance=True)
#     print('d.shape=',d.shape)
#     return np.mean(d)

def normalize_rows(x: np.ndarray):
    """
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return np.nan_to_num(x/np.linalg.norm(x, ord=2, axis=1, keepdims=True))


def cosine_distance(features1, features2):
    features1_nozero = features1[np.sum(features1, axis=1) != 0]
    features2_nozero = features2[np.sum(features2, axis=1) != 0]
    norm_f1 = normalize_rows(features1_nozero)
    norm_f2 = normalize_rows(features2_nozero)

    d = 1.0-np.abs(np.matmul(norm_f1, norm_f2.T))
    #print('d.shape=',d.shape)
    #print('np.min(d, axis=1).shape=',np.min(d, axis=1).shape)

    #argmin_img = np.argmin(d,axis=1)
    argmin_img =  np.argsort(d,axis=1)[0,:5]
    #print("The closest image is:",argmin_img+1,".jpg")
    #min_dis = str(d[0,argmin_img])
    min_dis = str(np.sort(d,axis=1)[0,:5])
    return argmin_img,min_dis


def distance_thresholding(d, eps):
    if d < eps:
        return d
    else:
        return 1

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    print('covmean.shape=',covmean.shape)

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_activation_statistics(images, sess, model_name, batch_size=1, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, model_name, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act
    
def _handle_path_memorization(path, sess, model_name, is_checksize, is_check_png):
    path = pathlib.Path(path)
    files =list(path.glob('*.jpg')) + list(path.glob('*.png'))
    imsize = model_params[model_name]['imsize']
    # In production we don't resize input images. This is just for demo purpose.
    is_check_png = False
    x = np.array([np.array(img_read_checks(fn, imsize, is_checksize, imsize, is_check_png)) for fn in files])
    m, s, features = calculate_activation_statistics(x, sess, model_name)
    del x #clean up memory
    return m, s, features, files

# check for image size
def img_read_checks(filename, resize_to, is_checksize=False, check_imsize = 256, is_check_png = False):
    im = Image.open(str(filename))
    #im = im.convert('RGB')
    if is_checksize and im.size != (check_imsize,check_imsize):
        raise KernelEvalException('The images are not of size '+str(check_imsize))
    if is_check_png and im.format != 'JPG':
        raise KernelEvalException('Only PNG images should be submitted.')

    if resize_to is None:
        return im
    else:
        return im.resize((resize_to,resize_to),Image.ANTIALIAS)

def calculate_kid_given_paths(paths, model_name, model_path, feature_path=None,files_path=None):
    ''' Calculates the KID of two paths. '''
    tf.compat.v1.reset_default_graph()
    create_model_graph(str(model_path))
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        m1, s1, features1, files1 = _handle_path_memorization(paths[0], sess, model_name, is_checksize = True, is_check_png = True)
        if feature_path is None:
            m2, s2, features2, files2 = _handle_path_memorization(paths[1], sess, model_name, is_checksize = False, is_check_png = False)
        else:
            features2 = np.load(feature_path)
            files2 = np.load(files_path, allow_pickle=True)
            #with np.load(feature_path) as f:
                #m2, s2, features2 = f['m'], f['s'], f['features']

        #np.save(paths[1]+"/monet_features.npy", features2)
        #files2 = np.asarray(files2)
        #np.save(paths[1]+"/monet_files.npy", files2)
        #print('m1,m2 shape=',(m1.shape,m2.shape),'s1,s2=',(s1.shape,s2.shape))
        #print('starting calculating FID')
        #fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        #print('done with FID, starting distance calculation')
        distance,value = cosine_distance(features1, features2)
        #files2_arr = np.asarray(files2)
        file_name = []
        for idx in distance:
           tmp = str(files2[idx].as_posix())
           tmp = re.search(r"[^//]+$", tmp) #regex: get the last string that starts with '/' (its the file name)
           tmp = tmp.group()
           file_name.append(tmp)
        file_name = str(file_name)
        file_name = re.sub('[\[\]\']', '', file_name)
        #file_name = str(files2[distance]) #get the path as a string
        #file_name = re.search(r"(?<=')(.*)(?=')", file_name) #regex:return whatever is inbetween '*' (its the relative path))
        #file_name = file_name.group()        
        #file_name = re.search(r"[^//]+$", file_name) #regex: get the last string that starts with '/' (its the file name)
        #file_name = file_name.group()

        return file_name,files1,value
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',dest='input_dir', type=str, help='input image dir', default='./Input/Paint/')
    parser.add_argument('--input_name',dest='input_name', type=str, help='input image name', required=True)
    parser.add_argument('--out_file_name',dest='out_file_name', type=str, help='the .txt file name', default='top5_photo_monet.txt')

    args = parser.parse_args()
    user_images_unzipped_path = args.input_dir + args.input_name
    #print(user_images_unzipped_path)
    
#user_images_unzipped_path = './Input/Images/monet_jpg_names/'
    
    root_folder_path = './calc_top_matches/'
    images_path = [user_images_unzipped_path,'./model_gen/Input/monet_jpg_names/']
    public_path = root_folder_path + 'metadata/classify_image_graph_def.pb'
    feature_path = root_folder_path + 'metadata/monet_features.npy'
    files_path = root_folder_path + 'metadata/monet_files.npy'
    fid_epsilon = 10e-15

    closest_image,real_image,val  = calculate_kid_given_paths(images_path, 'Inception', public_path,feature_path,files_path)
    #print("the correct file name:",closest_image)

    f = open(root_folder_path + args.out_file_name, "a")
    tmp = str(real_image[0].as_posix())
    tmp = re.search(r"[^//]+$", tmp) #regex: get the last string that starts with '/' (its the file name)
    tmp = tmp.group()
    
    val = re.sub('[\[\]\']', '', val)
    val = re.sub(' +', ', ', val)

    line = tmp + ", "+ closest_image+", "+ val+"\n"
    line = re.sub(',', '', line)
    f.write(line)
    f.close()

    #open and read the file after the appending:
    #f = open("image_matches.txt", "r")
    #print(f.read())
    #f.close()

    #distance_public  = calculate_kid_given_paths(images_path, 'Inception', public_path)
    #distance_public  = calculate_kid_given_paths(images_path, 'Inception', public_path,feature_path,files_path)
    #distance_public = distance_thresholding(distance_public, model_params['Inception']['cosine_distance_eps'])
    #print("FID_public: ", fid_value_public, "distance_thrash: ", distance_public, "multiplied_public: ", fid_value_public /(distance_public + fid_epsilon))
