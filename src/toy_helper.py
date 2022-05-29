"""Helper file to run the discover concept algorithm in the toy dataset."""
# lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
# matplotlib.use('GTK3Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from skimage.segmentation import felzenszwalb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from PIL import Image
from matplotlib import cm
from helper import CNN_cls
from torch import nn
import torch
from tqdm import tqdm, trange
from concept_models import *
seed(0)
batch_size = 64


def copy_save_image(x,f1,f2,a,b):
  # open the image
  Image1 = Image.fromarray(x.astype('uint8'))
  Image1.save(f1)
  # make a copy the image so that
  # the original image does not get affected
  Image1copy = Image1.copy()
  Image1copy = Image1copy.resize((240,240), Image.ANTIALIAS)
  left = 32*b
  right = left+116
  top = 32*a
  bottom = top+116

  region = Image1copy.crop((left,top,right,bottom))
  #im.paste(region, (50, 50, 100, 100))
  old_size = (116,116)
  new_size = (118,118)
  new_im = Image.new("RGB", new_size)
  new_im.paste(region, (1,1))
  new_im.save(f2)

def stack(imgs, direction):
  print('stacking {} imgs'.format(len(imgs)))
  # idx = len(imgs)//2
  # print('idx: ', idx)
  # imgs_comb1 = np.hstack( (np.asarray(i) for i in imgs[:idx] ) )
  # imgs_comb2 = np.hstack( (np.asarray(i) for i in imgs[-idx:] ) )
  # imgs_comb = np.vstack( (np.asarray(i) for i in [imgs_comb1, imgs_comb2] ) )
  if direction == 'h':
    imgs_comb = np.hstack( (np.asarray(i) for i in imgs ) )
  elif direction == 'v':
    imgs_comb = np.vstack( (np.asarray(i) for i in imgs ) )
  return imgs_comb

def copy_image_stacked(xx, aa, bb):
  cropped_imgs = []
  new_size = (118,118)
  for i in range(len(xx)):
    x = xx[i]
    a = aa[i]
    b = bb[i]
    # open the image
    Image1 = Image.fromarray(x.astype('uint8'))
    # make a copy the image so that
    # the original image does not get affected
    Image1copy = Image1.copy()
    Image1copy = Image1copy.resize((240,240), Image.ANTIALIAS)
    left = 32*b
    right = left+116
    top = 32*a
    bottom = top+116
    region = Image1copy.crop((left,top,right,bottom))
    new_im = Image.new("RGB", new_size)
    new_im.paste(region, (1,1))
    cropped_imgs.append(new_im)
  imgs_comb = stack(cropped_imgs, 'h')
  # save that beautiful picture
  imgs_comb = Image.fromarray( imgs_comb)
  return imgs_comb
  

def copy_save_image_all(x,f1,f2,a,b):

  # open the image
  Image1 = Image.fromarray(x.astype('uint8'))
  old_size = (240,240)
  new_size = (244,244)
  new_im = Image.new("RGB", new_size)
  new_im.paste(Image1, (2,2))
  new_im.save(f2)
  '''
  Image1.save(f1)
  # make a copy the image so that
  # the original image does not get affected
  Image1copy = Image1.copy()
  Image1copy = Image1copy.resize((240,240), Image.ANTIALIAS)
  left = 32*b
  right = left+116
  top = 32*a
  bottom = top+116

  region = Image1copy.crop((left,top,right,bottom))
  #im.paste(region, (50, 50, 100, 100))
  old_size = (116,116)
  new_size = (118,118)
  new_im = Image.new("RGB", new_size)
  new_im.paste(region, (1,1))
  '''
  #Image1.save(f2)

def load_xyconcept(n, pretrain):
  """Loads data and create label for toy dataset."""
  concept = np.load('concept_data.npy')[:n]
  y = np.zeros((n, 15))
  y[:, 0] = ((1 - concept[:, 0] * concept[:, 2]) + concept[:, 3]) > 0
  y[:, 1] = concept[:, 1] + (concept[:, 2] * concept[:, 3])
  y[:, 2] = (concept[:, 3] * concept[:, 4]) + (concept[:, 1] * concept[:, 2])
  y[:, 3] = np.bitwise_xor(concept[:, 0], concept[:, 1])
  y[:, 4] = concept[:, 1] + concept[:, 4]
  y[:, 5] = (1 - (concept[:, 0] + concept[:, 3] + concept[:, 4])) > 0
  y[:, 6] = np.bitwise_xor(concept[:, 1] * concept[:, 2], concept[:, 4])
  y[:, 7] = concept[:, 0] * concept[:, 4] + concept[:, 1]
  y[:, 8] = concept[:, 2]
  y[:, 9] = np.bitwise_xor(concept[:, 0] + concept[:, 1], concept[:, 3])
  y[:, 10] = (1 - (concept[:, 2] + concept[:, 4])) > 0
  y[:, 11] = concept[:, 0] + concept[:, 3] + concept[:, 4]
  y[:, 12] = np.bitwise_xor(concept[:, 1], concept[:, 2])
  y[:, 13] = (1 - (concept[:, 0] * concept[:, 4] + concept[:, 3])) > 0
  y[:, 14] = np.bitwise_xor(concept[:, 4], concept[:, 3])
  if not pretrain:
    x = np.load('x_data.npy') / 255.0
    return x, y, concept
  return 0, y, concept


def load_toy_data(args):
  if (not args.pretrained) or args.do_inference or args.visualize or args.eval_causal_effect:
    x, y, concept = load_xyconcept(args.n, False)
    print('Data Loaded')
    x = x.swapaxes(1, 3)
    print('x.shape: ', x.shape) 
    x_train = x[:args.n0, :, :, :]
    x_val = x[args.n0:, :, :, :]
  else:
    x, y, concept = load_xyconcept(args.n, True)
    x_train = 0
    x_val = 0
  y_train = y[:args.n0, :]
  y_val = y[args.n0:, :]
  return (x_train, y_train), (x_val, y_val)

def target_category_loss(x, category_index, nb_classes):
  return x * K.one_hot([category_index], nb_classes)


def load_model_stm_new(train_loader, valid_loader, device, epochs, save_folder, width=240, \
               height=240, channel=3, pretrain=True):
  """Loads pretrain model or train one."""
  model = CNN_cls(width, height, channel)
  save_dir = save_folder + 'cnn_cls_toy.pkl'

  if pretrain == False:
    print(model)
    print('{} parameters to train'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model.to(device)
    # raise Exception('end')
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    batch_history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }
    epoch_history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    for i in trange(epochs, unit="epoch", desc="Train"):
        model.train()
        with tqdm(train_loader, desc="Train") as tbatch:
          for i, (samples, targets) in enumerate(tbatch):
            optimizer.zero_grad()
            # model.zero_grad()
            predictions, _ = model(samples.to(device=device, dtype=torch.float))
            loss = criterion(predictions.squeeze(), targets.to(device=device, dtype=torch.float))
            acc = (predictions.round().squeeze() == targets.to(device=device, dtype=torch.float)).sum().item()/ (predictions.size(0)*predictions.size(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            batch_history["loss"].append(loss.item())
            batch_history["accuracy"].append(acc)
        epoch_history["loss"].append(np.mean(batch_history["loss"]))
        epoch_history["accuracy"].append(np.mean(batch_history["accuracy"]))
        model.eval()
        print("Validation...")
        with torch.no_grad():                  
          # validation loop
          with tqdm(valid_loader, desc="valid") as tbatch:
            for i, (samples, targets) in enumerate(tbatch):
              predictions, _ = model(samples.to(device=device, dtype=torch.float))
              loss = criterion(predictions.squeeze(), targets.to(device=device, dtype=torch.float))
              acc = (predictions.round().squeeze() == targets.to(device=device, dtype=torch.float)).sum().item() / (predictions.size(0)*predictions.size(1))
              batch_history["val_loss"].append(loss.item())
              batch_history["val_accuracy"].append(acc)
        epoch_history["val_loss"].append(np.mean(batch_history["val_loss"]))
        epoch_history["val_accuracy"].append(np.mean(batch_history["val_accuracy"]))
    torch.save(model, save_dir)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(epoch_history['loss'], label = 'train')
    ax[0].plot(epoch_history['val_loss'], label = 'val')
    ax[0].set_title('loss')
    ax[1].plot(epoch_history['accuracy'], label = 'train')
    ax[1].plot(epoch_history['val_accuracy'], label = 'val')
    ax[1].set_title('accuracy')
    plt.legend()
    plt.savefig(save_folder + 'cnn_cls_training_toy.png')
  else:
    model = torch.load(save_dir)
    model = model.to(device)    
  return model


def load_cls_model(args, device, data):
  # CLASSIFICATION MODEL
  (x_train, y_train), (x_val, y_val) = data
  if not args.pretrained:
      train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
      valid_data = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
      # shuffles to have a better pretraining
      train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, drop_last=False)
      valid_loader = DataLoader(valid_data, shuffle=True, batch_size=args.batch_size, drop_last=False)
      # trains model
      print(device)
      print('training prediction model')
      torch.cuda.empty_cache()
      model = load_model_stm_new(
          train_loader, valid_loader, device, args.epochs, args.save_dir, pretrain=args.pretrained)
  else:  
      print(device)
      # loads model
      torch.cuda.empty_cache()
      print('loading prediction model')
      model = load_model_stm_new(
          None, None, device, args.epochs, args.save_dir, pretrain=args.pretrained)
  return model

def get_pca_concept(f_train):
  pca = PCA()
  pca.fit(f_train)
  weight_pca = np.zeros((64, 15))
  for count, pc in enumerate(pca.components_):
    if count>14:
      break
    weight_pca[:, count] = pc
  return weight_pca

def get_kmeans_concept(f_train, n_concept):
  kmeans = KMeans(n_clusters=n_concept, random_state=0).fit(f_train)
  weight_cluster = kmeans.cluster_centers_.T
  return weight_cluster

def create_dependent_concept(original, p1):
  #p1: probablity of selecting 1 when original==True
  #1-p1: probablity of selecting 1 when original==False
  new = []
  for i in original:
    if i==True:
      new.append(np.random.choice([False, True], size = 1, p = [1-p1, p1]))
    elif i==False:
      new.append(np.random.choice([False, True], size = 1, p = [p1, 1-p1]))
    else:
      raise Exception('Encountered {}'.format(i))
  return np.array(new)

def create_dataset(n_sample, cov, p, return_directly = False, concept = False):
  """Creates toy dataset and save to disk."""
  if isinstance(concept, bool):
    if cov == False:
      # concept = np.reshape(np.random.randint(2, size=15 * n_sample),
      #                     (-1, 15)).astype(np.bool_)
      concept = np.reshape(np.random.choice(2, size=15 * n_sample, p = [0.7, 0.3]),
                          (-1, 15)).astype(np.bool_)
    else:
      # concept = np.reshape(np.random.randint(2, size=15 * n_sample),
      #                     (-1, 15)).astype(np.bool_)
      # print(concept.shape)
      concept = np.reshape(np.random.choice(2, size=15 * n_sample, p = [0.7, 0.3]),
                          (-1, 15)).astype(np.bool_)
      # print('concept shape: ', concept.shape)
      # raise Exception('end')
      i = 10
      concept[:, i] = create_dependent_concept(concept[:, i], p).reshape((-1,))
      for i in range(5):
        #change here for higher covariance
        concept[:, i+5] = create_dependent_concept(concept[:, i], p).reshape((-1,))
      # for i in range(5):
      #   concept[:, i+10] = create_dependent_concept(concept[:, i+5], p).reshape((-1,))
      print('Created co-dependent concepts')
    concept[:15, :15] = np.eye(15)
  
  fig = Figure(figsize=(2.4, 2.4))
  canvas = FigureCanvas(fig)
  axes = fig.gca()
  axes.set_xlim([0, 10])
  axes.set_ylim([0, 10])
  axes.axis('off')
  width, height = fig.get_size_inches() * fig.get_dpi()
  width = int(width)
  height = int(height)
  print(width)
  location = [(1.3, 1.3), (3.3, 1.3), (5.3, 1.3), (7.3, 1.3), (9.3, 1.3),
              (1.3, 3.3), (3.3, 3.3), (5.3, 3.3), (7.3, 2.3), (9.3, 3.3),
              (1.3, 5.3), (3.3, 5.3), (5.3, 5.3), (7.3, 5.3), (9.3, 5.3),
              (1.3, 7.3), (3.3, 7.3), (5.3, 7.3), (7.3, 7.3), (9.3, 7.3),
              (1.3, 9.3), (3.3, 9.3), (5.3, 9.3), (7.3, 9.3), (9.3, 9.3)]
  location_bool = np.zeros(25)
  x = np.zeros((n_sample, width, height, 3))
  color_array = ['green', 'red', 'blue', 'black', 'orange', 'purple', 'yellow']

  for i in range(n_sample):
    location_bool = np.zeros(25)
    if i % 1000 == 0:
      print('{} images are created'.format(i))
    if concept[i, 5] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'x',
          color=color_array[np.random.randint(100) % 7],
          markersize=10,
          mew=4)
    if concept[i, 6] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '3',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 7] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          's',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 8] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'p',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,)
    if concept[i, 9] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '_',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 10] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'd',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 11] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'd',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 12] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          11,
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 13] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'o',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 14] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '.',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 0] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '+',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 1] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '1',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 2] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '*',
          color=color_array[np.random.randint(100) % 7],
          markersize=30,
          mew=3)
    if concept[i, 3] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '<',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 4] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'h',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    canvas.draw()
    image = np.fromstring(
        canvas.tostring_rgb(), dtype='uint8').reshape(width, height, 3)

    x[i, :, :, :] = image
    fig = Figure(figsize=(2.4, 2.4))
    canvas = FigureCanvas(fig)
    axes = fig.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 10])
    axes.axis('off')
    # imgplot = plt.imshow(image)
    # plt.show()

  # create label by booling functions
  y = np.zeros((n_sample, 15))
  y[:, 0] = ((1 - concept[:, 0] * concept[:, 2]) + concept[:, 3]) > 0
  y[:, 1] = concept[:, 1] + (concept[:, 2] * concept[:, 3])
  y[:, 2] = (concept[:, 3] * concept[:, 4]) + (concept[:, 1] * concept[:, 2])
  y[:, 3] = np.bitwise_xor(concept[:, 0], concept[:, 1])
  y[:, 4] = concept[:, 1] + concept[:, 4]
  y[:, 5] = (1 - (concept[:, 0] + concept[:, 3] + concept[:, 4])) > 0
  y[:, 6] = np.bitwise_xor(concept[:, 1] * concept[:, 2], concept[:, 4])
  y[:, 7] = concept[:, 0] * concept[:, 4] + concept[:, 1]
  y[:, 8] = concept[:, 2]
  y[:, 9] = np.bitwise_xor(concept[:, 0] + concept[:, 1], concept[:, 3])
  y[:, 10] = (1 - (concept[:, 2] + concept[:, 4])) > 0
  y[:, 11] = concept[:, 0] + concept[:, 3] + concept[:, 4]
  y[:, 12] = np.bitwise_xor(concept[:, 1], concept[:, 2])
  y[:, 13] = (1 - (concept[:, 0] * concept[:, 4] + concept[:, 3])) > 0
  y[:, 14] = np.bitwise_xor(concept[:, 4], concept[:, 3])

  if return_directly:
    return x, y
  else:
    np.save('x_data.npy', x)
    np.save('y_data.npy', y)
    np.save('concept_data.npy', concept)
    return width, height


def get_groupacc(min_weight, f_train, f_val, concept,
                 n_concept, n_cluster, n0, verbose):
  """Gets the group accuracy for dicovered concepts."""
  #print(finetuned_model_pr.summary())
  #min_weight = finetuned_model_pr.layers[-5].get_weights()[0]
  
  loss_table = np.zeros((n_concept, 5))
  f_time_weight = np.matmul(f_train[:1000,:], min_weight)
  f_time_weight_val = np.matmul(f_val, min_weight)
  #f_time_weight[np.where(f_time_weight<0.8)]=0
  #f_time_weight_val[np.where(f_time_weight_val<0.8)]=0
  f_time_weight_m = np.max(f_time_weight,(1,2))
  f_time_weight_val_m = np.max(f_time_weight_val,(1,2))
  for count in range(n_concept):
    for count2 in range(5):

      #print('count 2 is {}'.format(count2))
      # count2 = max_cluster[count]
      mean0 = np.mean(
          f_time_weight_m[:,count][concept[:1000,count2] == 0]) * 100
      mean1 = np.mean(
          f_time_weight_m[:,count][concept[:1000,count2] == 1]) * 100

      if mean0 < mean1:
        pos = 1
      else:
        pos = -1
      best_err = 1e10
      best_bias = 0
      a = int((mean1 - mean0) / 10)
      if a == 0:
        a = pos

      
      for bias in range(int(mean0), int(mean1), a):
        if pos == 1:
          if np.sum(
              np.bitwise_xor(
                  concept[:1000, count2],
                  f_time_weight_m[:,count] >
                  bias / 100.)) < best_err:
            best_err = np.sum(
                np.bitwise_xor(
                    concept[:1000, count2],
                    f_time_weight_m[:,count] > bias / 100.))
            best_bias = bias
        else:
          if np.sum(
              np.bitwise_xor(
                  concept[:1000, count2],
                  f_time_weight_m[:,count] <
                  bias / 100.)) < best_err:
            best_err = np.sum(
                np.bitwise_xor(
                    concept[:1000, count2],
                    f_time_weight_m[:,count] < bias / 100.))
            best_bias = bias
      if pos == 1:
        loss_table[count, count2] = np.sum(
            np.bitwise_xor(
                concept[n0:, count2],
                f_time_weight_val_m[:,count] >
                best_bias / 100.)) / 12000
        if verbose:
          print(np.sum(
              np.bitwise_xor(
                  concept[n0:, count2],
                  f_time_weight_val_m[:,count] > best_bias / 100.))
                /12000)
      else:
        loss_table[count, count2] = np.sum(
            np.bitwise_xor(
                concept[n0:, count2],
                f_time_weight_val_m[:,count] <
                best_bias / 100.)) / 12000
        if verbose:
          print(np.sum(
              np.bitwise_xor(
                  concept[n0:, count2],
                  f_time_weight_val_m[:,count] < best_bias / 100.))
                /12000)
  print(np.amin(loss_table, axis=0))
  acc = np.mean(np.amin(loss_table, axis=0))
  print(acc)
  return acc

def get_groupacc_max(min_weight, f_train, f_val, concept,
                 n_concept, n_cluster, n0, verbose):
  """Gets the group accuracy for dicovered concepts."""
  #print(finetuned_model_pr.summary())
  #min_weight = finetuned_model_pr.layers[-5].get_weights()[0]

  loss_table = np.zeros((n_concept, 5))
  for count in range(n_concept):
    for count2 in range(5):
      #print('count 2 is {}'.format(count2))
      # count2 = max_cluster[count]
      #similarity bt x_train and topic count when concept count2 == 0 
      mean0 = np.mean(
          np.max(np.matmul(f_train[:1000,:], min_weight[:, count]),(1,2))[concept[:1000,count2] == 0]) * 100
      #similarity bt x_train and topic count when concept count2 == 1
      mean1 = np.mean(
          np.max(np.matmul(f_train[:1000,:], min_weight[:, count]),(1,2))[concept[:1000,count2] == 1]) * 100

      if mean0 < mean1:
        pos = 1
      else:
        pos = -1
      best_err = 1e10
      best_bias = 0
      a = int((mean1 - mean0) / 10)
      if a == 0:
        a = pos
      for bias in range(int(mean0), int(mean1), a):
        if pos == 1:
          if np.sum(
              np.bitwise_xor(
                  concept[:1000, count2],
                  np.max(np.matmul(f_train[:1000,:], min_weight[:, count]),(1,2)) >
                  bias / 100.)) < best_err:
            best_err = np.sum(
                np.bitwise_xor(
                    concept[:1000, count2],
                    np.max(np.matmul(f_train[:1000,:], min_weight[:, count]),(1,2)) >
                    bias / 100.))
            best_bias = bias
        else:
          if np.sum(
              np.bitwise_xor(
                  concept[:1000, count2],
                  np.max(np.matmul(f_train[:1000,:], min_weight[:, count]),(1,2)) <
                  bias / 100.)) < best_err:
            best_err = np.sum(
                np.bitwise_xor(
                    concept[:1000, count2],
                    np.max(np.matmul(f_train[:1000,:], min_weight[:, count]),(1,2)) <
                    bias / 100.))
            best_bias = bias
      if pos == 1:
        ans = np.sum(
            np.bitwise_xor(
                concept[n0:, count2],
                np.max(np.matmul(f_val, min_weight[:, count]),(1,2)) >
                best_bias / 100.)) / 12000
        loss_table[count, count2] = ans
        if verbose:
          print(ans)
      else:
        ans = np.sum(
            np.bitwise_xor(
                concept[n0:, count2],
                np.max(np.matmul(f_val, min_weight[:, count]),(1,2)) <
                best_bias / 100.)) / 12000
        loss_table[count, count2] = ans
        if verbose:
          print(ans)
  print(np.amin(loss_table, axis=0))
  acc = np.mean(np.amin(loss_table, axis=0))
  print(acc)
  return acc

def visualize_model(x, f_train, topic_vec, graph_save_folder, n_concept, method, topic_model = None, device = None):
  x_new = x.swapaxes(1, 3)
  if method == 'BCVAE':
    # some way to get probabilities size, 4, 4, n_concept
    shape = f_train.shape
    f_train = f_train.swapaxes(1, 3).flatten(start_dim = 0, end_dim = -2)
    zs, params = topic_model.encode(f_train.to(device))
    topic_prob = params.select(-1, 0).view(shape[0], shape[2], shape[3], -1).detach().cpu()
  else:
    print('x_new.shape: ', x_new.shape)
    # VISUALIZE THE NEAREST NEIGHBORS
    f_train_n = f_train[:10000]/(np.linalg.norm(f_train[:10000],axis=3,keepdims=True)+1e-9)
    f_train_n = f_train_n.swapaxes(1, 3)
    print('f_train_n.shape: ', f_train_n.shape) #100, 4, 4, 64
    topic_vec_n = topic_vec/(np.linalg.norm(topic_vec,axis=0,keepdims=True)+1e-9)
    print('topic_vec_n.shape: ', topic_vec_n.shape) #64, 5
    topic_prob = np.dot(f_train_n,topic_vec_n) #100, 4, 4, 5
    print('topic_prob.shape: ', topic_prob.shape)
    raise Exception('end')
  n_size = 4 #the final size
  imgs = []
  for i in range(n_concept):
      ind = np.argpartition(topic_prob[:,:,:,i].flatten(), -10)[-10:]
      # sim_list = topic_prob[:,:,:,i].flatten()[ind]
      xx = []
      aa = []
      bb = []
      for jc,j in enumerate(ind):
          j_int = int(np.floor(j/(n_size*n_size)))
          a = int((j-j_int*(n_size*n_size))/n_size)
          b = int((j-j_int*(n_size*n_size))%n_size)
          #if sim_list[jc]>0.95:
          xx.append(x_new[j_int,:,:,:])
          # print(xx)
          aa.append(a)
          bb.append(b)
          # toy_helper_v2.copy_save_image(x[j_int,:,:,:],f1,f2,a,b)
      # f = graph_save_folder + 'concept_{}.png'.format(i)
      # toy_helper_v2.copy_save_image_stacked(xx, f, aa, bb)
      imgs.append(copy_image_stacked(xx, aa, bb))
  img = stack(imgs, 'v')*255
  
  img = Image.fromarray(img)
  f = graph_save_folder + 'pytorch_concepts_{}.png'.format(n_concept)
  img.save(f)