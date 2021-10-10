def display_image(images:list, display=True, save=False, name=None):
  import cv2
  import numpy as np
  from matplotlib import pyplot as plt

  img1, img2, img3, *_ = images
    
    
  img1 = np.array(img1).astype(np.float32)
  img2 = np.array(img2).astype(np.float32) 
  img3 = np.array(img3).astype(np.float32) 
    

  img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
  img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)
  print(img1.shape, img2.shape, img3.shape)

  im_h = cv2.hconcat([img1, img2, img3])
  
  im_h = tf.nn.relu(im_h).numpy()
  im_h = np.clip(im_h, 0, 1)
    
  print(np.max(im_h))   
  print(np.min(im_h))

  plt.xticks([])
  plt.yticks([])

  if display:
    plt.imshow(im_h)

  if save:
    if name is not None:
      plt.imsave(name, im_h.astype(np.float32))
    else:
      raise AttributeError('plt.imsave expected to have a name to save the image')

  return im_h
