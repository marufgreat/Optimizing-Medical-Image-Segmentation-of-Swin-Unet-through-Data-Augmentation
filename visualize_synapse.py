import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
import glob

files = glob.glob("model_out/predictions/*")
for file_name in files:
  if "case" in file_name and "_img" in file_name:
    print(file_name)
    image = nb.load(file_name).get_fdata()
    pred_file = file_name.replace("img", "pred")
    prediction = nb.load(pred_file).get_fdata()
    gt_file = file_name.replace("img", "gt")
    gt = nb.load(gt_file).get_fdata()
    break


slice_num = 4 # visualizing 2D slice no. 4. Insert any slice number to see results for that specific slice.
plt.imshow(image[:,:,slice_num], 'gray')
plt.show()
plt.imshow(prediction[:,:,slice_num], 'gray')
plt.show()
plt.imshow(gt[:,:,slice_num], 'gray')
plt.show()