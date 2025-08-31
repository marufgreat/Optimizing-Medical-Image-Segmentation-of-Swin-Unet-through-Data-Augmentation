import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sample_file', type=str,
                    default='model_out/predictions/case0001_img.nii.gz', help='test sample to visualize')
parser.add_argument('--slice_number', type=int,
                    default=4, help='slice number')
args = parser.parse_args()


file_name = args.sample_file
print(file_name)
if "_img" in file_name:
    sample_name = file_name.replace("_img.nii.gz", "")
elif "_pred" in file_name:
    sample_name = file_name.replace("_pred.nii.gz", "")
elif "_gt" in file_name:
    sample_name = file_name.replace("_gt.nii.gz", "")

show_name = sample_name.split("/")[-1]
img_file = sample_name + "_img.nii.gz"
image = nb.load(img_file).get_fdata()
pred_file = img_file.replace("img", "pred")
prediction = nb.load(pred_file).get_fdata()
gt_file = file_name.replace("img", "gt")
gt = nb.load(gt_file).get_fdata()



slice_num = args.slice_number # visualizing 2D slice no. 4. Insert any slice number to see results for that specific slice.
fig = plt.figure(figsize=(10, 7))
fig.add_subplot(1, 3, 1) 
plt.imshow(image[:,:,slice_num], 'gray')
plt.title(show_name + " Slice " + str(slice_num))
#plt.show()
fig.add_subplot(1, 3, 2)
plt.imshow(prediction[:,:,slice_num], 'gray')
plt.title("Prediction")
#plt.show()
fig.add_subplot(1,3,3)
plt.imshow(gt[:,:,slice_num], 'gray')
plt.title("Ground Truth")
plt.show()