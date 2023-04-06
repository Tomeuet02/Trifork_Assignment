import cv2
import os
import numpy as np


def resize_imgs(read_folder_img, read_folder_kitti, nx, ny):
    print("Let's start resizing")
    # We read img's filenames
    for filename in os.listdir(read_folder_img):

        # Save path
        f = os.path.join(read_folder_img, filename)

        # Check if it is a file
        if os.path.isfile(f):

            # Read, resize and save img
            img = cv2.imread(f)
            new_img = cv2.resize(img, dsize=(nx, ny), interpolation=cv2.INTER_LINEAR)
            new_dir = os.path.join(read_folder_img + '_resized', filename)
            cv2.imwrite(new_dir, new_img)

            # Change path to the file with KITTI annotations
            base = os.path.splitext(filename)[0]
            new_name = os.path.join(read_folder_kitti, base + ".txt")
            # Read .txt as an array
            txt = np.genfromtxt(new_name, dtype='str')

            # Resize positions
            if txt.ndim == 2:
                for i in range(len(txt)):
                    txt[i, 4] = str(int(nx * float(txt[i, 4]) / img.shape[1]))
                    txt[i, 6] = str(int(nx * float(txt[i, 6]) / img.shape[1]))
                    txt[i, 5] = str(int(ny * float(txt[i, 5]) / img.shape[0]))
                    txt[i, 7] = str(int(ny * float(txt[i, 7]) / img.shape[0]))
            else:
                txt[4] = str(int(nx * float(txt[4]) / img.shape[1]))
                txt[6] = str(int(nx * float(txt[6]) / img.shape[1]))
                txt[5] = str(int(ny * float(txt[5]) / img.shape[0]))
                txt[7] = str(int(ny * float(txt[7]) / img.shape[0]))

            txt_dir = os.path.join(read_folder_kitti + '_resized', base + ".txt")
            np.savetxt(txt_dir, txt, fmt = "%s")
    print("Done :)")

resize_imgs('images', 'kitti_annotations', 284, 284)
