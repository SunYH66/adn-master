# --coding:utf-8--
import glob
import os.path
import ants
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

def prepare_timepoint_data():
    testA_path = sorted(glob.glob('/home/user/program/adn-master/data/test/testA/*.nii.gz'))
    testB_path = sorted(glob.glob('/home/user/program/adn-master/runs/nature_image/testB_visuals/*_de_enh.nii.gz'))

    data_4D_A = np.zeros((320, 320, 22, 6))
    for i in range(len(testA_path)):
        data_slice = i // 6
        data_time = i % 6
        print('data_slice:', data_slice)
        print('data_time:', data_time)
        data = sitk.GetArrayFromImage(sitk.ReadImage(testA_path[i]))
        data_4D_A[:, :, data_slice, data_time] = data


    data_4D_B = np.zeros((320, 320, 22, 16))
    for i in range(len(testB_path)):
        data_slice = i // 16
        data_time = i % 16
        print('data_slice:', data_slice)
        print('data_time:', data_time)
        data = sitk.GetArrayFromImage(sitk.ReadImage(testB_path[i]))
        data_4D_B[:, :, data_slice, data_time] = data

    data_4D = np.concatenate((data_4D_A, data_4D_B), axis=3)

    for i in range(22):
        sitk.WriteImage(sitk.GetImageFromArray(data_4D[..., i].transpose((2,0,1))),
                        '../../data/run_registration/de_enh_t{:0>2d}.nii.gz'.format(i + 1))


def register_de_enhanced_img():
    fix_img = ants.image_read(os.path.join('../../data/run_registration', 'de_enh_t11.nii.gz'))
    fix_img_apply = ants.image_read('/home/user/program/voxelmorph-legacy/original_Img/patient 2/original_Img_t11.nii')
    for i in range(1, 23):
        if i == 11:
            ants.image_write(fix_img, os.path.join('../../data/run_registration', 'warped_de_enh_t{:0>2d}.nii.gz'.format(i)))
        else:
            mov_img = ants.image_read(os.path.join('../../data/run_registration', 'de_enh_t{:0>2d}.nii.gz'.format(i)))
            mov_img_apply = ants.image_read('/home/user/program/voxelmorph-legacy/original_Img/patient 2/original_Img_t{:0>2d}.nii'.format(i))
            result = ants.registration(fixed=fix_img, moving=mov_img, type_of_transform='SyN')
            warped_img = result['warpedmovout']
            # transform = result['fwdtransforms']
            # warped_img = ants.apply_transforms(fixed=fix_img_apply, moving=mov_img_apply, transformlist=transform)
            ants.image_write(warped_img, os.path.join('../../data/run_registration', 'warped_de_enh_t{:0>2d}.nii.gz'.format(i)))


def generate_slice():
    file_path = sorted(glob.glob('/home/user/program/adn-master/data/run_registration/warped_*t*.nii.gz'))

    data_4D = np.zeros((320, 320, 22, 22))
    for i in range(len(file_path)):
        data = sitk.GetArrayFromImage(sitk.ReadImage(file_path[i])).transpose((2, 1, 0))
        data_4D[:, :, :, i] = data
    sitk.WriteImage(sitk.GetImageFromArray(data_4D[:, :, 12, :].transpose((2, 1, 0))),
                    '/home/user/program/adn-master/data/run_registration/warped_de_enh_s{:0>2d}.nii.gz'.format(12))


if __name__ == '__main__':
    generate_slice()
