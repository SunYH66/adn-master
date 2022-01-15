# --coding:utf-8--
import glob
import os.path
import ants
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

def prepare_timepoint_data():
    testA_path = sorted(glob.glob('/home/user/program/adn-master/runs/nature_image/testA_visuals_P1/*de_enh.nii.gz'))
    testB_path = sorted(glob.glob('/home/user/program/adn-master/runs/nature_image/testB_visuals_P1/*de_enh.nii.gz'))

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
                        '../../data/run_registration_P1/de_enh_t{:0>2d}.nii.gz'.format(i + 1))


def delete_slices():
    file_path = sorted(glob.glob('/home/user/program/voxelmorph-legacy/dataset/DCE_MRI_train/16/original_Img_t*.nii.gz'))
    img_small = np.zeros((320, 320, 22))
    for i in range(len(file_path)):
        img = sitk.GetArrayFromImage(sitk.ReadImage(file_path[i]))
        img_small = img[5:27, :, :]
        sitk.WriteImage(sitk.GetImageFromArray(img_small),
                        '/home/user/program/voxelmorph-legacy/dataset/DCE_MRI_train/16/ori_small_Img_t{:0>2d}.nii.gz'.format(i + 1))


def register_de_enhanced_img():
    fix_img = ants.image_read(os.path.join('../../data/run_registration_P2', 'de_enh_cycle_full_t11.nii.gz'))
    fix_img_apply = ants.image_read('/home/user/program/voxelmorph-legacy/dataset/DCE_MRI_train/16/original_Img_t341.nii.gz')
    for i in range(1, 23):
        if i == 11:
            ants.image_write(fix_img, os.path.join('../../data/run_registration_P2', 'warped_de_enh_cycle_t{:0>2d}.nii.gz'.format(i)))
            ants.image_write(fix_img_apply, os.path.join('../../data/run_registration_P2', 'warped_img_cycle_t{:0>2d}.nii.gz'.format(i)))
        else:
            mov_img = ants.image_read(os.path.join('../../data/run_registration_P2', 'de_enh_cycle_full_t{:0>2d}.nii.gz'.format(i)))
            mov_img_apply = ants.image_read('/home/user/program/voxelmorph-legacy/dataset/DCE_MRI_train/16/original_Img_t{:0>2d}.nii.gz'.format(i+330))

            result = ants.registration(fixed=fix_img, moving=mov_img, type_of_transform='SyN')
            # warped_img = result['warpedmovout']
            transform = result['fwdtransforms']
            warped_img_ori = ants.apply_transforms(fixed=fix_img_apply, moving=mov_img_apply, transformlist=transform)
            ants.image_write(warped_img_ori, os.path.join('../../data/run_registration_P2', 'warped_img_cycle_t{:0>2d}.nii.gz'.format(i)))
            # ants.image_write(warped_img, os.path.join('../../data/run_registration_P2', 'warped_de_enh_cycle_t{:0>2d}.nii.gz'.format(i)))


def ants_goupwise():
    file_paths = sorted(glob.glob('/home/user/program/adn-master/data/run_registration_P2/de_enh_full_t*.nii.gz'))
    a = sitk.GetArrayFromImage(sitk.ReadImage(file_paths[0])).transpose((2, 1, 0))
    print(a.shape)
    temp = np.zeros(a.shape)

    for i in range(len(file_paths)):
        temp += sitk.GetArrayFromImage(sitk.ReadImage(file_paths[i])).transpose((2, 1, 0))
    temp /= len(file_paths)

    for i in range(len(file_paths)):
        print(i)
        mov_Img = ants.image_read(file_paths[i])
        print(mov_Img.shape)
        fix_Img = ants.from_numpy(temp)
        moving_apply = ants.image_read('/home/user/program/voxelmorph-legacy/dataset/DCE_MRI_train/16/original_Img_t{:0>3d}.nii.gz'.format(i+331))
        results = ants.registration(moving=mov_Img, fixed=fix_Img, type_of_transform='SyN')
        warp = results['warpedmovout']
        transform = results['fwdtransforms']
        warped_img_ori = ants.apply_transforms(fixed=fix_Img, moving=moving_apply, transformlist=transform)
        # ants.image_write(warp, '/home/user/program/adn-master/data/run_registration_P2/warped_de_enh_ants_group_t%02d.nii.gz' %(i + 1))
        ants.image_write(warped_img_ori,
                         '/home/user/program/adn-master/data/run_registration_P2/warped_img_ants_group_t%02d.nii.gz' % (i + 1))


def generate_slice():
    file_path = sorted(glob.glob('/home/user/program/adn-master/data/run_registration_P2/warped_de_enh_ants_group_t*.nii.gz'))

    data_4D = np.zeros((320, 320, 32, 22))
    for i in range(len(file_path)):
        data = sitk.GetArrayFromImage(sitk.ReadImage(file_path[i])).transpose((2, 1, 0))
        print(data.shape)
        data_4D[:, :, :, i] = data
    sitk.WriteImage(sitk.GetImageFromArray(data_4D[:, :, 17, :].transpose((2, 1, 0))),
                    '/home/user/program/adn-master/data/run_registration_P2/warped_de_enh_ants_group_s{:0>2d}.nii.gz'.format(12))


def generate_slice2():
    data_4D = np.zeros((320, 320, 32, 22))
    for i in range(22):
        file_path = '/home/user/program/ants_test/MYde_enh_full_t%02ddeformed.nii.gz' % (i + 1)
        data = sitk.GetArrayFromImage(sitk.ReadImage(file_path)).transpose((2, 1, 0))
        data_4D[:, :, :, i] = data
    sitk.WriteImage(sitk.GetImageFromArray(data_4D[:, :, 12, :].transpose((2, 1, 0))),
                    '/home/user/program/ants_test/warped_de_enh_group_s{:0>2d}.nii.gz'.format(7))


def add_zero_slices():
    """Add zero slices to the head and bottom of the voxel."""
    file_path = sorted(glob.glob('/home/user/program/adn-master/data/run_registration_P1/de_enh_t*.nii.gz'))
    print(len(file_path))

    data_temp = np.zeros((320, 320, 32))
    for i in range(len(file_path)):
        img = sitk.GetArrayFromImage(sitk.ReadImage(file_path[i])).transpose((2, 1, 0))
        data_temp[:, :, 5:27] = img
        sitk.WriteImage(sitk.GetImageFromArray(data_temp.transpose((2, 1, 0))),
                        '/home/user/program/adn-master/data/run_registration_P1/de_enh_full_t%02d.nii.gz' % (i + 1))


def simpleITK_groupwise_reg():
    """Groupwise registration using SimpleITK package."""
    population = sorted(glob.glob('/home/user/program/adn-master/data/run_registration/de_enh_t*.nii.gz'))
    vectorOfImages = sitk.VectorOfImage()

    for filename in population:
        vectorOfImages.push_back(sitk.ReadImage(filename))

    image = sitk.JoinSeries(vectorOfImages)

    # Register
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(image)
    elastixImageFilter.SetMovingImage(image)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('groupwise'))
    resultImage = elastixImageFilter.Execute()
    sitk.WriteImage(resultImage, '/home/user/program/adn-master/data/run_registration/elastix_group_reg.nii.gz')


def reshape_image():
    """Reshape the image"""
    reg_imgs = sitk.GetArrayFromImage(sitk.ReadImage('/home/user/program/adn-master/data/run_registration/elastix_group_reg.nii.gz'))
    warped_de_enh_elastix = reg_imgs[:, :, 12, :]
    sitk.WriteImage(sitk.GetImageFromArray(warped_de_enh_elastix), '/home/user/program/adn-master/data/run_registration/warped_de_enh_elastix_s%02d.nii.gz' % 12)
    # for i in range(reg_imgs.shape[-1]):


def change_spacing():
    """"""
    file_paths = sorted(glob.glob('/home/user/program/voxelmorph-legacy/dataset/DCE_MRI_train/15/original_Img_t*.nii.gz'))

    for i in range(len(file_paths)):
        origin_Img = sitk.ReadImage(file_paths[i])
        origin_Img.SetSpacing((1, 1, 1))
        sitk.WriteImage(origin_Img,
                        '/home/user/program/voxelmorph-legacy/dataset/DCE_MRI_train/15/original_Img_spacing_t%02d.nii.gz' %(i + 1))

def save_DVFs():
    DVF_path = '/home/user/program/pytorch-CycleGAN-and-pix2pix-master/runs/try/train_visuals/pairwise_def_t013.nii.gz'
    dvf = sitk.GetArrayFromImage(sitk.ReadImage(DVF_path)).transpose((1,2,3,0))

    # dvf_slice = dvf[:, :, :, 12].transpose((1,0,2))
    sitk.WriteImage(sitk.GetImageFromArray(dvf), '/home/user/program/pytorch-CycleGAN-and-pix2pix-master/runs/try/train_visuals/'
                                                       'pairwise_def_t013_slice07.nii')


if __name__ == '__main__':
    save_DVFs()
