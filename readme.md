# Cross-dimension Affinity Distillation for 3D EM Neuron Segmentation
**Accepted by CVPR-2024**

Xiaoyu Liu, Miaomiao Cai, Yinda Chen, Yueyi Zhang, Te Shi, Ruobing Zhang, Xuejin Chen, Zhiwei Xiong* 

University of Science and Technology of China

*Corresponding Author


## Enviromenent

This code was tested with Pytorch 1.0.1 (later versions may work), CUDA 9.0, Python 3.7.4 and Ubuntu 16.04. 

If you have a [Docker](https://www.docker.com/) environment, we strongly recommend you to pull our image as follows：

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/em_seg/v54_higra:9.1
```

## Wafer4 Dataset
We have created the Wafer4 dataset, a novel electron microscopy (EM) dataset for neuronal segmentation, sourced from the mouse olfactory cortex region. Utilizing Multi-Beam-SEM technology, the dataset offers ultra-high-resolution imaging at 8x8x35 nanometers, showcasing intricate details of neuronal structures. The Wafer4 dataset comprises 125x1250x1250 voxels, providing finely annotated voxel-level instance segmentation for over 190 million voxels, facilitating precise segmentation of neuronal instances and aiding in detailed analysis of neuronal morphology and connectivity patterns. The dataset is divided into 100 sections for training and 25 sections for testing. Furthermore, the uniqueness of the Wafer4 dataset lies in its focus on the allocation cortex region, occupying a relatively small area (10%) of the brain cortex. By addressing the existing data gap concerning the allocation cortex, the Wafer4 dataset offers insights into this specific cortical region. Situated in layer VI of the cortex, the Wafer4 dataset captures deeper neuronal structures, providing a unique perspective on cortical organization and function.

We release the [Wafer4](https://drive.google.com/drive/folders/1QsMc71wWDozitktVDXSvZtu5OEP2JT5y?usp=drive_link) dataset in Google Driver.

**MEC Wafer4 is licensed under a CC-BY-NC 4.0 International [License](https://creativecommons.org/licenses/by-nc/4.0/legalcode)**. 


## Training stage

Take the training on the wafer4 dataset as an example.

```shell
cd scripts_2_5d_3d
python main_CAD.py --cfg=2d_wafer4_bs16_ps256_loss0.0_slice1.0_cross1.0_interaction1.0 --cfg_3d=3d_wafer4_h160_noft_lr_ratio1_ft10000
```

## Inference stage
We provide the trained models in the folder 'ckpt'

Take the testing on the wafer4 dataset as an example.
```shell
cd scripts_2_5d_3d
python inference_affs.py --cfg=2d_wafer4_bs16_ps256_loss0.0_slice1.0_cross1.0_interaction1.0 --model_name=wafer4 --mode=wafer4 --test_split=25
```


## Contact

If you have any problem with the released code and dataset, please contact me by email (liuxyu@mail.ustc.edu.cn).

## Citation
```shell
@inproceedings{liu2024cad,
  title={Cross-dimension Affinity Distillation for 3D EM Neuron Segmentation},
  author={Liu, Xiaoyu and Cai, Miaomiao and Chen, Yinda and Zhang, Yueyi and Shi, Te and Zhang, Ruobing and Chen, Xuejin and Xiong, Zhiwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
