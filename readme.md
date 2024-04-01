# Cross-dimension Affinity Distillation for 3D EM Neuron Segmentation
**Accepted by CVPR-2024**

Xiaoyu Liu, Miaomiao Cai, Yinda Chen, Yueyi Zhang, Te Shi, Ruobing Zhang, Xuejin Chen, Zhiwei Xiong* 

University of Science and Technology of China

*Corresponding Author


## Enviromenent

This code was tested with Pytorch 1.0.1 (later versions may work), CUDA 9.0, Python 3.7.4 and Ubuntu 16.04. 

If you have a [Docker](https://www.docker.com/) environment, we strongly recommend you to pull our image as follows：

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/renwu527/auto-emseg:v3.1
```

## Wafer4 Dataset
We introduce the Wafer4 dataset, a new electron microscopy (EM) neuron segmentation dataset collected from a region of the mouse medial entorhinal cortex. Utilizing Multi-Beam-SEM technology, the dataset is imaged at an impressive resolution of $8\times8\times35$ nm, providing intricate details of neural structures.

Comprising a volumetric size of $125\times1250\times1250$ voxels, the Wafer4 dataset offers fine-grained voxel-level annotations for over $1.9\times10^8$ voxels. This level of granularity enables precise segmentation of neural components, facilitating detailed analyses of neuronal morphology and connectivity patterns.

Notably, the Wafer4 dataset stands out due to its focus on the allocortex, a cerebral cortex region occupying a smaller area (10%) compared to the neocortex. While existing datasets primarily target the neocortex, particularly the somatosensory cortex, the Wafer4 dataset fills a crucial gap by providing insights into the allocortical region.

Positioned in layer VI of the cortical layers within the allocortex, the Wafer4 dataset captures neural structures in a deeper layer, offering unique perspectives on cortical organization and function. Layer VI's distinct location within the cortex and its potential associations with specific brain functions make it an intriguing area for exploration and analysis.

For ease of access and utilization by the research community, we will provide a comprehensive description of the Wafer4 dataset along with a link for accessing it in the updated paper. We anticipate that this unique dataset will serve as a valuable resource for advancing research in neural circuitry, cortical organization, and computational neuroscience.

We release the [Wafer4](https://drive.google.com/drive/folders/1QsMc71wWDozitktVDXSvZtu5OEP2JT5y?usp=drive_link) dataset in Google Driver.


## The code will be released soon ！




## Contact

If you have any problem with the released code and dataset, please contact me by email (liuxyu@mail.ustc.edu.cn).

