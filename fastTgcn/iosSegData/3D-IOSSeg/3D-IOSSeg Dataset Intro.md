# 3D-IOSSeg Dataset

## Introduction

To promote the development of digital orthodontics, we propose a standardized clinical orthodontic tooth segmentation dataset, named 3D-IOSSeg. 3D-IOSSeg utilizes the 3Shape Trios intraoral digital scanning system to directly obtain a three-dimensional digital model of the patient's oral cavity. This method reduces errors in the modeling process and provides a better reflection of the patient's oral condition.

Each patient sample consisted of two rows of upper/lower tooth data. Meanwhile, we used the most common form of triangular mesh to describe the internal surface of the mouth to improve accuracy. It is worth noting that each row of teeth (including the gums) consists of 100,000 to 300,000 points and 100,000 to 450,000 mesh cells, and there are more than 20 million cells in the entire dataset.


3D-IOSSeg takes 3D mesh data as the processing object. Each tooth sample of the 3D-IOSSeg is stored in a PLY file format that consist of a header, a list of vertex, and a list of face. Among them, the header defines the internal organization of the file, such as which face are used by the 3D sample to describe the mesh, and the data type and attribute arrangement order of the spatial coordinates in the vertex and face lists. The vertex list and face list are used to store vertex data  (spatial coordinates and normal vectors) and mesh data (vertex numbers and color labels).


Adults usually have 32 permanent teeth, which are divided into incisors, canines, premolars, and molars according to their function and morphology. In this work, we analyze the relevant data for a fine-grained division of 32 human teeth. According to the different positions of the upper and lower jaws and the left and right distribution, we divide the teeth into frontal incisors, lateral incisors, canines, first premolars, second premolars, first molars, second molars, and third molars, a total of 32 tooth categories. 


## How to use our dataset

We divide the data into two parts: a training set and a test set. The training data consists of samples from the maxilla and mandible of 60 patients, totaling 120 data points. The test set includes samples from the maxilla and mandible of 30 patients, totaling 60 data points. We name the data as "Index_L/U" to distinguish between the maxilla and mandible of different patients.

We will provide both the original annotated version and the downsampled version of the 3D-IOSSeg data. You are free to modify the organization of the data according to your specific needs. Using Meshlab (https://www.meshlab.net/) software, you can easily edit our data by rotating, coloring, downsampling, and modifying the format. Depending on your task requirements, you can downsample the mesh data or directly use our downsampled standard data. 