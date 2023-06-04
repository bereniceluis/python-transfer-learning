### split_dataset.ipynb  
- The file structure resulting from the partitioning process can be described as follows:
```data
├── dataset
│   ├── person1
│   │   ├── masked_faces
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   └── unmasked_faces
│   │       ├── image1.jpg
│   │       ├── image2.jpg
│   │       └── ...
│   ├── person2
│   │   ├── masked_faces
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   └── unmasked_faces
│   │       ├── image1.jpg
│   │       ├── image2.jpg
│   │       └── ...
│   └── ...
└── partition
    ├── train
    │   ├── person1
    │   │   ├── masked_faces
    │   │   │   ├── image1.jpg
    │   │   │   ├── image2.jpg
    │   │   │   └── ...
    │   │   └── unmasked_faces
    │   │       ├── image1.jpg
    │   │       ├── image2.jpg
    │   │       └── ...
    │   ├── person2
    │   │   ├── masked_faces
    │   │   │   ├── image1.jpg
    │   │   │   ├── image2.jpg
    │   │   │   └── ...
    │   │   └── unmasked_faces
    │   │       ├── image1.jpg
    │   │       ├── image2.jpg
    │   │       └── ...
    └── val
        ├── person1
        │   ├── masked_faces
        │   │   ├── image1.jpg
        │   │   ├── image2.jpg
        │   │   └── ...
        │   └── unmasked_faces
        │       ├── image1.jpg
        │       ├── image2.jpg
        │       └── ...
        ├── person2
        │   ├── masked_faces
        │   │   ├── image1.jpg
        │   │   ├── image2.jpg
        │   │   └── ...
        │   └── unmasked_faces
        │       ├── image1.jpg
        │       ├── image2.jpg
        │       └── ...
        └── ...
 ```
In the `dataset` irectory, you have subdirectories for each person containing their corresponding masked and unmasked face images.

The `partition` irectory contains the split of the dataset into training and validation sets. Inside the train directory, you have subdirectories for each person with their respective masked and unmasked face images used for training. Similarly, the val directory contains subdirectories for each person with their masked and unmasked face images used for validation.

This file structure allows for organized storage and easy access to the training and validation data during model training and evaluation.
     
