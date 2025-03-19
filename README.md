This repo contains the code for the WACV 2025 Paper, WARLearn: Weather Adaptive Representation Learning.


The base code for YOLOv3 is taken from this repo: YOLOv3-PyTorch[https://github.com/Lornatang/YOLOv3-PyTorch]

# WARLearn

The WARLearn code was developed on a Linux machine with an NVIDIA TITAN RTX 24GB GPU.

## Replicating the WARLearn Experiment

To replicate the WARLearn experiment, follow these steps:

1. Create a conda environment with Python 3.10.
   
    ```bash
    conda create -n warlearn_env python=3.10
    conda activate warlearn_env
    ```

2. Install the required packages using pip.

    ```bash
    pip install -r requirements.txt
    pip install tensorboard
    ```

### Datasets

- Pascal VOC: [http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/)
- RTTS: [https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2)
- ExDark: [https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset)

### Code References

The WARLearn code is built on top of the YOLOv3 code from [Lornatang/YOLOv3-PyTorch](https://github.com/Lornatang/YOLOv3-PyTorch).

We also utilized the simulation code from Image-Adaptive-YOLO [wenyyu/Image-Adaptive-YOLO](https://github.com/wenyyu/Image-Adaptive-YOLO) to run adverse weather baseline experiments.

### Training

1. Download the Pascal VOC dataset and place it in the following structure:

    ```plaintext
    VOC0712
    ├── test
    |    └──VOCdevkit
    |        └──VOC2007 (VOCtest_06-Nov-2007.tar)
    └── train
         └──VOCdevkit
             └──VOC2007 (VOCtrainval_06-Nov-2007.tar)
             └──VOC2012 (VOCtrainval_11-May-2012.tar)
    ```

2. Use `scripts/voc_annotation.py` to reformat the dataset.
3. Run the `train.py` script to train the model on clear weather data. We trained the clean data model for 600 epochs.
4. Store the feature vectors obtained from the backbone for each image in the training dataset into a dictionary and save it in "train_latent_dict.pickle". This can be done running the `test.py` script on the training data. Check commented lines 133-136 and 217-218 in `test.py`.

Run the adverse weather simulating scripts, (e.g., `scripts/data_make_foggy.py`) to generate synthetic foggy adverse weather training and test datasets from Pascal VOC training and test data respectively. Use `train_adverse.py` to fine-tune the YOLO clean model backbone for foggy weather. The fine-tuning was done for 10 epochs.

Similarly, generate low-light data using `scripts/data_make_lowlight.py` and then utilize `train_adverse.py` to fine-tune the YOLO clean model backbone for low-light weather conditions. The fine-tuning was done for 10 epochs.

Please refer to the paper for more details about the training parameters setup.

### Testing

1. Download RTTS dataset for real-world foggy images and ExDark for real-world low-light images.
2. Get the test samples from these datasets in the required format using the corresponding scripts in the `scripts` folder.
3. Modify the `test` path in `data/voc.data` to your test data list path.
4. Modify the `WEIGHTS_PATH` in `configs/test/YOLOV3_VOC.yaml` to the model `.pth.tar` path.
5. Run `test.py` to get the results for mAP@50(%).

## WARLearn Results

This code, along with the trained models and the test datasets will be made publicly available after the anonymous review process.

```bibtex
@InProceedings{Agarwal_2025_WACV,
    author    = {Agarwal, Shubham and Birman, Raz and Hadar, Ofer},
    title     = {WARLearn: Weather-Adaptive Representation Learning},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {4978-4987}
}
```
