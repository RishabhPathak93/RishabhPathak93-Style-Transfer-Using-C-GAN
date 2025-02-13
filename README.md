# CycleGAN for Style Transfer: Horse to Zebra Transformation

## **Project Overview**
This project implements a Cycle-Consistent Generative Adversarial Network (CycleGAN) for unpaired image-to-image translation. Specifically, it transforms images of horses into zebras and vice versa, without requiring paired training data.

CycleGAN achieves this by using two generator-discriminator pairs to learn mappings between two domains, enforcing consistency through cycle-consistency losses.

### **Key Features**
- Implements CycleGAN architecture with PyTorch.
- Includes detailed preprocessing and augmentation using Albumentations.
- Saves checkpoints for generators and discriminators during training.
- Optimized for GPU-based training; supports CPU with specific requirements.

---

## **Dataset**
The project uses the **Horse2Zebra** dataset for unpaired image-to-image translation.

### **Download the Dataset**
The dataset can be downloaded from Kaggle using the following link:
[Horse2Zebra Dataset](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset/code).

### **Dataset Structure**
Ensure the dataset is organized as follows:
```
data/
  train/
    horses/
      - horse_image1.jpg
      - horse_image2.jpg
      ...
    zebras/
      - zebra_image1.jpg
      - zebra_image2.jpg
      ...
  val/
    horses/
      - horse_val_image1.jpg
      - horse_val_image2.jpg
      ...
    zebras/
      - zebra_val_image1.jpg
      - zebra_val_image2.jpg
      ...
```

---

## **Prerequisites**

### **Clone the Repository**
Clone this project using the following GitHub link:
```
git clone https://github.com/RishabhPathak93/RishabhPathak93-Style-Transfer-Using-C-GAN.git
cd RishabhPathak93-Style-Transfer-Using-C-GAN
```

### **System Requirements**
- **GPU:** NVIDIA GPU with CUDA support (recommended).
- **CPU:** Minimum 16 GB RAM (if GPU is unavailable).
- **Training Time:** Approximately 96 hours on a CPU.

### **Python Dependencies**
Install the required Python libraries using the following commands:
```
pip install numpy
pip install pandas
pip install scikit-learn
pip install torch
pip install torchvision
pip install albumentations
pip install pillow
```

---

## **How to Run the Project**

1. **Prepare the Dataset**
   - Download the dataset and organize it as per the structure mentioned above.

2. **Edit Configuration**
   - Adjust hyperparameters in `config.py` as needed, such as learning rate, batch size, and model paths.

3. **Train the Model**
   Run the following command to start training:
   ```bash
   python train.py
   ```
   This will:
   - Train the CycleGAN model for the specified number of epochs (default: 10).
   - Save intermediate results in the `saved_images/` folder.
   - Save model checkpoints for generators and discriminators.

4. **Validate the Model**
   Modify the script to load trained models (`LOAD_MODEL=True` in `config.py`) and validate on the test dataset.

5. **Generated Images**
   Check the generated images during training in the `saved_images/` directory.

---

## **Model Architecture**

### **Generator**
The generator consists of:
- Initial convolution layers with instance normalization.
- Downsampling blocks.
- Residual blocks (9 layers by default).
- Upsampling blocks.
- Final convolutional layer with a `tanh` activation function.

### **Discriminator**
The discriminator uses PatchGAN and:
- Takes 256x256 input images.
- Outputs a grid of real/fake predictions (70x70 patches).

### **Loss Functions**
- **Adversarial Loss:** Helps generators create realistic images.
- **Cycle Consistency Loss:** Ensures that translations are consistent (e.g., Horse -> Zebra -> Horse).
- **Identity Loss:** Encourages the generator to preserve details when applied to images from the target domain.

---

## **Training Process**
During training:
1. **Discriminator Training:**
   - Discriminators learn to distinguish real images from fake ones generated by the generators.
   - Real images are labeled as 1, and generated images as 0.

2. **Generator Training:**
   - Generators are trained to fool the discriminators while adhering to cycle consistency and identity losses.

3. **Loss Monitoring:**
   - Discriminator loss (`D_loss`)
   - Generator loss (`G_loss`)
   - Cycle consistency loss

Checkpoints for generators (`genh.pth.tar`, `genz.pth.tar`) and discriminators are saved after every epoch.

---

## **Results**
Visual results of the model’s transformation are saved in the `saved_images/` directory during training. Here are examples:

- **Horse-to-Zebra:**
  ![Horse to Zebra]
  ![1](https://i.ibb.co/4R5nDhY/horse-0.png)
  ![2](https://i.ibb.co/dMwqJLH/horse-200.png)
  ![3](https://i.ibb.co/3ChfYzX/horse-400.png)
  ![4](https://i.ibb.co/9Wt19v4/horse-600.png)
  ![5](https://i.ibb.co/jwY5cpw/horse-1000.png)

- **Zebra-to-Horse:**
  ![Zebra to Horse]([(https://ibb.co/8jh9PB4))

---

## **Troubleshooting**
- **CUDA Out of Memory:** Reduce the batch size in `config.py`.
- **Slow Training:** Ensure GPU is being utilized. Use `torch.cuda.is_available()` to verify.
- **Model Divergence:** Experiment with learning rates or reduce `LAMBDA_CYCLE` and `LAMBDA_IDENTITY` values.

---

## **Future Improvements**
- Experiment with different datasets for domain adaptation.
- Add perceptual loss for improved style transfer quality.
- Implement advanced GAN architectures, such as StyleGAN.
- Use mixed precision training for faster convergence on GPUs.

---

## **References**
- [CycleGAN Paper](https://www.researchgate.net/publication/322060135_Unpaired_Image-to-Image_Translation_Using_Cycle-Consistent_Adversarial_Networks)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Albumentations](https://albumentations.ai/)

---

This project was implemented by **Rishabh Pathak**. Contributions and feedback are welcome!

