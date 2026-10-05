\# 🌾 Rice Leaf Disease Identifier using Deep Learning



An end-to-end computer vision project that leverages a fine-tuned \*\*MobileNetV3-Large\*\* neural network to diagnose 5 distinct rice leaf diseases along with identifying healthy crops. The system includes an interactive drag-and-drop web application built with \*\*Gradio\*\*.



\## 📊 Project Overview

\* \*\*Architecture:\*\* MobileNetV3-Large (Pre-trained on ImageNet)

\* \*\*Framework:\*\* PyTorch \& Torchvision

\* \*\*Classes Detected (6):\*\* Bacterial Leaf Blight, Brown Spot, Healthy, Leaf Blast, Leaf Scald, Narrow Brown Spot

\* \*\*Deployment Interface:\*\* Gradio Web Application



\---



\## 📈 Performance \& Evaluation Report



The model was tested on a completely unseen validation set consisting of 528 total images (88 images per class). It achieved an \*\*Overall Validation Accuracy of 88.83%\*\*.



\### Class Breakdown:



| Disease Class | Total Test Images | Correct Predictions | Accuracy |

| :--- | :---: | :---: | :---: |

| \*\*Bacterial Leaf Blight\*\* | 88 | 88 | \*\*100.00%\*\* |

| \*\*Brown Spot\*\* | 88 | 77 | \*\*87.50%\*\* |

| \*\*Healthy\*\* | 88 | 88 | \*\*100.00%\*\* |

| \*\*Leaf Blast\*\* | 88 | 63 | \*\*71.59%\*\* |

| \*\*Leaf Scald\*\* | 88 | 86 | \*\*97.73%\*\* |

| \*\*Narrow Brown Spot\*\* | 88 | 67 | \*\*76.14%\*\* |



\### Insights from Confusion Matrix:

\* The model achieved \*\*perfect 100% scores\*\* identifying \*\*Healthy\*\* leaves and \*\*Bacterial Leaf Blight\*\*.

\* \*\*Visual Ambiguity:\*\* The primary area of confusion lies between \*Leaf Blast\* and \*Brown Spot\* (21 instances misclassified). These diseases display highly similar physical brown lesions, mimicking common challenges faced by human agricultural experts in the field.



\---



\## 🛠️ Installation \& Setup



\### 1. Clone the repository

```bash

git clone \[https://github.com/YOUR\_USERNAME/YOUR\_REPOSITORY\_NAME.git](https://github.com/YOUR\_USERNAME/YOUR\_REPOSITORY\_NAME.git)

cd Rice-Disease-AI



\###2. Set up the Environment

Ensure you have Conda installed, then run the following commands in your prompt:



Bash

conda create -n rice\_ai python=3.10

conda activate rice\_ai

pip install torch torchvision gradio scikit-learn matplotlib



\###3. Dataset Setup

Download the dataset from Kaggle.



Place the train and validation directories inside your project root directory.



\###4. Run the Web Application

Bash

python app.py



\---

