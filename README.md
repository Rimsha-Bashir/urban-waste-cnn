# Automated Detection and Spatial Analysis of Illegal Waste Sites with CNNs

A deep learning approach to automatically classify aerial images and identify areas with illegal waste accumulation in urban environments.

----
## 1. Problem Statement

### Background
Illegal waste dumping and accumulation in urban areas is a significant environmental and public health concern. Traditional monitoring methods rely on manual inspections, which are costly, slow, and often incomplete. Rapid detection of waste hotspots using aerial imagery and machine learning can help city planners, environmental authorities, and public health agencies make faster, data-driven decisions to manage and mitigate waste effectively.

### Objectives
- Develop a **binary image classification model** to detect the presence of illegal waste sites (waste vs. no-waste).
- Train and compare **multiple CNN-based models** using transfer learning.
- Perform **exploratory data analysis (EDA)** on aerial images and annotations.
- Build a **reproducible training pipeline** suitable for cloud execution.
- Deploy the trained model as a **containerized inference service** on AWS.

### Challenges
- Large image sizes and high storage requirements.
- Class imbalance between waste and non-waste images.
- Variability in lighting, resolution, and background patterns across images.
- Limited availability of pixel-level annotations for all images.

### Expected Outcome
- Trained CNN models capable of accurately classifying aerial images as **waste / no-waste**.
- Quantitative comparison of multiple model architectures and hyperparameters.
- A containerized inference API deployable via AWS Lambda.
- A fully reproducible, cloud-based ML pipeline.
.....
  
----
## 2. Dataset Description

**Source:** [AerialWaste Dataset](https://aerialwaste.org/)

The AerialWaste dataset is a large-scale dataset designed for landfill discovery and waste detection using aerial and satellite imagery.

### Dataset Characteristics
- Images collected from multiple geographic regions.
- Images distributed across multiple folders (`images0` – `images5`).
- COCO-style JSON annotations provided for training and testing.

### Annotation Types
- **Binary classification:** Presence or absence of waste in an image.
- **Multi-label annotations:** Types of waste objects present (subset of images).
- **Segmentation masks:** Polygon annotations around waste objects (subset of images).

> **Note:**  
> The primary focus of this project is **binary image classification**.  
> Segmentation annotations are explored during EDA and discussed as a future extension.

### Citation
```bibtex
@article{torres2023aerialwaste,
  title={AerialWaste dataset for landfill discovery in aerial and satellite images},
  author={Torres, Rocio Nahime and Fraternali, Piero},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={63},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
....

----
## 3. Technology Stack

| Layer                   | Tools / Libraries                                    |
| ----------------------- | ---------------------------------------------------- |
| **Language**            | Python 3.10+                                         |
| **Data Processing**     | pandas, numpy, pycocotools, OpenCV                   |
| **Visualization**       | matplotlib, seaborn                                  |
| **Modeling**            | PyTorch, torchvision, timm (transfer learning), ONNX |
| **Training Platform**   | AWS SageMaker                                        |
| **Web Framework**       | FastAPI                                              |
| **Model Serialization** | TorchScript, ONNX                                    |
| **Containerization**    | Docker                                               |
| **Cloud Deployment**    | AWS Lambda, AWS ECR, AWS API Gateway                 |
| **Version Control**     | Git, GitHub                                          |


----
## 4. Folder Structure

![alt text](./images/folder-structure.png)

----
## 5. Application Workflow



---

## 6. Instructions to Reproduce

This section provides a **complete, end-to-end guide** to reproducing the deployed machine learning inference system, from local container-based testing to cloud deployment on **AWS Lambda** and public access via **API Gateway**.  
The final inference API is additionally integrated into a **Streamlit Cloud application** for user-facing testing and validation.

### 6.1 System Requirements

The following tools and services are required:

- **Docker ≥ 20.x**
- **AWS CLI ≥ 2.x**
- **Python ≥ 3.9**
- An active **AWS account** with permissions to create IAM users, roles, Lambda functions, ECR repositories, and API Gateway resources
- A **Streamlit Cloud account** (optinally, if you wish to deploy this project yourself.)


### 6.2 AWS Identity and Access Management (IAM)

For instructions on AWS configurations and IAM, check out [aws-configurations.md](./reproducability-instructions/aws-configurations.md)

---

### 6.3 **Local Inference Testing**

For instructions on local containerized inference testing, check out [local-testing.md](./reproducability-instructions/local-testing.md)

---

### **6.4 Cloud Deployment (AWS)**

For instructions on deploying the ML project on AWS, check out [cloud-deployment.md](./reproducability-instructions/cloud-deployment.md)

---

### **6.5 Streamlit Cloud Deployment (User-Facing Validation)**

To enable non-technical users to interact with the model, the AWS API endpoint is integrated into a Streamlit Cloud application.

Overview

- The Streamlit app provides an image upload interface
- Uploaded images are Base64-encoded client-side
- Requests are forwarded to the AWS API Gateway /predict endpoint
- Predictions are displayed in real time

Purpose

- A final system validation layer
- A user experience (UX) testbed
- A demonstration of real-world model consumption

The Streamlit application communicates only with the public API and does not require direct access to AWS credentials.


----
## 7. Future Enhancements

- Expanding the training dataset with additional labeled images to improve model generalization
- GPU-backed inference (Amazon ECS or SageMaker)
- Batch inference pipelines for large-scale evaluation
- Model versioning
- CI/CD automation for retraining and redeployment