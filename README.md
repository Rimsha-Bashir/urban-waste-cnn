# Automated Detection and Spatial Analysis of Illegal Waste Sites with CNNs

A deep learning approach to automatically classify aerial images and identify areas with illegal waste accumulation in urban environments.

---

## 1. Problem Statement

### Background
Illegal waste dumping and accumulation in urban areas is a significant environmental and public health concern. Traditional monitoring methods rely on manual inspections, which are costly, slow, and often incomplete. Rapid detection of waste hotspots using aerial imagery and machine learning can help city planners, environmental authorities, and public health agencies make faster, data-driven decisions to manage and mitigate waste effectively.

### Objectives
- Develop a data-driven model that **classifies aerial images** to identify areas with waste accumulation.
- Identify which image features contribute most to waste detection (for model interpretability).
.....

### Expected Outcome
- Trained **CNN models** capable of accurately classifying aerial images as waste / no-waste.
- Visualizations and heatmaps highlighting predicted waste areas.
.....
  
---

## 2. Dataset
**Source:** [AerialWaste](https://aerialwaste.org/)  
The dataset consists of more than 11,700 images from three different sources and contains annotations at different granularities:

- Binary labels: images are classified based on the presence or absence of waste.
- Multi-class multi-label: a subset of images is annotated based on the presence of specific waste objects.
- Weakly-supervised localization: a subset of images is annotated with ground truth segmentation masks surrounding relevant waste objects.

**Citation:** Torres, R. N., & Fraternali, P. (2023). AerialWaste dataset for landfill discovery in aerial and satellite images. *Scientific Data, 10*(1), 63. Nature Publishing Group UK London.
