# Deconstructing Research: Building PLS + SVM from Scratch

This repository is a implementation of a machine learning pipeline inspired by two real-world neuroscience and machine learning research papers to help better understand the research conducted at Dunkley Lab. This library was created by David Estrine and goes through key concepts like **Partial Least Squares (PLS)** and **Support Vector Machines (SVM)**.

---

## Background & Goal

The project is inspired by the methods and findings of the following two papers:

1. **Nature Scientific Reports**  
   ["Support vector machine classification of major depressive disorder using diffusion-weighted neuroimaging and graph theory"](https://www.nature.com/articles/s41598-020-62713-5)  
   *→ Used SVM on structural connectome features to classify depression vs. healthy controls.*

2. **medRxiv (Preprint)**  
   ["Cognitive and Functional Correlates of Network Dysconnectivity in Depression"](https://www.medrxiv.org/content/10.1101/2024.11.15.24317356v1.full.pdf)  
   *→ Applied PLS to reduce imaging features and link to cognitive/clinical outcomes.*

The **goal** of this project is to replicate the core logic of their machine learning methodology using only native Python to fully understand and showcase the underlying math. You can find detailed notes in the `notes/` folder, thoroughly explaining the concepts learned.
