# Deconstructing Research: Building PLS + SVM from Scratch

This repository is a implementation of a machine learning pipeline inspired by two real-world neuroscience and machine learning research papers to help better understand the research conducted at Dunkley Lab. This library was created by David Estrine and goes through key concepts like **Partial Least Squares (PLS)** and **Support Vector Machines (SVM)**.

---

## Background & Goal

The project is inspired by the methods and findings of the following two papers:

1. ["Support vector machine classification of major depressive disorder using diffusion-weighted neuroimaging and graph theory"](https://www.nature.com/articles/s41598-020-62713-5)  
   *→ Developing an objective method to diagnose/distinguish combat-related PTSD.*

   *→ Implementing a machine learning framework including SVM for classification.*

   *→ Using data from MEG scans from individuals with combat-related PTSD.*

3. ["Cognitive and Functional Correlates of Network Dysconnectivity in Depression"](https://www.medrxiv.org/content/10.1101/2024.11.15.24317356v1.full.pdf)  
   *→ Aimed to understand the neural mechanisms underlying chronic pain & its correlation with mental health issues like anxiety, depression and PTSD.*

   *→ Used MEG scans to assess neural synchrony, applying PLSR to predict chronic pain severity and its interaction with mental health symptoms.*

The **goal** of this project is to replicate the core logic of the machine learning methodology using Python to fully understand the papers. You can find detailed notes in the `research_notes/` folder, thoroughly explaining the concepts learned.
