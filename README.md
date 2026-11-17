# SubstrateFinder – Cardiac Electrophysiology Inverse Problem

Team project for *Scientific Computing Tools for Advanced Mathematical Modelling*,
Politecnico di Milano (2026), with Simone Milani and Lorenzo Sciacca.

**Goal:** locate and reconstruct slow-conduction regions in the atria (atrial fibrillation)
from electrogram recordings.

**Method:** anisotropic eikonal model solved with the Fast Marching Method; a physics-based
Nelder–Mead inversion replaced by neural networks trained offline; an 8-dimensional
DeepSDF-style generative model of the conduction field with an encoder from activation times;
data augmentation based on the physical symmetries of the problem.

**Results:** focal localisation solved almost exactly; lesion centre, radius and full field
recovered with a single network forward pass, with no solver calls at inference time.

**Report:** [full report](docs/report.pdf)

---
# SubstrateFinder

# BeatFinder

Repository of the first guided project of the course "Scientific Computing Tools for Advanced Mathematical Modelling" AY 2025/2026.

# Introduction

Atrial fibrillation (AF) is a highly prevalent cardiac arrhythmia characterized by episodes of irregular electrical activity in the atria, which may impair atrial contraction. Despite extensive clinical and academic research efforts, the mechanisms underlying atrial fibrillation remain a subject of debate. Several theories have been proposed to explain AF mechanisms, including the multiple wavelet theory (Moe, 1962), focal activity (Jaïs et al., 1997), and driver domains (Haïssaguerre et al., 2014).

In recent years, technological advances have led to the development of new clinical tools that open new frontiers in the study of AF. In particular, the latest generation of mapping catheters, equipped with multiple splines, enables the acquisition of high-density endocavitary signals measuring the local electrical activity of atrial tissue.

This project aims to develop innovative scientific computing tools to reconstruct the electrophysiological substrate underlying atrial fibrillation from sparse measurements.

# Rules

Each group needs to create its branch of the main branch (name_of_your_new_branch=Surname1Surname2Surname3). 

Create the branch on your local machine and switch in this branch :

$ git checkout -b [name_of_your_new_branch]

Then push the branch on GitHub :

$ git push origin [name_of_your_new_branch]

Please upload only the code required for project evaluation and avoid including unnecessary material (e.g., images or additional datasets). Add an appropriate .gitignore file to your branch.

The data and code provided must be used exclusively within the scope of this course. For any other use, please notify the lecturer and request permission. Implementations and reports derived from course materials must not be published online or used for job applications or interviews without prior permission.
