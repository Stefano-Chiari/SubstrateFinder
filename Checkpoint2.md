# Problem

The second checkpoint requires the development of an algorithm to identify slow conduction sites from the traces of given 20 unipolar signals. 

The algorithm must return the following outputs:

1. **slow conduction center** 
2. **slow conduction radius**

Therefore output should be in the following form:
[x_SC,y_SC,r_SC] 

The algorithm must be uploaded into the group's branch of the main repository. The implementation should follow the signature proposed in the checkpoint1.ipynb file. 

## Evaluation

The checkpoint's evaluation will be performed on a test dataset, not at your disposal. Your algorithm performance would allow for a maximum of 6 points. The evaluation metric will consider the accuracy and the efficiency of the annotation algorithm.
The group of students implementing the algorithm with the best efficiency-accuracy balance will receive a bonus point. 

## Dataset

The dataset comprises a set of 20 unipolar traces acquired in 5 different scenarios. Impulse sites and anisotropy ratios have been provided for these cases.

## Reference estimations

Here you can find some estimates of the center and radius of the slow conduction area for each scenario:
- scenario 1: [-0.407, 0.299, 0.578]
- scenario 2: [0.81, 0.113, 0.394]
- scenario 4: [-0.816, -0.686, 0.274]
- scenario 4: [0.197, -0.242, 0.042]
- scenario 5: [0.647, -0.69, 0.244]
 