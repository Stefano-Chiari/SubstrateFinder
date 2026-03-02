# Problem

The first checkpoint requires the development of an algorithm to identify focal activity from the traces of the given 20 unipolar signals. 

The algorithm must return the following outputs:

1. **Activation Annotations**: for each of the 20 signals, generate a list of activation times. These activation times correspond to the local minima of the first derivative of the signal, indicating the moments of depolarization at each electrode.
2. **Focal proximity**: assign a score of 1 to the sensor closest to the focal activity and 0 to all others. If the focal activity is within two sensors (up to a specified tolerance), assign 0.5 to each of the two sensors.
This output should be guided by signal morphology: focal activity is expected to correspond to a QS morphology (i.e., a single dominant negative deflection).

## Evaluation

The checkpoint's evaluation will be performed on a test dataset, not at your disposal. Your algorithm performance would allow for a maximum of 6 points. The evaluation metric will consider the accuracy and the efficiency of the annotation algorithm.
The group of students implementing the algorithm with the best efficiency-accuracy balance will receive a bonus point. 

## Dataset

The dataset comprises a set of 20 unipolar traces acquired in 5 different scenarios.


