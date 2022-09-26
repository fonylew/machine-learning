
a description of your classification problems, and why you feel that they are interesting. Think hard about this. To be at all interesting the problems should be non-trivial on the one hand, but capable of admitting comparisons and analysis of the various algorithms on the other. 

the training and testing error rates you obtained running the various learning algorithms on your problems. At the very least you should include graphs that show performance on both training and test data as a function of training size (note that this implies that you need to design a classification problem that has more than a trivial amount of data) and--for the algorithms that are iterative--training times/iterations. Both of these kinds of graphs are referred to as learning curves, BTW.

analyses of your results. 
Why did you get the results you did? 
Compare and contrast the different algorithms. 
What sort of changes might you make to each of those algorithms to improve performance? 
How fast were they in terms of wall clock time? 
Iterations? 
Would cross validation help (and if it would, why didn't you implement it?)? 
How much performance was due to the problems you chose? 
How about the values you choose for learning rates, stopping criteria, pruning methods, and so forth (and why doesn't your analysis show results for the different values you chose? 
Please do look at more than one. And please make sure you understand it, it only counts if the results are meaningful)? 
Which algorithm performed best? 
How do you define best? 
Be creative and think of as many questions you can, and as many answers as you can but a lot of the questions boil down to: why... WHY WHY WHY?

** need a baseline with all features are used if you train with selected features
** if you mentioned something you used, you better explain its impact/why you used it


1 Decision Trees

with some form of pruning and describe split attributes
- pruning
- leaf

2 Neural Networks --> Stick with Relu
- hidden layers
- activation functions

3 Boosting
- number of weak leaners
- number of bags
with much more aggressive about pruning


4 Support Vector Machines
- kernel type
with at least two kernel functions
C and gamma depends on each other. If you tune one, you may/should tun another, i.e., tune C and gamma together. (For one of my datasets, gamma=auto instead of default scale did make a different) 

5 K-Nearest Neighbors
- k
with different values of k

Compare and contrast between your algorithms and datasets.

Explain why your datasets are interesting from the perspective of ML 


AdaBoost: n_estimators =  30
F1 Score:  0.8447857411929649
Training Accuracy:  0.8254755996691481
Testing Accuracy:  0.838150289017341
Confusion Matrix
[[311  44   8   0]
 [ 15  87   6   7]
 [  0   2  17   2]
 [  0   0   0  20]]


AdaBoost: n_estimators =  42
F1 Score:  0.8694975270950295
Training Accuracy:  0.8519437551695617
Testing Accuracy:  0.8651252408477842
Confusion Matrix
[[320  42   1   0]
 [ 15  89   1  10]
 [  0   0  20   1]
 [  0   0   0  20]]

 AdaBoost: n_estimators =  60
F1 Score:  0.8722368416999388
Training Accuracy:  0.8486352357320099
Testing Accuracy:  0.8689788053949904
Confusion Matrix
[[323  39   1   0]
 [ 17  89   2   7]
 [  0   0  19   2]
 [  0   0   0  20]]

 
 AdaBoost (max_depth = 2): n_estimators =  24
F1 Score:  0.8737675273238762
Training Accuracy:  0.8709677419354839
Testing Accuracy:  0.8670520231213873
Confusion Matrix
[[315  47   1   0]
 [  7  97   1  10]
 [  0   1  18   2]
 [  0   0   0  20]]

 AdaBoost (max_depth = 2): n_estimators =  42
F1 Score:  0.8739766107123705
Training Accuracy:  0.8701406120760959
Testing Accuracy:  0.8670520231213873
Confusion Matrix
[[310  52   1   0]
 [  5  99   1  10]
 [  0   0  21   0]
 [  0   0   0  20]]

 AdaBoost (max_depth = 2): n_estimators =  50
F1 Score:  0.8903058046144933
Training Accuracy:  0.8759305210918115
Testing Accuracy:  0.8863198458574181
Confusion Matrix
[[325  37   1   0]
 [  8  94   3  10]
 [  0   0  21   0]
 [  0   0   0  20]]

Conclusion and Inferences.

1.This case study is majorly based on cases in the Western world.

2.Cases show that more than 50% of people surveyed in countries like US,Australia and Canada undergo treatment for mental ailments.

3.People who are not more prone to work at home are usually bored and filled with anxiety leading to degradation in mental health.

4.People who are in the early 30's usually undergo treatment but there are extreme cases like 8 years and 72 years people recieving the same treatment.

5.It is interesting to find that people face mental trauma regardless of whether they are self employed or not.

6.The surveyed people agree that their mental health somewhat affects their productivity at work.

7.People feel that their employers somewhat easily sanction leave for mental health issues.The reason maybe that the employer does not want to take any risk of overloading the patient with work.

8.People feel that sharing about their mental or physical health with employers would help them a bit but they are reluctant to share the same with their coworkers.They would prefer to share with only some of the coworkers.

9.People dont know whether the employer considers mental health issues as seriously as the physical ones.The ambiguity still remains about people's reaction towards mental health.



TODO:
- Graph เทียบ param แต่ละตัว ROC AUC
- bias varience

References
Awwalu, J., Ghazvini, A. and Abu Bakar, A., 2014. Performance Comparison of Data Mining Algorithms: A Case Study on Car Evaluation Dataset. International Journal of Computer Trends and Technology, [online] 13(2), pp.78-82. Available at: <https://www.ijcttjournal.org/Volume13/number-2/IJCTT-V13P117.pdf> [Accessed 11 September 2020].

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. Available at: <https://archive.ics.uci.edu/ml/datasets/car+evaluation> [Accessed 9 September 2020].

Zupan, B., n.d. Car Dataset - Function Decomposition Page At AI Lab, FRI. [online] File.biolab.si. Available at: <https://file.biolab.si/biolab/app/hint/car_dataset.html> [Accessed 12 September 2020].


https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html


https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py

https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

    

((3), 'relu'),
    ((5), 'relu'),
    ((10), 'relu'),
    ((16), 'relu'),
    ((27), 'relu'),
    ((32), 'relu'),
    ((64), 'relu'),
    ((128), 'relu'),
    ((256), 'relu'),
    ((512), 'relu'),
    ((3, 3), 'relu'),
    ((8, 8), 'relu'),
    ((16, 16), 'relu'),
    ((27, 27), 'relu'),
    ((32, 32), 'relu'),
    ((64, 64), 'relu'),
    ((128, 128), 'relu'),
    ((256, 256), 'relu'),
    ((512, 512), 'relu'),
    ((2, 2, 2), 'relu'),
    ((3, 3, 3), 'relu'),
    ((5, 5, 5), 'relu'),
    ((8, 8, 8), 'relu'),
    ((8, 16, 8), 'relu'),
    ((16, 16, 16), 'relu'),
    ((27, 27, 27), 'relu'),
    ((32, 32, 32), 'relu'),
    ((64, 64, 64), 'relu'),
    ((128, 128, 128), 'relu'),
    ((256, 256, 256), 'relu'),
    ((512, 512, 512), 'relu'),