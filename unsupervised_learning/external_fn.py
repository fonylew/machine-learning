import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import learning_curve, validation_curve
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

def expectation_maximization(X_train, X_test, y_train, y_test, init_means, no_iter = 1000, component_list = [3,4,5,6,7,8,9,10,11], num_class = 7, toshow = 1):


    array_aic = []
    array_bic = []
    array_homo =[]
    array_comp = []
    array_sil = []
    array_avg_log = []


    for num_classes in component_list:

        clf = GaussianMixture(n_components=num_classes,covariance_type='spherical', max_iter=no_iter, init_params= 'kmeans')
        #     clf = KMeans(n_clusters= num_classes, init='k-means++')

        clf.fit(X_train)

        y_test_pred = clf.predict(X_test)
        #Per sample average log likelihood
        avg_log = clf.score(X_test)
        array_avg_log.append(avg_log)


        #AIC on the test data
        aic = clf.aic(X_test)
        array_aic.append(aic)

        #BIC on the test data
        bic = clf.bic(X_test)
        array_bic.append(bic)

        #Homogenity score on the test data
        homo = metrics.homogeneity_score(y_test, y_test_pred)
        array_homo.append(homo)

        #Completeness score
        comp = metrics.completeness_score(y_test, y_test_pred)
        array_comp.append(comp)

        #Silhoutette score
        sil = metrics.silhouette_score(X_test, y_test_pred, metric='euclidean')
        array_sil.append(sil)



    #Generating plots

    fig1,ax1 = plt.subplots()
    ax1.plot(component_list, array_aic)
    ax1.plot(component_list, array_bic)
    plt.legend(['AIC', 'BIC'])
    plt.xlabel('Number of clusters')
    plt.title('AIC/BIC curve for Expected Maximization')

    fig2,ax2 = plt.subplots()
    ax2.plot(component_list, array_homo)
    ax2.plot(component_list, array_comp)
    ax2.plot(component_list, array_sil)
    plt.legend(['homogenity','completeness','silhoutette'])
    plt.xlabel('Number of clusters')
    plt.title('Performance evaluation scores for Expected Maximization')


    fig3, ax3 = plt.subplots()
    ax3.plot(component_list, array_avg_log)
    plt.xlabel('Number of clusters')
    plt.title('Per sample average log likelihood for Expected Maximization')


    if(toshow == 1):
        plt.show()

    #Training and testing accuracy for K = number of classes
    
    clf = GaussianMixture(n_components=num_class ,covariance_type='spherical', max_iter=no_iter, init_params= 'kmeans')

    #Assigning the initial means as the mean feature vector for the class
    clf.means_init = init_means

    clf.fit(X_train)

    #Training accuracy
    y_train_pred = clf.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    print('Training accuracy for Expected Maximization for K = {}:  {}'.format(num_class, train_accuracy))

    #Testing accuracy
    y_test_pred = clf.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    print('Testing accuracy for Expected Maximization for K = {}:  {}'.format(num_class, test_accuracy))

    return component_list, array_aic, array_bic, array_homo, array_comp, array_sil, array_avg_log

    
def kmeans(X_train, X_test, y_train, y_test, init_means, no_iter = 1000, component_list =[3,4,5,6,7,8,9,10,11], num_class = 7, toshow=  1):

    array_homo =[]
    array_comp = []
    array_sil = []
    array_var = []

    for num_classes in component_list:
        
        clf = KMeans(n_clusters= num_classes, init='k-means++')
        clf.fit(X_train)
        y_test_pred = clf.predict(X_test)
        
          
        #Homogenity score on the test data
        homo = metrics.homogeneity_score(y_test, y_test_pred)
        array_homo.append(homo)
        
        #Completeness score
        comp = metrics.completeness_score(y_test, y_test_pred)
        array_comp.append(comp)
        
        #Silhoutette score
        sil = metrics.silhouette_score(X_test, y_test_pred, metric='euclidean')
        array_sil.append(sil)

        #Variance explained by the cluster
        var = clf.score(X_test)
        array_var.append(var)
        
        

    #Generating plots
    fig4,ax4 = plt.subplots()
    ax4.plot(component_list, array_homo)
    ax4.plot(component_list, array_comp)
    ax4.plot(component_list, array_sil)
    plt.legend(['homogenity','completeness','silhoutette'])
    plt.xlabel('Number of clusters')
    plt.title('Performance evaluation scores for KMeans')


    fig5, ax5 = plt.subplots()
    ax5.plot(component_list, array_var)
    plt.title('Variance explained by each cluster for KMeans')
    plt.xlabel('Number of cluster')

    if(toshow == 1):

        plt.show()


    #Training and testing accuracy for K = num_class

    #Assigning the initial means as the mean feature vector for the class
    init_mean = init_means
    clf = KMeans(n_clusters= num_class, init = init_mean)

    clf.fit(X_train)

    #Training accuracy
    y_train_pred = clf.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    print('Training accuracy for KMeans for K = {}:  {}'.format(num_class, train_accuracy))

    #Testing accuracy
    y_test_pred = clf.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    print('Testing accuracy for KMeans for K = {}:  {}'.format(num_class, test_accuracy))


    return component_list, array_homo, array_comp, array_sil, array_var


# plot_learning_curve function from official scikit-learn documentation
# ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")
    axes[0].set_ylim(0.0, 1.1)

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


# ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

def plot_validation_curve(estimator, title, X, y, param_name, param_range, scoring="accuracy", axes=None, ylim=None, cv=10,
                            n_jobs=-1):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        scoring=scoring, n_jobs=n_jobs, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    ax = plt.subplot()

    plt.title("Validation Curve")
    plt.xlabel(param_name)
    plt.ylabel(scoring + " Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw, )
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    
    return plt


# ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

def plot_roc_auc_curve(estimator, x_test, y_test, y_pred, title="Receiver operating characteristic (ROC Curve)"):
    probs = estimator.predict_proba(x_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    return plt
