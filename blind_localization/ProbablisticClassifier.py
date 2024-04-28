import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt

class ProbablisticClassifier:
    def __init__(self):
        self.channel_features = None
        self.distances = None
        self.regions = None

    def fit(self, X_train, y_train, distance=False):
        """
        channel_features: len(channel_idx_train) * 5 array
        distances: len(channel_idx_train) * len(channel_idx_train) array (selected_D)
        channel_idx_train + regions = channel_region_map
        """
        if distance:
            self.channel_features = X_train[0]
            self.distances = X_train[1]
            self.n_features = X_train[0].shape[1]
        else:
            self.channel_features = X_train
            self.n_features = X_train.shape[1]
        
        self.regions = y_train
        self.region_ls = np.unique(y_train)
        self.n_regions = len(self.region_ls)

        self.estimate_priors()
        self.estimate_feature_dist()
        if distance: 
            self.estimate_distance_dist()

    def predict(self, channel_features_test, distances_test, distance=False):
        assert channel_features_test.shape[1] == self.channel_features.shape[1], "train/test set feature dimensions don't match"
        log_likelihoods = np.ones((len(channel_features_test), self.n_regions))
        labels = np.zeros((len(channel_features_test, ), ))

        for i in tqdm(range(len(channel_features_test))):
            log_likelihood = np.zeros(self.n_regions, )
            
            for a in range(self.n_regions):
                # a = test_regions[a] if test_regions else a
                log_likelihood[a] += np.log(multivariate_normal.pdf(channel_features_test[i], mean=self.feature_means[a], cov=self.feature_cov[a]))
                log_likelihood[a] += np.log(self.priors[a])

                for j in range(len(self.channel_features)):
                    b = np.where(self.region_ls == self.regions[j])[0][0]
                    log_likelihood[a] += np.log(multivariate_normal.pdf(self.channel_features[j], mean=self.feature_means[b], cov=self.feature_cov[b]))

                    if distance:
                        log_likelihood[a] += np.log(norm.pdf(distances_test[i][j], loc=self.dist_means[a][b], scale=np.sqrt(self.dist_cov[a][b])))
                    log_likelihood[a] += np.log(self.priors[b])

            log_likelihoods[i] = log_likelihood
            labels[i] = self.region_ls[int(np.argmax(log_likelihood))]

        return log_likelihoods, labels

    def estimate_priors(self):
        self.priors = np.zeros((self.n_regions, ))

        for i in range(self.n_regions):
            self.priors[i] = len(self.regions[self.regions == self.region_ls[i]])

        self.priors = self.priors / sum(self.priors)

    def estimate_feature_dist(self):
        self.feature_means = np.zeros((self.n_regions, self.n_features))  # number of train regions * number of features
        self.feature_cov = np.zeros((self.n_regions, self.n_features, self.n_features))  # number of train regions * cov table

        for i in range(self.n_regions):
            xcorr = self.channel_features[self.regions == self.region_ls[i]]
            if len(xcorr) < 2:
                raise ValueError("Covariance calculation requires at least two observations per class. ")
            self.feature_means[i] = np.mean(xcorr, axis=0)
            self.feature_cov[i] = np.cov(xcorr.T)
        
        self.correct_cov_matrix()

    def correct_cov_matrix(self):
        for i in range(len(self.feature_cov)):
            self.feature_cov[i] = (self.feature_cov[i] + self.feature_cov[i].T)/2
            cov_matrix = self.feature_cov[i]
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues[eigenvalues < 0] = 0
            self.feature_cov[i] = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            self.feature_cov[i] += 1e-3 * np.eye(self.feature_cov[i].shape[0])

    def estimate_distance_dist(self):
        self.dist_means = np.zeros((self.n_regions, self.n_regions))  # number of test regions * number of train regions
        self.dist_var = np.zeros((self.n_regions, self.n_regions))  # number of test regions * number of train regions

        for c_i in range(self.n_regions):
            for c_j in range(self.n_regions):
                dist = self.distances[np.ix_(self.regions == self.region_ls[c_i], self.regions == self.region_ls[c_j])]
                self.dist_means[c_i][c_j] = np.mean(dist)
                self.dist_var[c_i][c_j] = np.var(dist)