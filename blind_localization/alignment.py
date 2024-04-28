import numpy as np


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X

    Inputs: X (target), Y (input)

    Outputs: d (residual sum of square errors), Z (transformed Y), tform(transformation matrix)
    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros((n, m-my))),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # translation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform


def find_class_mean_target(X_train_target, y_train_target):
    class_means = []
    r_ls = []

    for r in range(5):
        xcorr = X_train_target[0][y_train_target == r]
        if len(xcorr) > 0:
            class_mean = np.mean(xcorr, axis=0)
            class_means.append(class_mean)
            r_ls.append(r)

    class_means = np.array(class_means)
    return class_means, r_ls

def find_class_mean_source(channel_features, channel_ridx_map, r_ls):
    class_means = []
    r_ls_new = []

    for r in r_ls:
        xcorr = channel_features[channel_ridx_map[:, 0]][channel_ridx_map[:, 1] == r][:, :-1]
        if len(xcorr) > 0:
            class_mean = np.mean(xcorr, axis=0)
            class_means.append(class_mean)
            r_ls_new.append(r)

    class_means = np.array(class_means)
    return class_means, r_ls_new

def supervised_align(channel_features, channel_ridx_map, X_train_target, y_train_target, shared_regions):
    """
    aligning mean of each class in target session to mean of each class in source session
    """
    class_mean_target, r_ls = find_class_mean_target(X_train_target, y_train_target)
    class_mean_source, r_ls_new = find_class_mean_source(channel_features, channel_ridx_map, r_ls)

    d, Z, tform = procrustes(class_mean_source, class_mean_target, scaling=True, reflection='best')
    return tform

def transform(data, tform, test_regions=None):
    if test_regions is None:
        return tform["scale"] * data @ tform["rotation"] + tform["translation"]
    
    transformed_data = tform["scale"] * data[:, test_regions] @ tform["rotation"] + tform["translation"]
    data[:, test_regions] = transformed_data
    return data

def supervised_align_all(X_source, y_source, X_target, y_target, n_regions=5, scaling=True):
    matched_X_source, matched_X_target = [], []

    for r in range(n_regions):
        X_source_r = X_source[y_source == r]
        X_target_r = X_target[y_target == r]

        for i in range(len(X_source_r)):
            for j in range(len(X_target_r)):
                if np.random.rand() < 0.1:
                    matched_X_source.append(X_source_r[i])
                    matched_X_target.append(X_target_r[j])

    matched_X_source = np.array(matched_X_source)
    matched_X_target = np.array(matched_X_target)

    d, Z, tform = procrustes(matched_X_source, matched_X_target, scaling=scaling, reflection='best')
    return tform