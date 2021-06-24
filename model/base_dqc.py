# import math
import scipy
import numpy as np

from scipy import integrate
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D

# vectorize the scipy quad function
integrate_vec = np.vectorize(integrate.quad)


def wavefunction(x_i, sigma):
    '''
    Returns a function which accepts an N-dimensional argument for a center x_i and standard deviation sigma.
    '''
    s_sq = np.float64(sigma) ** 2
    a = (2 * s_sq) ** -1
    return lambda x: (np.sqrt(np.pi / a) ** -(0.5) * np.exp(-a * (euclidean(x, x_i) ** 2)))
    # return lambda x: np.exp((-(x-x_i)**2)/(2*(sigma**2)))


# wavefunction_vec = np.vectorize(wavefunction) # vectorized wavefunction for multiple data inputs

# Output of vectorized kinetic, potential, and wavefunction terms is an array of functions.
# This allows you to pass a large array to all of the components.
apply_vectorized = np.vectorize(lambda f, x: f(x))


def multi_wavefunction(data_array, sigma):
    wfs = []
    for i in range(len(data_array)):
        wfs.append(wavefunction(data_array[i], sigma))
    return wfs


def total_wavefunction(data_array, sigma):
    '''
    Returns the total wavefunction which is the sum of all component wavefunctions.
    '''
    # arg of output must be full data array, not a single data point
    wfs = multi_wavefunction(data_array, sigma)

    def call(x):
        term = 0
        for wf in wfs:
            term += wf(x)
        return term

    return lambda x: call(x)
    # return lambda x: apply_vectorized(wfs, x)


def kinetic_term(x_i, sigma, m):
    """
    Returns the kinetic term of the Schrodinger equation with a Gaussian wavefunction input.
    """
    s_sq = np.float64(sigma) ** 2
    a = (2 * s_sq) ** -1

    return lambda x: np.divide(1, 2 * m, dtype=np.float64) * (4 * a ** 2) * (euclidean(x, x_i)) ** 2 * np.exp(-a * (euclidean(x, x_i) ** 2))  # - 2*a*np.exp(-a*np.dot(x-x_i,x-x_i)))).prod()
    # return lambda x: -0.5*1/sigma**2 *normalization**2 * np.linalg.norm((x - x_i)) * np.exp(np.linalg.norm((x-x_i))/(2*sigma**2))
    # kinetic_vec = np.vectorize(kinetic_term) # Allows for multiple datapoint inputs.


def multi_kterm(data_array, sigma, m):
    ks = []
    for i in range(len(data_array)):
        ks.append(kinetic_term(data_array[i], sigma, m))
    return ks


def total_kinetic_term(data_array, sigma, m):
    """
    Returns the the kinetic term function which evaluates the total kinetic term as a sum of all kinetic term components.
    """
    kts = multi_kterm(data_array, sigma, m)

    def call(x):
        term = 0
        for ks in kts:
            term += ks(x)
        return term

    return lambda x: call(x)
    # return lambda x: apply_vectorized(kfs, x)


def potential(x_i, sigma):
    """
    Returns a single component potential term with center x_i and standard deviation sigma.
    """
    wf = wavefunction(x_i, sigma)
    kt = kinetic_term(x_i, sigma)
    # d=(np.shape(x_i[0]))[0]
    zero = np.zeros(np.shape(x_i))
    # d_term = np.divide(d,2,dtype=np.float32)
    term_3 = lambda x: (2 * (sigma) ** 2 * wf(x)) ** -1 * kt(x)

    # E = d_term - term_3(zero)
    # return lambda x: E - d_term + term_3(x)
    return lambda x: term_3(x)


potential_vec = np.vectorize(potential)


def total_potential(data_array, sigma, m):
    """
    Calculates the DQC Schrodinger potential for a given point x and given initial value x_i.
    From Horn and Gotlieb (2001), the potential should be E - d/2 + 1/2(sigma)(psi)* sum((x-x_i)**2 * exp(-(x-x_i)**2/2(sigma)**2)).
    The second and third terms are referred to as the 2nd and 3rd terms in the code.
    """

    kts = total_kinetic_term(data_array, sigma, m)
    wfs = total_wavefunction(data_array, sigma)
    d = (np.shape(data_array[0]))[0]
    term_3 = lambda x: (wfs(x)) ** -1 * kts(x)
    # print(term_3(data_array[0]))
    data_sum = np.zeros(np.shape(data_array[0]))
    for data in data_array:
        data_sum += data
    # print(term_3(zero))
    E = d / 2  # - term_3(data_sum)
    # print(E)
    # return lambda x:  wfs(x,data_array,sigma), lambda x: kts(x,data_array,sigma,m)
    return lambda x: E - d / 2 + term_3(x)


def calculate_N(data_array, sigma):
    """
    Computes the N matrix of <psi_j|psi_i> values by looping over the array of data centers data_array with standard deviation sigma.
    """
    s_sq = np.float64(sigma) ** 2
    a = (2 * s_sq) ** -1

    def N_element(x, y, sigma):
        return np.exp(-0.5 * a * (euclidean(x, y) ** 2))

    # N_element_vec = np.vectorize(N_element)
    dim = len(data_array)
    N = np.empty((dim, dim))
    for i in range(dim):
        for j in range(dim):
            x_i = data_array[i]
            x_j = data_array[j]
            N[i][j] = N_element(x_i, x_j, sigma)

    return N


def truncate_data(data_array, N):
    evals, evects = np.linalg.eig(N)
    evects = [evects[i] for i in range(len(evals)) if evals[i] > 10 ** -5]
    evals = [eig for eig in evals if eig > 10 ** -5]
    truncated_data = np.empty((len(evects), len(data_array[0])))
    for i in range(len(evects)):
        truncated_data[i] = data_array[i]

    return truncated_data


def calculate_H(data_array, sigma, m):
    dim = len(data_array)
    H = np.empty((dim, dim))
    s_sq = np.float64(sigma) ** 2
    a = (2 * s_sq) ** -1
    vt = total_potential(data_array, sigma, m)

    def p_sq_expectation(x, y, sigma):
        # return np.dot(x-y,x-y)/(2*sigma**2) * np.exp(-np.dot(x-y,x-y)/(4*sigma**2))
        return np.divide(1, 2 * m, dtype=np.float64) * (euclidean(x, y) ** 2) * 0.5 * a * np.exp(-0.5 * a * (euclidean(x, y) ** 2))

    p_vec = np.vectorize(p_sq_expectation)

    def v_expectation(x, y, sigma):
        return np.exp(-0.5 * a * (euclidean(x, y) ** 2) * vt(0.5 * (x + y)))

    # v_vec = np.vectorize(v_expectation)
    for i in range(dim):
        for j in range(dim):
            x_i = data_array[i]
            x_j = data_array[j]
            p_term = p_sq_expectation(x_i, x_j, sigma)
            # v_term = np.exp(-a*(euclidean(x_i,x_j)**2))*vt(0.5*(x_i+x_j))
            #  if p_term == np.nan or math.isnan(v_term) is True:
            #      print(x_i)
            #                print(x_j)
            # term = p_term.astype(np.float64) + v_term.astype(np.float64)
            H[i][j] = p_term
    return H


def calculate_X(data_array, sigma):
    s_sq = np.float64(sigma) ** 2
    a = (2 * s_sq) ** -1

    dim = len(data_array)
    data_dim = len(data_array[0])
    X = np.empty((data_dim, dim, dim), dtype=np.complex64)

    # exp_vec = np.vectorize(np.exp)
    def x_expec_t1(x, y):
        return 0.5 * (x + y)

    def x_expec_t2(x, y, sigma):
        return np.exp(-0.5 * a * (euclidean(x, y)) ** 2)

    # x_vec = np.vectorize(x_expec_t2)
    for i in range(dim):
        for j in range(dim):
            x_i = data_array[i]
            x_j = data_array[j]
            x = x_expec_t1(x_i, x_j)
            x = x * x_expec_t2(x_i, x_j, sigma)
            x = x.astype(np.complex64)
            for k in range(data_dim):
                X[k][i][j] = x[k]
    return X


def diagonalize_H(H):
    evals, evects = np.linalg.eig(H)
    H_diag = np.identity(np.shape(H)[0])
    for i in range(len(evals)):
        H_diag[i] = evals[i] * H_diag[i]
    P = evects.T
    return P, H_diag


def basis_transform(A, N, is_expectation=False):
    N_inv = np.linalg.inv(N)
    N_inv_half = scipy.linalg.sqrtm(N_inv)
    if is_expectation == False:
        A = np.dot(A, N_inv_half)
        A = np.dot(N_inv_half, A)
    else:
        for j in range(np.shape(A)[0]):
            A[j] = np.dot(A[j], N_inv_half)
            A[j] = np.dot(N_inv_half, A[j])
    return A


def trajectory(data_array, P, H, X, N, steps=10 ** 5, delta=1, sigma=0.07, m=0.2, stride=100):
    # N = calculate_N(data_array,sigma)
    N_inv = np.linalg.inv(N)
    # data_array = truncate_data(data_array,N)
    # N = calculate_N(data_array,sigma)
    # X = calculate_X(data_array,sigma)
    # X = X.astype(np.complex64)
    #    H = np.dot(H,scipy.linalg.sqrtm(N_inv))
    #    H = np.dot(scipy.linalg.sqrtm(N_inv),H)
    # H = calculate_H(data_array,sigma,m)
    # P,H = diagonalize_H(H)
    P = P.astype(np.complex64)
    H = H.astype(np.complex64)
    P_inv = scipy.linalg.inv(P)
    # HP = np.dot(H,P)
    # PinvHP = np.dot(P_inv,HP)
    # PinvHP = np.dot(PinvHP,scipy.linalg.sqrtm(N_inv))
    # PinvHP = np.dot(scipy.linalg.sqrtm(N_inv),PinvHP)
    data_dims = np.shape(data_array)
    traj = np.zeros((steps // stride, data_dims[1], data_dims[0], data_dims[0]), dtype=np.complex64)
    N_traj = np.zeros((steps // stride, data_dims[0], data_dims[0]), dtype=np.complex64)
    traj[0] = X
    expH = np.exp(1j * delta * H.diagonal(), dtype=np.complex64)
    expH = np.dot(np.diag(expH), P_inv)
    expH = np.dot(P, expH)
    expHconj = np.conj(expH).T
    #   expHconj = np.exp(-1j*delta*H.diagonal(),dtype=np.complex64)
    #   expHconj = np.dot(np.diag(expHconj),P)
    #   expHconj = np.dot(P_inv,expHconj)
    expHpsi = np.dot(expH, N)
    expHconjpsi = np.conj(expHpsi)
    index = 0
    for i in range(steps):
        dt = i * delta
        # expH = np.exp(1j*dt*H.diagonal(),dtype = np.complex64)
        # expHpsi = np.dot(np.diag(expH),N)
        expHpsi = np.dot(expH, expHpsi)
        expHconjpsi = np.conj(expHpsi)
        if stride - index == 0:
            index = i // stride
            for j in range(data_dims[1]):
                if index != 0:
                    traj[index][j] = np.dot(X[j], expHpsi)
                    traj[index][j] = np.dot(expHconjpsi, traj[index][j])
            if index != 0:
                N_traj[index] = expHpsi
                # traj[index] = expHpsi
            # traj1[index] = expHconjpsi
            index = 0
        index += 1
    return traj, N_traj


def cart2sph(x, y, z, normalized=True):
    hxy = np.hypot(x, y)
    # if normalized is False:
    #     r = np.hypot(hxy, z)
    # else:

    r = 1
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = np.sin(el)
    return x, y, z


cart2sph_vec = np.vectorize(cart2sph)
sph2cart_vec = np.vectorize(sph2cart)


def plot_steps(traj, color_selections=None, spherical=True, lims=None, sets=[[0, 1, 2]]):
    for step in range(len(traj)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if color_selections != None:
            for sel, col in color_selections:
                x = traj[step][0].diagonal()[sel[0]:sel[1]]
                y = traj[step][1].diagonal()[sel[0]:sel[1]]
                z = traj[step][2].diagonal()[sel[0]:sel[1]]
                x = np.real(x).astype(np.float64)
                y = np.real(y).astype(np.float64)
                z = np.real(z).astype(np.float64)
                norm = np.sqrt(x ** 2 + y ** 2 + z ** 2) ** -1
                # az,el,r = cart2sph_vec(x*norm,y*norm,z*norm)
                # x,y,z = sph2cart_vec(az,el,r)
                ax.scatter(x * norm, y * norm, z * norm, c=col)
            if lims != None:
                ax.set_xlim(lims[0][0], lims[0][1])
                ax.set_ylim(lims[1][0], lims[1][1])
                # ax.set_zlim(lims[2][0],lims[2][0])
            ax.set_title("Step" + ' ' + str(step))
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            # plt.show()
        else:
            x = traj[step:step + 1, :1, :, :][0][0][0]
            y = traj[step:step + 1, 1:2, :, :][0][0][0]
            z = traj[step:step + 1, 2:3, :, :][0][0][0]
            x = x.astype(np.float64)
            y = y.astype(np.float64)
            z = z.astype(np.float64)
            r, el, az = cart2sph_vec(x, y, z)
            x, y, z = sph2cart_vec(r, el, az)
            ax.scatter(x, y, z)
            if lims != None:
                ax.set_xlim(lims[0][0], lims[0][1])
                ax.set_ylim(lims[1][0], lims[1][1])
                # ax.set_zlim(lims[2][0],lims[2][0])
            #     ax = fig.add_subplot(111, projection='3d')

            ax.set_title("Step" + ' ' + str(step))
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            # plt.show()
        _ = plt.savefig('step_' + str(step) + '.png')
        # plt.show()
        _ = plt.clf()

        
def reverse_entropy(traj, N_traj):
    S = np.array(np.zeros(traj.shape[0]))

    for step in range(traj.shape[0]):
        N_step = np.matrix(np.real(N_traj[step]))
        psi = np.matrix(np.zeros([traj.shape[1],traj.shape[2]]))
        for d in range(traj.shape[1]):
            psi[d,:] = np.matrix(np.real(traj[step][d])).diagonal()


        p_i = psi * N_step.T * N_step * psi.T
        p_i = np.array([np.float(p_i[i]*p_i[i].T) for i in range(traj.shape[1])])

        for p in p_i:
            if p <= 0:
                pass
            S[step] = S[step] - 1/p * np.log(1/p)
    return S        
