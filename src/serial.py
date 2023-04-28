import numpy as np

def qr_algorithm(A, eps=1e-12, max_iter=1000):
    """
    Computes the eigenvalues and eigenvectors of a square matrix A using the QR algorithm.
    """
    n = A.shape[0]
    V = np.eye(n)

    for i in range(max_iter):
        # Compute the QR decomposition of A.
        Q = np.zeros((n, n))
        R = np.zeros((n, n))
        for j in range(n):
            Q[:, j] = A[:, j]
            for k in range(j):
                R[k, j] = np.dot(Q[:, k], A[:, j])
                Q[:, j] -= R[k, j] * Q[:, k]
            R[j, j] = np.linalg.norm(Q[:, j])
            Q[:, j] /= R[j, j]

        # Update A and V.
        A = R @ Q
        V = V @ Q

        # Check if the subdiagonal elements are small enough.
        subdiag = np.abs(A - np.triu(A, 1)).max()
        if subdiag < eps:
            break

    # Extract the eigenvalues from the diagonal of A.
    eigenvalues = np.diag(A)
    return eigenvalues, V

def lanczos_algorithm(A, k):
    n = A.shape[0]
    V = np.zeros((n, k))
    T = np.zeros((k, k))
    alpha = np.zeros(k)
    beta = np.zeros(k)
    w = np.zeros(n)

    for j in range(k): 
        beta[j] = np.linalg.norm(w)
        if(beta[j] != 0):
            v = w / beta[j]
        else:
            v = np.random.randn(n)
            v = v / np.linalg.norm(v)
        V[:, j] = v
        w_p = A @ v
        alpha[j] = np.dot(w_p.transpose(), v)
        w = w_p - alpha[j] * V[:, j] - beta[j] * V[:, j-1]

    for i in range(k):
        T[i, i] = alpha[i]
        if i < k - 1:
            T[i, i+1] = beta[i+1]
            T[i+1, i] = beta[i+1]
    # print("-----------------alpha--------------------")
    # print(alpha)
    # print("----------------beta---------------------")
    # print(beta)
    # print("---------------T----------------------")
    # print(T)
    # print("--------lap------------")
    # print(lap)        

    eigvals, eigvecs =qr_algorithm(T)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = V @ eigvecs[:, idx]

    return eigvals, eigvecs


# Create a random symmetric matrix
size = 10  # matrix dimension
adj = np.random.randint(2, size=(size, size))  # random binary matrix
adj = np.triu(adj, 1) + np.triu(adj, 1).T  # make symmetric
diag = np.diag(np.sum(adj, axis=1))  # create degree matrix
lap = diag - adj # create Laplacian matrix

evals, evecs = np.linalg.eigh(lap)
print(evals)
# print(evecs)

eigvals, eigvecs = lanczos_algorithm(lap, size)
print("Smallest eigenvalues: \n", eigvals, "\n")
print("Smallest eigenvectors: \n", eigvecs, "\n")

# Graph partitioning using eigen vector
lap_div_1 = lap.copy()
lap_div_2 = lap.copy()
nodes_set_1 = []
nodes_set_2 = []
deleted_1 = 0
deleted_2 = 0
for i in range(size):
    if eigvecs[i][1] < 0:
        nodes_set_2.append(i)
        lap_div_1 = np.delete(lap_div_1, i - deleted_1, axis=0)
        lap_div_1 = np.delete(lap_div_1, i - deleted_1, axis=1)
        deleted_1 += 1
    else:
        nodes_set_1.append(i)
        lap_div_2 = np.delete(lap_div_2, i - deleted_2, axis=0)
        lap_div_2 = np.delete(lap_div_2, i - deleted_2, axis=1)
        deleted_2 += 1
for i in range(lap_div_1.shape[0]):
    val = np.sum(lap_div_1[:, i])
    lap_div_1[i][i] = -1 * (val - lap_div_1[i][i])
for i in range(lap_div_2.shape[0]):
    val = np.sum(lap_div_2[:, i])
    lap_div_2[i][i] = -1 * (val - lap_div_2[i][i])

print("Original Laplacian matrix:\n", lap, "\n")
print("1st set of nodes:\n", nodes_set_1, "\n")
print("1st divided Laplacian matrix:\n", lap_div_1, "\n")
print("2nd set of nodes:\n", nodes_set_2, "\n")
print("2nd divided Laplacian matrix:\n", lap_div_2, "\n")
