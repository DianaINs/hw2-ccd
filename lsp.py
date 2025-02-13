import numpy as np

def lstsq_ne(A, b):
    ATA = A.T @ A  
    ATb = A.T @ b  
    x = np.linalg.solve(ATA, ATb)  
    residuals = A @ x - b
    cost = np.sum(residuals ** 2)  
    var = np.linalg.inv(ATA)  
    return x, cost, var

def lstsq_svd(A, b, rcond=None):
    
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_inv = np.zeros_like(S)
    mask = S > (rcond * np.max(S)) if rcond else np.ones_like(S, dtype=bool)
    S_inv[mask] = 1 / S[mask]
    x = Vt.T @ np.diag(S_inv) @ U.T @ b
    residuals = A @ x - b
    cost = np.sum(residuals ** 2)

    S2_inv = np.zeros_like(S)
    S2_inv[mask] = 1 / (S[mask] ** 2)
    var = Vt.T @ np.diag(S2_inv) @ Vt
    return x, cost, var

def lstsq(A, b, method="ne", **kwargs):
   
    if method == "ne":
        return lstsq_ne(A, b)
    elif method == "svd":
        return lstsq_svd(A, b, **kwargs)
    else:
        raise ValueError("Invalid method. Choose 'ne' or 'svd'")


