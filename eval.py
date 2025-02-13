import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
import lsp

def main():
    np.random.seed(42)
    
    N = 500      
    M = 20       
    num_experiments = 10000
    sigma_noise = np.sqrt(0.01)  
    
    A = np.random.randn(N, M)
    x_true = np.random.randn(M)
    
    x_estimates = np.empty((num_experiments, M))
    all_residuals = []    
    residual_norms = np.empty(num_experiments)  
    
    for i in range(num_experiments):
        noise = np.random.normal(0, sigma_noise, size=N)
        b = A.dot(x_true) + noise
        
        x_est, cost, var = lsp.lstsq(A, b, method='ne')
        x_estimates[i, :] = x_est
        
        r = b - A.dot(x_est)
        all_residuals.extend(r)
        residual_norms[i] = np.dot(r, r)
    
    
    x_mean = np.mean(x_estimates, axis=0)
    x_std  = np.std(x_estimates, axis=0, ddof=1)
    cov_theoretical = 0.01 * np.linalg.inv(A.T.dot(A))
    std_theoretical = np.sqrt(np.diag(cov_theoretical))
    
    print("Истинный вектор x:")
    print(x_true)
    print("\nВыборочное среднее оценок x:")
    print(x_mean)
    print("\nВыборочные стандартные отклонения оценок x:")
    print(x_std)
    print("\nТеоретические стандартные отклонения оценок x:")
    print(std_theoretical)
    
    
    all_residuals = np.array(all_residuals)
    fig, ax = plt.subplots(figsize=(8,6))
    bins = 50
    ax.hist(all_residuals, bins=bins, density=True, alpha=0.6, 
            label='Экспериментальное распределение')
    

    xs = np.linspace(all_residuals.min(), all_residuals.max(), 200)
    ax.plot(xs, norm.pdf(xs, loc=0, scale=sigma_noise), 'r-', 
            label=f'N(0, {sigma_noise:.3f})')
    ax.set_xlabel("Невязка")
    ax.set_ylabel("Плотность вероятности")
    ax.set_title("Распределение компонент невязок")
    ax.legend()
    plt.savefig("norm.png")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(residual_norms, bins=bins, density=True, alpha=0.6, 
            label=r'Экспериментальное распределение $||r||^2$')
    
    df = N - M
    xs = np.linspace(residual_norms.min(), residual_norms.max(), 200)
    pdf_chi2 = chi2.pdf(xs / 0.01, df) / 0.01
    ax.plot(xs, pdf_chi2, 'r-', 
            label=f'chi² PDF (df={df}, scale=0.01)')
    ax.set_xlabel(r"$||r||^2$")
    ax.set_ylabel("Плотность вероятности")
    ax.set_title("Распределение суммы квадратов невязок")
    ax.legend()
    plt.savefig("chi2.png")
    plt.close()

if __name__ == "__main__":
    main()
