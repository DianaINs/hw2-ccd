import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from astropy.io import fits
import lsp

def main():
    if len(sys.argv) < 2:
        print("Usage: python ccd.py <path_to_ccd_v2.fits>")
        sys.exit(1)
        
    filename = sys.argv[1]
    print("Обрабатываем файл:", filename)
    
    with fits.open(filename) as hdul:
        data = hdul[0].data
    
    num_pairs = data.shape[0]
    
    bias_pair = data[0]
    bias_level = np.mean(bias_pair)
    
    x_vals = []
    sigma_delta_x2_vals = []
    for i in range(1, num_pairs):
        pair = data[i]  
        avg_signal = np.mean(pair) - bias_level
        diff = pair[0] - pair[1]
        sigma_delta_x2 = np.var(diff, ddof=1)
        x_vals.append(avg_signal)
        sigma_delta_x2_vals.append(sigma_delta_x2)
    
    x_vals = np.array(x_vals)
    sigma_delta_x2_vals = np.array(sigma_delta_x2_vals)
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(x_vals, sigma_delta_x2_vals, color='blue', label='Экспериментальные точки')
    
    A_matrix = np.column_stack((np.ones_like(x_vals), x_vals))
    b_vector = sigma_delta_x2_vals
    params, cost, cov = lsp.lstsq(A_matrix, b_vector, method='ne')
    A_est, B_est = params
    x_fit = np.linspace(x_vals.min(), x_vals.max(), 200)
    y_fit = A_est + B_est * x_fit
    ax.plot(x_fit, y_fit, 'r-', label='Линейная модель')
    ax.set_xlabel('Средний сигнал x ')
    ax.set_ylabel(r'$\sigma_{\Delta x}^2$')
    ax.legend()
    ax.set_title(r'Зависимость $\sigma_{\Delta x}^2$ от сигнала')
    plt.savefig('ccd.png')
    plt.close()
    
    g_est = B_est / 2.0
    sigma_r_est = np.sqrt(A_est / (2 * g_est**2))
    
    var_A = cov[0, 0]
    var_B = cov[1, 1]
    cov_AB = cov[0, 1]
    
    g_err = 0.5 * np.sqrt(var_B)
    
    dsr_dA = 1.0 / (B_est * np.sqrt(2 * A_est)) if A_est > 0 and B_est != 0 else 0
    dsr_dB = -np.sqrt(2 * A_est) / (B_est**2) if B_est != 0 else 0
    sigma_r_err = np.sqrt((dsr_dA**2)*var_A + (dsr_dB**2)*var_B + 2*dsr_dA*dsr_dB*cov_AB)
 
    n_data = len(x_vals)
    n_boot = 10000
    g_boot = np.empty(n_boot)
    sigma_r_boot = np.empty(n_boot)
    rng = np.random.default_rng(42)
    for i in range(n_boot):
        indices = rng.integers(0, n_data, n_data) 
        x_sample = x_vals[indices]
        y_sample = sigma_delta_x2_vals[indices]
        A_sample = np.column_stack((np.ones_like(x_sample), x_sample))
        params_sample, cost_sample, _ = lsp.lstsq(A_sample, y_sample, method='ne')
        A_sample_est, B_sample_est = params_sample
        g_sample = B_sample_est / 2.0
       
        if A_sample_est > 0 and g_sample > 0:
            sigma_r_sample = np.sqrt(A_sample_est / (2 * g_sample**2))
        else:
            sigma_r_sample = np.nan
        g_boot[i] = g_sample
        sigma_r_boot[i] = sigma_r_sample
    
    valid = ~np.isnan(sigma_r_boot)
    g_boot = g_boot[valid]
    sigma_r_boot = sigma_r_boot[valid]
    
    g_err_bootstrap = np.std(g_boot, ddof=1)
    sigma_r_err_bootstrap = np.std(sigma_r_boot, ddof=1)
    
    results = {
        "ron": sigma_r_est,
        "ron_err": sigma_r_err,
        "ron_err_bootstrap": sigma_r_err_bootstrap,
        "gain": g_est,
        "gain_err": g_err,
        "gain_err_bootstrap": g_err_bootstrap
    }
    
    with open("ccd.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("Результаты сохранены в файле ccd.json")

if __name__ == '__main__':
    main()



   
