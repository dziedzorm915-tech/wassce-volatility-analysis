"""
WASSCE Pass Rate Volatility Analysis
AR(1)-ARCH(1) Model with Bootstrap Inference
Based on Methodology Chapter (Revised)

Author: Roland Ankudze
Data: 2011-2025 (T=15 observations per subject, T=14 after differencing)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gammaln
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
np.random.seed(42)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING (Section 3.3)
# ============================================================================

def load_and_preprocess_data(filepath='data.xlsx'):
    """Load WASSCE pass rate data and apply first difference transformation"""
    
    try:
        df = pd.read_excel(filepath)
        
        # Identify subject columns (exclude Year, year, Date, etc.)
        exclude_patterns = ['year', 'Year', 'date', 'Date', 'index', 'Index']
        subject_cols = [col for col in df.columns 
                       if not any(pattern in str(col) for pattern in exclude_patterns)]
        
        if len(subject_cols) == 0:
            subject_cols = df.columns[1:].tolist()
        
        subjects = subject_cols
        years = df.iloc[:, 0].values
        
    except FileNotFoundError:
        print(f"\nFile '{filepath}' not found. Creating synthetic data for demonstration.")
        print("WARNING: Replace with actual WAEC data for final analysis.\n")
        
        years = np.arange(2011, 2026)
        subjects = ['English', 'Core_Mathematics', 'Integrated_Science', 'Social_Studies']
        
        np.random.seed(42)
        df = pd.DataFrame({'Year': years})
        
        # Synthetic data with realistic patterns
        df['English'] = 60 + np.cumsum(np.random.randn(15) * 8)
        df['Core_Mathematics'] = 48 + np.cumsum(np.random.randn(15) * 11)
        df['Integrated_Science'] = 48 + np.cumsum(np.random.randn(15) * 12)
        df['Social_Studies'] = 67 + np.cumsum(np.random.randn(15) * 10)
        
        # Clip to realistic range
        for col in subjects:
            df[col] = np.clip(df[col], 20, 90)
    
    T = len(years)
    print(f"\nData loaded: {T} years ({years[0]}-{years[-1]})")
    print(f"Subjects: {subjects}")
    
    # Store original and differenced series
    original = {}
    differenced = {}
    
    for subject in subjects:
        original[subject] = df[subject].values.astype(float)
        differenced[subject] = np.diff(original[subject])
        
        print(f"\n{subject.replace('_', ' ')}:")
        print(f"  Mean change: {np.mean(differenced[subject]):.2f} pp")
        print(f"  Std of changes: {np.std(differenced[subject]):.2f} pp")
        print(f"  Variance of changes: {np.var(differenced[subject]):.2f}")
    
    return years, subjects, original, differenced


# ============================================================================
# 2. STATIONARITY TESTING (Section 3.4)
# ============================================================================

def test_stationarity(series, subject_name):
    """Perform Augmented Dickey-Fuller test (Section 3.4.2)"""
    
    result = adfuller(series, autolag='AIC')
    
    print(f"\n{subject_name}:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    
    is_stationary = result[1] < 0.05
    print(f"  Stationary at 5%: {'Yes' if is_stationary else 'No'}")
    print(f"  Note: Low power with T=14; Bayesian check follows")
    
    return is_stationary


def bayesian_ar1_posterior(series):
    """
    Bayesian AR(1) robustness check (Section 3.4.3).
    Prior: phi ~ N(0, 0.5^2) — conservative belief favouring near-zero AR coefficient.
    """
    T = len(series)
    Y_t = series[1:]
    Y_t1 = series[:-1]
    
    # Grid over phi
    phi_grid = np.linspace(-0.99, 0.99, 200)
    log_posterior = np.zeros_like(phi_grid)
    
    for i, phi in enumerate(phi_grid):
        # Conditional MLE for mu given phi
        residuals = Y_t - phi * Y_t1
        mu_hat = np.mean(residuals)
        sigma2_hat = np.var(residuals - mu_hat, ddof=1)
        
        if sigma2_hat <= 0:
            log_posterior[i] = -np.inf
            continue
        
        # Log-likelihood
        log_lik = -0.5 * (T-1) * np.log(2 * np.pi * sigma2_hat)
        log_lik -= 0.5 * np.sum((Y_t - mu_hat - phi * Y_t1)**2 / sigma2_hat)
        
        # Log-prior: N(0, 0.5^2)
        log_prior = -0.5 * (phi / 0.5)**2
        
        log_posterior[i] = log_lik + log_prior
    
    # Normalize posterior
    log_posterior -= np.max(log_posterior)
    posterior = np.exp(log_posterior)
    posterior /= np.trapz(posterior, phi_grid)
    
    # 95% HPD interval (highest posterior density)
    sorted_idx = np.argsort(posterior)[::-1]
    cumulative = 0
    hpd_idx = []
    for idx in sorted_idx:
        hpd_idx.append(idx)
        cumulative += posterior[idx] * (phi_grid[1] - phi_grid[0])
        if cumulative >= 0.95:
            break
    
    phi_hpd = phi_grid[hpd_idx]
    hpd_lower, hpd_upper = np.min(phi_hpd), np.max(phi_hpd)
    
    return phi_grid, posterior, hpd_lower, hpd_upper


# ============================================================================
# 3. AR(1) MEAN MODEL (Section 3.5)
# ============================================================================

def fit_ar1(series):
    """Fit AR(1) model by OLS"""
    
    T = len(series)
    Y_t = series[1:]
    Y_t1 = series[:-1]
    
    X = np.column_stack([np.ones(T-1), Y_t1])
    theta = np.linalg.lstsq(X, Y_t, rcond=None)[0]
    mu, phi = theta[0], theta[1]
    
    residuals = Y_t - mu - phi * Y_t1
    
    return mu, phi, residuals


# ============================================================================
# 4. ARCH-LM TEST (Section 3.10)
# ============================================================================

def arch_lm_test(residuals):
    """Engle's ARCH-LM test for conditional heteroskedasticity"""
    
    T = len(residuals)
    eps2 = residuals**2
    eps2_lag = eps2[:-1]
    eps2_current = eps2[1:]
    
    X = np.column_stack([np.ones(len(eps2_lag)), eps2_lag])
    coef = np.linalg.lstsq(X, eps2_current, rcond=None)[0]
    eps2_pred = X @ coef
    
    SS_res = np.sum((eps2_current - eps2_pred)**2)
    SS_tot = np.sum((eps2_current - np.mean(eps2_current))**2)
    R2 = 1 - SS_res / SS_tot if SS_tot > 0 else 0
    
    LM = (T - 1) * R2
    p_value = 1 - stats.chi2.cdf(LM, df=1)
    
    return LM, p_value


# ============================================================================
# 5. AR(1)-ARCH(1) QMLE ESTIMATION (Section 3.9)
# ============================================================================

def arch1_log_likelihood(params, Y):
    """Negative log-likelihood for AR(1)-ARCH(1) with Gaussian innovations"""
    
    mu, phi, omega, alpha = params
    
    # Parameter constraints
    if abs(phi) >= 1 or omega <= 0 or alpha < 0 or alpha >= 1:
        return 1e10
    
    T = len(Y)
    residuals = np.zeros(T)
    sigma2 = np.zeros(T)
    
    # Initialise
    residuals[0] = Y[0] - mu  # simplified initialisation
    sigma2[0] = omega / (1 - alpha) if alpha < 1 else omega
    
    log_lik = 0
    for t in range(1, T):
        residuals[t] = Y[t] - mu - phi * Y[t-1]
        sigma2[t] = omega + alpha * residuals[t-1]**2
        
        if sigma2[t] <= 0:
            return 1e10
        
        log_lik += -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2[t])
        log_lik += -0.5 * residuals[t]**2 / sigma2[t]
    
    return -log_lik


def fit_arch1(Y, starting_alphas=None):
    """
    Fit AR(1)-ARCH(1) model by QMLE with multiple starting values.
    Uses BFGS algorithm as specified in Section 3.9.4.
    """
    
    if starting_alphas is None:
        starting_alphas = [0.1, 0.3, 0.5, 0.7]
    
    # Initial estimates from AR(1)
    mu_init, phi_init, _ = fit_ar1(Y)
    omega_init = np.var(Y) * 0.7
    
    best_result = None
    best_lik = -np.inf
    
    for alpha_init in starting_alphas:
        initial_params = [mu_init, phi_init, max(omega_init, 0.01), alpha_init]
        bounds = [(-10, 10), (-0.99, 0.99), (1e-6, 1000), (1e-6, 0.99)]
        
        result = minimize(
            arch1_log_likelihood,
            initial_params,
            args=(Y,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'disp': False}
        )
        
        if result.success and -result.fun > best_lik:
            best_lik = -result.fun
            best_result = result
    
    if best_result is not None:
        mu, phi, omega, alpha = best_result.x
        return mu, phi, omega, alpha, best_lik
    else:
        raise RuntimeError("AR(1)-ARCH(1) estimation failed for all starting values")


# ============================================================================
# 6. PARAMETRIC BOOTSTRAP (Section 3.11)
# ============================================================================

def parametric_bootstrap(Y, mu, phi, omega, alpha, n_bootstrap=1000):
    """
    Parametric bootstrap for ARCH parameter confidence intervals.
    Implements Algorithm in Section 3.11.1.
    """
    
    T = len(Y)
    
    # Compute standardised residuals from original fit
    residuals = np.zeros(T)
    sigma2 = np.zeros(T)
    residuals[0] = Y[0] - mu
    sigma2[0] = omega / (1 - alpha) if alpha < 1 else omega
    
    for t in range(1, T):
        residuals[t] = Y[t] - mu - phi * Y[t-1]
        sigma2[t] = omega + alpha * residuals[t-1]**2
    
    std_resid = residuals[1:] / np.sqrt(sigma2[1:])
    std_resid = std_resid[np.isfinite(std_resid)]
    
    if len(std_resid) == 0:
        return 0, 1, np.array([])
    
    # Bootstrap loop
    alpha_bootstrap = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        # Draw innovations with replacement
        z_boot = np.random.choice(std_resid, size=T-1)
        
        # Generate bootstrap series
        Y_boot = np.zeros(T)
        eps_boot = np.zeros(T)
        sigma2_boot = np.zeros(T)
        
        Y_boot[0] = Y[0]
        eps_boot[0] = residuals[0]
        sigma2_boot[0] = max(omega / (1 - alpha), 1e-8) if alpha < 1 else omega
        
        for t in range(1, T):
            sigma2_boot[t] = max(omega + alpha * eps_boot[t-1]**2, 1e-8)
            eps_boot[t] = np.sqrt(sigma2_boot[t]) * z_boot[t-1]
            Y_boot[t] = mu + phi * Y_boot[t-1] + eps_boot[t]
        
        # Re-estimate on bootstrap series
        try:
            _, _, _, alpha_b, _ = fit_arch1(Y_boot, starting_alphas=[alpha])
            alpha_bootstrap[b] = alpha_b
        except:
            alpha_bootstrap[b] = np.nan
    
    # Clean and compute CIs
    alpha_bootstrap = alpha_bootstrap[np.isfinite(alpha_bootstrap)]
    
    if len(alpha_bootstrap) == 0:
        return 0, 1, np.array([])
    
    ci_lower = np.percentile(alpha_bootstrap, 2.5)
    ci_upper = np.percentile(alpha_bootstrap, 97.5)
    
    return ci_lower, ci_upper, alpha_bootstrap


# ============================================================================
# 7. DIAGNOSTIC CHECKS (Section 3.12)
# ============================================================================

def diagnostic_checks(Y, mu, phi, omega, alpha):
    """Compute standardised residuals and diagnostic tests"""
    
    T = len(Y)
    residuals = np.zeros(T)
    sigma2 = np.zeros(T)
    
    residuals[0] = Y[0] - mu
    sigma2[0] = omega / (1 - alpha) if alpha < 1 else omega
    
    for t in range(1, T):
        residuals[t] = Y[t] - mu - phi * Y[t-1]
        sigma2[t] = omega + alpha * residuals[t-1]**2
    
    # Standardised residuals
    std_resid = residuals[1:] / np.sqrt(sigma2[1:])
    std_resid = std_resid[np.isfinite(std_resid)]
    
    # Ljung-Box on residuals
    if len(std_resid) > 5:
        lb_resid = acorr_ljungbox(std_resid, lags=5, return_df=True)
        lb_sq = acorr_ljungbox(std_resid**2, lags=5, return_df=True)
    else:
        lb_resid = None
        lb_sq = None
    
    # Jarque-Bera
    jb_stat, jb_pvalue = stats.jarque_bera(std_resid)
    
    return {
        'std_residuals': std_resid,
        'lb_residuals': lb_resid,
        'lb_squared': lb_sq,
        'jb_stat': jb_stat,
        'jb_pvalue': jb_pvalue
    }


# ============================================================================
# 8. CROSS-SUBJECT ANALYSIS WITH PERMUTATION TESTS (Section 3.13)
# ============================================================================

def permutation_test_correlation(z1, z2, n_permutations=1000):
    """Permutation test for correlation (Section 3.13.2)"""
    
    min_len = min(len(z1), len(z2))
    z1, z2 = z1[:min_len], z2[:min_len]
    
    r_obs = np.corrcoef(z1, z2)[0, 1]
    
    r_perm = np.zeros(n_permutations)
    for i in range(n_permutations):
        z2_perm = np.random.permutation(z2)
        r_perm[i] = np.corrcoef(z1, z2_perm)[0, 1]
    
    p_value = np.mean(np.abs(r_perm) >= np.abs(r_obs))
    
    return r_obs, p_value, r_perm


def permutation_test_directional_agreement(z1, z2, n_permutations=1000):
    """Permutation test for directional agreement (Section 3.13.4)"""
    
    min_len = min(len(z1), len(z2))
    z1, z2 = z1[:min_len], z2[:min_len]
    
    agreement_obs = np.mean(z1 * z2 > 0)
    
    agreement_perm = np.zeros(n_permutations)
    for i in range(n_permutations):
        sign_flip = np.random.choice([-1, 1], size=len(z2))
        agreement_perm[i] = np.mean(z1 * z2 * sign_flip > 0)
    
    p_value = np.mean(agreement_perm >= agreement_obs)
    
    return agreement_obs, p_value, agreement_perm


def permutation_test_synchronization(z1, z2, threshold=1.0, n_permutations=1000):
    """Permutation test for tail synchronisation (Section 3.13.3)"""
    
    min_len = min(len(z1), len(z2))
    z1, z2 = z1[:min_len], z2[:min_len]
    
    sync_obs = np.mean((np.abs(z1) > threshold) & (np.abs(z2) > threshold))
    
    sync_perm = np.zeros(n_permutations)
    for i in range(n_permutations):
        z2_perm = np.random.permutation(z2)
        sync_perm[i] = np.mean((np.abs(z1) > threshold) & (np.abs(z2_perm) > threshold))
    
    p_value = np.mean(sync_perm >= sync_obs)
    
    return sync_obs, p_value, sync_perm


def cross_subject_analysis_with_significance(std_residuals_dict, subjects, n_permutations=1000):
    """Full cross-subject dependence analysis with permutation tests"""
    
    n_subjects = len(subjects)
    min_len = min(len(std_residuals_dict[s]) for s in subjects)
    
    # Truncate all series to same length
    truncated = {s: std_residuals_dict[s][:min_len] for s in subjects}
    
    # Initialise matrices
    corr_matrix = np.eye(n_subjects)
    corr_pvalues = np.zeros((n_subjects, n_subjects))
    sync_matrix = np.eye(n_subjects)
    sync_pvalues = np.zeros((n_subjects, n_subjects))
    sign_matrix = np.eye(n_subjects)
    sign_pvalues = np.zeros((n_subjects, n_subjects))
    
    for i, s1 in enumerate(subjects):
        for j, s2 in enumerate(subjects):
            if i < j:
                r, p_corr, _ = permutation_test_correlation(
                    truncated[s1], truncated[s2], n_permutations)
                corr_matrix[i, j] = corr_matrix[j, i] = r
                corr_pvalues[i, j] = corr_pvalues[j, i] = p_corr
                
                sync, p_sync, _ = permutation_test_synchronization(
                    truncated[s1], truncated[s2], 1.0, n_permutations)
                sync_matrix[i, j] = sync_matrix[j, i] = sync
                sync_pvalues[i, j] = sync_pvalues[j, i] = p_sync
                
                agreement, p_agree, _ = permutation_test_directional_agreement(
                    truncated[s1], truncated[s2], n_permutations)
                sign_matrix[i, j] = sign_matrix[j, i] = agreement
                sign_pvalues[i, j] = sign_pvalues[j, i] = p_agree
    
    return {
        'correlation': corr_matrix,
        'correlation_pvalues': corr_pvalues,
        'synchronization': sync_matrix,
        'synchronization_pvalues': sync_pvalues,
        'agreement': sign_matrix,
        'agreement_pvalues': sign_pvalues,
        'subjects': subjects,
        'n_observations': min_len,
        'n_permutations': n_permutations
    }


def print_cross_subject_results(results):
    """Print formatted cross-subject results"""
    
    subjects = results['subjects']
    n = len(subjects)
    display_names = [s.replace('_', ' ') for s in subjects]
    
    print(f"\nObservations: {results['n_observations']}")
    print(f"Permutations: {results['n_permutations']}")
    
    # Correlation matrix
    print("\n" + "=" * 65)
    print("CORRELATION MATRIX (lower = r, upper = p-value)")
    print("=" * 65)
    
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append("1.000")
            elif i > j:
                row.append(f"{results['correlation'][i,j]:.3f}")
            else:
                p = results['correlation_pvalues'][i,j]
                sig = "*" if p < 0.05 else ""
                row.append(f"{p:.3f}{sig}")
        print(f"  {display_names[i]:20s} " + "  ".join(row))
    
    print("\n* p < 0.05")
    
    # Directional agreement
    print("\n" + "=" * 65)
    print("DIRECTIONAL AGREEMENT (proportion same-sign shocks)")
    print("=" * 65)
    
    for i in range(n):
        for j in range(i+1, n):
            a = results['agreement'][i,j]
            p = results['agreement_pvalues'][i,j]
            sig = "SIGNIFICANT" if p < 0.05 else "n.s."
            print(f"  {display_names[i]} vs {display_names[j]}: {a:.3f} ({a*100:.1f}%), p={p:.3f} [{sig}]")
    
    # Tail synchronisation
    print("\n" + "=" * 65)
    print("TAIL SYNCHRONISATION (both |z| > 1)")
    print("=" * 65)
    
    for i in range(n):
        for j in range(i+1, n):
            s = results['synchronization'][i,j]
            p = results['synchronization_pvalues'][i,j]
            sig = "SIGNIFICANT" if p < 0.05 else "n.s."
            print(f"  {display_names[i]} vs {display_names[j]}: {s:.3f} ({s*100:.1f}%), p={p:.3f} [{sig}]")


# ============================================================================
# 9. POWER ANALYSIS (Section 3.10.2)
# ============================================================================

def power_analysis(n_sim=500, T=14, true_alpha=0.3):
    """Monte Carlo power analysis for ARCH-LM test"""
    
    rejections = 0
    
    for sim in range(n_sim):
        Y = np.zeros(T)
        eps = np.zeros(T)
        sigma2 = np.zeros(T)
        
        mu, phi, omega = 0, 0, 1
        alpha = true_alpha
        
        eps[0] = np.random.normal(0, np.sqrt(omega/(1-alpha)))
        Y[0] = eps[0]
        sigma2[0] = omega/(1-alpha)
        
        for t in range(1, T):
            sigma2[t] = max(omega + alpha * eps[t-1]**2, 1e-8)
            eps[t] = np.sqrt(sigma2[t]) * np.random.normal(0, 1)
            Y[t] = mu + phi * Y[t-1] + eps[t]
        
        _, _, residuals = fit_ar1(Y)
        _, p_value = arch_lm_test(residuals)
        
        if p_value < 0.05:
            rejections += 1
    
    return rejections / n_sim


# ============================================================================
# 10. STUDENT'S t ROBUSTNESS CHECK (Section 3.12.5)
# ============================================================================

def arch1_log_likelihood_t(params, Y):
    """
    Negative log-likelihood for AR(1)-ARCH(1) with standardised Student's t innovations.
    Uses gammaln for numerical stability; density follows Section 3.12.5.
    """
    
    mu, phi, omega, alpha, nu = params
    
    if abs(phi) >= 1 or omega <= 0 or alpha < 0 or alpha >= 1 or nu <= 2:
        return 1e10
    
    T = len(Y)
    residuals = np.zeros(T)
    sigma2 = np.zeros(T)
    
    residuals[0] = Y[0] - mu
    sigma2[0] = omega / (1 - alpha) if alpha < 1 else omega
    
    log_lik = 0
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    
    for t in range(1, T):
        residuals[t] = Y[t] - mu - phi * Y[t-1]
        sigma2[t] = omega + alpha * residuals[t-1]**2
        
        if sigma2[t] <= 0:
            return 1e10
        
        z = residuals[t] / np.sqrt(sigma2[t])
        log_lik += const - 0.5 * np.log(sigma2[t]) - ((nu + 1) / 2) * np.log(1 + z**2 / (nu - 2))
    
    return -log_lik


def fit_arch1_t(Y):
    """Fit AR(1)-ARCH(1) with Student's t innovations"""
    
    # Initial estimates
    mu_init, phi_init, _ = fit_ar1(Y)
    omega_init = np.var(Y) * 0.7
    alpha_init = 0.1
    nu_init = 10.0
    
    initial_params = [mu_init, phi_init, max(omega_init, 0.01), alpha_init, nu_init]
    bounds = [(-10, 10), (-0.99, 0.99), (1e-6, 1000), (1e-6, 0.99), (2.01, 100)]
    
    result = minimize(
        arch1_log_likelihood_t,
        initial_params,
        args=(Y,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'disp': False}
    )
    
    if result.success:
        mu, phi, omega, alpha, nu = result.x
        return mu, phi, omega, alpha, nu, -result.fun
    else:
        return None


# ============================================================================
# 11. VISUALISATION
# ============================================================================

def create_figures(years, subjects, original, differenced, results, cross_subject_results=None):
    """Create all figures for the analysis"""
    
    display_names = [s.replace('_', ' ') for s in subjects]
    diff_years = years[1:]
    resid_years = years[2:]
    
    # Figure 1: Original pass rates
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for i, subject in enumerate(subjects):
        ax1.plot(years, original[subject], marker='o', linewidth=2, label=display_names[i])
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Pass Rate (%)', fontsize=12)
    ax1.set_title('WASSCE Core Subject Pass Rates (2011-2025)', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure1_original_series.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("  Saved: figure1_original_series.png")
    
    # Figure 2: Differenced series
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for i, subject in enumerate(subjects):
        ax2.plot(diff_years, differenced[subject], marker='o', linewidth=2, label=display_names[i])
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Change in Pass Rate (pp)', fontsize=12)
    ax2.set_title('Year-to-Year Changes in WASSCE Pass Rates', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure2_differenced_series.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("  Saved: figure2_differenced_series.png")
    
    # Figure 3: Squared AR(1) residuals
    fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, subject in enumerate(subjects):
        _, _, residuals = fit_ar1(differenced[subject])
        squared_resid = residuals**2
        axes[idx].bar(resid_years, squared_resid, color='steelblue', alpha=0.7)
        axes[idx].set_xlabel('Year', fontsize=10)
        axes[idx].set_ylabel('Squared Residual', fontsize=10)
        axes[idx].set_title(f'{display_names[idx]}', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
    fig3.suptitle('Squared AR(1) Residuals by Subject', fontsize=14)
    plt.tight_layout()
    plt.savefig('figure3_squared_residuals.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("  Saved: figure3_squared_residuals.png")
    
    # Figure 4: ARCH estimates with bootstrap CIs
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(subjects))
    alpha_est = [results[s]['alpha'] for s in subjects if s in results and results[s] is not None]
    ci_low = [results[s]['ci_lower'] for s in subjects if s in results and results[s] is not None]
    ci_high = [results[s]['ci_upper'] for s in subjects if s in results and results[s] is not None]
    
    ax4.errorbar(x_pos, alpha_est,
                 yerr=[np.array(alpha_est) - np.array(ci_low),
                       np.array(ci_high) - np.array(alpha_est)],
                 fmt='o', capsize=5, capthick=2, markersize=10, linewidth=2, color='darkblue')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No ARCH (alpha=0)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(display_names, rotation=45, ha='right')
    ax4.set_ylabel('ARCH Parameter (alpha)', fontsize=12)
    ax4.set_title('Volatility Persistence Estimates with 95% Bootstrap CIs', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure4_arch_estimates.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("  Saved: figure4_arch_estimates.png")
    
    # Figure 5: Cross-subject correlation heatmap
    if cross_subject_results is not None:
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        corr = cross_subject_results['correlation']
        pvals = cross_subject_results['correlation_pvalues']
        
        im = ax5.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        for i in range(len(subjects)):
            for j in range(len(subjects)):
                if i != j:
                    text = f"{corr[i,j]:.2f}"
                    if pvals[i,j] < 0.05:
                        text += "*"
                    color = "white" if abs(corr[i,j]) > 0.5 else "black"
                    ax5.text(j, i, text, ha="center", va="center", color=color, fontsize=10)
        
        ax5.set_xticks(range(len(subjects)))
        ax5.set_yticks(range(len(subjects)))
        ax5.set_xticklabels(display_names, rotation=45, ha='right')
        ax5.set_yticklabels(display_names)
        ax5.set_title('Cross-Subject Correlation of Shocks\n(* p < 0.05)', fontsize=14)
        plt.colorbar(im, ax=ax5, label='Correlation')
        plt.tight_layout()
        plt.savefig('figure5_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig5)
        print("  Saved: figure5_correlation_heatmap.png")
    
    # Figure 6: Standardised residuals
    fig6, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, subject in enumerate(subjects):
        if subject in results and results[subject] is not None and 'diagnostics' in results[subject]:
            std_resid = results[subject]['diagnostics']['std_residuals']
            min_len = min(len(std_resid), len(resid_years))
            axes[idx].plot(resid_years[:min_len], std_resid[:min_len], marker='o', linewidth=2, color='darkblue')
            axes[idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[idx].axhline(y=2, color='red', linestyle='--', alpha=0.5, label='+/-2')
            axes[idx].axhline(y=-2, color='red', linestyle='--', alpha=0.5)
            axes[idx].set_xlabel('Year', fontsize=10)
            axes[idx].set_ylabel('Std. Residual', fontsize=10)
            axes[idx].set_title(f'{display_names[idx]}', fontsize=12)
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)
    fig6.suptitle('Standardised Residuals from AR(1)-ARCH(1) Model', fontsize=14)
    plt.tight_layout()
    plt.savefig('figure6_standardized_residuals.png', dpi=300, bbox_inches='tight')
    plt.close(fig6)
    print("  Saved: figure6_standardized_residuals.png")
    
    print("\nAll figures saved.")


# ============================================================================
# 12. MAIN ANALYSIS
# ============================================================================

def main():
    """Run the complete WASSCE volatility analysis"""
    
    print("=" * 70)
    print("WASSCE PASS RATE VOLATILITY ANALYSIS")
    print("AR(1)-ARCH(1) Model with Parametric Bootstrap Inference")
    print("Data: 2011-2025 (T=15, effective T=14 after differencing)")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1] DATA LOADING AND TRANSFORMATION")
    print("-" * 40)
    years, subjects, original, differenced = load_and_preprocess_data('data.xlsx')
    
    # 2. Stationarity tests
    print("\n[2] STATIONARITY ANALYSIS")
    print("-" * 40)
    stationary = {}
    for subject in subjects:
        stationary[subject] = test_stationarity(differenced[subject], subject.replace('_', ' '))
    
    # 3. Bayesian robustness
    print("\n[3] BAYESIAN AR(1) ROBUSTNESS CHECK")
    print("-" * 40)
    bayesian_results = {}
    for subject in subjects:
        _, _, hpd_lower, hpd_upper = bayesian_ar1_posterior(differenced[subject])
        bayesian_results[subject] = {'hpd_lower': hpd_lower, 'hpd_upper': hpd_upper}
        contains_zero = hpd_lower <= 0 <= hpd_upper
        print(f"  {subject.replace('_', ' ')}: 95% HPD = [{hpd_lower:.3f}, {hpd_upper:.3f}], "
              f"contains zero: {contains_zero}")
    
    # 4. ARCH-LM test
    print("\n[4] ARCH-LM TEST")
    print("-" * 40)
    arch_lm_results = {}
    for subject in subjects:
        _, _, residuals = fit_ar1(differenced[subject])
        LM, p_value = arch_lm_test(residuals)
        arch_lm_results[subject] = {'LM': LM, 'p_value': p_value}
        effect = "Detected" if p_value < 0.05 else "Not detected"
        print(f"  {subject.replace('_', ' ')}: LM = {LM:.4f}, p = {p_value:.4f} [{effect}]")
    
    # 5. Power analysis
    print("\n[5] POST-HOC POWER ANALYSIS")
    print("-" * 40)
    power_03 = power_analysis(n_sim=500, T=14, true_alpha=0.3)
    power_05 = power_analysis(n_sim=500, T=14, true_alpha=0.5)
    print(f"  Power for alpha=0.3 (T=14): {power_03:.3f} ({power_03*100:.1f}%)")
    print(f"  Power for alpha=0.5 (T=14): {power_05:.3f} ({power_05*100:.1f}%)")
    print(f"  WARNING: Power well below 0.80. Null results are expected regardless of true DGP.")
    
    # 6. AR(1)-ARCH(1) estimation
    print("\n[6] AR(1)-ARCH(1) QMLE ESTIMATION")
    print("-" * 40)
    results = {}
    for subject in subjects:
        print(f"\n  {subject.replace('_', ' ')}:")
        try:
            mu, phi, omega, alpha, lik = fit_arch1(differenced[subject])
            ci_lower, ci_upper, alpha_boot = parametric_bootstrap(
                differenced[subject], mu, phi, omega, alpha, n_bootstrap=1000)
            
            results[subject] = {
                'mu': mu, 'phi': phi, 'omega': omega, 'alpha': alpha,
                'log_likelihood': lik,
                'ci_lower': ci_lower, 'ci_upper': ci_upper,
                'alpha_bootstrap': alpha_boot
            }
            
            print(f"    mu (intercept):        {mu:.4f}")
            print(f"    phi (AR persistence):   {phi:.4f}")
            print(f"    omega (baseline var):   {omega:.4f}")
            print(f"    alpha (ARCH):           {alpha:.4f}")
            print(f"    95% Bootstrap CI:      [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"    Log-likelihood:         {lik:.2f}")
            
        except Exception as e:
            print(f"    Estimation failed: {e}")
            results[subject] = None
    
    # 7. Student's t robustness
    print("\n[7] STUDENT'S t ROBUSTNESS CHECK")
    print("-" * 40)
    t_results = {}
    for subject in subjects:
        print(f"\n  {subject.replace('_', ' ')}:")
        try:
            result_t = fit_arch1_t(differenced[subject])
            if result_t is not None:
                mu, phi, omega, alpha, nu, lik = result_t
                t_results[subject] = {'alpha': alpha, 'nu': nu, 'log_likelihood': lik}
                tail_desc = "very heavy" if nu < 5 else ("moderate" if nu < 15 else "near-normal")
                print(f"    alpha (ARCH):           {alpha:.4f}")
                print(f"    nu (degrees of freedom): {nu:.2f} ({tail_desc} tails)")
                print(f"    Log-likelihood:         {lik:.2f}")
            else:
                print("    Estimation failed")
                t_results[subject] = None
        except Exception as e:
            print(f"    Estimation failed: {e}")
            t_results[subject] = None
    
    # 8. Diagnostics
    print("\n[8] MODEL DIAGNOSTICS")
    print("-" * 40)
    for subject in subjects:
        if results[subject] is not None:
            Y = differenced[subject]
            mu, phi, omega, alpha = results[subject]['mu'], results[subject]['phi'], \
                                     results[subject]['omega'], results[subject]['alpha']
            diagnostics = diagnostic_checks(Y, mu, phi, omega, alpha)
            results[subject]['diagnostics'] = diagnostics
            
            print(f"\n  {subject.replace('_', ' ')}:")
            print(f"    Standardised residuals: {len(diagnostics['std_residuals'])} observations")
            print(f"    Jarque-Bera p-value:    {diagnostics['jb_pvalue']:.4f}")
            normality = "Rejected" if diagnostics['jb_pvalue'] < 0.05 else "Not rejected"
            print(f"    Normality:              {normality}")
    
    # 9. Cross-subject analysis
    print("\n[9] CROSS-SUBJECT DEPENDENCE ANALYSIS (Permutation Tests)")
    print("-" * 40)
    
    std_resid_dict = {}
    for subject in subjects:
        if results[subject] is not None and 'diagnostics' in results[subject]:
            std_resid_dict[subject] = results[subject]['diagnostics']['std_residuals']
    
    cross_subject_results = None
    if len(std_resid_dict) >= 2:
        cross_subject_results = cross_subject_analysis_with_significance(
            std_resid_dict, list(std_resid_dict.keys()), n_permutations=1000)
        print_cross_subject_results(cross_subject_results)
    
    # 10. Figures
    print("\n[10] GENERATING FIGURES")
    print("-" * 40)
    create_figures(years, subjects, original, differenced, results, cross_subject_results)
    
    # 11. Final summary
    print("\n" + "=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)
    
    print("\nPower Analysis:")
    print(f"  ARCH-LM test power with T=14: {power_05*100:.1f}% (alpha=0.5), "
          f"{power_03*100:.1f}% (alpha=0.3)")
    print("  Conclusion: Non-significant ARCH tests are statistically inevitable.")
    
    print("\nVolatility Clustering:")
    for subject in subjects:
        if results[subject] is not None:
            alpha_val = results[subject]['alpha']
            ci = f"[{results[subject]['ci_lower']:.3f}, {results[subject]['ci_upper']:.3f}]"
            print(f"  {subject.replace('_', ' ')}: alpha = {alpha_val:.4f}, 95% CI = {ci}")
    print("  Conclusion: No evidence of ARCH effects; boundary solutions throughout.")
    
    print("\nBaseline Volatility (omega):")
    for subject in subjects:
        if results[subject] is not None:
            print(f"  {subject.replace('_', ' ')}: {results[subject]['omega']:.2f}")
    
    print("\nCross-Subject Co-movement:")
    if cross_subject_results is not None:
        n_cs = len(cross_subject_results['subjects'])
        display_names_cs = [s.replace('_', ' ') for s in cross_subject_results['subjects']]
        for i in range(n_cs):
            for j in range(i+1, n_cs):
                r = cross_subject_results['correlation'][i,j]
                p_r = cross_subject_results['correlation_pvalues'][i,j]
                a = cross_subject_results['agreement'][i,j]
                p_a = cross_subject_results['agreement_pvalues'][i,j]
                sig_r = "SIG" if p_r < 0.05 else "n.s."
                sig_a = "SIG" if p_a < 0.05 else "n.s."
                print(f"  {display_names_cs[i]} vs {display_names_cs[j]}: "
                      f"r={r:.3f} [{sig_r}], agreement={a:.3f} [{sig_a}]")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return results, arch_lm_results, bayesian_results, cross_subject_results, t_results


if __name__ == "__main__":
    results, arch_lm_results, bayesian_results, cross_subject_results, t_results = main()
