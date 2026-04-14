"""
Chapter 4: Results and Discussion - WASSCE Volatility Analysis
Based on Chapter 3 Methodology

Specifications:
- AR(1)-ARCH(1) model
- Quasi-Maximum Likelihood Estimation (QMLE)
- Parametric bootstrap with B = 1000 (as per Chapter 3)
- Engle LM test with bootstrap inference
- Cross-subject analysis (correlation, synchronisation, directional agreement)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# Colors for consistent plotting
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

print("="*80)
print("CHAPTER 4: RESULTS AND DISCUSSION")
print("WASSCE Volatility Clustering Analysis")
print("Based on Chapter 3 Methodology")
print("="*80)


# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n[1] Loading and preprocessing data...")

# Load data
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Find columns automatically
year_col = None
english_col = None
math_col = None
science_col = None
social_col = None

for col in df.columns:
    col_lower = str(col).lower()
    if 'year' in col_lower:
        year_col = col
    elif 'english' in col_lower:
        english_col = col
    elif 'math' in col_lower or 'mathematics' in col_lower:
        math_col = col
    elif 'science' in col_lower or 'integrated' in col_lower:
        science_col = col
    elif 'social' in col_lower:
        social_col = col

# Create clean dataframe
df_clean = pd.DataFrame()
df_clean['Year'] = pd.to_numeric(df[year_col], errors='coerce')
df_clean['English'] = pd.to_numeric(df[english_col], errors='coerce')
df_clean['Mathematics'] = pd.to_numeric(df[math_col], errors='coerce')
df_clean['Integrated_Science'] = pd.to_numeric(df[science_col], errors='coerce')
df_clean['Social_Studies'] = pd.to_numeric(df[social_col], errors='coerce')

# Fix Social Studies if needed (handle commas)
if df_clean['Social_Studies'].dtype == 'object':
    df_clean['Social_Studies'] = df_clean['Social_Studies'].astype(str).str.replace(',', '.').astype(float)

df_clean = df_clean.dropna().reset_index(drop=True)
df_clean['Year'] = df_clean['Year'].astype(int)

subjects = ['English', 'Mathematics', 'Integrated_Science', 'Social_Studies']
subject_labels = ['English', 'Mathematics', 'Integrated Science', 'Social Studies']

# First differencing (as per Chapter 3)
df_diff = df_clean.copy()
for subj in subjects:
    df_diff[f'{subj}_diff'] = df_clean[subj].diff()
df_diff = df_diff.dropna().reset_index(drop=True)

print(f"   Original sample: T = {len(df_clean)} years ({df_clean['Year'].min()}-{df_clean['Year'].max()})")
print(f"   Effective sample after differencing: T = {len(df_diff)} years")
print(f"   Bootstrap replications: B = 1000 (as specified in Chapter 3)")


# ============================================================================
# 2. AR(1)-ARCH(1) MODEL IMPLEMENTATION
# ============================================================================

def arch1_loglikelihood(params, y):
    """Negative log-likelihood for AR(1)-ARCH(1) model"""
    mu, phi, alpha0, alpha1 = params
    
    if abs(phi) >= 1 or alpha0 <= 0 or alpha1 < 0 or alpha1 >= 1:
        return 1e10
    
    n = len(y)
    eps = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = alpha0 / (1 - alpha1) if alpha1 < 1 else alpha0
    
    loglik = 0
    for t in range(1, n):
        eps[t] = y[t] - mu - phi * y[t-1]
        sigma2[t] = alpha0 + alpha1 * (eps[t-1]**2)
        if sigma2[t] > 0:
            loglik += -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2[t]) - 0.5 * (eps[t]**2 / sigma2[t])
        else:
            return 1e10
    return -loglik


def fit_arch1(series, subject_name=""):
    """Fit AR(1)-ARCH(1) model using MLE"""
    try:
        y = series.values
        n = len(y)
        
        mu_init = np.mean(y)
        phi_init = np.corrcoef(y[1:], y[:-1])[0, 1] if n > 1 else 0
        alpha0_init = max(np.var(y) * 0.1, 0.01)
        
        bounds = [(None, None), (-0.99, 0.99), (0.001, None), (0, 0.99)]
        
        best_result = None
        best_loglik = 1e10
        
        for alpha1_start in [0.1, 0.3, 0.5]:
            result = minimize(
                arch1_loglikelihood, 
                [mu_init, phi_init, alpha0_init, alpha1_start],
                args=(y,),
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 500, 'disp': False}
            )
            if result.success and result.fun < best_loglik:
                best_loglik = result.fun
                best_result = result
        
        if best_result is not None and best_result.success:
            mu, phi, alpha0, alpha1 = best_result.x
            
            eps = np.zeros(n)
            sigma2 = np.zeros(n)
            sigma2[0] = alpha0 / (1 - alpha1) if alpha1 < 1 else alpha0
            
            for t in range(1, n):
                eps[t] = y[t] - mu - phi * y[t-1]
                sigma2[t] = alpha0 + alpha1 * (eps[t-1]**2)
            
            std_resid = eps[1:] / np.sqrt(sigma2[1:])
            cond_var = sigma2[1:]
            
            loglik = -best_loglik
            aic = 2 * 4 - 2 * loglik
            
            return {
                'alpha1': alpha1, 'alpha0': alpha0, 'phi': phi, 'mu': mu,
                'std_resid': std_resid, 'cond_var': cond_var,
                'loglik': loglik, 'aic': aic, 'converged': True
            }
        return {'converged': False, 'alpha1': np.nan}
    except Exception as e:
        print(f"      Error: {e}")
        return {'converged': False, 'alpha1': np.nan}


# ============================================================================
# 3. PARAMETRIC BOOTSTRAP FOR α₁ (B = 1000)
# ============================================================================

def parametric_bootstrap(series, n_bootstrap=1000, subject_name=""):
    """Parametric bootstrap for α₁ confidence intervals (B = 1000)"""
    try:
        original_fit = fit_arch1(series, subject_name)
        if not original_fit['converged']:
            print(f"      ERROR: Original model for {subject_name} did not converge")
            return np.nan, np.nan, np.nan
        
        mu = original_fit['mu']
        phi = original_fit['phi']
        alpha0 = original_fit['alpha0']
        alpha1_hat = original_fit['alpha1']
        y = series.values
        n = len(y)
        
        # Get standardized residuals from original fit
        eps = np.zeros(n)
        sigma2 = np.zeros(n)
        sigma2[0] = alpha0 / (1 - alpha1_hat) if alpha1_hat < 1 else alpha0
        for t in range(1, n):
            eps[t] = y[t] - mu - phi * y[t-1]
            sigma2[t] = alpha0 + alpha1_hat * (eps[t-1]**2)
        std_resid = eps[1:] / np.sqrt(sigma2[1:])  # Length = n-1 = 13
        
        print(f"      Original α₁ = {alpha1_hat:.4f}, std_resid length = {len(std_resid)}")
        
        alpha1_bootstrap = []
        
        for b in range(n_bootstrap):
            try:
                # FIXED: Sample exactly len(std_resid) residuals (not n)
                boot_resids = np.random.choice(std_resid, size=len(std_resid), replace=True)
                
                # Generate bootstrap series
                y_boot = np.zeros(n)
                y_boot[0] = y[0]
                eps_boot = np.zeros(n)
                sigma2_boot = np.zeros(n)
                sigma2_boot[0] = alpha0 / (1 - alpha1_hat) if alpha1_hat < 1 else alpha0
                eps_boot[0] = np.sqrt(max(sigma2_boot[0], 1e-8)) * boot_resids[0]
                
                for t in range(1, n):
                    sigma2_boot[t] = alpha0 + alpha1_hat * (eps_boot[t-1]**2)
                    eps_boot[t] = np.sqrt(max(sigma2_boot[t], 1e-8)) * boot_resids[t-1]  # Use boot_resids[t-1]
                    y_boot[t] = mu + phi * y_boot[t-1] + eps_boot[t]
                
                # Re-estimate on bootstrap sample
                boot_fit = fit_arch1(pd.Series(y_boot), f"{subject_name}_boot")
                if boot_fit['converged'] and 0 <= boot_fit['alpha1'] < 1:
                    alpha1_bootstrap.append(boot_fit['alpha1'])
            except Exception as e:
                continue
            
            if (b + 1) % 200 == 0:
                print(f"      {subject_name}: {b+1}/{n_bootstrap} replications completed, valid={len(alpha1_bootstrap)}")
        
        print(f"      Bootstrap completed: {len(alpha1_bootstrap)} valid samples out of {n_bootstrap}")
        
        if len(alpha1_bootstrap) > 100:
            ci_lower = np.percentile(alpha1_bootstrap, 2.5)
            ci_upper = np.percentile(alpha1_bootstrap, 97.5)
            print(f"      95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            return alpha1_hat, ci_lower, ci_upper
        else:
            print(f"      WARNING: Only {len(alpha1_bootstrap)} valid samples (<100)")
            return alpha1_hat, np.nan, np.nan
    except Exception as e:
        print(f"      Bootstrap error for {subject_name}: {e}")
        return np.nan, np.nan, np.nan

# ============================================================================
# 4. ENGLE LM TEST WITH BOOTSTRAP INFERENCE (B = 1000)
# ============================================================================

def engle_lm_bootstrap(series, n_bootstrap=1000, seed=42):
    """Engle LM test for ARCH effects with bootstrap inference (B = 1000)"""
    np.random.seed(seed)
    
    y = series.values
    n = len(y)
    
    # Step 1: AR(1) residuals under null
    y_lag = np.roll(y, 1)
    y_lag[0] = np.nan
    mask = ~np.isnan(y_lag)
    y_clean, y_lag_clean = y[mask], y_lag[mask]
    X = np.column_stack([np.ones(len(y_lag_clean)), y_lag_clean])
    beta = np.linalg.lstsq(X, y_clean, rcond=None)[0]
    residuals_null = y_clean - X @ beta
    
    # Step 2: Observed LM statistic
    resid_sq = residuals_null**2
    resid_sq_lag = np.roll(resid_sq, 1)
    resid_sq_lag[0] = np.nan
    mask2 = ~np.isnan(resid_sq_lag)
    resid_sq_clean, resid_sq_lag_clean = resid_sq[mask2], resid_sq_lag[mask2]
    X2 = np.column_stack([np.ones(len(resid_sq_lag_clean)), resid_sq_lag_clean])
    beta2 = np.linalg.lstsq(X2, resid_sq_clean, rcond=None)[0]
    residuals2 = resid_sq_clean - X2 @ beta2
    r_squared_obs = 1 - np.var(residuals2) / np.var(resid_sq_clean)
    lm_obs = len(resid_sq_clean) * r_squared_obs
    
    # Step 3: Bootstrap under null (wild bootstrap)
    lm_bootstrap = []
    residuals_centered = residuals_null - np.mean(residuals_null)
    
    for b in range(n_bootstrap):
        w = np.random.choice([-1, 1], size=len(residuals_centered))
        y_boot = X @ beta + residuals_centered * w
        
        y_boot_lag = np.roll(y_boot, 1)
        y_boot_lag[0] = np.nan
        mask_boot = ~np.isnan(y_boot_lag)
        y_boot_clean, y_boot_lag_clean = y_boot[mask_boot], y_boot_lag[mask_boot]
        X_boot = np.column_stack([np.ones(len(y_boot_lag_clean)), y_boot_lag_clean])
        beta_boot = np.linalg.lstsq(X_boot, y_boot_clean, rcond=None)[0]
        resid_boot = y_boot_clean - X_boot @ beta_boot
        
        resid_sq_boot = resid_boot**2
        resid_sq_lag_boot = np.roll(resid_sq_boot, 1)
        resid_sq_lag_boot[0] = np.nan
        mask_boot2 = ~np.isnan(resid_sq_lag_boot)
        resid_sq_clean_boot, resid_sq_lag_clean_boot = resid_sq_boot[mask_boot2], resid_sq_lag_boot[mask_boot2]
        
        if len(resid_sq_clean_boot) > 1:
            X2_boot = np.column_stack([np.ones(len(resid_sq_lag_clean_boot)), resid_sq_lag_clean_boot])
            beta2_boot = np.linalg.lstsq(X2_boot, resid_sq_clean_boot, rcond=None)[0]
            resid2_boot = resid_sq_clean_boot - X2_boot @ beta2_boot
            r_squared_boot = 1 - np.var(resid2_boot) / np.var(resid_sq_clean_boot)
            lm_boot = len(resid_sq_clean_boot) * r_squared_boot
            lm_bootstrap.append(lm_boot)
    
    p_bootstrap = np.mean(np.array(lm_bootstrap) >= lm_obs)
    return lm_obs, p_bootstrap


# ============================================================================
# 5. FIGURE GENERATION FUNCTIONS
# ============================================================================

def generate_figure_4_1(df_clean, subjects, subject_labels, colors):
    """Figure 4.1: Raw pass rates over time (2011-2025)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, subj in enumerate(subjects):
        ax.plot(df_clean['Year'], df_clean[subj], 'o-', linewidth=2, markersize=6, 
                color=colors[i], label=subject_labels[i])
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.set_title('Figure 4.1: WASSCE Pass Rates by Subject, 2011-2025', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2011, 2026)
    plt.tight_layout()
    plt.savefig('Figure_4_1_Pass_Rates.png', dpi=300)
    plt.close()
    print("   Saved: Figure_4_1_Pass_Rates.png")


def generate_figure_4_2(results_dict, subjects, subject_labels, colors, years_plot):
    """Figure 4.2: Standardised residuals from AR(1)-ARCH(1) model"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Residuals correspond to 2013-2025 (skip 2012)
    years_resid = years_plot[1:]
    
    for i, subj in enumerate(subjects):
        if subj in results_dict and results_dict[subj]['converged']:
            std_resid = results_dict[subj]['std_resid']
            
            # Ensure lengths match
            min_len = min(len(years_resid), len(std_resid))
            years_aligned = years_resid[:min_len]
            resid_aligned = std_resid[:min_len]
            
            axes[i].plot(years_aligned, resid_aligned, 'o-', linewidth=1.5, markersize=5, color=colors[i])
            axes[i].axhline(y=0, color='black', linewidth=0.8)
            axes[i].axhline(y=2, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
            axes[i].axhline(y=-2, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
            axes[i].fill_between(years_aligned, -2, 2, alpha=0.1, color='gray')
            axes[i].set_title(f'{subject_labels[i]} - Standardised Residuals', fontweight='bold', fontsize=12)
            axes[i].set_xlabel('Year', fontsize=10)
            axes[i].set_ylabel('$\\hat{z}_t$', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(2011, 2026)
            axes[i].set_ylim(-3, 3)
    
    plt.suptitle('Figure 4.2: Standardised Residuals from the Estimated AR(1)-ARCH(1) Model',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('Figure_4_2_Standardised_Residuals.png', dpi=300)
    plt.close()
    print("   Saved: Figure_4_2_Standardised_Residuals.png")


def generate_figure_4_3(results_dict, subjects, subject_labels, colors, years_plot):
    """Figure 4.3: Conditional variance overlay for all subjects"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    years_resid = years_plot[1:]
    
    for i, subj in enumerate(subjects):
        if subj in results_dict and results_dict[subj]['converged']:
            cond_var = results_dict[subj]['cond_var']
            min_len = min(len(years_resid), len(cond_var))
            years_aligned = years_resid[:min_len]
            cond_var_aligned = cond_var[:min_len]
            
            ax.plot(years_aligned, cond_var_aligned, 'o-', linewidth=2, markersize=6,
                    color=colors[i], label=subject_labels[i])
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Conditional Variance ($\\hat{\\sigma}^2_t$)', fontsize=12)
    ax.set_title('Figure 4.3: Conditional Variance (Volatility) Over Time — All Subjects',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2011, 2026)
    plt.tight_layout()
    plt.savefig('Figure_4_3_Conditional_Variance.png', dpi=300)
    plt.close()
    print("   Saved: Figure_4_3_Conditional_Variance.png")


def generate_figure_4_4(results_dict, subjects, subject_labels, colors, years_plot):
    """Figure 4.4: Conditional variance by subject (individual panels)"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    years_resid = years_plot[1:]
    
    for i, subj in enumerate(subjects):
        if subj in results_dict and results_dict[subj]['converged']:
            cond_var = results_dict[subj]['cond_var']
            min_len = min(len(years_resid), len(cond_var))
            years_aligned = years_resid[:min_len]
            cond_var_aligned = cond_var[:min_len]
            
            axes[i].plot(years_aligned, cond_var_aligned, 'o-', linewidth=2, markersize=6, color=colors[i])
            axes[i].fill_between(years_aligned, 0, cond_var_aligned, alpha=0.3, color=colors[i])
            axes[i].set_title(f'{subject_labels[i]} - Conditional Variance', fontweight='bold', fontsize=12)
            axes[i].set_xlabel('Year', fontsize=10)
            axes[i].set_ylabel('$\\hat{\\sigma}^2_t$', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(2011, 2026)
    
    plt.suptitle('Figure 4.4: Conditional Variance (Volatility) by Subject',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('Figure_4_4_Conditional_Variance_Individual.png', dpi=300)
    plt.close()
    print("   Saved: Figure_4_4_Conditional_Variance_Individual.png")


def generate_figure_4_5(results_dict, subjects, subject_labels):
    """Figure 4.5: Q-Q plots of standardised residuals vs normal distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, subj in enumerate(subjects):
        if subj in results_dict and results_dict[subj]['converged']:
            std_resid = results_dict[subj]['std_resid']
            stats.probplot(std_resid, dist="norm", plot=axes[i])
            axes[i].set_title(f'{subject_labels[i]}', fontweight='bold', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlabel('Theoretical Quantiles', fontsize=10)
            axes[i].set_ylabel('Sample Quantiles', fontsize=10)
    
    plt.suptitle('Figure 4.5: Q-Q Plots of Standardised Residuals vs. Normal Distribution',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('Figure_4_5_QQ_Plots.png', dpi=300)
    plt.close()
    print("   Saved: Figure_4_5_QQ_Plots.png")


def generate_figure_4_6(corr_matrix, subject_labels, subjects):
    """Figure 4.6: Cross-subject shock correlation heatmap"""
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', 
                cmap='RdBu_r', center=0, square=True, 
                xticklabels=subject_labels, yticklabels=subject_labels,
                cbar_kws={'label': 'Correlation'}, ax=ax)
    ax.set_title('Figure 4.6: Cross-Subject Shock Correlation Matrix',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Figure_4_6_Correlation_Heatmap.png', dpi=300)
    plt.close()
    print("   Saved: Figure_4_6_Correlation_Heatmap.png")


# ============================================================================
# 6. RUN ALL ANALYSES
# ============================================================================

print("\n[2] Estimating AR(1)-ARCH(1) models...")

results = {}
alpha1_vals = []
ci_lower = []
ci_upper = []

for i, subj in enumerate(subjects):
    print(f"\n   Processing {subject_labels[i]}...")
    series = df_diff[f'{subj}_diff'].dropna()
    
    # Fit model
    results[subj] = fit_arch1(series, subj)
    if results[subj]['converged']:
        print(f"      Model converged: α₁ = {results[subj]['alpha1']:.4f}, AIC = {results[subj]['aic']:.2f}")
    else:
        print(f"      WARNING: Model did not converge")
    
    # Run bootstrap with B = 1000
    a1, low, high = parametric_bootstrap(series, n_bootstrap=1000, subject_name=subj)
    alpha1_vals.append(a1)
    ci_lower.append(low)
    ci_upper.append(high)
    
    if not np.isnan(low):
        print(f"      Bootstrap (B=1000): 95% CI = [{low:.4f}, {high:.4f}], Excludes zero: {low > 0}")


# ============================================================================
# 7. PRINT ALL RESULTS TABLES
# ============================================================================

print("\n" + "="*80)
print("TABLE 4.1: SUMMARY STATISTICS FOR DIFFERENCED PASS RATES")
print("="*80)
print(f"{'Subject':<20} {'Mean':>10} {'Std Dev':>10} {'Min':>10} {'Max':>10}")
print("-"*60)
for i, subj in enumerate(subjects):
    diff_series = df_diff[f'{subj}_diff']
    print(f"{subject_labels[i]:<20} {diff_series.mean():>10.4f} {diff_series.std():>10.4f} {diff_series.min():>10.4f} {diff_series.max():>10.4f}")

print("\n" + "="*80)
print("TABLE 4.2: AUGMENTED DICKEY-FULLER TEST")
print("="*80)
print(f"{'Subject':<20} {'ADF Statistic':>15} {'p-value':>10} {'Stationary':>10}")
print("-"*60)
for i, subj in enumerate(subjects):
    series = df_diff[f'{subj}_diff']
    adf_stat, p_value, _, _, _, _ = adfuller(series, autolag='AIC')
    stationary = "Yes" if p_value < 0.05 else "No"
    print(f"{subject_labels[i]:<20} {adf_stat:>15.4f} {p_value:>10.4f} {stationary:>10}")

print("\n" + "="*80)
print("TABLE 4.3: ARCH(1) TEST WITH BOOTSTRAP INFERENCE (B = 1000)")
print("="*80)
print(f"{'Subject':<20} {'LM Statistic':>15} {'Bootstrap p-value':>20} {'ARCH Effect':>15}")
print("-"*70)

for i, subj in enumerate(subjects):
    series = df_diff[f'{subj}_diff']
    lm_stat, p_boot = engle_lm_bootstrap(series, n_bootstrap=1000, seed=42)
    arch_effect = "Not Detected" if p_boot > 0.05 else "Detected"
    print(f"{subject_labels[i]:<20} {lm_stat:>15.4f} {p_boot:>20.4f} {arch_effect:>15}")

print("\n" + "="*80)
print("TABLE 4.4: AR(1)-ARCH(1) PARAMETER ESTIMATES")
print("="*80)
print(f"{'Subject':<20} {'μ':>10} {'φ':>10} {'α₀':>10} {'α₁':>10} {'AIC':>10}")
print("-"*70)
for i, subj in enumerate(subjects):
    if results[subj]['converged']:
        print(f"{subject_labels[i]:<20} {results[subj]['mu']:>10.4f} {results[subj]['phi']:>10.4f} {results[subj]['alpha0']:>10.4f} {results[subj]['alpha1']:>10.4f} {results[subj]['aic']:>10.2f}")
    else:
        print(f"{subject_labels[i]:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

print("\n" + "="*80)
print("TABLE 4.5: BOOTSTRAP 95% CONFIDENCE INTERVALS FOR α₁ (B = 1000)")
print("="*80)
print(f"{'Subject':<20} {'α̂₁':>10} {'95% CI Lower':>15} {'95% CI Upper':>15} {'Excludes Zero':>15}")
print("-"*75)
for i, subj in enumerate(subjects):
    if not np.isnan(ci_lower[i]):
        excludes = "Yes" if ci_lower[i] > 0 else "No"
        print(f"{subject_labels[i]:<20} {alpha1_vals[i]:>10.4f} {ci_lower[i]:>15.4f} {ci_upper[i]:>15.4f} {excludes:>15}")
    else:
        print(f"{subject_labels[i]:<20} {alpha1_vals[i]:>10.4f} {'N/A':>15} {'N/A':>15} {'N/A':>15}")

print("\n" + "="*80)
print("TABLE 4.6: LJUNG-BOX TEST (LAG 5)")
print("="*80)
print(f"{'Subject':<20} {'LB (Residuals) p':>20} {'LB (Squared) p':>20} {'Model Adequate':>15}")
print("-"*75)
for i, subj in enumerate(subjects):
    if results[subj]['converged']:
        std_resid = results[subj]['std_resid']
        lb_resid = acorr_ljungbox(std_resid, lags=[5], return_df=True)
        lb_sq = acorr_ljungbox(std_resid**2, lags=[5], return_df=True)
        p_resid = lb_resid['lb_pvalue'].values[0]
        p_sq = lb_sq['lb_pvalue'].values[0]
        adequate = "Yes" if p_resid > 0.05 and p_sq > 0.05 else "Partial"
        print(f"{subject_labels[i]:<20} {p_resid:>20.4f} {p_sq:>20.4f} {adequate:>15}")

print("\n" + "="*80)
print("TABLE 4.7: JARQUE-BERA NORMALITY TEST")
print("="*80)
print(f"{'Subject':<20} {'JB Statistic':>15} {'p-value':>10} {'Normal':>10}")
print("-"*60)
for i, subj in enumerate(subjects):
    if results[subj]['converged']:
        std_resid = results[subj]['std_resid']
        jb_stat, jb_p = stats.jarque_bera(std_resid)
        normal = "No" if jb_p < 0.05 else "Yes"
        print(f"{subject_labels[i]:<20} {jb_stat:>15.4f} {jb_p:>10.4f} {normal:>10}")

print("\n" + "="*80)
print("TABLE 4.8: HALF-LIFE OF VOLATILITY SHOCKS")
print("="*80)
print(f"{'Subject':<20} {'α₁':>10} {'Half-Life (years)':>20}")
print("-"*55)
for i, subj in enumerate(subjects):
    a1 = alpha1_vals[i] if not np.isnan(alpha1_vals[i]) else results[subj]['alpha1'] if results[subj]['converged'] else np.nan
    if not np.isnan(a1) and a1 > 0 and a1 < 1:
        half_life = np.log(0.5) / np.log(a1)
        print(f"{subject_labels[i]:<20} {a1:>10.4f} {half_life:>20.2f}")
    else:
        print(f"{subject_labels[i]:<20} {a1:>10.4f} {'N/A':>20}")

print("\n" + "="*80)
print("TABLE 4.9: UNCONDITIONAL VARIANCE")
print("="*80)
print(f"{'Subject':<20} {'α₀':>10} {'α₁':>10} {'Unconditional Variance':>25}")
print("-"*70)
for i, subj in enumerate(subjects):
    if results[subj]['converged']:
        alpha0 = results[subj]['alpha0']
        alpha1 = results[subj]['alpha1']
        if alpha1 < 1:
            uncond_var = alpha0 / (1 - alpha1)
            print(f"{subject_labels[i]:<20} {alpha0:>10.4f} {alpha1:>10.4f} {uncond_var:>25.4f}")
        else:
            print(f"{subject_labels[i]:<20} {alpha0:>10.4f} {alpha1:>10.4f} {'Not finite':>25}")
    else:
        print(f"{subject_labels[i]:<20} {'N/A':>10} {'N/A':>10} {'N/A':>25}")


# ============================================================================
# 8. CROSS-SUBJECT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("CROSS-SUBJECT ANALYSIS")
print("="*80)

# Get standardized residuals matrix
std_resid_dict = {}
for subj in subjects:
    if results[subj]['converged']:
        std_resid_dict[subj] = results[subj]['std_resid']

# Align lengths
min_len = min([len(v) for v in std_resid_dict.values()])
std_resid_matrix = np.array([v[:min_len] for v in std_resid_dict.values()])
corr_matrix = np.corrcoef(std_resid_matrix)

print("\nTABLE 4.10: CORRELATION MATRIX OF STANDARDISED SHOCKS")
print("-"*60)
print(f"{'Subject':<20}", end="")
for label in subject_labels:
    print(f"{label[:12]:>12}", end="")
print()
print("-"*60)
for i, label in enumerate(subject_labels):
    print(f"{label:<20}", end="")
    for j in range(len(subject_labels)):
        print(f"{corr_matrix[i, j]:12.4f}", end="")
    print()

print("\nTABLE 4.11: LARGE SHOCK SYNCHRONISATION RATES (|z| > 1)")
print("-"*60)
threshold = 1.0
large_shocks = (np.abs(std_resid_matrix) > threshold).astype(int)
n_years = std_resid_matrix.shape[1]

print(f"{'Subject':<20}", end="")
for label in subject_labels:
    print(f"{label[:12]:>12}", end="")
print()
print("-"*60)
for i, label in enumerate(subject_labels):
    print(f"{label:<20}", end="")
    for j in range(len(subject_labels)):
        sync = (large_shocks[i] & large_shocks[j]).sum() / n_years
        print(f"{sync:12.3f}", end="")
    print()

print("\nTABLE 4.12: DIRECTIONAL AGREEMENT RATES (SAME SIGN)")
print("-"*60)
print(f"{'Subject':<20}", end="")
for label in subject_labels:
    print(f"{label[:12]:>12}", end="")
print()
print("-"*60)
for i, label in enumerate(subject_labels):
    print(f"{label:<20}", end="")
    for j in range(len(subject_labels)):
        same_sign = ((std_resid_matrix[i] * std_resid_matrix[j]) > 0).sum() / n_years
        print(f"{same_sign:12.3f}", end="")
    print()


# ============================================================================
# 9. GENERATE ALL FIGURES
# ============================================================================

print("\n[3] Generating figures for Chapter 4...")

years_plot = df_diff['Year'].values

generate_figure_4_1(df_clean, subjects, subject_labels, COLORS)
generate_figure_4_2(results, subjects, subject_labels, COLORS, years_plot)
generate_figure_4_3(results, subjects, subject_labels, COLORS, years_plot)
generate_figure_4_4(results, subjects, subject_labels, COLORS, years_plot)
generate_figure_4_5(results, subjects, subject_labels)
generate_figure_4_6(corr_matrix, subject_labels, subjects)


# ============================================================================
# 10. COMPLETION
# ============================================================================

print("\n" + "="*80)
print("COMPUTATION COMPLETE")
print(f"Bootstrap replications: B = 1000 (as specified in Chapter 3)")
print("All figures saved as PNG files")
print("="*80)