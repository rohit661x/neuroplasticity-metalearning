import numpy as np
from scipy.optimize import curve_fit

def exponential_decay(t, A, K, C):
    """
    Model: y(t) = A * exp(-K * t) + C
    We look for K (the rate constant).
    """
    return A * np.exp(-K * t) + C

def derive_constant(time_points, metric_series):
    """
    Fit an exponential curve to the metric evolution to find the rate constant.
    
    Args:
        time_points (np.array): Time indices (or seconds).
        metric_series (np.array): Values of the metric over time (e.g., Modularity).
        
    Returns:
        dict: {
            'k': fitted rate constant,
            'r2': goodness of fit
        }
    """
    # Remove NaNs
    valid = np.isfinite(metric_series)
    t = time_points[valid]
    y = metric_series[valid]
    
    if len(t) < 5:
        return {'k': 0, 'r2': 0}
        
    try:
        # Initial guess: A = range, K = 0.001, C = final value
        p0 = [y[0] - y[-1], 0.001, y[-1]]
        
        popt, pcov = curve_fit(exponential_decay, t, y, p0=p0, maxfev=5000)
        
        # Calculate R^2
        residuals = y - exponential_decay(t, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'A': popt[0],
            'k': popt[1], # This is our candidate CONSTANT
            'C': popt[2],
            'r2': r2
        }
    except Exception as e:
        print(f"Fitting failed: {e}")
        return {'k': 0, 'r2': 0}
