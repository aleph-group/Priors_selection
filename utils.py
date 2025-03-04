import arviz 
import numpy as np


def plot_MC_corr(X_trace, ax, title="X"):
    """ 
    Plot the autocorrelation of the MC X_trace if shape (n_samples, 1), and plot the autocorrelation of the min, median and
    max variance pixels if shape (n_samples, n_pixels). Displays the effective sampling size in plot title.
    """
    if X_trace.shape[1] > 1:  # shape (n_samples, n_pixels)
        X_var = np.var(X_trace, axis=0)   
        # track pixels with max, min, med variances
        sort_ind = np.argsort(X_var)
        X_max_var = X_trace[:, sort_ind[-1]]
        X_min_var = X_trace[:, sort_ind[0]]
        X_med_var = X_trace[:, sort_ind[len(sort_ind) // 2]]
        ds = dict(Xmax=X_max_var, Xmin=X_min_var, Xmed=X_med_var)
        # corresponding effective sample sizes
        ess_max = arviz.ess(X_max_var.reshape(-1))
        ess_min = arviz.ess(X_min_var.reshape(-1))
        ess_med  = arviz.ess(X_med_var.reshape(-1))
        arviz.plot_autocorr(ds, var_names=['Xmin', 'Xmed', 'Xmax'], ax=ax)
        ax[0].set_title(title + " min: {:.1f}".format(ess_min))
        ax[1].set_title(title + " med: {:.1f}".format(ess_med))
        ax[2].set_title(title + " max: {:.1f}".format(ess_max))
    else:  # shape (n_samples,)
        ess = arviz.ess(X_trace.reshape(-1))
        ds = dict(X=X_trace.reshape(-1))
        arviz.plot_autocorr(ds, var_names=['X'], ax=ax)
        ax[0].set_title(title + ": {:.1f}".format(ess))