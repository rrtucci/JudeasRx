from math import sqrt

def next_mu_sigma(n, next_x, mu, sigma):
    """
    This function calculates the next mean and standard deviation (i.e.,
    \mu_{n+1}, \sigma_{n+1}) in terms of the previous ones (i.e., \mu_n,
    \sigma_n) and the number n of samples and the next point (i.e., x_{n+1}).

    \mu_{n+1} = (n*\mu_n + x_{n+1})/(n+1)

    \sigma^2_{n+1} = ((n-1)\sigma^2_n + (x_{n+1}-\mu_{n+1})^2)/n

    Parameters
    ----------
    n : int
    mu : float
        \mu_n
    sigma : float
        \sigma_n
    next_x : float
        x_{n+1}

    Returns
    -------
    float, float
        \mu_{n+1}, \sigma_{n+1}

    """
    if n > 0:
        next_mu = (n * mu + next_x) / (n + 1)
        next_delta_x_sq = (next_x - next_mu) ** 2
        next_sigma_sq = ((n-1)*(sigma**2) + next_delta_x_sq)/n
    else:
        next_mu = next_x
        next_sigma_sq = 0.
    return next_mu, sqrt(next_sigma_sq)
