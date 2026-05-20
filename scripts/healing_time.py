import numpy as np

def calculate_healing_time(z00, z01, z02, alpha_minus_beta, method='newton', tol=1e-5, max_iter=20):
    """
    Calculates the minimum healing time T0.
    z00: Initial size
    z01: Initial rate of change
    z02: Acceleration
    alpha_minus_beta: Difference between reparative and aggressive potentials
    """
    
    # Initial approximation
    t_n = (6 * z00 / alpha_minus_beta)**(1/3)
    
    print(f"Initial approximation T(0): {t_n:.4f} days\n")
    
    for i in range(1, max_iter + 1):
        if method == 'simple':
            # Simple iteration: T = Phi(T)
            inside = z00 + t_n * z01 + 0.5 * (t_n**2) * z02
            t_next = ( (6 / alpha_minus_beta) * abs(inside) )**(1/3)
        
        elif method == 'newton':
            # Newton's method: T = T - F(T)/F'(T)
            f_t = ((alpha_minus_beta / 6) * t_n**3) - (z00 + t_n * z01 + 0.5 * z02 * t_n**2)
            df_t = (0.5 * alpha_minus_beta * t_n**2) - z01 - t_n * z02
            t_next = t_n - f_t / df_t
            
        rel_error = abs(t_next - t_n) / t_n
        t_n = t_next
        
        print(f"Iteration {i}: T = {t_n:.4f} (Rel. Error: {rel_error:.6%})")
        
        if rel_error < tol:
            print(f"\nConverged to {t_n:.4f} days in {i} iterations.")
            return t_n

    return t_n

# Example parameters from the text
params = {
    "z00": 10.0,
    "z01": -0.1,
    "z02": 0.0,
    "alpha_minus_beta": 0.001
}

result = calculate_healing_time(**params, method='newton')