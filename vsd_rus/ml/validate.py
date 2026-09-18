for theta in sample_stratified_thetas(50):
    X_sur = surrogate.predict(theta)
    X_ode = run_full_ode(theta)  # WholeBodyModel + LSODA

    rel_err = {
        k: abs(X_sur[k] - X_ode[k]) / max(abs(X_ode[k]), 1e-6)
        for k in X_sur
    }
    print(f"d_vsd={theta['d_vsd']:.2f}  "
          f"max_err={max(rel_err.values()):.3f}")