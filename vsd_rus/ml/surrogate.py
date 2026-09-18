class Surrogate:
    def __init__(self, bundle):
        self.models   = bundle["models"]
        self.features = bundle["features"]
        self.targets  = bundle["targets"]
        self.transforms = bundle.get("transforms", {})
        self.R_vsd_range = bundle.get("R_vsd_range", [0.5, 15.8])

    # ------------------------------------------------------------------
    # Dict-интерфейс (для дашборда)
    # ------------------------------------------------------------------
    def predict(self, theta_dict) -> dict:
        X = self._prepare_features(theta_dict).reshape(1, -1)
        out = {}
        for t in self.targets:
            v = float(self.models[t].predict(X)[0])
            if t == "Qp_Qs":
                v = 10 ** v
            out[t] = v
        return out

    # ------------------------------------------------------------------
    # Array-интерфейс (для Optuna в Stage 4b)
    # ------------------------------------------------------------------
    def predict_array(self, theta_vec):
        theta_vec = np.atleast_2d(theta_vec)
        X = np.column_stack([
            theta_vec[:, 1],   # E_max_lv
            theta_vec[:, 2],   # E_max_rv
            theta_vec[:, 3],   # R_sys
            theta_vec[:, 4],   # flow_sensitivity
            theta_vec[:, 5],   # C_sys_art
            np.log10(np.clip(d_vsd_to_R_vsd(theta_vec[:, 0]), 1e-3, 1e2)),
        ])
        Y = np.empty((X.shape[0], len(self.targets)))
        for j, t in enumerate(self.targets):
            v = self.models[t].predict(X)
            if t == "Qp_Qs":
                v = 10 ** v
            Y[:, j] = v
        return Y[0] if Y.shape[0] == 1 else Y

    # ------------------------------------------------------------------
    # Batch-интерфейс (алиас)
    # ------------------------------------------------------------------
    def predict_batch(self, thetas):
        return self.predict_array(thetas)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_features(self, theta_dict):
        return np.array([
            theta_dict["E_max_lv"],
            theta_dict["E_max_rv"],
            theta_dict["R_sys"],
            theta_dict["flow_sensitivity"],
            theta_dict["C_sys_art"],
            np.log10(np.clip(
                d_vsd_to_R_vsd(theta_dict["d_vsd"]), 1e-3, 1e2)),
        ])