from __future__ import annotations
import numpy as np
from scipy import sparse
from cvxopt import matrix, solvers


class SCPOptimiser:
    # --------------------------------------------------------------------- #
    # configuration
    # --------------------------------------------------------------------- #
    def __init__(self,
                 safety_margin_obstacles: float = 0.02,
                 safety_margin_boundary:  float = 0.02,
                 fix_start: bool = True,
                 fix_goal:  bool = True):
        self.safety_margin_obstacles = safety_margin_obstacles
        self.safety_margin_boundary  = safety_margin_boundary
        self.fix_start = fix_start
        self.fix_goal  = fix_goal

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def optimise(self,
                 X_ref: np.ndarray,
                 obstacles: list[dict],
                 workspace: dict,
                 workspace_type: str = "cuboidal",
                 lambda_dev: float = 10.0,
                 lambda_smooth: float = 0.1,
                 Δ0: float = 0.05,
                 Δ_max: float = 0.20,
                 max_iter: int = 20,
                 tol: float = 1e-3) -> np.ndarray:
        """
        Sequential convex programme that enforces *exact* obstacle avoidance.

        Parameters
        ----------
        X_ref         (T,3)  : reference way-points (also used as initial guess)
        obstacles     list   : [{'x','y','z','dimensions':(dx,dy,dz)}, …]
        workspace     dict   : cuboid: {'x':(..),'y':(..),'z':(..)}
                                arm   : {'centre':(cx,cy,cz), 'link_lengths':[..]}
        workspace_type        : "cuboidal" | "arm"
        lambda_dev, lambda_smooth  : cost weights
        Δ0 / Δ_max            : start / cap trust-region radius (metres)
        max_iter, tol         : SCP stopping criteria
        """
        Xk = X_ref.copy()
        Δ  = float(Δ0)

        for it in range(max_iter):
            P, q = self._build_objective(Xk, X_ref, lambda_dev, lambda_smooth)
            G, h = self._build_constraints(Xk, obstacles, workspace,
                                           workspace_type, Δ)

            sol = self._solve_qp(P, q, G, h)
            if sol is None:
                # infeasible trust region → shrink and retry
                Δ *= 0.5
                continue

            Xnew = sol.reshape(-1, 3)
            step = np.max(np.abs(Xnew - Xk))
            Xk   = Xnew

            # grow trust region gradually if converging nicely
            if step < Δ / 2:
                Δ = min(Δ * 1.2, Δ_max)

            # stopping test
            if step < tol:
                print(f"[SCP] converged in {it+1} iterations, max step={step:.4g}")
                break
        return Xk

    # ------------------------------------------------------------------ #
    # private helpers
    # ------------------------------------------------------------------ #
    def _build_objective(self, X, X_ref, λ_dev, λ_smooth):
        T = X.shape[0]
        n = 3 * T
        I = sparse.eye(n, format="csr")

        # first-difference operator (T-1)×T  →  smoothness ∝ ‖▽¹X‖²
        D1_1d = sparse.diags([-np.ones(T-1), np.ones(T-1)],
                             offsets=[0, 1], shape=(T-1, T), format="csr")
        D1 = sparse.kron(sparse.eye(3), D1_1d, format="csr")

        P = 2 * (λ_dev * I + λ_smooth * (D1.T @ D1))
        q = -2 * λ_dev * X_ref.flatten()
        return P, q

    # -------- constraint assembly --------------------------------------- #
    def _build_constraints(self, Xk, obstacles, workspace,
                           workspace_type, Δ):
        T = Xk.shape[0]
        n = 3 * T
        G_rows, h_vals = [], []

        # ---- 1) workspace constraints (unchanged across SCP iters) ---- #
        if workspace_type == "cuboidal":
            xmin, xmax = workspace["x"]
            ymin, ymax = workspace["y"]
            zmin, zmax = workspace["z"]
            m = self.safety_margin_boundary
            for t in range(T):
                ix, iy, iz = 3*t, 3*t+1, 3*t+2
                for idx, ub in zip([ix, iy, iz],
                                   [xmax-m, ymax-m, zmax-m]):
                    row = sparse.lil_matrix((1, n)); row[0, idx] =  1
                    G_rows.append(row); h_vals.append( ub )
                for idx, lb in zip([ix, iy, iz],
                                   [xmin+m, ymin+m, zmin+m]):
                    row = sparse.lil_matrix((1, n)); row[0, idx] = -1
                    G_rows.append(row); h_vals.append(-lb)

        elif workspace_type == "arm":
            c   = np.asarray(workspace["centre"], float)
            R   = np.sum(workspace["link_lengths"]) - self.safety_margin_boundary
            for t in range(T):
                ix, iy, iz = 3*t, 3*t+1, 3*t+2
                # linearise circle:  (q-c)⋅d ≤ ½(R² + d⋅(2c-qk))
                d = Xk[t] - c
                if np.linalg.norm(d) < 1e-6:           # avoid zero gradient
                    d[:] = (1.0, 0.0, 0.0)
                row = sparse.lil_matrix((1, n))
                row[0, ix:iz+1] =  d
                rhs = 0.5 * (R**2 + d @ (2*c - Xk[t]))
                G_rows.append(row); h_vals.append(rhs)
        else:
            raise ValueError("workspace_type must be 'cuboidal' or 'arm'")

        # ---- 2) obstacle half-spaces (linearised per iteration) ------- #
        m_obs = self.safety_margin_obstacles
        for t, (xr, yr, zr) in enumerate(Xk):
            ix, iy, iz = 3*t, 3*t+1, 3*t+2
            for obs in obstacles:
                xmin, xmax, ymin, ymax, zmin, zmax = self._inflated_bounds(obs, m_obs)
                faces = [(+1, 0, xmin, xr - xmin),   # (sign, axis, rhs_face, dist)
                         (-1, 0, -xmax, xmax - xr),
                         (+1, 1, ymin, yr - ymin),
                         (-1, 1, -ymax, ymax - yr),
                         (+1, 2, zmin, zr - zmin),
                         (-1, 2, -zmax, zmax - zr)]

                # keep violated or near-violated faces (dist < ε)
                for sign, axis, rhs_face, dist in faces:
                    ε = 1e-4     # small buffer to keep active at boundary
                    if dist < ε:
                        row = sparse.lil_matrix((1, n))
                        row[0, 3*t + axis] = sign
                        G_rows.append(row)
                        h_vals.append(rhs_face)

        # ---- 3) trust region ----------------------------------------- #
        for t, qk in enumerate(Xk):
            for j in range(3):
                idx = 3*t + j
                row = sparse.lil_matrix((1, n)); row[0, idx] =  1
                G_rows.append(row); h_vals.append(qk[j] + Δ)
                row = sparse.lil_matrix((1, n)); row[0, idx] = -1
                G_rows.append(row); h_vals.append(-qk[j] + Δ)

        # ---- 4) fixed start / goal equality -------------------------- #
        A_rows, b_vals = [], []
        if self.fix_start:
            for j in range(3):
                row = sparse.lil_matrix((1, n)); row[0,       j] = 1
                A_rows.append(row); b_vals.append(Xk[0, j])
        if self.fix_goal:
            for j in range(3):
                row = sparse.lil_matrix((1, n)); row[0, n-3+j] = 1
                A_rows.append(row); b_vals.append(Xk[-1, j])

        # stack
        G = sparse.vstack(G_rows, format="csr")
        h = np.asarray(h_vals)
        A = sparse.vstack(A_rows, format="csr") if A_rows else None
        b = np.asarray(b_vals)                  if b_vals else None
        return (P, q, G, h, A, b) if False else (G, h), (A, b)

    # -------- face helpers -------------------------------------------- #
    @staticmethod
    def _inflated_bounds(obs, margin):
        dx, dy, dz = obs["dimensions"]
        return (obs["x"] - dx/2 - margin,
                obs["x"] + dx/2 + margin,
                obs["y"] - dy/2 - margin,
                obs["y"] + dy/2 + margin,
                obs["z"] - dz/2 - margin,
                obs["z"] + dz/2 + margin)

    # -------- CVXOPT QP solve wrapper --------------------------------- #
    def _solve_qp(self, P, q, G, h):
        P_c = matrix(P.todense())
        q_c = matrix(q)
        G_c = matrix(G.todense())
        h_c = matrix(h)
        try:
            sol = solvers.qp(P_c, q_c, G_c, h_c)
            if sol["status"] != "optimal":
                return None
            return np.asarray(sol["x"]).flatten()
        except ValueError:          # infeasible or ill-posed
            return None
        


if __name__=="__main__":
    import numpy as np

    # reference trajectory: straight line
    T = 40
    X_ref = np.linspace([0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.3], T)

    # one box obstacle
    obstacles = [dict(x=0.25, y=0.25, z=0.15,
                    dimensions=(0.15, 0.15, 0.20))]

    # workspace (cuboid table)
    workspace = dict(x=(-0.1, 0.6),
                    y=(-0.1, 0.6),
                    z=( 0.0, 0.4))

    opt = SCPOptimiser()
    X_opt = opt.optimise(X_ref, obstacles, workspace,
                        workspace_type="cuboidal")
    print("first point:", X_opt[0], "  last:", X_opt[-1])

