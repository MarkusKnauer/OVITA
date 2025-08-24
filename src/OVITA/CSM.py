from scipy import sparse
import optuna
import json
import numpy as np
from cvxopt import matrix, solvers  # pip install cvxopt
import cvxpy as cp
import copy
from OVITA.util import (
    convert_to_vector,
    compare_trajectory,
    resample_trajectory,
    smooth_trajectory_spline,
)

solvers.options['maxiters'] = 1000 


class BarrierFunctions:
    def __init__(self):
        pass

    def cost_deviation(self, X_flat, X_ref):
        X = X_flat.reshape(-1, 4)
        return np.sum((X - X_ref) ** 2)

    def cost_smoothness(self, X_flat):
        """
        Penalizes non-smooth trajectories by summing squared second differences
        (i.e., curvature) across the trajectory. Excludes the first and last two points.
        """
        X = X_flat.reshape(-1, 3)  # Shape: (T, 3)
        second_diffs = X[2:] - 2 * X[1:-1] + X[:-2]  # Shape: (T-2, 3)
        return np.sum(second_diffs**2)

    def optimize_trajectory_cvxopt(
        self,
        X_ref: np.ndarray,
        obstacles: list,
        workspace: dict,
        workspace_type: str = "cuboidal",  # "cuboidal"  or  "arm"
        lambda_deviation: float = 10.0,
        lambda_smooth: float = 0.1,
    ):
        """
        Convex QP trajectory optimiser using CVXOPT, with

        • linearised obstacle avoidance               (common)
        • linearised workspace constraint, chosen by  `workspace_type`

            workspace_type="cuboidal"   → axis-aligned box bounds
            workspace_type="arm"→ spherical reachability (sum-of-links radius)

        Parameters
        ----------
        X_ref : (T,3) array
            Reference way-points.
        obstacles : list of dicts
            Each has keys {'x','y','z','dimensions' : (dx,dy,dz)} in world frame.
        workspace : dict
            If workspace_type == "cuboidal":
                {'x':(xmin,xmax), 'y':(...), 'z':(...)}
            elif workspace_type == "arm":
                {'centre':(cx,cy,cz), 'link_lengths':[l0,l1,...]}
        """
        # ------------------------------------------------------------------------- #
        # 0. Sizes & common quadratic objective
        # ------------------------------------------------------------------------- #
        T = X_ref.shape[0]  # number of way-points
        d_dim= X_ref.shape[1]
        n = d_dim * T  # decision variables  [x₀ y₀ z₀ vel …]
        I = sparse.eye(n)

        # 1-st difference operator ▽¹ (for path-length² / smoothness term)
        D1_1d = sparse.diags(
            [-np.ones(T - 1), np.ones(T - 1)],
            offsets=[0, 1],
            shape=(T - 1, T),
            format="csr",
        )
        D1 = sparse.kron(sparse.eye(d_dim), D1_1d, format="csr")

        P = 2 * (lambda_deviation * I + lambda_smooth * (D1.T @ D1))
        q = -2 * lambda_deviation * X_ref.flatten()

        # ------------------------------------------------------------------------- #
        # 1. Inequality constraints  G x ≤ h
        # ------------------------------------------------------------------------- #
        G_rows, h_vals = [], []

        # === (a) Workspace boundary – chose formulation ========================== #
        if workspace_type == "cuboidal":
            try:
                x_min, x_max = workspace["x"]
                y_min, y_max = workspace["y"]
                z_min, z_max = workspace["z"]
                c=np.array([(x_min+x_max)/2,(y_min+y_max)/2,(z_min+z_max)/2])
            except KeyError as e:  # give a helpful error early
                raise ValueError(f"Missing key for box workspace: {e}")
            m_bnd = self.safety_margin_boundary

            for t in range(T):
                ix, iy, iz = d_dim * t, d_dim * t + 1, d_dim * t + 2
                # xr, yr, zr = X_ref[t]   #Change
                
                # upper faces  (p ≤ max − m)
                for idx, ub in zip(
                    [ix, iy, iz], [x_max - m_bnd, y_max - m_bnd, z_max - m_bnd]
                ):
                    # dist=np.linalg.norm(np.array([xr, yr, zr]) - (c+np.array())  #Chnage
                    row = sparse.lil_matrix((1, n))
                    # row[0, idx] = dist/2  #Chnage
                    row[0, idx] = 1
                    G_rows.append(row)
                    h_vals.append(ub)
                # lower faces  (−p ≤ −(min + m)  ⇒  p ≥ min + m)
                for idx, lb in zip(
                    [ix, iy, iz], [x_min + m_bnd, y_min + m_bnd, z_min + m_bnd]
                ):
                    row = sparse.lil_matrix((1, n))
                    # row[0, idx]=-dist/2#Chnage
                    row[0, idx] = -1
                    G_rows.append(row)
                    h_vals.append(-lb)

        elif workspace_type == "arm":
            try:
                c = np.asarray(workspace["centre"], dtype=float)
                R_Max = workspace["r_max"]#float(np.sum(workspace["link_lengths"]))
                R_Min=workspace["r_min"]
            except KeyError as e:
                raise ValueError(f"Missing key for spherical workspace: {e}")
            R_Max -= self.safety_margin_boundary  # shrink by margin
            R_Min += self.safety_margin_boundary

            for t in range(T):
                ix, iy, iz = d_dim * t, d_dim * t + 1, d_dim * t + 2
                xr, yr, zr, _ = X_ref[t]
                d = np.array([xr, yr, zr]) - c
                if np.linalg.norm(d) < 1e-6:
                    d = np.array([1.0, 0.0, 0.0])  # avoid zero gradient

                # gx_lhs = 2.0 * d  # ∇g(x_ref)
                # gx_rhs = 2.0 * d.dot([xr, yr, zr]) + R**2 - d.dot(d)

                # row = sparse.lil_matrix((1, n))
                # row[0, ix] = gx_lhs[0]
                # row[0, iy] = gx_lhs[1]
                # row[0, iz] = gx_lhs[2]
                # G_rows.append(row)
                # h_vals.append(gx_rhs)
                # Outer sphere constraint: ||x - c||^2 <= R_max^2
                gx_lhs_outer = 2.0 * d
                gx_rhs_outer = 2.0 * d.dot([xr, yr, zr]) + R_Max**2 - d.dot(d)

                row_outer = sparse.lil_matrix((1, n))
                row_outer[0, ix] = gx_lhs_outer[0]
                row_outer[0, iy] = gx_lhs_outer[1]
                row_outer[0, iz] = gx_lhs_outer[2]
                G_rows.append(row_outer)
                h_vals.append(gx_rhs_outer)

                # Inner sphere constraint: -||x - c||^2 <= -R_min^2
                gx_lhs_inner = -2.0 * d
                gx_rhs_inner = -2.0 * d.dot([xr, yr, zr]) + d.dot(d) - R_Min**2

                row_inner = sparse.lil_matrix((1, n))
                row_inner[0, ix] = gx_lhs_inner[0]
                row_inner[0, iy] = gx_lhs_inner[1]
                row_inner[0, iz] = gx_lhs_inner[2]
                G_rows.append(row_inner)
                h_vals.append(gx_rhs_inner)
        elif workspace_type=="general":
            pass
        else:
            raise ValueError(
                f"workspace_type must be 'box' or 'sphere', got {workspace_type!r}"
            )

        # === (b) Boundary Gradient based obstacle separation ====================== #
        m_obs = self.safety_margin_obstacles
        for t in range(T):
            ix, iy, iz = d_dim * t, d_dim * t + 1, d_dim * t + 2
            xr, yr, zr, _ = X_ref[t]

            for obs in obstacles:
                xmin = obs["x"] - obs["dimensions"][0] / 2 - m_obs
                xmax = obs["x"] + obs["dimensions"][0] / 2 + m_obs
                ymin = obs["y"] - obs["dimensions"][1] / 2 - m_obs
                ymax = obs["y"] + obs["dimensions"][1] / 2 + m_obs
                zmin = obs["z"] - obs["dimensions"][2] / 2 - m_obs
                zmax = obs["z"] + obs["dimensions"][2] / 2 + m_obs
                R=np.linalg.norm(np.array([xmax, ymax, zmax]) - np.array([obs["x"],obs["y"],obs["z"]]))
                if (xr>xmin and xr<xmax and yr>ymin and yr<ymax and zr>zmin and zr<zmax):
                    d = np.array([xr, yr, zr]) - np.array([obs["x"],obs["y"],obs["z"]])
                    if np.linalg.norm(d) < 1e-6:
                        d = np.array([1.0, 0.0, 0.0])  # avoid zero gradient

                    gx_lhs = -2.0 * d  # ∇g(x_ref)
                    gx_rhs =  -(2.0 * d.dot([xr, yr, zr]) + R**2 - d.dot(d))

                    row = sparse.lil_matrix((1, n))
                    row[0, ix] = gx_lhs[0]
                    row[0, iy] = gx_lhs[1]
                    row[0, iz] = gx_lhs[2]
                    G_rows.append(row)
                    h_vals.append(gx_rhs)

            vel_idx = d_dim * (t - 1) + (d_dim - 1)         # index of last velocity element
            row_vel = sparse.lil_matrix((1, n))
            row_vel[0, vel_idx] = 1
            G_rows.append(row_vel)
            h_vals.append(self.max_vel)
        G = sparse.vstack(G_rows, format="csr")
        h = np.asarray(h_vals)
        
        # ------------------------------------------------------------------------- #
        # 2.  Equality constraints  (optional fixed start / goal)
        # ------------------------------------------------------------------------- #
        A_rows, b_vals = [], []
        if self.fix_start:
            for k in range(3):
                row = sparse.lil_matrix((1, n))
                row[0, k] = 1
                A_rows.append(row)
                b_vals.append(X_ref[0, k])
        if self.fix_goal:
            for k in range(3):
                row = sparse.lil_matrix((1, n))
                row[0, -d_dim + k] = 1
                A_rows.append(row)
                b_vals.append(X_ref[-1, k])

        A = sparse.vstack(A_rows, format="csr") if A_rows else None
        b = np.asarray(b_vals) if b_vals else None

        # ------------------------------------------------------------------------- #
        # 3.  Solve with CVXOPT
        # ------------------------------------------------------------------------- #
        P_cvx = matrix(P.todense())
        q_cvx = matrix(q)
        G_cvx = matrix(G.todense())
        h_cvx = matrix(h)

        if A is not None:
            sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, matrix(A.todense()), matrix(b))
        else:
            sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)

        return np.asarray(sol["x"]).reshape(-1, 4)
    
 
    def cost_selection(self, trajectory):
        return self.cost_deviation(trajectory, self.trajectory)

    def store_data(
        self,
        trajectory,
        workspace_type,
        workspace_bounds,
        obstacles,
        safety_margin_obstacles,
        safety_margin_boundary,
        fix_start=False, 
        fix_goal=False,
        max_vel=3.0
    ):
        self.workspace_type = workspace_type
        self.workspace_bounds = workspace_bounds
        self.trajectory = trajectory
        self.obstacles = obstacles
        self.safety_margin_obstacles = safety_margin_obstacles
        self.safety_margin_boundary = safety_margin_boundary
        self.fix_start = fix_start
        self.fix_goal = fix_goal
        self.max_vel=max_vel

    def optuna_optimize_trajectory(self, trial):
        lambda_deviation = trial.suggest_float("lambda_deviation", 0.1, 100, log=True)
        lambda_smooth = trial.suggest_float("lambda_smooth", 0.001, 10.0, log=True)

        optimized_trajectory = self.optimize_trajectory_cvxopt(
            self.trajectory,
            self.obstacles,
            self.workspace_bounds,
            self.workspace_type,
            lambda_deviation,
            lambda_smooth,
        )

        trial.set_user_attr("optimised_trajectory", optimized_trajectory)

        # how to choose the best trajectorys
        cost = self.cost_selection(optimized_trajectory)

        return cost


def constraint_satisfaction_module(
    trajectory,
    workspace_type,
    workspace_bounds,
    obstacles,
    safety_margin_obstacles,
    safety_margin_boundary,
    fix_start=False, 
    fix_goal=False, 
    max_vel=3.0,
    apply_spline_smooth=False
):
    cbf = BarrierFunctions()
    cbf.store_data(
        trajectory,
        workspace_type,
        workspace_bounds,
        obstacles,
        safety_margin_obstacles,
        safety_margin_boundary,
        fix_start, 
        fix_goal, 
        max_vel 
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(
        cbf.optuna_optimize_trajectory,
        n_trials=50,
        n_jobs=6,
    )  # Try 20 trials or more

    optimized_trajectory = study.best_trial.user_attrs["optimised_trajectory"]
    modified_trajectory= optimized_trajectory
    # modified_trajectory = np.hstack((optimized_trajectory, trajectory[:, -1:]))
    if apply_spline_smooth:
        modified_trajectory = smooth_trajectory_spline(
            smooth_trajectory_spline(modified_trajectory, num_points=40), num_points=100
        )

    return modified_trajectory

# Example Usage
if __name__ == "__main__":
    workspace_type = "arm"  # choose from cuboidal or arm
    if workspace_type == "cuboidal":
        workspace_bounds = {
            "x": [-2.5, 2.5],
            "y": [-2.5, 2.5],
            "z": [-0.01, 0.01],
        }
    elif workspace_type == "arm":
        workspace_bounds = {
            "centre": [-0.5, 0, 0.6],
            "r_max": 1.2,
            "r_min": 0.15
        }
    # elif workspace_type=="general":
    #     workspace_bounds={"x": [-1, 1],
    #         "y": [-0.25, 1],
    #         "z": [-1, 0],}
    else:
        print(f"{workspace_type} not implemented. Choose from 'arm' or 'cuboidal'")

    # add 0.01 to the safety margin
    safety_margin_obstacles = 0.01
    safety_margin_boundary = 0.01

    file_path = "dataset/robot_subset\Manipulator/trajectory_4.json"
    with open(file_path, "r") as file:
        data = json.loads(file.read())

    temp = data
    trajectory = resample_trajectory(copy.deepcopy(data.get('trajectory', [])),100)

    obstacles = temp["objects"]
    for item in obstacles:
        item['name']=item['name'].lower()
        if 'dimensions' not in item.keys():
            item.update({'dimensions': [0.05]*3})
    compare_trajectory(
        trajectory,
        trajectory,
        "test",
        obstacles,
        visualize_workspace=True,
        workspace_type=workspace_type,
        workspace_bounds=workspace_bounds,
    )

    modified_trajectory = constraint_satisfaction_module(
        trajectory=np.array(trajectory),
        workspace_type=workspace_type,
        workspace_bounds=workspace_bounds,
        obstacles=obstacles,
        safety_margin_obstacles=safety_margin_obstacles,
        safety_margin_boundary=safety_margin_boundary,
    )
    compare_trajectory(
        trajectory,
        modified_trajectory,
        "test_CBF",
        obstacles,
        visualize_workspace=True,
        workspace_type=workspace_type,
        workspace_bounds=workspace_bounds,
    )
    # import ipdb

    # ipdb.set_trace()
