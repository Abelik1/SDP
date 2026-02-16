def backend_run(config, system, analyzer):
    

    eq_id = config.get("selected_equation_id")
    eq_name = config.get("selected_equation_name", "")
    vars_str = config.get("variables_str", config.get("variables", ""))
    custom_func_text = config.get("custom_function", "")

    print("\n[backend_run] Selected equation:", eq_id, f"({eq_name})")
    print("[backend_run] Variables string:", vars_str)
    print("[backend_run] Custom text length:", len(custom_func_text))

    vars_dict = parse_variables_string(vars_str)
    print("[backend_run] Parsed variables:", vars_dict)

    # -----------------------
    # Pull UI-controlled vars
    # -----------------------
    dA = int(vars_dict.get("dA", system.dA))
    dAp = int(vars_dict.get("dAp", system.dAp))
    beta = float(vars_dict.get("beta", system.beta))

    symmetric_flag = bool(vars_dict.get("symmetric", True))
    if symmetric_flag and dA != dAp:
        symmetric_flag = False  # can't be symmetric if dims differ

    reset_system = bool(vars_dict.get("reset_system", False))

    solver_override = str(vars_dict.get("solver", system.solver_default))
    eps_eq_global = float(vars_dict.get("eps_eq_global", system.eps_eq_global))
    eps_eq_local = float(vars_dict.get("eps_eq_local", system.eps_eq_local))
    eps_gibbs = float(vars_dict.get("eps_gibbs", getattr(system, "eps_gibbs", 1e-8)))

    # -----------------------
    # Rebuild system if needed
    # -----------------------
    needs_rebuild = (
        reset_system
        or (system.dA != dA)
        or (system.dAp != dAp)
        or (abs(system.beta - beta) > 1e-15)
        or (str(system.solver_default) != solver_override)
        or (not np.allclose(system.H_Ap, system.H_A) and symmetric_flag)
    )

    if needs_rebuild:
        log_info(
            "System rebuild",
            f"dA={dA}, dAp={dAp}, beta={beta}, solver={solver_override}, symmetric={symmetric_flag}\n"
            f"eps_eq_global={eps_eq_global}, eps_eq_local={eps_eq_local}, eps_gibbs={eps_gibbs}"
        )
        system, analyzer = build_system_and_analyzer(
            dA=dA,
            dAp=dAp,
            beta=beta,
            solver=solver_override,
            tol=system.tol_default,
            symmetric=symmetric_flag,
            eps_eq_global=eps_eq_global,
            eps_eq_local=eps_eq_local,
            eps_gibbs=eps_gibbs,
        )
    else:
        # Always update eps values even if no rebuild
        system.eps_eq_global = eps_eq_global
        system.eps_eq_local = eps_eq_local
        system.eps_gibbs = eps_gibbs
        system.solver_default = solver_override

    # seed (negative = don't touch global rng)
    seed = int(vars_dict.get("seed", -1))
    if seed >= 0:
        np.random.seed(seed)

    d = system.dA * system.dAp

    # Helper: global dephasing in energy eigenbasis
    def dephase_global_in_energy_basis(rho):
        H_A = system.H_A
        H_Ap = system.H_Ap
        eA, UA = np.linalg.eigh((H_A + dagger(H_A)) / 2.0)
        eAp, UAp = np.linalg.eigh((H_Ap + dagger(H_Ap)) / 2.0)
        U_tot = np.kron(UA, UAp)
        rho_e = dagger(U_tot) @ rho @ U_tot
        rho_e_deph = np.diag(np.diag(rho_e))
        return 0.5 * (U_tot @ rho_e_deph @ dagger(U_tot) + dagger(U_tot @ rho_e_deph @ dagger(U_tot)))

    # ====================================================
    # Dispatch
    # ====================================================

    if eq_id == "tfd_vs_dephased":
        try:
            results = analyzer.analyze_tfd_vs_dephased(
                solver=system.solver_default,
                tol=system.tol_default,
                verbose=False,
            )
            rep_tfd = results["tfd"]
            rep_deph = results["tfd_dephased"]

            tfd_state = analyzer.factory.tfd_state()
            tfd_deph = rep_deph["distance_to_LT"]["sigma_closest"]  # should be itself (LT)
            if tfd_deph is None:
                tfd_deph = dephase_global_in_energy_basis(tfd_state)

            conv_report = analyzer.analyze_pair(
                tfd_state,
                tfd_deph,
                label="TFD_to_TFD_deph",
                solver=system.solver_default,
                tol=system.tol_default,
                verbose=False,
                eps_eq_global=system.eps_eq_global,
                eps_eq_local=system.eps_eq_local,
            )

            gp = conv_report["feasibility"]["Global_GP"]
            lgp = conv_report["feasibility"]["Local_GP"]

            text = (
                "TFD vs dephased TFD\n\n"
                "=== Monotones ===\n"
                f"TFD: I(A:B)={rep_tfd['monotones']['I_rho']:.4f}, D(ρ||γ⊗γ)={rep_tfd['monotones']['D_rho_vs_gamma']:.4f}\n"
                f"Dephased: I(A:B)={rep_deph['monotones']['I_rho']:.4f}, D(ρ||γ⊗γ)={rep_deph['monotones']['D_rho_vs_gamma']:.4f}\n\n"
                "=== Convertibility TFD → dephased ===\n"
                f"Global GP: feasible={gp['feasible']} (status={gp['status']})\n"
                f"Local  GP: feasible={lgp['feasible']} (status={lgp['status']})\n"
            )
            log_info("TFD vs Dephased TFD", text)
            print("Finished TFD vs dephased analysis.")
        except Exception as e:
            log_error(
                "TFD Error",
                "TFD analysis failed. This typically means symmetric=True is required (H_A == H_A').\n\n"
                f"Details:\n{e}"
            )

    elif eq_id == "random_pair_gp_lgp":
        tau = random_state(d)
        tau_p = random_state(d)
        report = analyzer.analyze_pair(
            tau,
            tau_p,
            label="random_pair",
            solver=system.solver_default,
            tol=system.tol_default,
            verbose=False,
        )
        gp = report["feasibility"]["Global_GP"]
        lgp = report["feasibility"]["Local_GP"]

        text = (
            "Random τ → τ' convertibility\n\n"
            f"Global GP: feasible={gp['feasible']} (status={gp['status']})\n"
            f"Local  GP: feasible={lgp['feasible']} (status={lgp['status']})\n\n"
            f"D_tau  = {report['monotones']['D_tau_vs_gamma']:.4f}\n"
            f"D_taup = {report['monotones']['D_taup_vs_gamma']:.4f}\n"
            f"I_tau  = {report['monotones']['I_tau']:.4f}\n"
            f"I_taup = {report['monotones']['I_taup']:.4f}\n"
        )
        log_info("Random Pair GP / LGP Test", text)

        fig, ax = plt.subplots()
        ax.scatter(
            [report["monotones"]["D_tau_vs_gamma"], report["monotones"]["D_taup_vs_gamma"]],
            [report["monotones"]["I_tau"], report["monotones"]["I_taup"]],
        )
        ax.set_xlabel("D(ρ || γ⊗γ)")
        ax.set_ylabel("I(A:B)")
        ax.set_title("Random pair in (D,I) plane")
        ax.annotate("τ", (report["monotones"]["D_tau_vs_gamma"], report["monotones"]["I_tau"]))
        ax.annotate("τ'", (report["monotones"]["D_taup_vs_gamma"], report["monotones"]["I_taup"]))
        path = save_plot(fig, "random_pair_DI.png")
        log_info("Random Pair Plot", f"Saved:\n{path}")

    elif eq_id == "mix_with_gamma":
        # Choose a "source" state: TFD if symmetric and dims match, else random.
        try:
            rho0 = analyzer.factory.tfd_state()
            label = "TFD"
        except Exception:
            rho0 = random_state(d)
            label = "random"

        lam_grid = np.linspace(0.0, 1.0, int(vars_dict.get("num_samples", 25)))
        reports = analyzer.scan_mixture_with_gamma(
            rho0,
            lam_grid,
            solver=system.solver_default,
            tol=system.tol_default,
            verbose=False,
        )

        lams = [r["lambda"] for r in reports]
        Dvals = [r["monotones"]["D_rho_vs_gamma"] for r in reports]
        Ivals = [r["monotones"]["I_rho"] for r in reports]
        distLT = [r["distance_to_LT"]["distance"] for r in reports]

        fig, ax = plt.subplots()
        ax.plot(lams, Dvals, label="D(ρ||γ⊗γ)")
        ax.plot(lams, Ivals, label="I(A:B)")
        ax.plot(lams, distLT, label="dist_to_LT")
        ax.set_xlabel("λ in (1−λ)ρ + λ γ⊗γ")
        ax.set_title(f"Mix with γ⊗γ starting from {label}")
        ax.legend()
        path = save_plot(fig, "mix_with_gamma.png")
        log_info("Mixture with γ⊗γ", f"Saved plot:\n{path}")
        print("Finished mixture with gamma analysis.")

    elif eq_id == "closest_lt_distance":
        rho = random_state(d)
        rep = analyzer.analyze_single_state(
            rho,
            label="random_rho",
            solver=system.solver_default,
            tol=system.tol_default,
            verbose=False,
        )
        text = (
            "Distance to LT (random state)\n\n"
            f"Is LT? {rep['LT_membership']['is_LT']}\n"
            f"Dist to LT (trace): {rep['distance_to_LT']['distance']:.4e} (status={rep['distance_to_LT']['status']})\n"
        )
        if rep["distance_to_classical_LT"]["distance"] is not None:
            text += (
                f"Dist to classical LT (trace): {rep['distance_to_classical_LT']['distance']:.4e} "
                f"(status={rep['distance_to_classical_LT']['status']})\n"
            )
        log_info("Closest LT Distance", text)
        print("Finished closest LT distance analysis.") 

    elif eq_id == "lt_region_geometry":
        num_samples = int(vars_dict.get("num_samples", 50))
        classical_flag = bool(vars_dict.get("classical", False))

        extremals = analyzer.sample_extremal_lt_states(
            num_samples=num_samples,
            classical=classical_flag,
            solver=system.solver_default,
            tol=system.tol_default,
            verbose=False,
        )
        if not extremals:
            log_warning("LT Region Geometry", "No extremal LT states found (all SDPs failed).")
            return

        D_vals = [rep["monotones"]["D_rho_vs_gamma"] for rep in extremals]
        I_vals = [rep["monotones"]["I_rho"] for rep in extremals]

        fig, ax = plt.subplots()
        ax.scatter(D_vals, I_vals)
        ax.set_xlabel("D(ρ || γ⊗γ)")
        ax.set_ylabel("I(A:B)")
        ax.set_title("Extremal LT boundary (support-function samples)")
        path = save_plot(fig, "lt_region_geometry_DI.png")
        log_info("LT Region Geometry", f"Sampled {len(extremals)} extremals. Saved:\n{path}")
        print("Finished LT region geometry analysis.")

    elif eq_id == "lt_interior_geometry":
        num_samples = int(vars_dict.get("num_samples", 200))
        use_classical_metric = bool(vars_dict.get("classical", False))

        projected = []
        for _ in range(num_samples):
            rho = random_state(d)

            sigma_LT, _, _ = system.closest_lt_state(
                rho, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False
            )
            if sigma_LT is None:
                continue

            D_val, I_val, _, _ = system.monotones(sigma_LT)

            if use_classical_metric and system.dims == (2, 2):
                _, dist_cl, _ = system.closest_lt_state(
                    rho, classical=True, solver=system.solver_default, tol=system.tol_default, verbose=False
                )
                if dist_cl is None:
                    continue
                z_raw = float(dist_cl)
                z_name = "dist_to_classical_LT"
            else:
                cm = system.correlation_metrics(sigma_LT)
                z_raw = float(cm["C_fro"])
                z_name = "||C||_F  where  rho = gamma⊗gamma + C"

            projected.append({"rho": sigma_LT, "D": D_val, "I": I_val, "Z_raw": z_raw})

        if not projected:
            log_warning("LT Interior Geometry", "No LT projections succeeded.")
            return

        D_vals = np.array([p["D"] for p in projected], dtype=float)
        I_vals = np.array([p["I"] for p in projected], dtype=float)
        Z_raw  = np.array([p["Z_raw"] for p in projected], dtype=float)

        z_min = float(np.min(Z_raw))
        z_max = float(np.max(Z_raw))
        denom = (z_max - z_min) if (z_max - z_min) > 1e-15 else 1.0

        # Scale as "max -> min" so you get visible separation even for tiny raw ranges:
        # z_plot = 0 means max(raw), z_plot = 1 means min(raw)
        Z_plot = (z_max - Z_raw) / denom

        # 3D plot (D, I, scaled Z)
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(D_vals, I_vals, Z_plot, c=Z_plot)
        ax.set_xlabel("D(ρ || γ⊗γ)")
        ax.set_ylabel("I(A:B)")
        ax.set_zlabel(f"{z_name} (scaled: 0=max, 1=min)\nraw range [{z_min:.3e}, {z_max:.3e}]")
        ax.set_title("LT interior: random → LT projection")

        cbar = fig.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label(f"{z_name} (scaled max→min; raw in title)")

        path = save_plot(fig, "lt_interior_geometry_3d.png")
        log_info("LT Interior Geometry", f"Projected {len(projected)} states. Saved:\n{path}")
        fig.show()
        print("Finished LT interior geometry analysis.")

    
    elif eq_id == "lt_geometry_combined":
        # Final figure: interior + boundary + classical line (if qubits)
        num_samples = int(vars_dict.get("num_samples", 200))
        classical_flag = bool(vars_dict.get("classical", False))

        n_interior = max(20, num_samples)
        n_boundary = max(20, min(100, num_samples // 2))

        # Interior points via projection
        interior = []
        for _ in range(n_interior):
            rho = random_state(d)
            sigma_LT, _, _ = system.closest_lt_state(
                rho, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False
            )
            if sigma_LT is None:
                continue
            D_val, I_val, _, _ = system.monotones(sigma_LT)
            interior.append((D_val, I_val))

        # Boundary extremals
        extremals = analyzer.sample_extremal_lt_states(
            num_samples=n_boundary,
            classical=classical_flag,
            solver=system.solver_default,
            tol=system.tol_default,
            verbose=False,
        )
        boundary = [(rep["monotones"]["D_rho_vs_gamma"], rep["monotones"]["I_rho"]) for rep in extremals]

        fig, ax = plt.subplots()
        if interior:
            ax.scatter([p[0] for p in interior], [p[1] for p in interior], alpha=0.25, label="interior (proj)")
        if boundary:
            ax.scatter([p[0] for p in boundary], [p[1] for p in boundary], alpha=0.9, label="boundary (extremal)")

        # classical LT line for qubits (if available)
        if system.dims == (2, 2):
            reports_cl = analyzer.scan_classical_LT_line_qubit(
                num_points=60,
                solver=system.solver_default,
                tol=system.tol_default,
                verbose=False,
            )
            D_cl = [rep["monotones"]["D_rho_vs_gamma"] for rep in reports_cl]
            I_cl = [rep["monotones"]["I_rho"] for rep in reports_cl]
            ax.plot(D_cl, I_cl, linestyle="--", linewidth=1.5, label="classical LT line")

        ax.set_xlabel("D(ρ || γ⊗γ)")
        ax.set_ylabel("I(A:B)")
        ax.set_title("LT geometry: boundary + interior")
        ax.legend()
        path = save_plot(fig, "lt_geometry_combined.png")
        log_info("LT Geometry Combined", f"Saved:\n{path}")
        print("Finished LT geometry combined analysis.")

    elif eq_id == "lt_convertibility_graph":
        # Build an LT ensemble and compute GP vs LGP reachability graphs
        num_samples = int(vars_dict.get("num_samples", 25))
        use_classical = bool(vars_dict.get("classical", False))

        N_target = max(8, min(30, num_samples))  # clamp for runtime
        states = []
        labels = []

        GAxGAp = np.kron(system.gammaA, system.gammaAp)

        # --- anchors: bottom element gamma⊗gamma
        states.append(0.5 * (GAxGAp + dagger(GAxGAp)))
        labels.append("gamma⊗gamma")

        # --- anchors: TFD and dephased TFD when available
        try:
            tfd = analyzer.factory.tfd_state()
            tfd_deph = dephase_global_in_energy_basis(tfd)
            states.append(tfd)
            labels.append("TFD")
            states.append(tfd_deph)
            labels.append("TFD_dephased")
        except Exception:
            pass

        # Optional: include classical LT samples only if the checkbox is enabled
        if use_classical and system.dims == (2, 2):
            n_cl = min(6, max(2, N_target // 4))
            reports_cl = analyzer.scan_classical_LT_line_qubit(
                num_points=n_cl, solver=system.solver_default, tol=system.tol_default, verbose=False
            )
            for rep in reports_cl:
                a = rep.get("a", None)
                if a is None:
                    continue
                rho_cl = analyzer.factory.classical_LT_point_qubit(a=a)
                states.append(rho_cl)
                labels.append(f"classical a={a:.2f}")

        # Extremal LT boundary samples
        n_ext = min(10, max(3, N_target // 3))
        extremals = analyzer.sample_extremal_lt_states(
            num_samples=n_ext, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False
        )
        for k, rep in enumerate(extremals):
            states.append(rep["rho"])
            labels.append(f"ext {k}")

        # Interior LT points via projection
        while len(states) < N_target:
            rho = random_state(d)
            sigma_LT, _, _ = system.closest_lt_state(
                rho, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False
            )
            if sigma_LT is None:
                continue
            states.append(sigma_LT)
            labels.append(f"proj {len(states)-1}")

        N = len(states)
        log_info("Convertibility Graph", f"Testing pairwise reachability on N={N} LT states (can take time).")

        # Precompute correlation strengths / signatures
        I_node = np.zeros(N, dtype=float)
        D_node = np.zeros(N, dtype=float)
        Cfro_node = np.zeros(N, dtype=float)
        Csvals_top = []

        for i in range(N):
            D_i, I_i, _, _ = system.monotones(states[i])
            cm = system.correlation_metrics(states[i])
            D_node[i] = float(D_i)
            I_node[i] = float(I_i)
            Cfro_node[i] = float(cm["C_fro"])
            Csvals_top.append(cm["C_svals_top"])

        # Adjacency + local residuals
        G = np.zeros((N, N), dtype=int)
        L = np.zeros((N, N), dtype=int)
        L_res = np.full((N, N), np.inf, dtype=float)

        # Pairwise convertibility with monotone pre-screen:
        # If I(i) < I(j), then i -> j is impossible under any GP (global or local),
        # because local GP ⊂ global GP and D=I on LT.
        mono_tol = 1e-10

        for i in range(N):
            for j in range(N):
                if i == j:
                    G[i, j] = 1
                    L[i, j] = 1
                    L_res[i, j] = 0.0
                    continue

                if I_node[i] + mono_tol < I_node[j]:
                    G[i, j] = 0
                    L[i, j] = 0
                    L_res[i, j] = np.inf
                    continue

                g_ok, _ = system.check_global_gp_feasible(
                    states[i], states[j], solver=system.solver_default, tol=system.tol_default, verbose=False
                )
                G[i, j] = 1 if g_ok else 0

                l_ok, l_status, l_det = system.check_local_gp_feasible(
                    states[i], states[j],
                    solver=system.solver_default, tol=system.tol_default, verbose=False,
                    return_details=True
                )
                L[i, j] = 1 if l_ok else 0
                L_res[i, j] = float(l_det.get("residual", np.inf))

        # Incomparability under LGP
        unordered = 0
        incomparable = 0
        for i in range(N):
            for j in range(i + 1, N):
                unordered += 1
                if (L[i, j] == 0) and (L[j, i] == 0):
                    incomparable += 1
        inc_rate = incomparable / unordered if unordered else 0.0

        # Heatmaps
        fig, ax = plt.subplots()
        ax.imshow(G, interpolation="nearest", aspect="auto")
        ax.set_title("Adjacency: Global GP (i → j)")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        pathG = save_plot(fig, "convertibility_global_heatmap.png")

        fig, ax = plt.subplots()
        ax.imshow(L, interpolation="nearest", aspect="auto")
        ax.set_title(f"Adjacency: Local GP (i → j) | incomparability={inc_rate:.2%}")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        pathL = save_plot(fig, "convertibility_local_heatmap.png")

        # ---- SCC decomposition (Kosaraju) on local graph ----
        def _scc_kosaraju(adj: np.ndarray):
            n = adj.shape[0]
            g = [list(np.where(adj[i] == 1)[0]) for i in range(n)]
            rg = [list(np.where(adj[:, i] == 1)[0]) for i in range(n)]

            seen = [False] * n
            order = []

            def dfs1(v):
                seen[v] = True
                for u in g[v]:
                    if not seen[u]:
                        dfs1(u)
                order.append(v)

            for v in range(n):
                if not seen[v]:
                    dfs1(v)

            comp = [-1] * n
            comps = []

            def dfs2(v, cid):
                comp[v] = cid
                comps[-1].append(v)
                for u in rg[v]:
                    if comp[u] == -1:
                        dfs2(u, cid)

            for v in reversed(order):
                if comp[v] == -1:
                    comps.append([])
                    dfs2(v, len(comps) - 1)
            return comps

        comps = _scc_kosaraju(L)
        comps_sorted = sorted(comps, key=len, reverse=True)

        # Print SCC summary + node signatures
        lines = []
        lines.append(f"Local GP SCCs: {len(comps_sorted)} components")
        for k, comp in enumerate(comps_sorted[:10]):
            if len(comp) <= 1:
                continue
            lines.append(f"\nSCC #{k} | size={len(comp)}")
            for idx in sorted(comp):
                sv = Csvals_top[idx]
                sv_str = ", ".join([f"{x:.3e}" for x in sv])
                lines.append(
                    f"  [{idx:02d}] {labels[idx]:>14} | I={I_node[idx]:.6f} | ||C||_F={Cfro_node[idx]:.3e} | svals(C)~[{sv_str}]"
                )
        log_info("Local GP SCC structure", "\n".join(lines))

        # Print a compact edge list (local feasible), prioritizing high-I sources
        edges = []
        for i in range(N):
            for j in range(N):
                if i != j and L[i, j] == 1:
                    edges.append((i, j, I_node[i], I_node[j], L_res[i, j]))
        edges.sort(key=lambda t: (-t[2], t[4]))  # high I(source), then low residual

        edge_lines = ["Top Local-GP edges (source→target):"]
        for (i, j, Ii, Ij, res) in edges[:25]:
            edge_lines.append(
                f"  {labels[i]}[{i}] → {labels[j]}[{j}] | ΔI={Ii-Ij:+.3e} | I: {Ii:.6f}->{Ij:.6f} | residual={res:.3e}"
            )
        log_info("Local GP edge sample", "\n".join(edge_lines))

        # Directed graph in 3D (embedded coords) with node color = I
        coords_rng = np.random.default_rng(0)
        coords = np.array([embed_state_3d(system, r, rng=coords_rng) for r in states])

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=I_node)

        # draw edges if not too dense
        if N <= 25:
            for i in range(N):
                for j in range(N):
                    if i != j and L[i, j] == 1:
                        ax.plot(
                            [coords[i, 0], coords[j, 0]],
                            [coords[i, 1], coords[j, 1]],
                            [coords[i, 2], coords[j, 2]],
                            linewidth=0.6,
                            alpha=0.35,
                        )

        # Meaningful axis labels when (2,2)
        if system.dims == (2, 2):
            ax.set_xlabel("⟨σx⊗σx⟩")
            ax.set_ylabel("⟨σy⊗σy⟩")
            ax.set_zlabel("⟨σz⊗σz⟩")
        else:
            ax.set_xlabel("embed x")
            ax.set_ylabel("embed y")
            ax.set_zlabel("embed z")

        ax.set_title("Local GP reachability graph (3D embed), color=I(A:B)")
        cbar = fig.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label("I(A:B) (correlation strength on LT)")

        pathGraph = save_plot(fig, "convertibility_local_graph_3d.png")

        log_info(
            "Convertibility Graph Results",
            f"Incomparability rate (Local GP): {inc_rate:.2%}\n"
            f"Saved heatmaps:\n- {pathG}\n- {pathL}\nSaved 3D graph:\n- {pathGraph}"
        )
        print("Finished LT convertibility graph analysis.")

    elif eq_id == "lt_family_ray_validation":
        if system.dims != (2, 2):
            log_warning("LT family ray validation", "This experiment is implemented for dims=(2,2) only.")
            return

        label = str(vars_dict.get("label", "XX"))
        num_points = int(vars_dict.get("num_points", 21))
        include_negative = bool(vars_dict.get("include_negative", False))
        pair_mode = str(vars_dict.get("pair_mode", "adjacent"))
        p_shrink = float(vars_dict.get("p_shrink", 0.98))

        fam = analyzer.scan_lt_ray_family_pauli(
            label=label,
            num_points=num_points,
            include_negative=include_negative,
            p_shrink=p_shrink,
        )
        rep = analyzer.validate_local_gp_monotones_on_ray(
            fam["p_list"],
            fam["states"],
            pair_mode=pair_mode,
            solver=system.solver_default,
            tol=system.tol_default,
            eps_map_local=system.eps_eq_local,
            eps_gibbs=system.eps_gibbs,
            verbose=False,
        )

        title = f"Ray family ({label}), β={system.beta}"
        plot_lt_family_scan(fam["p_list"], rep["observables"], title, f"ray_{label}")

        v = rep["violations"]
        text = (
            f"Family: ρ(p)=γ⊗γ+pC0 with C0=(1/4){label}\n"
            f"Analytic PSD bounds: p∈[{fam['p_bounds'][0]:.6g}, {fam['p_bounds'][1]:.6g}] (scan uses shrink={p_shrink})\n"
            f"Scan points: {len(fam['p_list'])}, pair_mode={pair_mode}\n"
            f"Local-GP tests run: {len(rep['edges'])}\n"
            f"Feasible edges: {sum(1 for e in rep['edges'] if e['local_feasible'])}\n"
            f"Monotone-inequality violations among feasible edges: {len(v)}\n\n"
            "Saved plots:\n"
            f"- png/ray_{label}_I_vs_p.png\n"
            f"- png/ray_{label}_C_norms_vs_p.png\n"
            f"- png/ray_{label}_T_svals_vs_p.png (if applicable)\n"
        )
        if len(v) > 0:
            text += "\nFirst violation (debug):\n" + str(v[0])
        log_info("LT Family Ray Validation", text)

    elif eq_id == "lt_family_diagT_validation":
        if system.dims != (2, 2):
            log_warning("LT family diagT validation", "This experiment is implemented for dims=(2,2) only.")
            return

        # Parse t0 as 'tx;ty;tz' or 'tx,ty,tz' or separate keys
        t0_raw = vars_dict.get("t0", None)
        if t0_raw is not None:
            t0_str = str(t0_raw)
            import re
            parts = [p for p in re.split(r"[;,]", t0_str) if p.strip()]
            if len(parts) != 3:
                raise ValueError("t0 must be 'tx;ty;tz' or 'tx,ty,tz'")
            t0 = (float(parts[0]), float(parts[1]), float(parts[2]))
        else:
            t0 = (
                float(vars_dict.get("tx", 1.0)),
                float(vars_dict.get("ty", 0.0)),
                float(vars_dict.get("tz", 0.0)),
            )

        num_points = int(vars_dict.get("num_points", 30))
        include_negative = bool(vars_dict.get("include_negative", False))
        pair_mode = str(vars_dict.get("pair_mode", "adjacent"))
        p_shrink = float(vars_dict.get("p_shrink", 0.98))

        fam = analyzer.scan_lt_diagT_family(
            t0=t0,
            num_points=num_points,
            include_negative=include_negative,
            p_shrink=p_shrink,
        )
        rep = analyzer.validate_local_gp_monotones_on_ray(
            fam["p_list"],
            fam["states"],
            pair_mode=pair_mode,
            solver=system.solver_default,
            tol=system.tol_default,
            eps_map_local=system.eps_eq_local,
            eps_gibbs=system.eps_gibbs,
            verbose=False,
        )

        t0x, t0y, t0z = t0
        tag = f"{t0x:g}_{t0y:g}_{t0z:g}".replace("-", "m").replace(".", "p")
        title = f"diagT ray (t0={t0}), β={system.beta}"
        plot_lt_family_scan(fam["p_list"], rep["observables"], title, f"diagT_{tag}")

        v = rep["violations"]
        text = (
            f"Family: ρ(p)=γ⊗γ + p*(t0x XX + t0y YY + t0z ZZ)/4 with t0={t0}\n"
            f"Analytic PSD bounds: p∈[{fam['p_bounds'][0]:.6g}, {fam['p_bounds'][1]:.6g}] (scan uses shrink={p_shrink})\n"
            f"Scan points: {len(fam['p_list'])}, pair_mode={pair_mode}\n"
            f"Local-GP tests run: {len(rep['edges'])}\n"
            f"Feasible edges: {sum(1 for e in rep['edges'] if e['local_feasible'])}\n"
            f"Monotone-inequality violations among feasible edges: {len(v)}\n\n"
            "Saved plots:\n"
            f"- png/diagT_{tag}_I_vs_p.png\n"
            f"- png/diagT_{tag}_C_norms_vs_p.png\n"
            f"- png/diagT_{tag}_T_svals_vs_p.png\n"
        )
        if len(v) > 0:
            text += "\nFirst violation (debug):\n" + str(v[0])
        log_info("LT Family DiagT Validation", text)

    
    elif eq_id == "lt_C_diagT_plane_characterise":
        # 2D characterisation of the LT "+C" slice in a diag-correlation plane (qubits only)
        if system.dims != (2, 2):
            log_warning("LT C-plane characterisation", "Implemented for dims=(2,2) only.")
            return

        axes_raw = str(vars_dict.get("plane_axes", "xz"))
        axes_str = axes_raw.replace(",", "").replace(" ", "").lower()
        if len(axes_str) != 2:
            axes_str = "xz"
        axes = (axes_str[0], axes_str[1])

        num_angles = int(vars_dict.get("num_angles", 360))
        num_points = int(vars_dict.get("num_points", 40))
        interior_scale = float(vars_dict.get("interior_scale", 0.5))
        seed = int(vars_dict.get("seed", 0))

        bd = analyzer.diagT_plane_boundary(
            axes=axes,
            num_angles=num_angles,
            include_negative=True,
            tol=1e-12,
        )

        # Plot boundary in (t_axis1, t_axis2)
        axmap = {"x": 0, "y": 1, "z": 2}
        k1 = axmap[axes[0]]
        k2 = axmap[axes[1]]

        fig, ax = plt.subplots()
        ax.plot(bd["t_max"][:, k1], bd["t_max"][:, k2], linewidth=1.5, label="boundary (p_max)")
        ax.plot(bd["t_min"][:, k1], bd["t_min"][:, k2], linewidth=1.0, linestyle="--", label="boundary (p_min)")
        ax.set_xlabel(f"t{axes[0]}")
        ax.set_ylabel(f"t{axes[1]}")
        ax.set_title(f"Feasible LT slice in diagT plane ({axes[0]}{axes[1]})")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        pathB = save_plot(fig, f"lt_C_plane_{axes[0]}{axes[1]}_boundary.png")

        # Sample interior points along random directions (scaled from p_max)
        rng = np.random.default_rng(seed)
        idx = rng.integers(low=0, high=bd["u"].shape[0], size=num_points)
        states = []
        labels = []
        tcoords = []

        for ii in idx:
            u = bd["u"][ii]
            pmax = bd["p_max"][ii]
            p = interior_scale * float(pmax) * float(rng.uniform(0.2, 1.0))
            C0 = system.qubit_C_from_diag_T(u[0], u[1], u[2])
            rho = system.lt_ray_state(C0, p)
            states.append(rho)
            t = p * u
            tcoords.append(t)
            labels.append(f"t=({t[0]:+.2f},{t[1]:+.2f},{t[2]:+.2f})")

        tcoords = np.array(tcoords)

        # Verify convexity numerically: midpoint feasibility for random pairs
        mid_ok = 0
        trials = min(50, num_points * (num_points - 1) // 2)
        for _ in range(trials):
            i, j = rng.integers(0, num_points, size=2)
            if i == j:
                continue
            mid = 0.5 * (states[i] + states[j])
            w = np.linalg.eigvalsh(0.5 * (mid + dagger(mid)))
            if np.min(np.real(w)) >= -1e-10:
                mid_ok += 1
        conv_rate = mid_ok / max(1, trials)
        
        # I_vals = [mutual_information(rho, system.dims) for rho in states]
        # order = np.argsort(-np.array(I_vals))
        # states = [states[k] for k in order]
        # Convertibility adjacency (Global GP vs Local GP)
        N = len(states)
        I_node = np.zeros(N, dtype=float)
        svalsT = np.zeros((N, 3), dtype=float)

        for i in range(N):
            _, I, _, _ = system.monotones(states[i], tol=1e-12)
            I_node[i] = I
            # For diag-T family, singular values are |tx|,|ty|,|tz|
            tx, ty, tz = tcoords[i]
            s = np.sort(np.abs([tx, ty, tz]))[::-1]
            svalsT[i] = s

        G = np.zeros((N, N), dtype=int)
        L = np.zeros((N, N), dtype=int)
        mono_tol = 1e-10

        for i in range(N):
            for j in range(N):
                if i == j:
                    G[i, j] = 1
                    L[i, j] = 1
                    continue
                if I_node[i] + mono_tol < I_node[j]:
                    G[i, j] = 0
                    L[i, j] = 0
                    continue

                g_ok, _ = system.check_global_gp_feasible(
                    states[i], states[j], solver=system.solver_default, tol=system.tol_default, verbose=False
                )
                G[i, j] = 1 if g_ok else 0

                l_ok, _ = system.check_local_gp_feasible(
                    states[i], states[j], solver=system.solver_default, tol=system.tol_default, verbose=False, return_details=False
                )
                L[i, j] = 1 if l_ok else 0

        fig, ax = plt.subplots()
        ax.imshow(G, interpolation="nearest", aspect="auto")
        ax.set_title("Adjacency: Global GP (i → j)")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        pathG = save_plot(fig, f"lt_C_plane_{axes[0]}{axes[1]}_global_heatmap.png")

        fig, ax = plt.subplots()
        ax.imshow(L, interpolation="nearest", aspect="auto")
        ax.set_title("Adjacency: Local GP (i → j)")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        pathL = save_plot(fig, f"lt_C_plane_{axes[0]}{axes[1]}_local_heatmap.png")

        # Check singular-value contraction heuristic on local edges
        sv_viol = 0
        sv_edges = 0
        for i in range(N):
            for j in range(N):
                if i != j and L[i, j] == 1:
                    sv_edges += 1
                    if not np.all(svalsT[i] + 1e-10 >= svalsT[j]):
                        sv_viol += 1
        viol_rate = sv_viol / max(1, sv_edges)

        text = (
            f"Saved boundary plot: {pathB}\n"
            f"Saved adjacency heatmaps:\n- {pathG}\n- {pathL}\n\n"
            f"Convexity spot-check (midpoint PSD) pass-rate: {conv_rate:.2%} over {trials} random pairs.\n"
            f"Local edges checked: {sv_edges} | singular-value contraction violations: {sv_viol} ({viol_rate:.2%}).\n\n"
            f"Note: this is a *2D slice* of the full LT '+C' region in diag-T coordinates.\n"
        )
        log_info("LT +C characterisation (2D diagT plane)", text)

    elif eq_id == "lt_C_diagT_3d_characterise":
        # 3D characterisation of the LT "+C" slice in diag correlation coordinates (qubits only)
        if system.dims != (2, 2):
            log_warning("LT C-3D characterisation", "Implemented for dims=(2,2) only.")
            return

        num_dirs = int(vars_dict.get("num_dirs", 1500))
        num_points = int(vars_dict.get("num_points", 40))
        interior_scale = float(vars_dict.get("interior_scale", 0.5))
        seed = int(vars_dict.get("seed", 0))

        bd = analyzer.diagT_3d_boundary(num_dirs=num_dirs, include_negative=False, tol=1e-12)

        # Boundary scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(bd["t_max"][:, 0], bd["t_max"][:, 1], bd["t_max"][:, 2], s=4, alpha=0.6)
        ax.set_xlabel("tx")
        ax.set_ylabel("ty")
        ax.set_zlabel("tz")
        ax.set_title("Feasible LT region boundary (diag-T slice, p_max surface)")
        pathB = save_plot(fig, "lt_C_diagT_3d_boundary.png")

        # Sample interior points and build a small convertibility graph
        rng = np.random.default_rng(seed)
        idx = rng.integers(low=0, high=bd["u"].shape[0], size=num_points)
        states = []
        tcoords = []

        for ii in idx:
            u = bd["u"][ii]
            pmax = bd["p_max"][ii]
            p = interior_scale * float(pmax) * float(rng.uniform(0.2, 1.0))
            C0 = system.qubit_C_from_diag_T(u[0], u[1], u[2])
            rho = system.lt_ray_state(C0, p)
            states.append(rho)
            tcoords.append(p * u)

        tcoords = np.array(tcoords)
        
        # I_vals = [mutual_information(rho, system.dims) for rho in states]
        # order = np.argsort(-np.array(I_vals))
        # states = [states[k] for k in order]
        
        N = len(states)
        I_node = np.zeros(N, dtype=float)
        svalsT = np.zeros((N, 3), dtype=float)

        for i in range(N):
            _, I, _, _ = system.monotones(states[i], tol=1e-12)
            I_node[i] = I
            tx, ty, tz = tcoords[i]
            svalsT[i] = np.sort(np.abs([tx, ty, tz]))[::-1]

        G = np.zeros((N, N), dtype=int)
        L = np.zeros((N, N), dtype=int)
        mono_tol = 1e-10

        for i in range(N):
            for j in range(N):
                if i == j:
                    G[i, j] = 1
                    L[i, j] = 1
                    continue
                if I_node[i] + mono_tol < I_node[j]:
                    G[i, j] = 0
                    L[i, j] = 0
                    continue
                g_ok, _ = system.check_global_gp_feasible(
                    states[i], states[j], solver=system.solver_default, tol=system.tol_default, verbose=False
                )
                G[i, j] = 1 if g_ok else 0

                l_ok, _ = system.check_local_gp_feasible(
                    states[i], states[j], solver=system.solver_default, tol=system.tol_default, verbose=False, return_details=False
                )
                L[i, j] = 1 if l_ok else 0

        fig, ax = plt.subplots()
        ax.imshow(G, interpolation="nearest", aspect="auto")
        ax.set_title("Adjacency: Global GP (i → j)")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        pathG = save_plot(fig, "lt_C_diagT_3d_global_heatmap.png")

        fig, ax = plt.subplots()
        ax.imshow(L, interpolation="nearest", aspect="auto")
        ax.set_title("Adjacency: Local GP (i → j)")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        pathL = save_plot(fig, "lt_C_diagT_3d_local_heatmap.png")

        # Check singular-value contraction heuristic on local edges
        sv_viol = 0
        sv_edges = 0
        for i in range(N):
            for j in range(N):
                if i != j and L[i, j] == 1:
                    sv_edges += 1
                    if not np.all(svalsT[i] + 1e-10 >= svalsT[j]):
                        sv_viol += 1
        viol_rate = sv_viol / max(1, sv_edges)

        text = (
            f"Saved 3D boundary plot: {pathB}\n"
            f"Saved adjacency heatmaps:\n- {pathG}\n- {pathL}\n"
            f"Local edges checked: {sv_edges} | singular-value contraction violations: {sv_viol} ({viol_rate:.2%}).\n"
        )
        log_info("LT +C characterisation (3D diagT)", text)

    elif eq_id == "local_gp_closure_test":
        # Objective-2 verification: LT is closed under local Gibbs-preserving maps
        num_states = int(vars_dict.get("num_states", 6))
        num_channels = int(vars_dict.get("num_channels", 4))
        seed = int(vars_dict.get("seed", 0))

        rng = np.random.default_rng(seed)
        d = system.dA * system.dAp

        # Build a small set of LT states by projecting random states to LT
        lt_states = []
        statuses = []
        for _ in range(num_states):
            X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            rho0 = X @ dagger(X)
            rho0 = rho0 / np.trace(rho0)
            sigma, val, st = system.closest_lt_state(rho0, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)
            if sigma is None:
                continue
            lt_states.append(0.5 * (sigma + dagger(sigma)))
            statuses.append(st)

        if len(lt_states) == 0:
            log_warning("Local GP closure test", "Failed to generate LT states via projection.")
            return

        # Sample random local GP channels and apply; measure LT errors
        errsA = []
        errsAp = []
        diagA = None
        diagAp = None

        for k in range(num_channels):
            JA, stA, dA_diag = system.find_random_local_gp_channel(which="A", seed=seed + 10 + k, verbose=False)
            JAp, stAp, dAp_diag = system.find_random_local_gp_channel(which="Ap", seed=seed + 200 + k, verbose=False)
            if JA is None or JAp is None:
                continue
            diagA = dA_diag
            diagAp = dAp_diag

            for rho in lt_states:
                rhoA = system.apply_local_channel_A(rho, JA)
                ok, okA, okAp, mA, mAp = system.lt_membership(rhoA, tol=1e-7)
                errsA.append(float(np.linalg.norm(mA - system.gammaA, "fro") + np.linalg.norm(mAp - system.gammaAp, "fro")))

                rhoB = system.apply_local_channel_Ap(rho, JAp)
                ok, okA, okAp, mA, mAp = system.lt_membership(rhoB, tol=1e-7)
                errsAp.append(float(np.linalg.norm(mA - system.gammaA, "fro") + np.linalg.norm(mAp - system.gammaAp, "fro")))

        if len(errsA) == 0 or len(errsAp) == 0:
            log_warning("Local GP closure test", "No successful random local GP channels were generated.")
            return

        text = (
            f"Generated LT states via projection: {len(lt_states)} (statuses: {set(statuses)})\n"
            f"Applications tested (A-channels): {len(errsA)} | mean LT error={np.mean(errsA):.3e}, max={np.max(errsA):.3e}\n"
            f"Applications tested (A'-channels): {len(errsAp)} | mean LT error={np.mean(errsAp):.3e}, max={np.max(errsAp):.3e}\n"
            f"\nExample channel diagnostics (A): {diagA}\n"
            f"Example channel diagnostics (A'): {diagAp}\n"
        )
        log_info("Objective 2: LT closure under local GP (numerical)", text)

    elif eq_id == "extract_local_channels":
        # Objective-3 support: extract concrete local channels (JA,JAp) from an LGP-feasible mapping
        if system.dims != (2, 2):
            log_warning("Extract local channels", "Implemented for dims=(2,2) only (for easy LT family construction).")
            return

        label = str(vars_dict.get("label", "XX")).strip().upper()
        p_src = float(vars_dict.get("p_src", 0.5))
        p_tgt = float(vars_dict.get("p_tgt", 0.2))

        C0 = system.qubit_C0_from_pauli_label(label)
        # clamp to PSD interval
        pmin, pmax = system.lt_ray_p_bounds(C0, tol=1e-12)
        p_src = min(max(p_src, pmin), pmax)
        p_tgt = min(max(p_tgt, pmin), pmax)

        tau = system.lt_ray_state(C0, p_src)
        tau_p = system.lt_ray_state(C0, p_tgt)

        feasible, status, det = system.check_local_gp_feasible(
            tau, tau_p,
            solver=system.solver_default,
            tol=system.tol_default,
            verbose=False,
            return_details=True
        )

        if not feasible or det.get("J_A") is None or det.get("J_Ap") is None:
            log_warning("Extract local channels", f"LGP infeasible or no channels returned. Status: {status}")
            return

        JA = det["J_A"]
        JAp = det["J_Ap"]
        omega = det["omega"]

        # Verify sequential mapping numerically
        omega_num = system.apply_local_channel_A(tau, JA)
        tau_num = system.apply_local_channel_Ap(omega_num, JAp)

        map_err1 = float(np.linalg.norm(omega_num - omega, "fro"))
        map_err2 = float(np.linalg.norm(tau_num - tau_p, "fro"))

        diagA = system.choi_diagnostics(JA, d_in=system.dA, d_out=system.dA, gamma_in=system.gammaA, gamma_out=system.gammaA)
        diagAp = system.choi_diagnostics(JAp, d_in=system.dAp, d_out=system.dAp, gamma_in=system.gammaAp, gamma_out=system.gammaAp)

        # Save to npy for report
        out = {
            "label": label,
            "p_src": p_src,
            "p_tgt": p_tgt,
            "status": status,
            "residual": det.get("residual", None),
            "J_A": JA,
            "J_Ap": JAp,
            "omega": omega,
            "diagA": diagA,
            "diagAp": diagAp,
            "omega_num": omega_num,
            "tau_num": tau_num,
            "map_err_step1": map_err1,
            "map_err_step2": map_err2,
        }
        np.save("local_gp_channels.npy", out, allow_pickle=True)

        text = (
            f"LGP feasible: {feasible} | {status}\n"
            f"Ray label={label} | p_src={p_src:.6f} → p_tgt={p_tgt:.6f}\n"
            f"Step-1 ω error (recompute): {map_err1:.3e}\n"
            f"Step-2 τ error (recompute): {map_err2:.3e}\n\n"
            f"Saved local_gp_channels.npy (contains J_A, J_Ap, ω, diagnostics).\n"
            f"Diagnostics A: {diagA}\n"
            f"Diagnostics A': {diagAp}\n"
        )
        log_info("Objective 3: extracted local GP channels", text)


    elif eq_id == "extract_global_channel":
        # Try to find one mapping where global GP is feasible, then dump the Choi matrix.
        # Prefer TFD->dephased if available.
        candidates = []
        try:
            tau = analyzer.factory.tfd_state()
            tau_p = dephase_global_in_energy_basis(tau)
            candidates.append(("TFD→dephased", tau, tau_p))
        except Exception:
            pass

        # Add some LT→LT projected candidates
        for _ in range(6):
            r1 = random_state(d)
            r2 = random_state(d)
            t1, _, _ = system.closest_lt_state(r1, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)
            t2, _, _ = system.closest_lt_state(r2, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)
            if t1 is not None and t2 is not None:
                candidates.append(("projLT→projLT", t1, t2))

        picked = None
        details = None
        for name, tau, tau_p in candidates:
            ok, status, det = system.check_global_gp_feasible(
                tau, tau_p,
                solver=system.solver_default,
                tol=system.tol_default,
                return_details=True
            )
            if ok and det.get("J") is not None:
                picked = (name, tau, tau_p, status)
                details = det
                break

        if picked is None:
            log_warning(
                "Extract Global Channel",
                "Couldn't find a feasible global GP mapping in the quick candidate set. "
                "Try increasing num_samples or relaxing eps_eq_global / eps_gibbs."
            )
            return

        name, tau, tau_p, status = picked
        J = details["J"]

        # Verify CPTP approximately: TP is enforced; CP => J ⪰ 0 (numerical)
        # Verify mapping and Gibbs preservation directly
        GAxGAp = np.kron(system.gammaA, system.gammaAp)
        Yg = system.choi_apply_numpy(J, GAxGAp, d_in=d, d_out=d)
        Ym = system.choi_apply_numpy(J, 0.5*(tau+dagger(tau)), d_in=d, d_out=d)

        gibbs_err = np.linalg.norm(Yg - GAxGAp, "fro")
        map_err = np.linalg.norm(Ym - 0.5*(tau_p+dagger(tau_p)), "fro")

        # Save Choi matrix
        ensure_png_dir()
        choi_path = os.path.join("png", "global_gp_choi.npy")
        np.save(choi_path, J)

        text = (
            f"Picked mapping: {name}\n"
            f"Status: {status}\n\n"
            f"Choi saved: {choi_path}\n"
            f"Gibbs error ||Φ(γ⊗γ)-γ⊗γ||_F = {gibbs_err:.3e}\n"
            f"Map   error ||Φ(τ)-τ'||_F       = {map_err:.3e}\n"
        )
        log_info("Extract Global GP Channel", text)
        print("Finished global channel extraction analysis.")

    elif eq_id == "sanity_checks":
        # Produce a small 'capstone-grade' table of errors + monotone change.
        # Use one mapping (prefer TFD->dephased).
        try:
            tau = analyzer.factory.tfd_state()
            tau_p = dephase_global_in_energy_basis(tau)
            label = "TFD→dephased"
        except Exception:
            # fallback: two LT projected states
            r1 = random_state(d)
            r2 = random_state(d)
            tau, _, _ = system.closest_lt_state(r1, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)
            tau_p, _, _ = system.closest_lt_state(r2, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)
            label = "projLT→projLT"

        if tau is None or tau_p is None:
            log_warning("Sanity checks", "Could not construct test states.")
            return

        # LT errors
        tauA = analyzer.system.lt_membership(tau)[3]
        tauAp = analyzer.system.lt_membership(tau)[4]
        lt_err_A = np.linalg.norm(tauA - system.gammaA, "fro")
        lt_err_Ap = np.linalg.norm(tauAp - system.gammaAp, "fro")

        # Global GP channel residuals
        g_ok, g_status, g_det = system.check_global_gp_feasible(
            tau, tau_p, solver=system.solver_default, tol=system.tol_default, return_details=True
        )
        J = g_det.get("J", None)
        if J is None:
            log_warning("Sanity checks", "Global GP channel solve failed (no J). Try relaxing tolerances.")
            return

        GAxGAp = np.kron(system.gammaA, system.gammaAp)
        Yg = system.choi_apply_numpy(J, GAxGAp, d_in=d, d_out=d)
        Ym = system.choi_apply_numpy(J, 0.5*(tau+dagger(tau)), d_in=d, d_out=d)

        gp_err = np.linalg.norm(Yg - GAxGAp, "fro")
        map_err = np.linalg.norm(Ym - 0.5*(tau_p+dagger(tau_p)), "fro")

        # Local GP residual (gap score)
        l_ok, l_status, l_det = system.check_local_gp_feasible(
            tau, tau_p, solver=system.solver_default, tol=system.tol_default, return_details=True
        )
        l_res = float(l_det.get("residual", np.inf))

        # Monotone change
        D_tau, I_tau, _, _ = system.monotones(tau)
        D_taup, I_taup, _, _ = system.monotones(tau_p)
        dD = D_tau - D_taup

        # Table-like print
        lines = [
            f"Mapping: {label}",
            "",
            f"LT error ||tau_A - gamma||_F      = {lt_err_A:.3e}",
            f"LT error ||tau_A' - gamma||_F     = {lt_err_Ap:.3e}",
            "",
            f"Global GP feasible? {g_ok} | {g_status}",
            f"GP constraint error ||Φ(γ⊗γ)-γ⊗γ||_F = {gp_err:.3e}",
            f"Map error ||Φ(tau)-tau'||_F           = {map_err:.3e}",
            "",
            f"Local GP feasible?  {l_ok} | {l_status}",
            f"Local best residual (step-2 objective) = {l_res:.3e}",
            "",
            f"Monotones: D(tau)={D_tau:.4f}, D(tau')={D_taup:.4f}, ΔD = {dD:.4f} (should be ≥ 0 under GP)",
            f"          I(tau)={I_tau:.4f}, I(tau')={I_taup:.4f}",
        ]
        text = "\n".join(lines)
        log_info("Sanity checks", text)
        print("Finished sanity checks analysis.")
    elif eq_id == "custom":
        # Custom JSON spec pasted in the GUI custom box.
        # Required:
        #   {"experiment":"lt_structured_family_hierarchy", "family":"ray", "label":"XX", "num_p":21, ...}
        # or:
        #   {"experiment":"lt_structured_family_hierarchy", "family":"diagT", "tx":1, "ty":0, "tz":1, ...}
        try:
            if not custom_func_text.strip():
                raise ValueError("Custom JSON spec is empty.")
            spec = json.loads(custom_func_text)
        except Exception as e:
            lines = [
                "Custom mode expects a JSON object in the Custom box.",
                "",
                "Example (Pauli ray):",
                '{"experiment":"lt_structured_family_hierarchy","family":"ray","label":"XX","num_p":21,"pair_mode":"decreasing","mono_tol":1e-8,"p_shrink":0.98}',
                "",
                "Example (diagT ray):",
                '{"experiment":"lt_structured_family_hierarchy","family":"diagT","tx":1,"ty":0,"tz":1,"num_p":21,"pair_mode":"decreasing","mono_tol":1e-8,"p_shrink":0.98}',
                "",
                f"Parse error: {e}",
            ]
            log_error("Custom JSON parse error", "\n".join(lines))
            return

        exp = str(spec.get("experiment", "")).strip()
        if exp != "lt_structured_family_hierarchy":
            lines = [
                "Unknown custom experiment.",
                "",
                "Supported:",
                '  experiment="lt_structured_family_hierarchy"',
                "",
                f"Got: {exp}",
            ]
            log_warning("Custom", "\n".join(lines))
            return

        try:
            summary, _ = _structured_family_hierarchy_run(system, analyzer, spec)
            log_info("LT Structured-Family Hierarchy", summary)
        except Exception as e:
            log_error("LT Structured-Family Hierarchy error", str(e))
            return

    else:
        log_warning(
            "No backend attached",
            "No specific backend implemented for this experiment id yet.\n\n"
            f"ID: {eq_id}\nName: {eq_name}\n\n"
            f"Variables: {vars_dict}"
        )
