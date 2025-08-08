# main.py
import json
from pathlib import Path
import collections
import re
import time

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ortools.sat.python import cp_model


# ========================= Streamlit setup =========================
st.set_page_config(page_title="Hay Day Optimizer", layout="wide")
st.title("Hay Day Optimizer (Streamlit)")

BASE_DIR = Path(__file__).parent


# ========================= Helpers =========================
def load_local_json(filename, fallback=None):
    p = BASE_DIR / filename
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return fallback


def save_local_json(filename, data) -> bool:
    try:
        with open(BASE_DIR / filename, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def get_trailing_number(s):
    m = re.search(r"\d+$", s)
    return int(m.group(0)) if m else None


# ========================= JSON state =========================
if "raw_data" not in st.session_state:
    st.session_state.raw_data = load_local_json("hayday_goods_with_storage.json", {})
if "my_machines" not in st.session_state:
    st.session_state.my_machines = load_local_json("my_machines.json", {})
if "my_goods" not in st.session_state:
    st.session_state.my_goods = load_local_json("my_goods.json", {})
if "my_storage" not in st.session_state:
    st.session_state.my_storage = load_local_json("my_storage.json", {"Barn": 0, "Silo": 0})

raw_data = st.session_state.raw_data
my_machines = st.session_state.my_machines
my_goods = st.session_state.my_goods
my_storage = st.session_state.my_storage

# Machine names from catalog (for convenience)
catalog_machines = sorted({v.get("machine") for v in raw_data.values() if v.get("machine")})

# ========================= Tabs (Optimizer first so it opens by default) =========================
tab_run, tab_inputs, tab_catalog = st.tabs(
    ["Run Optimization", "Inputs", "Goods Catalog Editor"]
)

# ========================= RUN TAB =========================
with tab_run:
    st.markdown("### Features & How it works")
    st.markdown(
        """
- **What it does:** Builds a production schedule minimizing the **makespan** with OR-Tools CP-SAT.
- **Respects machines:** Each job uses a specific machine type; multiple identical machines are handled via optional intervals and `AddNoOverlap`.
- **Handles recipes:** If a good has ingredients, those are auto-expanded into prerequisite tasks; parents can’t start before children finish.
- **Targets & removals:** For each target good, the model activates exactly the number of leaf tasks needed and marks extras as removed.
- **Storage constraints:** At any timepoint (task ends), the count of items sitting in **Barn** or **Silo** won’t exceed your capacities (including your initial stock).
- **Output:** A Gantt-style chart per machine lane, plus a table showing storage utilization over time.
        """
    )

    # Recompute machines from catalog
    all_machines = []
    for _, details in raw_data.items():
        m = details.get("machine")
        if m and m not in all_machines:
            all_machines.append(m)

    default_goods = ['Affogato','Asparagus','Cabbage soup']
    goods_data = st.multiselect(
        "Select top-level goods to schedule",
        options=sorted(list(raw_data.keys())),
        default=default_goods,
    )

    col_run1, col_run2, col_run3 = st.columns([1, 1, 1])
    with col_run1:
        horizon = st.number_input("Horizon (time units)", min_value=1, value=10000, step=1)
    with col_run2:
        time_limit_s = st.number_input("Time limit (seconds, optional)", min_value=0, value=0, help="0 = no time limit")
    with col_run3:
        log_progress = st.checkbox("Verbose solver log", value=False)

    BARN_CAPACITY = int(my_storage.get("Barn", 0))
    SILO_CAPACITY = int(my_storage.get("Silo", 0))

    run = st.button("Run Optimization")

    if run:
        if not raw_data or not my_machines:
            st.error("Missing required JSON(s). Ensure goods_with_storage and my_machines are provided.")
            st.stop()
        if not goods_data:
            st.error("Pick at least one top-level good to schedule.")
            st.stop()

        model = cp_model.CpModel()

        class Task:
            def __init__(self, id_, name, start, end, machine, l_presences, parent):
                self.id = id_
                self.name = name
                self.start = start
                self.end = end
                self.machine = machine
                self.l_presences = l_presences
                self.parent = parent
                self.children = []
                self.is_active = model.NewBoolVar(f"is_active{id_}")
                self.removed_explicit = model.NewBoolVar(f"removed{id_}")

        machine_to_intervals = collections.defaultdict(list)

        # Expand selected goods
        next_id = 0
        queue = []
        all_tasks = []

        for good in goods_data:
            queue.append((good, next_id, -1))
            next_id += 1

        while queue:
            current_name, current_id, parent_id = queue.pop(0)

            dur = int(raw_data[current_name]["time"])
            start_var = model.NewIntVar(0, horizon, f"start{current_id}")
            end_var = model.NewIntVar(0, horizon, f"end{current_id}")
            _interval = model.NewIntervalVar(start_var, dur, end_var, f"interval{current_id}")

            machine_var = model.NewIntVar(0, len(all_machines), f"machine{current_id}")

            mtype = raw_data[current_name]["machine"]
            num_specific = int(my_machines.get(mtype, {}).get("number", 0))
            l_presences = []

            for i in range(num_specific):
                l_start = model.NewIntVar(0, horizon, f"l_start{i}_{current_id}")
                l_end = model.NewIntVar(0, horizon, f"l_end{i}_{current_id}")
                l_pres = model.NewBoolVar(f"l_presence{i}_{current_id}")
                l_int = model.NewOptionalIntervalVar(
                    l_start, dur, l_end, l_pres, f"l_interval{i}_{current_id}"
                )

                model.Add(start_var == l_start).OnlyEnforceIf(l_pres)
                model.Add(end_var == l_end).OnlyEnforceIf(l_pres)
                model.Add(machine_var == i).OnlyEnforceIf(l_pres)

                l_presences.append(l_pres)
                machine_to_intervals[f"{mtype}{i}"].append(l_int)

            if l_presences:
                model.AddExactlyOne(l_presences)

            t = Task(current_id, current_name, start_var, end_var, machine_var, l_presences, parent_id)
            all_tasks.append(t)

            if parent_id != -1:
                all_tasks[parent_id].children.append(t)

                model.Add(all_tasks[parent_id].start >= all_tasks[current_id].end).OnlyEnforceIf(
                    all_tasks[parent_id].is_active, all_tasks[current_id].is_active
                )
                model.AddImplication(all_tasks[parent_id].is_active.Not(), all_tasks[current_id].is_active.Not())

                model.AddBoolAnd(
                    [all_tasks[current_id].is_active.Not(), all_tasks[parent_id].is_active]
                ).OnlyEnforceIf(all_tasks[current_id].removed_explicit)

                model.AddBoolOr(
                    [all_tasks[current_id].is_active, all_tasks[parent_id].is_active.Not()]
                ).OnlyEnforceIf(all_tasks[current_id].removed_explicit.Not())
            else:
                model.Add(all_tasks[current_id].removed_explicit == all_tasks[current_id].is_active.Not())

            for ing, qty in raw_data[current_name]["ingredients"].items():
                if ing != current_name:
                    for _ in range(int(qty)):
                        queue.append((ing, next_id, current_id))
                        next_id += 1

        # No-overlap + ordering disambiguation
        for machine, intervals in machine_to_intervals.items():
            model.AddNoOverlap(intervals)
            for i_idx, i_int in enumerate(intervals):
                for j_idx, j_int in enumerate(intervals):
                    if i_idx == j_idx:
                        continue

                    def _id_from_intname(nm):
                        if nm.startswith("l_interval"):
                            return int(nm.split("_")[-1])
                        return int(nm.replace("interval", ""))

                    id_i = _id_from_intname(i_int.Name())
                    id_j = _id_from_intname(j_int.Name())

                    a = model.NewBoolVar(f"a_{id_i}_{id_j}_{machine}")
                    model.Add(all_tasks[id_j].start >= all_tasks[id_i].end).OnlyEnforceIf(a)
                    model.Add(all_tasks[id_j].start < all_tasks[id_i].end).OnlyEnforceIf(a.Not())

                    b = model.NewBoolVar(f"b_{id_i}_{id_j}_{machine}")
                    model.Add(all_tasks[id_i].start >= all_tasks[id_j].end).OnlyEnforceIf(b)
                    model.Add(all_tasks[id_i].start < all_tasks[id_j].end).OnlyEnforceIf(b.Not())

                    idx = get_trailing_number(machine) or 0
                    if (
                        all_tasks[id_i].l_presences
                        and all_tasks[id_j].l_presences
                        and idx < len(all_tasks[id_i].l_presences)
                        and idx < len(all_tasks[id_j].l_presences)
                    ):
                        model.AddBoolOr([a, b]).OnlyEnforceIf(
                            all_tasks[id_i].l_presences[idx], all_tasks[id_j].l_presences[idx]
                        )

        # Demand / removal alignment
        for task in all_tasks:
            same_name_removed = []
            inactive_parents = []
            for t in all_tasks:
                if t.name == task.name:
                    same_name_removed.append(t.removed_explicit)
                    if t.parent != -1:
                        inactive_parents.append(all_tasks[t.parent].is_active.Not())

            if task.name in my_goods:
                need = int(my_goods[task.name].get("number", 0))
                if need > 0:
                    num_inactive_parents = sum(inactive_parents) if inactive_parents else 0
                    min_var = model.NewIntVar(0, 10000, f"min_var_{task.id}")
                    if task.parent != -1:
                        model.AddMinEquality(
                            min_var,
                            [len(same_name_removed) - num_inactive_parents, need],
                        )
                    else:
                        model.AddMinEquality(min_var, [len(same_name_removed), need])
                    model.Add(sum(same_name_removed) == min_var)
                else:
                    model.Add(sum(same_name_removed) == 0)
            else:
                model.Add(sum(same_name_removed) == 0)

        # Storage capacity
        times = [t.end for t in all_tasks]
        storage = []
        initial_silo_storage = 0
        initial_barn_storage = 0
        initial_silo_storage_used = []
        initial_barn_storage_used = []

        for good, spec in my_goods.items():
            if good in raw_data:
                if raw_data[good]["storage"] == "barn":
                    initial_barn_storage += int(spec.get("number", 0))
                elif raw_data[good]["storage"] == "silo":
                    initial_silo_storage += int(spec.get("number", 0))

        for i in range(len(times)):
            silo_temp_arr = []
            barn_temp_arr = []

            for j in range(len(all_tasks)):
                temp = model.NewBoolVar(f"{i}_{j}")

                end_on_time = model.NewBoolVar(f"end_on_time_{i}_{j}")
                model.Add(all_tasks[j].end <= times[i]).OnlyEnforceIf(end_on_time)
                model.Add(all_tasks[j].end > times[i]).OnlyEnforceIf(end_on_time.Not())

                if all_tasks[j].parent != -1:
                    parent_not_started = model.NewBoolVar(f"parent_not_started_{i}_{j}")
                    model.Add(all_tasks[all_tasks[j].parent].start >= times[i]).OnlyEnforceIf(parent_not_started)
                    model.Add(all_tasks[all_tasks[j].parent].start <  times[i]).OnlyEnforceIf(parent_not_started.Not())

                    model.AddBoolAnd([parent_not_started, end_on_time]).OnlyEnforceIf(temp)
                    model.AddBoolOr([parent_not_started.Not(), end_on_time.Not()]).OnlyEnforceIf(temp.Not())

                    a = model.NewBoolVar(f"parent_started_removed_{i}_{j}")
                    model.AddBoolAnd([all_tasks[j].removed_explicit, parent_not_started.Not()]).OnlyEnforceIf(a)
                    model.AddBoolOr([all_tasks[j].removed_explicit.Not(), parent_not_started]).OnlyEnforceIf(a.Not())

                    if raw_data[all_tasks[j].name]["storage"] == "barn":
                        initial_barn_storage_used.append(a)
                    elif raw_data[all_tasks[j].name]["storage"] == "silo":
                        initial_silo_storage_used.append(a)
                else:
                    temp = end_on_time

                if raw_data[all_tasks[j].name]["storage"] == "barn":
                    barn_temp_arr.append(temp)
                elif raw_data[all_tasks[j].name]["storage"] == "silo":
                    silo_temp_arr.append(temp)

            if barn_temp_arr:
                model.Add(sum(barn_temp_arr) + initial_barn_storage - sum(initial_barn_storage_used) <= BARN_CAPACITY)
            if silo_temp_arr:
                model.Add(sum(silo_temp_arr) + initial_silo_storage - sum(initial_silo_storage_used) <= SILO_CAPACITY)

            storage.append([silo_temp_arr, barn_temp_arr])

        # Objective
        obj_var = model.NewIntVar(0, horizon, "makespan")
        cand_ends = []
        for i in range(next_id):
            c = model.NewIntVar(0, horizon, f"cand_end_{i}")
            model.Add(c == 0).OnlyEnforceIf(all_tasks[i].is_active.Not())
            model.Add(c == all_tasks[i].end).OnlyEnforceIf(all_tasks[i].is_active)
            cand_ends.append(c)
        model.AddMaxEquality(obj_var, cand_ends)
        model.Minimize(obj_var)

        # Solve with spinner (and optional time limit + logs)
        solver = cp_model.CpSolver()
        if time_limit_s > 0:
            solver.parameters.max_time_in_seconds = float(time_limit_s)
        solver.parameters.log_search_progress = bool(log_progress)
        solver.parameters.num_search_workers = 8

        start_t = time.time()
        with st.spinner("Optimizing… this can take a bit."):
            status = solver.Solve(model)
        elapsed = time.time() - start_t

        # Results
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            st.success(f"{'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'} — solved in {elapsed:.2f}s")
            st.metric("Makespan", solver.Value(obj_var))

            # Gantt
            st.subheader("Optimized Schedule (Gantt)")
            schedule_data = []
            for i in range(next_id):
                if solver.Value(all_tasks[i].is_active):
                    start_v = solver.Value(all_tasks[i].start)
                    end_v = solver.Value(all_tasks[i].end)
                    mname = raw_data[all_tasks[i].name]["machine"]
                    mindex = solver.Value(all_tasks[i].machine)
                    machine_label = f"{mname}{mindex}"
                    schedule_data.append((all_tasks[i].name, start_v, end_v, machine_label))

            if schedule_data:
                df = pd.DataFrame(schedule_data, columns=["Task", "Start", "End", "Machine"])
                df["Duration"] = df["End"] - df["Start"]

                fig = go.Figure()
                for task_name, g in df.groupby("Task"):
                    fig.add_trace(
                        go.Bar(
                            y=g["Machine"],
                            x=g["Duration"],
                            base=g["Start"],
                            orientation="h",
                            name=task_name,
                            hovertemplate=(
                                "Task=%{fullData.name}<br>"
                                "Machine=%{y}<br>"
                                "Start=%{base}<br>"
                                "End=%{x:+.0f}+%{base}=%{customdata}<extra></extra>"
                            ),
                            customdata=g["End"],
                        )
                    )

                fig.update_layout(
                    barmode="stack",
                    xaxis_title="Time",
                    yaxis_title="Machine",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=40, r=20, t=40, b=40),
                    legend_title_text="Task",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No active tasks to plot.")

            # Storage table (AFTER Gantt)
            st.subheader("Storage Usage Over Time")
            storage_records = []
            for i in range(len(times)):
                silo_arr = [solver.Value(j) for j in storage[i][0]]
                barn_arr = [solver.Value(j) for j in storage[i][1]]
                t_val = solver.Value(times[i])
                storage_records.append((t_val, sum(silo_arr), sum(barn_arr)))

            st.dataframe(
                {
                    "Time": [t for t, _, _ in storage_records],
                    "Silo Used (Δ)": [s for _, s, _ in storage_records],
                    "Barn Used (Δ)": [b for _, _, b in storage_records],
                },
                use_container_width=True,
            )
        else:
            st.error(f"No feasible solution found (elapsed {elapsed:.2f}s). Try increasing horizon, capacities, or machines.")


# ========================= INPUTS TAB =========================
with tab_inputs:
    st.markdown("### Storage Capacities")
    c1, c2 = st.columns(2)
    with c1:
        my_storage["Barn"] = c1.number_input("Barn Capacity", min_value=0, value=int(my_storage.get("Barn", 0)))
    with c2:
        my_storage["Silo"] = c2.number_input("Silo Capacity", min_value=0, value=int(my_storage.get("Silo", 0)))

    c3, c4 = st.columns(2)
    if c3.button("Save my_storage.json"):
        ok = save_local_json("my_storage.json", my_storage)
        st.success("Saved my_storage.json") if ok else st.error("Failed to save my_storage.json")
    c4.download_button("Download my_storage.json", data=json.dumps(my_storage, indent=2),
                       file_name="my_storage.json", mime="application/json")

    # Machines
    st.markdown("---")
    st.markdown("### Machines You Own")
    st.caption("Machine names should match the `machine` field in your catalog.")
    rows = []
    seen = set()
    for name, spec in my_machines.items():
        count = int(spec.get("number", 0)) if isinstance(spec, dict) else int(spec or 0)
        rows.append({"Machine": name, "number": count})
        seen.add(name)
    for m in catalog_machines:
        if m and m not in seen:
            rows.append({"Machine": m, "number": 0})

    df_m = pd.DataFrame(rows).sort_values("Machine").reset_index(drop=True) if rows else pd.DataFrame(columns=["Machine", "number"])
    edited_df_m = st.data_editor(
        df_m,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Machine": st.column_config.TextColumn(required=True),
            "number": st.column_config.NumberColumn(min_value=0, step=1, required=True),
        },
        key="machines_editor",
    )
    new_machines = {}
    for _, row in edited_df_m.iterrows():
        name = str(row["Machine"]).strip()
        if name:
            new_machines[name] = {"number": int(row["number"])}
    st.session_state.my_machines = new_machines
    my_machines = new_machines

    c1, c2 = st.columns(2)
    if c1.button("Save my_machines.json"):
        ok = save_local_json("my_machines.json", my_machines)
        st.success("Saved my_machines.json") if ok else st.error("Failed to save my_machines.json")
    c2.download_button("Download my_machines.json", data=json.dumps(my_machines, indent=2),
                       file_name="my_machines.json", mime="application/json")

    # Initial Inventory (starting stock before scheduling)
    st.markdown("---")
    st.markdown("### Initial Inventory")
    all_goods_list = sorted(list(raw_data.keys()))

    rows = [{"Good": k, "number": int(v.get("number", 0))} for k, v in my_goods.items()]
    df_g = pd.DataFrame(rows).sort_values("Good").reset_index(drop=True) if rows else pd.DataFrame(columns=["Good", "number"])

    edited_df_g = st.data_editor(
        df_g,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Good": st.column_config.TextColumn(required=True),
            "number": st.column_config.NumberColumn(min_value=0, step=1, required=True),
        },
        key="goods_editor",
    )

    new_goods = {}
    for _, row in edited_df_g.iterrows():
        name = str(row["Good"]).strip()
        if name and name in raw_data:
            new_goods[name] = {"number": int(row["number"])}
    st.session_state.my_goods = new_goods
    my_goods = new_goods

    c1, c2 = st.columns(2)
    if c1.button("Save my_goods.json"):
        ok = save_local_json("my_goods.json", my_goods)
        st.success("Saved my_goods.json") if ok else st.error("Failed to save my_goods.json")
    c2.download_button("Download my_goods.json", data=json.dumps(my_goods, indent=2),
                       file_name="my_goods.json", mime="application/json")


# ========================= GOODS CATALOG TAB =========================
with tab_catalog:
    st.markdown("### Goods Catalog Editor")
    if not raw_data:
        st.warning("No goods loaded.")
    else:
        good_names = sorted(list(raw_data.keys()))
        selected_good = st.selectbox("Select good to edit", good_names, index=0 if good_names else None)

        if selected_good:
            item = raw_data[selected_good]
            c1, c2, c3 = st.columns(3)
            with c1:
                choices = sorted(set(catalog_machines + [item.get("machine", "")]))
                new_machine = st.selectbox("Machine", options=choices, index=choices.index(item.get("machine", "")) if item.get("machine", "") in choices else 0)
            with c2:
                new_storage = st.selectbox("Storage", options=["barn", "silo"], index=0 if item.get("storage", "barn") == "barn" else 1)
            with c3:
                new_time = st.number_input("Craft Time", min_value=0, value=int(item.get("time", 0)), step=1)

            st.markdown("**Ingredients**")
            ing_dict = item.get("ingredients", {}) or {}
            ing_rows = [{"Ingredient": k, "qty": int(v)} for k, v in ing_dict.items()]
            df_ing = pd.DataFrame(ing_rows).sort_values("Ingredient").reset_index(drop=True) if ing_rows else pd.DataFrame(columns=["Ingredient", "qty"])

            add_ings = st.multiselect("Add ingredients", options=[g for g in raw_data.keys() if g != selected_good])
            for a in add_ings:
                if a not in df_ing["Ingredient"].values:
                    df_ing.loc[len(df_ing)] = {"Ingredient": a, "qty": 1}

            edited_df_ing = st.data_editor(
                df_ing,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "Ingredient": st.column_config.TextColumn(required=True),
                    "qty": st.column_config.NumberColumn(min_value=1, step=1, required=True),
                },
                key=f"ing_editor_{selected_good}",
            )

            new_ing = {}
            for _, row in edited_df_ing.iterrows():
                name = str(row["Ingredient"]).strip()
                if name and name in raw_data:
                    new_ing[name] = int(row["qty"])

            cA, cB, cC = st.columns(3)
            if cA.button("Apply changes to selected item"):
                raw_data[selected_good]["machine"] = new_machine
                raw_data[selected_good]["storage"] = new_storage
                raw_data[selected_good]["time"] = int(new_time)
                raw_data[selected_good]["ingredients"] = new_ing
                st.session_state.raw_data = raw_data
                st.success(f"Updated '{selected_good}'")

            if cB.button("Save hayday_goods_with_storage.json"):
                ok = save_local_json("hayday_goods_with_storage.json", raw_data)
                st.success("Saved hayday_goods_with_storage.json") if ok else st.error("Failed to save hayday_goods_with_storage.json")

            cC.download_button("Download hayday_goods_with_storage.json",
                               data=json.dumps(raw_data, indent=2),
                               file_name="hayday_goods_with_storage.json",
                               mime="application/json")
