import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.titlepad'] = 10
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import itertools
import os
from tools.pd2csv import pd2csv
import argparse


def plot_from_tables(jodie_table, batch_table, x_axis_max, timeout, fig_name, title):

    Gtime = []
    for i in jodie_table['GTime'].values:
        if i >= timeout:
            Gtime += [float('inf')]
        else:
            Gtime += [i]
    Btime_SR = []
    for i in jodie_table['BTime_SR'].values:
        if i >= timeout:
            Btime_SR += [float('inf')]
        else:
            Btime_SR += [i]
    prox_time = []
    for i in batch_table['BTime_SR_prox_100.0'].values:
        if i >= timeout:
            prox_time += [float('inf')]
        else:
            prox_time += [i]
    adam_time = []
    for i in batch_table['BTime_SR_adam_160.0'].values:
        if i >= timeout:
            adam_time += [float('inf')]
        else:
            adam_time += [i]
    djadam_time = []
    for i in batch_table['BTime_SR_dj-adam_260.0'].values:
        if i >= timeout:
            djadam_time += [float('inf')]
        else:
            djadam_time += [i]
    Gtime.sort()
    Btime_SR.sort()
    prox_time.sort()
    adam_time.sort()
    djadam_time.sort()

    # check that they have the same length.
    assert len(Btime_SR) == len(prox_time)
    assert len(adam_time) == len(prox_time)
    assert len(adam_time) == len(djadam_time)
    print(f"{title} has {len(prox_time)} entries")

    starting_point = min(Gtime[0], Btime_SR[0], prox_time[0], adam_time[0])

    method_2_color = {}
    method_2_color['MIPplanet'] = 'red'
    method_2_color['Gurobi BaBSR'] = 'green'
    method_2_color['Proximal BaBSR'] = 'skyblue'
    method_2_color['Supergradient BaBSR'] = 'gold'
    method_2_color['DSG+ BaBSR'] = 'darkmagenta'
    fig = plt.figure(figsize=(10,10))
    ax_value = plt.subplot(1, 1, 1)
    ax_value.axhline(linewidth=3.0, y=100, linestyle='dashed', color='grey')

    y_min = 0
    y_max = 100
    ax_value.set_ylim([y_min, y_max+5])

    min_solve = float('inf')
    max_solve = float('-inf')
    for timings in [Gtime, Btime_SR, prox_time, adam_time, djadam_time]:
        min_solve = min(min_solve, min(timings))
        finite_vals = [val for val in timings if val != float('inf')]
        if len(finite_vals) > 0:
            max_solve = max(max_solve, max([val for val in timings if val != float('inf')]))


    axis_min = starting_point
    #ax_value.set_xscale("log")
    axis_min = min(0.5 * min_solve, 1)
    ax_value.set_xlim([axis_min, x_axis_max])

    # Plot all the properties
    linestyle_dict = {
        'MIPplanet': 'solid',
        'Gurobi BaBSR': 'solid',
        'Proximal BaBSR': 'solid',
        'Supergradient BaBSR': 'solid',
        'DSG+ BaBSR': 'solid',
    }

    for method, m_timings in [('MIPplanet', Gtime), ('Gurobi BaBSR', Btime_SR), ('Proximal BaBSR', prox_time),
                              ('Supergradient BaBSR', adam_time), ('DSG+ BaBSR', djadam_time)]:
        # Make it an actual cactus plot
        xs = [axis_min]
        ys = [y_min]
        prev_y = 0
        for i, x in enumerate(m_timings):
            if x <= x_axis_max:
                # Add the point just before the rise
                xs.append(x)
                ys.append(prev_y)
                # Add the new point after the rise, if it's in the plot
                xs.append(x)
                new_y = 100*(i+1)/len(m_timings)
                ys.append(new_y)
                prev_y = new_y
        # Add a point at the end to make the graph go the end
        xs.append(x_axis_max)
        ys.append(prev_y)

        ax_value.plot(xs, ys, color=method_2_color[method],
                      linestyle=linestyle_dict[method], label=method, linewidth=4.0)

    ax_value.set_ylabel("% of properties verified", fontsize=22)
    ax_value.set_xlabel("Computation time [s]", fontsize=22)
    plt.xscale('log', nonposx='clip')
    ax_value.legend(fontsize=19.5)
    plt.grid(True)
    plt.title(title)
    plt.savefig(fig_name, format='pdf', dpi=300)


def plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, fig_name, title, create_csv=False,
                            linestyles=None, one_vs_all=False, vline=None, idx_key="Idx", exclude_sat=False):

    mpl.rcParams['font.size'] = 12
    tables = []
    if create_csv and not os.path.exists(folder + "/csv/"):
        os.makedirs(folder + "/csv/")
    for filename in file_list:
        m = pd.read_pickle(folder + filename).dropna(how="all")
        if create_csv:
            pd2csv(folder, folder + "/csv/", filename)
        tables.append(m)
    # keep only the properties in common
    for m in tables:
        if not one_vs_all:
            m["unique_id"] = m[idx_key].map(int).map(str) + "_" + m["prop"].map(int).map(str)
        else:
            m["unique_id"] = m[idx_key].map(int).map(str)
        m["Eps"] = m[idx_key].map(float)
    for m1, m2 in itertools.product(tables, tables):
        m1.drop(m1[(~m1['unique_id'].isin(m2["unique_id"]))].index, inplace=True)

    if exclude_sat:
        exclude_sat_props(tables)

    # Set all timeouts to <timeout> seconds.
    timeout_setter(tables, timeout)

    timings = []
    for idx in range(len(tables)):
        timings.append([])
        for i in tables[idx][time_name_list[idx]].values:
            if i >= timeout:
                timings[-1].append(float('inf'))
            else:
                timings[-1].append(i)
        timings[-1].sort()
    # check that they have the same length.
    for m1, m2 in itertools.product(timings, timings):
        assert len(m1) == len(m2)
    print(len(m1))

    starting_point = timings[0][0]
    for timing in timings:
        starting_point = min(starting_point, timing[0])

    fig = plt.figure(figsize=(6, 6))
    ax_value = plt.subplot(1, 1, 1)
    ax_value.axhline(linewidth=3.0, y=100, linestyle='dashed', color='grey')

    if vline is not None:
        ax_value.axvline(linewidth=1, x=vline, linestyle='solid', color='red')

    y_min = 0
    y_max = 100
    ax_value.set_ylim([y_min, y_max + 5])

    min_solve = float('inf')
    max_solve = float('-inf')
    for timing in timings:
        min_solve = min(min_solve, min(timing))
        finite_vals = [val for val in timing if val != float('inf')]
        if len(finite_vals) > 0:
            max_solve = max(max_solve, max([val for val in timing if val != float('inf')]))

    axis_min = starting_point
    axis_min = min(0.5 * min_solve, 1)
    ax_value.set_xlim([axis_min, timeout + 1])

    for idx, (clabel, timing) in enumerate(zip(labels, timings)):
        # Make it an actual cactus plot
        xs = [axis_min]
        ys = [y_min]
        prev_y = 0
        for i, x in enumerate(timing):
            if x <= timeout:
                # Add the point just before the rise
                xs.append(x)
                ys.append(prev_y)
                # Add the new point after the rise, if it's in the plot
                xs.append(x)
                new_y = 100 * (i + 1) / len(timing)
                ys.append(new_y)
                prev_y = new_y
        # Add a point at the end to make the graph go the end
        xs.append(timeout)
        ys.append(prev_y)

        linestyle = linestyles[idx] if linestyles is not None else "solid"
        ax_value.plot(xs, ys, color=colors[idx], linestyle=linestyle, label=clabel, linewidth=3.0)

    ax_value.set_ylabel("% of properties verified", fontsize=15)
    ax_value.set_xlabel("Computation time [s]", fontsize=15)
    plt.xscale('log', nonposx='clip')
    ax_value.legend(fontsize=9.5, loc="lower right")
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()

    figures_path = "./plots/"
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    plt.savefig(figures_path + fig_name, format='pdf', dpi=300)


def timeout_setter(tables, timeout, exclude_timeouts=False, exclude_branch_timeouts=False):

    if exclude_branch_timeouts:
        table_locs = tables[0].apply(lambda x: (x == '!!!').any(), axis=1)  # dummy condition to get all false
        for table in tables:
            if exclude_branch_timeouts not in table.columns.tolist()[1]:
                table_locs = table_locs | table.apply(lambda x: (x == 'timeout').any(), axis=1)

    for table in tables:
        for ccol in table.columns.tolist():
            if 'Time' in ccol:
                time_name = ccol
                if not exclude_timeouts:
                    table.loc[table[ccol] >= timeout, time_name] = timeout
                    table.loc[table[ccol] >= timeout, sat_name] = 'timeout'
            if 'Bran' in ccol:
                bran_name = ccol
            if "SAT" in ccol:
                sat_name = ccol
        for column in table:
            if "SAT" in column:
                if exclude_timeouts:
                    table.loc[table[column] == 'timeout', time_name] = float("nan")
                    table.loc[table[column] == 'timeout', bran_name] = float("nan")
                else:
                    table.loc[table[column] == 'timeout', time_name] = timeout
                    if exclude_branch_timeouts:
                        # exclude from branching computations cases where some algorithms (except those whose name
                        # contains exclude_branch_timeouts) have timed out
                        table.loc[table_locs, bran_name] = float("nan")


def exclude_sat_props(tables):
    for table in tables:
        for column in table:
            if "SAT" in column:
                table.drop(table[table[column] == 'True'].index, inplace=True)


def to_latex_table(folder, bases, file_list, labels, plot_names, timeout, one_vs_all=False, exclude_timeouts=False,
                   skip_nbranches=False, exclude_branch_timeouts=False, idx_key="Idx", exclude_sat=False):

    # latex_tables
    latex_tables = []
    for base, plt_name in zip(bases, plot_names):
        # Create latex table.
        tables = []
        for filename in file_list:
            m = pd.read_pickle(folder + f"{base}_" + filename).dropna(how="all")
            tables.append(m)
        # keep only the properties in common
        for m in tables:
            if not one_vs_all:
                m["unique_id"] = m[idx_key].map(int).map(str) + "_" + m["prop"].map(int).map(str)
            else:
                m["unique_id"] = m[idx_key].map(int).map(str)
            m["Eps"] = m[idx_key].map(float)
        for m1, m2 in itertools.product(tables, tables):
            m1.drop(m1[(~m1['unique_id'].isin(m2["unique_id"]))].index, inplace=True)

        if exclude_sat:
            exclude_sat_props(tables)

        # Set all timeouts to <timeout> seconds.
        timeout_setter(tables, timeout, exclude_timeouts=exclude_timeouts,
                       exclude_branch_timeouts=exclude_branch_timeouts)

        full_table = tables[0]
        for c_table in tables[1:]:
            full_table = pd.merge(full_table, c_table, on=[idx_key, 'prop', 'Eps', 'unique_id'], how='inner')

        # Create summary table.
        summary_dict = {}
        for column in full_table:
            excluded_columns = [idx_key, 'prop', 'Eps', 'unique_id']
            if not skip_nbranches:
                col_condition = lambda x: x not in excluded_columns
            else:
                col_condition = lambda x: x not in excluded_columns and "Bran" not in column
            if col_condition(column):
                if "SAT" not in column:
                    c_mean = full_table[column].mean()
                    summary_dict[column] = c_mean
                else:
                    # Handle SAT status
                    n_timeouts = len(full_table.loc[full_table[column] == 'timeout'])
                    m_len = len(full_table)
                    summary_dict[column + "_perc_timeout"] = n_timeouts / m_len * 100

        # Re-sort by method, exploiting that the columns are ordered per method.
        latex_table_dict = {}
        for counter, key in enumerate(summary_dict):
            c_key = key.split("_")[0]
            if c_key == "FS":
                continue
            if c_key in latex_table_dict:
                latex_table_dict[c_key].append(summary_dict[key])
            else:
                latex_table_dict[c_key] = [summary_dict[key]]

        latex_table = pd.DataFrame(latex_table_dict)
        latex_table = latex_table.rename(columns={"BSAT": "%Timeout", "BBran": "Sub-problems", "BTime": "Time [s]"})
        latex_tables.append(latex_table[latex_table.columns[::-1]])

    merged_plot_names = [cname.split(" ")[0] for cname in plot_names]
    merged_latex_table = pd.concat(latex_tables, axis=1, keys=merged_plot_names)
    # print(merged_latex_table)
    converted_latex_table = merged_latex_table.to_latex(float_format="%.2f")
    print(f"latex table.\n Row names: {labels} \n Table: \n{converted_latex_table}")


def iclr_plots(use_autostrat=False):

    stratstring = "_auto_strat" if use_autostrat else ""

    # Plots for OVAL-CIFAR
    folder = './cifar_results/'
    timeout = 3600
    bases = ["base_100", "wide_100", "deep_100"]
    plot_names = ["Base model", "Wide large model", "Deep large model"]
    time_base = "BTime_SR"
    file_list_nobase = [
        "SR_prox_100-pinit-eta100.0-feta100.0.pkl",
        "SR_bigm-adam_180-pinit-ilr0.01,flr0.0001.pkl",
        "SR_cut_100_no_easy-pinit-ilr0.001,flr1e-06-cut_add2.0-diilr0.01,diflr0.0001.pkl",
        f"SR_cut_100{stratstring}-pinit-ilr0.001,flr1e-06-cut_add2.0-diilr0.01,diflr0.0001.pkl",
        "anderson-mip.pkl",
        f"SR_gurobi-anderson_1{stratstring}.pkl",
        "eran.pkl"
    ]
    time_name_list = [
        f"{time_base}_prox_100",
        f"{time_base}_bigm-adam_180",
        f"{time_base}_cut_100_no_easy",
        f"{time_base}_cut_100{stratstring}",
        f"BTime_anderson-mip",
        f"{time_base}_gurobi-anderson_1{stratstring}",
        f"{time_base}_eran"
    ]
    labels = [
        "BDD+ BaBSR",
        "Big-M BaBSR",
        "Active Set BaBSR",
        "Big-M + Active Set BaBSR",
        r"MIP $\mathcal{A}_k$",
        "G. Planet + G. 1 cut BaBSR",
        "ERAN"
    ]
    line_styles = [
        "dotted",
        "solid",
        "solid",
        "solid",
        "dotted",
        "dotted",
        "dotted",
    ]
    for base, plt_name in zip(bases, plot_names):
        file_list = [f"{base}_" + cfile for cfile in file_list_nobase]
        plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, base + ".pdf", plt_name,
                                create_csv=False, linestyles=line_styles)
    to_latex_table(folder, bases, file_list_nobase, labels, plot_names, timeout)

    plt.show()


def jmlr_plots(use_autostrat=False):

    stratstring = "_auto_strat" if use_autostrat else ""

    # Plots for OVAL-CIFAR
    bases = ["base_100", "wide_100", "deep_100"]
    plot_names = ["Base model", "Wide large model", "Deep large model"]
    folder = './cifar_results/'
    timeout = 3600
    time_base = "BTime_SR"
    file_list_nobase = [
        f"SR_cut_100{stratstring}-pinit-ilr0.001,flr1e-06-cut_add2.0-diilr0.01,diflr0.0001.pkl",
        f"SR_cut_600{stratstring}-pinit-ilr0.001,flr1e-06-cut_add2.0-diilr0.01,diflr0.0001.pkl",
        f"SR_sp-fw_1000{stratstring}-pinit-fw_start10.0-diilr0.01,diflr0.0001.pkl",
        f"SR_cut_1650{stratstring}-pinit-ilr0.001,flr1e-06-cut_add2.0-diilr0.01,diflr0.0001.pkl",
        f"SR_sp-fw_4000{stratstring}-pinit-fw_start10.0-diilr0.01,diflr0.0001.pkl",
    ]
    time_name_list = [
        f"{time_base}_cut_100{stratstring}",
        f"{time_base}_cut_600{stratstring}",
        f"{time_base}_sp-fw_1000{stratstring}",
        f"{time_base}_cut_1650{stratstring}",
        f"{time_base}_sp-fw_4000{stratstring}",
    ]
    labels = [
        "Big-M + Active Set 100 it.",
        "Big-M + Active Set 600 it.",
        "Big-M + Saddle Point 1000 it.",
        "Big-M + Active Set 1650 it.",
        "Big-M + Saddle Point 4000 it.",
    ]
    line_styles = [
        "dotted",
        "solid",
        "solid",
        "solid",
        "solid",
    ]
    for base, plt_name in zip(bases, plot_names):
        file_list = [f"{base}_" + cfile for cfile in file_list_nobase]
        plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, base + "spfw.pdf", plt_name,
                                create_csv=False, linestyles=line_styles)
    to_latex_table(
        folder, bases, file_list_nobase, labels, plot_names, timeout, exclude_timeouts=False, one_vs_all=True)

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    iclr_plots(use_autostrat=True)
    jmlr_plots(use_autostrat=True)
