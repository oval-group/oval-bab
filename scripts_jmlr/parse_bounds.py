#!/usr/bin/env python
import argparse
import glob
import itertools
import seaborn as sns
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import math


def parse_results(file_content, sp_idx, method_name):
    results = []
    lines = file_content.split('\n')
    timing = float(lines[0])
    all_bounds = lines[1].split('\t')
    all_bounds = list(map(float, all_bounds))

    for neur_idx, bound in enumerate(all_bounds):
        results.append({
            "Neuron index": neur_idx,
            "Sample": sp_idx,
            "Value": bound,
            "Method": method_name,
            "Time": timing
        })
    return results


def do_violin_plots(datapoints, filename, labels, planet_only):
    datapoints = datapoints.reset_index(level=1)

    fig, (time_ax, bound_ax) = plt.subplots(2, 1, sharex='col',
                                            figsize=(20, 7))
    sns.boxplot(x="Method", y="Time", data=datapoints, ax=time_ax, order=labels, showfliers=False)
    time_ax.set(yscale="log")

    bound_label = "Planet optimality gap" if planet_only else "Improvement from 1 cut"

    for clabel in labels:
        slice = datapoints[(datapoints['Method'] == clabel)]
        slice = slice[slice[bound_label].between(slice[bound_label].quantile(.001), slice[bound_label].quantile(.999))]
        datapoints[(datapoints['Method'] == clabel)] = slice

    sns.violinplot(x="Method", y=bound_label, data=datapoints,
                   inner=None, width=1, scale='area', ax=bound_ax,
                   order=labels, cut=0)
    if planet_only:
        bound_ax.set(yscale="log")
    else:
        # bound_ax.set(yscale="symlog")
        bound_ax.set(yscale="linear")
    bound_ax.xaxis.set_label_text("")

    target_format = "pdf" if filename.endswith('.pdf') else "png"
    plt.tight_layout()
    plt.savefig(filename, format=target_format, dpi=50)
    plt.savefig(filename.replace(target_format, 'pdf'), pad_inches=0)


def main(results_folder, output_image_prefix, files_to_load, legend_names, hex_plots,
         planet_only=False, violin_opts=None, mnist=False, max_idx=None):

    all_results = []
    for sp_idx in sorted(os.listdir(results_folder)):
        sp_results = []
        if max_idx is not None and int(sp_idx) >= max_idx:
            break
        for method_name, filename in zip(legend_names, files_to_load):
            file_to_load = os.path.join(results_folder, sp_idx, filename)
            if os.path.exists(file_to_load):
                with open(file_to_load, 'r') as res_file:
                    sp_results.extend(parse_results(res_file.read(),
                                                    int(sp_idx),
                                                    method_name))
            else:
                break
        else:
            all_results.extend(sp_results)

    all_results = pd.DataFrame(all_results)
    n_completed = all_results[all_results["Method"] == "Gurobi\nPlanet"]["Sample"].max()
    print(f'Completed images: {n_completed}')
    all_results["Optimization Problem"] = all_results["Sample"].astype(str).str.cat(
        all_results["Neuron index"].astype(str), sep=' - ')
    all_results.drop(columns=["Neuron index", "Sample"], inplace=True)
    if planet_only:
        planet_bounds = all_results[all_results["Method"] == 'Gurobi\nPlanet'].drop(columns=["Method", "Time"])
        planet_bounds.set_index(["Optimization Problem"], inplace=True)
        planet_bounds = planet_bounds.squeeze()
        all_results.set_index(["Optimization Problem", "Method"], inplace=True)

        all_results["Planet optimality gap"] = all_results["Value"] - planet_bounds
    if not planet_only:
        gur1cut_bounds = all_results[all_results["Method"] == 'Gurobi\n1 cut'].drop(columns=["Method", "Time"])
        planet_bounds = all_results[all_results["Method"] == 'Gurobi\nPlanet'].drop(columns=["Method", "Time"])
        gur1cut_bounds.set_index(["Optimization Problem"], inplace=True)
        planet_bounds.set_index(["Optimization Problem"], inplace=True)
        gur1cut_bounds = gur1cut_bounds.squeeze()
        planet_bounds = planet_bounds.squeeze()
        all_results.set_index(["Optimization Problem", "Method"], inplace=True)

        all_results["Improvement from 1 cut"] = gur1cut_bounds - all_results["Value"]
        all_results["Gap to Planet"] = all_results["Value"] - planet_bounds
        all_results["Improvement from Planet"] = planet_bounds - all_results["Value"]

    if violin_opts is None:
        do_violin_plots(all_results, output_image_prefix + "_violins.png", legend_names, planet_only)
    else:
        for cname, clabels in zip(violin_opts["names"], violin_opts["labels"]):
            do_violin_plots(all_results, output_image_prefix + cname + "_violins.png", clabels, planet_only)

    # do_pairwise_plot(all_results, output_image_prefix)

    for hex_plot in hex_plots:
        do_selected_pairwise_hex(all_results, output_image_prefix, hex_plot, mnist=mnist)


def do_selected_pairwise_hex(datapoints, filename_prefix, plot_type, mnist=False):

    allowed_plots = [
        "anderson_bigm",
        "anderson_planet_improvement",
        "anderson_planet_improvement_jmlr",
        "anderson_1cut_improvement",
        "anderson_fc_1cut_improvement",
    ]
    if plot_type not in allowed_plots:
        raise ValueError("Wrong plot type for do_selected_pairwise_hex.")

    sns.set(font_scale=2.5)
    sns.set_palette(sns.color_palette("Set2"))
    sns.set_style("ticks")

    datapoints = datapoints.reset_index(level=1)
    gridsize = 35

    if plot_type == "uai_planet_cheap":
        comparisons = [
            ("WK", "IBP"),
            ("CROWN", "WK")
        ]
        bnd_scale = 'log'
        time_extent = (1e-3, 4e-3, 1e-3, 4e-3)
        to_plot = "Planet optimality gap"

    elif plot_type == "uai_planet_exp":
        comparisons = [
            ("Proximal\n400 steps", "Supergradient\n740 steps"),
            ("Supergradient\n740 steps", "DSG+\n1040 steps"),
            ("Supergradient\n740 steps", "Dec-DSG+\n1040 steps"),
            ("Dec-DSG+\n1040 steps", "DSG+\n1040 steps"),
            ("Proximal\n400 steps", "CROWN"),
            ("Proximal\n100 steps", "CROWN"),
        ]
        bnd_scale = 'log'
        time_extent = (1e-3, 4, 1e-3, 4)
        to_plot = "Planet optimality gap"

    elif plot_type == "anderson_bigm":
        comparisons = [
            ("BDD+\n400 steps", "Big-M\n850 steps"),
        ]
        time_extent = (1, 4, 1, 4)
        bnd_scale = 'log'
        to_plot = "Gap to Planet"

    elif plot_type in ["anderson_planet_improvement", "anderson_planet_improvement_jmlr"]:
        if not mnist:
            comparisons = [
                ("Big-M\n850 steps", "Active Set\n80 steps"),
            ]
            if "jmlr" in plot_type:
                comparisons.append(("Big-M\n850 steps", "Saddle Point\n80 steps"))
        else:
            # mnist
            comparisons = [
                ("Gurobi\n1 cut", "Active Set\n1650 steps"),
                ("Gurobi\n1 cut", "Active Set\n2500 steps"),
            ]
            if "jmlr" in plot_type:
                comparisons += [
                ("Gurobi\n1 cut", "Saddle Point\n7500 steps"),
                ("Gurobi\n1 cut", "AS-SP\n1050 + 5500 steps"),
            ]
        time_extent = (1, 4, 1, 4) if not mnist else (20, 130, 20, 130)
        # time_extent = (2, 3, 2, 3)
        bnd_scale = 'linear'
        to_plot = "Improvement from Planet"

    elif plot_type == "anderson_1cut_improvement":
        # Both Cuts and SP-FW.
        comparisons = [
            ("Active Set\n1650 steps", "Saddle Point\n4000 steps"),
            ("Active Set\n600 steps", "Saddle Point\n1000 steps"),
            ("Active Set\n1050 steps", "Saddle Point\n2000 steps"),
        ]
        if not mnist:
            comparisons.append(("Active Set\n1650 steps", "AS-SP\n600 + 3000 steps"))
        else:
            comparisons.append(("Active Set\n2500 steps", "Saddle Point\n7500 steps"))
            comparisons.append(("Active Set\n2500 steps", "AS-SP\n1050 + 5500 steps"))
        time_extent = (8, 40, 8, 40) if not mnist else (8, 90, 8, 90)
        bnd_scale = 'linear'
        to_plot = "Improvement from 1 cut"

    elif plot_type == "anderson_fc_1cut_improvement":
        # Both Cuts and SP-FW.
        comparisons = [
            ("Active Set\n500 steps", "Saddle Point\n200 steps"),
            ("Active Set\n1200 steps", "Saddle Point\n500 steps"),
            ("Active Set\n2400 steps", "Saddle Point\n1000 steps"),
        ]
        time_extent = (30, 800, 30, 800)
        bnd_scale = 'linear'
        to_plot = "Improvement from 1 cut"

    target_dir = filename_prefix + "_pairwise_comparisons"
    os.makedirs(target_dir, exist_ok=True)

    for meth1, meth2 in comparisons:

        filename = "{meth1}_vs_{meth2}_hex.pdf".format(meth1=meth1, meth2=meth2)
        filename = filename.replace("\n", '_')
        filename = filename.replace(" ", '_')
        filename = os.path.join(target_dir, filename)
        target_format = "pdf" if filename.endswith(".pdf") else "png"
        meth1_data = datapoints[datapoints.Method == meth1]
        meth2_data = datapoints[datapoints.Method == meth2]

        fig, (time_ax, bound_ax) = plt.subplots(1, 2, figsize=(20, 10))
        timescale = 'linear'
        if abs(math.log10(meth1_data.Time.mean()) - math.log10(meth2_data.Time.mean())) > 2:
            timescale = 'log'
            ctime_extent = (math.log10(time_extent[0]), math.log10(time_extent[1]),
                           math.log10(time_extent[2]), math.log10(time_extent[3]))
        else:
            ctime_extent = time_extent

        time_ax.hexbin(x=meth1_data.Time, y=meth2_data.Time, cmap="Greens", gridsize=gridsize, mincnt=1,
                       xscale=timescale, yscale=timescale, bins='log', extent=ctime_extent)
        time_ax.xaxis.set_label_text(meth1.replace('\n', ' '))
        time_ax.yaxis.set_label_text(meth2.replace('\n', ' '))
        time_ax.set_title("Timing (in s)")
        xlim = time_ax.get_xlim()
        ylim = time_ax.get_ylim()
        rng = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        time_ax.plot(rng, rng, ls="--", c=".3")
        plt.setp(time_ax.get_xticklabels(which='both'), rotation=30, horizontalalignment='right')

        datamin = min(meth1_data[to_plot].min(), meth2_data[to_plot].min())
        datamax = max(meth1_data[to_plot].max(), meth2_data[to_plot].max())
        if bnd_scale == "log":
            bnd_extent = (math.log10(datamin), math.log10(datamax), math.log10(datamin), math.log10(datamax))
        else:
            bnd_extent = (datamin, datamax, datamin, datamax)
        bound_ax.hexbin(x=meth1_data[to_plot], y=meth2_data[to_plot], gridsize=gridsize, mincnt=1, cmap="Greens",
                        xscale=bnd_scale, yscale=bnd_scale, bins='log', extent=bnd_extent)
        bound_ax.xaxis.set_label_text(meth1.replace('\n', ' '))
        bound_ax.yaxis.set_label_text(meth2.replace('\n', ' '))
        bound_ax.set_title(to_plot)
        xlim = bound_ax.get_xlim()
        ylim = bound_ax.get_ylim()
        rng = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        bound_ax.plot(rng, rng, ls="--", c=".3")
        plt.setp(bound_ax.get_xticklabels(which='both'), rotation=30, horizontalalignment='right')

        if plot_type != "uai_planet":
            rng_x = (xlim[0], 0)
            rng_y = (ylim[0], 0)
            val = (0, 0)
            bound_ax.plot(rng_x, val, ls="--", c=".3", alpha=0.5)
            bound_ax.plot(val, rng_y, ls="--", c=".3", alpha=0.5)

        plt.tight_layout()
        plt.savefig(filename, format=target_format, dpi=100)

        plt.close(fig)


def jmlr_cifar_plots():

    temp_dir = "../results/anderson_jmlr/"
    os.system(f"mkdir -p {temp_dir}")

    # Define the filenames to be loaded for this plot.
    files_to_load = [
        "Gurobi-fromintermediate-fixed.txt",
        "Anderson-1cuts-fromintermediate-fixed.txt",
        "Proximal_finalmomentum_400-fromintermediate.txt",
        "Big-M_850-fromintermediate.txt",
    ]
    for citer in [80, 600, 1050, 1650]:
        files_to_load.append(f"Cuts_{citer}-fromintermediate.txt")
    files_to_load.append("CutsCPU_600-fromintermediate.txt")
    for citer in [80, 1000, 2000, 4000]:
        files_to_load.append(f"SP-FW_{citer}-fromintermediate.txt")
    files_to_load.append("SP-FW_Cuts_3000-600-fromintermediate.txt")
    files_to_load.append(f"SP-FWCPU_1000-fromintermediate.txt")
    # sensitivity to selection criterion
    for citers in [600, 1050, 1650]:
        files_to_load.append(f"Cuts_randmask_{citers}-fromintermediate.txt")
    # sensitivity to selection frequency
    freq_steps = [(300, [550, 900, 1500]), (600, [650, 1150, 1850])]
    for freq, cut_steps_list in freq_steps:
        for cut_steps in cut_steps_list:
            files_to_load.append(f"Cuts_freq{freq}_{cut_steps}-fromintermediate.txt")

    # Properly label the filenames to be loaded.
    legend_names = [
        "Gurobi\nPlanet",
        "Gurobi\n1 cut",
        "BDD+\n400 steps",
        "Big-M\n850 steps",
    ]
    for citer in [80, 600, 1050, 1650]:
        legend_names.append(f"Active Set\n{citer} steps")
    legend_names.append("Active Set CPU\n600 steps")
    for citer in [80, 1000, 2000, 4000]:
        legend_names.append(f"Saddle Point\n{citer} steps")
    legend_names.append("AS-SP\n600 + 3000 steps")
    legend_names.append("Saddle Point CPU\n1000 steps")
    # sensitivity to selection criterion
    for citers in [600, 1050, 1650]:
        legend_names.append(f"Ran. Selection\n{citers} steps")
    # sensitivity to selection frequency
    for freq, cut_steps_list in freq_steps:
        for cut_steps in cut_steps_list:
            legend_names.append(fr"AS, $\omega={freq}$" + f"\n{cut_steps} steps")

    labels_as = legend_names[:9]
    labels_spfw = legend_names[:4] + legend_names[9:15]

    labels_criterion = ["Gurobi\nPlanet"]
    labels_frequency = []
    for i, citers in enumerate([600, 1050, 1650]):
        labels_criterion.append(f"Active Set\n{citers} steps")
        labels_frequency.append(f"Active Set\n{citers} steps")
        labels_criterion.append(f"Ran. Selection\n{citers} steps")
        for freq, cut_steps_list in freq_steps:
            labels_frequency.append(fr"AS, $\omega={freq}$" + f"\n{cut_steps_list[i]} steps")
    violin_opts = {
        "labels": [labels_as, labels_spfw, labels_criterion, labels_frequency],
        "names": ["as", "spfw", "sensitivity", "frequency"],
    }

    # Madry and SGD experiments.
    folders = [("../results/madry8/", f"{temp_dir}/madry8_and"), ("../results/sgd8/", f"{temp_dir}/sgd8_and")]
    hex_plots = ["anderson_bigm", "anderson_planet_improvement_jmlr", "anderson_1cut_improvement"]
    for results_folder, output_image_prefix in folders:
        sns.set(font_scale=1.5); sns.set_palette(sns.color_palette("Set2")); sns.set_style("whitegrid")
        main(results_folder, output_image_prefix, files_to_load, legend_names, hex_plots, violin_opts=violin_opts)


def jmlr_fc_cifar_plots(max_idx=None):

    temp_dir = "../results/anderson_jmlr_fc/"
    os.system(f"mkdir -p {temp_dir}")

    # Define the filenames to be loaded for this plot.
    files_to_load = [
        "Gurobi-fromintermediate-fixed.txt",
        "Anderson-1cuts-fromintermediate-fixed.txt",
        "Proximal_finalmomentum_400-fromintermediate.txt",
    ]
    strings = ["Cuts_{}-fromintermediate_fixed.txt", "SP-FW_{}-fromintermediate.txt"]
    citers = [[500, 1200, 2400], [200, 500, 1000]]
    for idx in range(len(citers[0])):
        for string, citer in zip(strings, citers):
            files_to_load.append(string.format(citer[idx]))

    # Properly label the filenames to be loaded.
    legend_names = [
        "Gurobi\nPlanet",
        "Gurobi\n1 cut",
        "BDD+\n400 steps",
    ]
    strings = ["Active Set\n{} steps", "Saddle Point\n{} steps"]
    citers = [[500, 1200, 2400], [200, 500, 1000]]
    for idx in range(len(citers[0])):
        for string, citer in zip(strings, citers):
            legend_names.append(string.format(citer[idx]))
    violin_opts = {
        "labels": [legend_names],
        "names": ["spfw"],
    }

    # Madry and SGD experiments.
    folders = [("../results/adv_fc_2/", f"{temp_dir}/adv_fc_2")]
    hex_plots = ["anderson_fc_1cut_improvement"]
    for results_folder, output_image_prefix in folders:
        sns.set(font_scale=1.5); sns.set_palette(sns.color_palette("Set2")); sns.set_style("whitegrid")
        main(results_folder, output_image_prefix, files_to_load, legend_names, hex_plots, violin_opts=violin_opts,
             max_idx=max_idx)


def jmlr_mnist_plots():

    temp_dir = "../results/anderson_jmlr_madry/"
    os.system(f"mkdir -p {temp_dir}")

    # Define the filenames to be loaded for this plot.
    files_to_load = [
        "Gurobi-fromintermediate-fixed.txt",
        "Anderson-1cuts-fromintermediate-fixed.txt",
        "Proximal_finalmomentum_400-fromintermediate.txt",
        "Big-M_850-fromintermediate.txt",
    ]
    for citer in [80, 600, 1050, 1650, 2500]:
        files_to_load.append(f"Cuts_{citer}-fromintermediate.txt")
    for citer in [80, 1000, 2000, 4000, 7500]:
        files_to_load.append(f"SP-FW_{citer}-fromintermediate.txt")
    files_to_load.append("SP-FW_Cuts_5500-1050-fromintermediate.txt")

    # Properly label the filenames to be loaded.
    legend_names = [
        "Gurobi\nPlanet",
        "Gurobi\n1 cut",
        "BDD+\n400 steps",
        "Big-M\n850 steps",
    ]
    for citer in [80, 600, 1050, 1650, 2500]:
        legend_names.append(f"Active Set\n{citer} steps")
    for citer in [80, 1000, 2000, 4000, 7500]:
        legend_names.append(f"Saddle Point\n{citer} steps")
    legend_names.append("AS-SP\n1050 + 5500 steps")

    labels_as = legend_names[:9]
    labels_spfw = legend_names[:3] + legend_names[9:]
    violin_opts = {
        "labels": [labels_as, labels_spfw],
        "names": ["as", "spfw"],
    }

    # MNIST wide net experiments.
    folders = [("../results/mnist_wide/", f"{temp_dir}/mnist_wide_and")]

    hex_plots = ["anderson_bigm", "anderson_planet_improvement_jmlr", "anderson_1cut_improvement"]
    for results_folder, output_image_prefix in folders:
        sns.set(font_scale=1.5); sns.set_palette(sns.color_palette("Set2")); sns.set_style("whitegrid")
        main(results_folder, output_image_prefix, files_to_load, legend_names, hex_plots, violin_opts=violin_opts,
             mnist=True)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_idx', type=int)
    args = parser.parse_args()

    jmlr_cifar_plots()
    jmlr_mnist_plots()
    jmlr_fc_cifar_plots(args.max_idx)
