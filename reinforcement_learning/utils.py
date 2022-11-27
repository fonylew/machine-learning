import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Plot customization
plt.rcParams['axes.grid'] = True
plt.style.use('seaborn-colorblind')

IMAGE_DIR = 'plots/'

def magic_plot(results, task, method, style, x_list):
    fig, axs = plt.subplots(1 , 3,figsize=(18, 4), constrained_layout=True,)
    for i, x in enumerate(x_list):
        axs[0].plot(results[i]["Iteration"], results[i]["Mean V"], style)
    axs[0].set_ylabel("Mean Value")
    axs[0].set_xlabel("Iteration")
    axs[0].legend(x_list, loc='best')
    axs[0].set_title(f"{method}: max value [{task}]")

    for i, x in enumerate(x_list):
        axs[1].plot(results[i]["Iteration"], results[i]["Max V"], style)
    axs[1].set_ylabel("Max Value")
    axs[1].set_xlabel("Iteration")
    axs[1].legend(x_list, loc='best')
    axs[1].set_title(f"{method}: mean value [{task}]")

    for i, x in enumerate(x_list):
        axs[2].plot(results[i]["Iteration"], results[i]["times"], style)
    axs[2].set_ylabel("Time (seconds)")
    axs[2].set_xlabel("Iteration")
    axs[2].legend(x_list, loc='best')
    axs[2].set_title(f"{method}: Time elapse [{task}]")
    plt.show()

def record(run_stats, variables):
    times = []
    output_dict = {v:[] for v in variables}
    output_dict["times"] = times
    for result in run_stats:
        times.append(result["Time"])
        for v in result:
            if v in variables:
                output_dict[v].append(result[v])
    return output_dict

def compose_discounts(significant_digits):
    prev_discount = 0
    discounts = []
    for i in range(1,significant_digits + 1):
        discounts.append(round(prev_discount + 9*(10**-i),i))
        prev_discount = discounts[-1]
    return discounts


def plot_data(x_var, y_var, x_label, y_label, title, legend=[], figure_size=(4,3), style="o-"):
    plt.rcParams["figure.figsize"] = figure_size
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(legend, loc='best')
    plt.plot(x_var, y_var, style)
    plt.show()

def plot_data_legend(x_vars, x_label, all_y_vars, y_var_labels, y_label, title, y_bounds=None, style="o-"):
    colors = ['red','orange','black','green','blue','violet']
    plt.rcParams["figure.figsize"] = (4,3)
    i = 0
    for y_var in all_y_vars:
        plt.plot(x_vars, y_var, style, color=colors[i % 6], label=y_var_labels[i])
        i += 1
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if y_bounds != None:
        plt.ylim(y_bounds)
    plt.legend()
    plt.show()

def plot_heatmap_policy(policy, V, rows, columns):
    policy_labels = np.empty([rows, columns], dtype='<U10')
    for row in range(rows):
        for col in range(columns):
            state = row * columns + col
            policy_labels[row, col] += '<--' * (policy[state] == 0)
            policy_labels[row, col] += 'v' * (policy[state] == 1)
            policy_labels[row, col] += '-->' * (policy[state] == 2)
            policy_labels[row, col] += '^' * (policy[state] == 3)

    sns.heatmap(V.reshape(rows, columns), annot=policy_labels, fmt='', linewidths=.5)


def plot_heatmap_value_function(V, rows, columns):
    sns.heatmap(V.reshape(rows, columns), annot=True, fmt='.3f', linewidths=.5)


def plot_value_convergence(convergence, iteration):
    plt.plot(np.arange(1, iteration + 1), convergence)


def plot_value_function(V, iteration, show_label=False):
    if show_label:
        plt.plot(np.arange(1, len(V) + 1), V, label='iteration = {}'.format(iteration))
    else:
        plt.plot(np.arange(1, len(V) + 1), V)


def plot_optimal_policy(policy):
    plt.bar(np.arange(0, len(policy)), policy, color='blue')


def save_figure(title):
    plt.savefig(IMAGE_DIR + title)
    plt.close()


def set_plot_title_labels(title, x_label='', y_label='', legend=False):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if legend:
        plt.legend(loc='best')
