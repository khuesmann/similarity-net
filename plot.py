import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def get_durations(latents):
    durations = {}
    for run in latents:
        durations[run] = latents[run].shape[0]
    return durations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot results")

    parser.add_argument("-m", "--model", required=True, type=str)
    parser.add_argument("-ds", "--dataset", required=True, type=str)
    parser.add_argument("-cp", "--results_path", default='results', type=str)
    parser.add_argument("-lp", "--legend_position", default=2, type=int)
    parser.add_argument("-td", "--two_dimensional", default=False, type=bool)
    parser.add_argument("-mt", "--max_timesteps", default=None, type=int)
    parser.add_argument("-sl", "--show_legend", default=True, type=int)

    args = parser.parse_args()

    results_path = os.path.join(args.results_path, f'{args.model}-{args.dataset}')

    with open(os.path.join(results_path, 'all_latents.pkl'), 'rb') as f:
        all_latents = pickle.load(f)

    with open(os.path.join(results_path, 'all_attentions.pkl'), 'rb') as f:
        all_attentions = pickle.load(f)

    run_names = [n.replace('.npy', '') for n in list(all_latents.keys())]
    if len(run_names[0]) > 20:
        run_names = ['run{}'.format(i) for i, n in enumerate(run_names)]

    PC = 0
    PC2 = 1

    # available = ['default'] + plt.style.available
    # with plt.style.context('seaborn-whitegrid'):
    with plt.style.context('tableau-colorblind10'):
        fig, axs = plt.subplots(1, 3, figsize=(5, 2), dpi=300)

        for i, c in enumerate(all_latents.values()):
            if args.two_dimensional:
                l1, = axs[0].plot(c[:args.max_timesteps, PC], c[:args.max_timesteps, PC2], label=run_names[i])
            else:
                l1, = axs[0].plot(c[:args.max_timesteps, PC], label=run_names[i])
        # axs[0].set_title('SimilarityNet')
        axs[0].text(.0, .95, 'a)', horizontalalignment='left', transform=axs[0].transAxes)
        axs[0].get_yaxis().set_visible(False)
        if args.two_dimensional:
            axs[0].get_xaxis().set_visible(False)

        try:
            distance_correlation = np.load(os.path.join(results_path, 'distance_correlation.npy'))
            # plt.imshow(distance_correlation)
            # plt.show()

            with open(os.path.join(results_path, 'mds_correlation.pkl'), 'rb') as f:
                mds_correlation = pickle.load(f)

            for i, c in enumerate(mds_correlation):
                if args.two_dimensional:
                    l2, = axs[2].plot(c[:args.max_timesteps, PC], c[:args.max_timesteps, PC2], label=run_names[i])
                else:
                    l2, = axs[2].plot(c[:args.max_timesteps, PC], label=run_names[i])
            axs[2].text(.0, .95, 'c)', horizontalalignment='left', transform=axs[2].transAxes)
            axs[2].get_yaxis().set_visible(False)
            if args.two_dimensional:
                axs[2].get_xaxis().set_visible(False)
        except Exception as e:
            print(e)
        try:

            with open(os.path.join(results_path, 'mds_field_similarity.pkl'), 'rb') as f:
                mds_field_similarity = pickle.load(f)
            mds_eigen_field_similarity = np.load(os.path.join(results_path, 'mds_eigen_field_similarity.npy'))

            for i, c in enumerate(mds_field_similarity):
                if args.two_dimensional:
                    l3, = axs[1].plot(c[:args.max_timesteps, PC], c[:args.max_timesteps, PC2], label=run_names[i])
                else:
                    l3, = axs[1].plot(c[:args.max_timesteps, PC], label=run_names[i])
            # axs[1].set_title('MDS (Field Similarity)')
            axs[1].text(.0, .95, 'b)', horizontalalignment='left', transform=axs[1].transAxes)
            axs[1].get_yaxis().set_visible(False)
            if args.two_dimensional:
                axs[1].get_xaxis().set_visible(False)
        except Exception as e:
            print(e)
            pass

        # if len(all_latents) <= 10 and args.show_legend:
        #    axs[args.legend_position].legend()

        # fig.subplots_adjust(bottom=0.3)

        # axs[1].legend(handles=[l1, l2, l3], labels=run_names, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=3)

        plt.tight_layout(pad=0)
        plt.savefig(os.path.join('tmp', args.dataset + '.png'))
        plt.show()
