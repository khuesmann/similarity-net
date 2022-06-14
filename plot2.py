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
    # model attributes
    parser.add_argument("-m", "--model", required=True, type=str)
    parser.add_argument("-ds", "--dataset", required=True, type=str)
    parser.add_argument("-cp", "--results_path", default='results', type=str)
    parser.add_argument("-lp", "--legend_position", default=2, type=int)
    parser.add_argument("-td", "--two_dimensional", default=False, type=bool)

    args = parser.parse_args()

    results_path = os.path.join(args.results_path, f'{args.model}-{args.dataset}')

    with open(os.path.join(results_path, 'all_latents.pkl'), 'rb') as f:
        all_latents = pickle.load(f)

    with open(os.path.join(results_path, 'all_attentions.pkl'), 'rb') as f:
        all_attentions = pickle.load(f)

    durations = get_durations(all_latents)
    run_names = [n.replace('.npy', '') for n in list(all_latents.keys())]

    # available = ['default'] + plt.style.available
    available = ['fivethirtyeight']
    available = ['tableau-colorblind10']
    for i, style in enumerate(available):
        with plt.style.context(style):

            fig, axs = plt.subplots(len(run_names), 3, figsize=(6, 6), dpi=300, sharex=True, subplot_kw=dict(frameon=True))
            # plt.subplots_adjust(hspace=.0)
            PC = 0
            lat_min = min([l[..., PC].min() for l in list(all_latents.values())])
            lat_max = max([l[..., PC].max() for l in list(all_latents.values())])
            d = (lat_max - lat_min) * 0.05
            lat_min -= d
            lat_max += d

            for j, run_name in enumerate(run_names):
                lat = list(all_latents.values())[j][:, PC]
                lat2 = list(all_latents.values())[j][:, 1]
                if args.two_dimensional:
                    axs[j][0].plot(lat, lat2, label=run_names[i])
                else:
                    axs[j][0].plot(lat, label=run_name, linewidth=.3)
                axs[j][0].get_yaxis().set_ticks([])
                axs[j][0].set_ylim([lat_min, lat_max])
                if j == len(run_names) - 1:
                    axs[j][0].label_outer()
                axs[j][0].set_ylabel(run_name if len(run_name) < 8 else f'r{str(j).zfill(2)}')
                if j == 0:
                    axs[j][0].set_title('SimilarityNet')
                    # axs[j][0].text(.5, .8, 'SimilarityNet', horizontalalignment='center', transform=axs[j][0].transAxes)

                try:
                    distance_correlation = np.load(os.path.join(results_path, 'distance_correlation.npy'))
                    # plt.imshow(distance_correlation)
                    # plt.show()

                    with open(os.path.join(results_path, 'mds_correlation.pkl'), 'rb') as f:
                        mds_correlation = pickle.load(f)
                    mds_corr_min = min([m[:, 0].min() for m in mds_correlation])
                    mds_corr_max = max([m[:, 0].max() for m in mds_correlation])
                    d = (mds_corr_max - mds_corr_min) * 0.05
                    mds_corr_min -= d
                    mds_corr_max += d

                    mds_corr = mds_correlation[j][:, 0]
                    mds_corr2 = mds_correlation[j][:, 0]
                    if args.two_dimensional:
                        axs[j][2].plot(mds_corr, mds_corr2, label=run_name, linewidth=.3)
                    else:
                        axs[j][2].plot(mds_corr, label=run_name, linewidth=.3)
                    axs[j][2].get_yaxis().set_ticks([])
                    axs[j][2].set_ylim([mds_corr_min, mds_corr_max])
                    if j == 0:
                        axs[j][2].set_title('MDS (corr)')
                        # axs[j][2].text(.5, .8, 'MDS (corr)', horizontalalignment='center', transform=axs[j][2].transAxes)
                except Exception as e:
                    print(e)
                    pass
                try:
                    with open(os.path.join(results_path, 'mds_field_similarity.pkl'), 'rb') as f:
                        mds_field_similarity = pickle.load(f)
                    mds_eigen_field_similarity = np.load(os.path.join(results_path, 'mds_eigen_field_similarity.npy'))

                    mds_field = mds_field_similarity[j][:, 0]
                    mds_field2 = mds_field_similarity[j][:, 1]
                    mds_field_min = min([m[:, 0].min() for m in mds_field_similarity])
                    mds_field_max = max([m[:, 0].max() for m in mds_field_similarity])
                    d = (mds_field_max - mds_field_min) * 0.05
                    mds_field_min -= d
                    mds_field_max += d
                    if args.two_dimensional:
                        axs[j][1].plot(mds_field, mds_field2, label=run_name, linewidth=.3)
                    else:
                        axs[j][1].plot(mds_field, label=run_name, linewidth=.3)
                    axs[j][1].get_yaxis().set_ticks([])
                    axs[j][1].set_ylim([mds_field_min, mds_field_max])
                    if j == 0:
                        axs[j][1].set_title('MDS (field)')
                        # axs[j][1].text(.5, .8, 'MDS (field)', horizontalalignment='center', transform=axs[j][1].transAxes)

                except Exception as e:
                    print(e)
                    pass

                # if len(all_latents) < 10:
                #     axs[args.legend_position].legend()

            plt.tight_layout(pad=0)
            plt.savefig(os.path.join('tmp/', args.dataset + '.png'))
            plt.show()
