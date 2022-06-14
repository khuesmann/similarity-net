import argparse
import os
import pickle

import numpy as np

from mds import calculate_distance_matrix, calculate_mds
from model.similarity_net import SimilarityNet
from utils import load_runs, batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create results")
    # model attributes
    parser.add_argument("-m", "--model", required=True, type=str)
    parser.add_argument("-ds", "--dataset", required=True, type=str)
    parser.add_argument("-mr", "--max_runs", default=10, type=int)
    parser.add_argument("-fs", "--calculate_mds_field_similarity", default=False, type=int)
    parser.add_argument("-cs", "--calculate_mds_correlation", default=False, type=int)

    args = parser.parse_args()

    result_path = os.path.join('results', f'{args.model}-{args.dataset}')
    os.makedirs(result_path, exist_ok=True)

    with open(f'checkpoint/{args.model}/args.pkl', 'rb') as fiModel:
        persisted_args = pickle.load(fiModel)

    model = SimilarityNet.make(persisted_args)
    model.checkpoint_manager(f'checkpoint/{args.model}/model.ckpt')

    runs, run_files, max_timesteps, durations = load_runs(f'data/{args.dataset}/samples/train', max_runs=args.max_runs)

    # distance_methods = ['correlation', 'field_similarity']
    distance_methods = []
    if args.calculate_mds_field_similarity:
        distance_methods.append('field_similarity')
    if args.calculate_mds_correlation:
        distance_methods.append('correlations')

    for distance_method in distance_methods:
        d_path = os.path.join(result_path, f'distance_{distance_method}.npy')
        if not os.path.exists(d_path):
            print('Calculate distance matrix ' + d_path)
            distance_matrix = calculate_distance_matrix(runs, distance_method, threads=5)
            np.save(d_path, distance_matrix)
        else:
            distance_matrix = np.load(d_path)
        mds_path = os.path.join(result_path, f'mds_{distance_method}.pkl')
        mds_eigen_path = os.path.join(result_path, f'mds_eigen_{distance_method}.npy')
        if not os.path.exists(mds_path):
            X, eigen = calculate_mds(distance_matrix, durations)
            np.save(mds_eigen_path, eigen)
            with open(mds_path, 'wb') as handle:
                pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)

    latent_save_path = os.path.join(result_path, 'all_latents.pkl')
    attention_save_path = os.path.join(result_path, 'all_attentions.pkl')
    batches = batch(runs, 1)
    all_latents = {}
    all_decoder_attentions = {}
    latent_dim = None
    for i, vr in enumerate(batches):
        mask = []
        for m in np.ma.make_mask(vr + 1)[0]:
            mask.append(m[0])
        mask = np.expand_dims(mask, 0)
        predictions, latents, decoder_attentions = model(vr, mask, training=False)
        all_latents[run_files[i]] = latents[0].numpy()[0, :durations[i], :]
        all_decoder_attentions[run_files[i]] = decoder_attentions[0].numpy()[0, :durations[i], :]
        if latent_dim is None:
            latent_dim = all_latents[run_files[i]].shape[-1]

    with open(latent_save_path, 'wb') as handle:
        pickle.dump(all_latents, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(attention_save_path, 'wb') as handle:
        pickle.dump(all_decoder_attentions, handle, protocol=pickle.HIGHEST_PROTOCOL)
