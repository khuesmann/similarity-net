# SimilarityNet: A Deep Neural Network for Similarity Analysis Within Spatio-temporal Ensembles

## 0. Requirements
* tensorflow==2.6
* tqdm
* matplotlib
* numpy
* netCDF4 (optional)
* joblib (for MDS calculation)

## 1. Generate training data
Open and run dataset_generator.ipynb to generate the phantom training dataset.

## 2. Generate Synthetic Dataset
Open and run syntethic_dataset.ipynb to generate the synthetic dataset. This will save all runs to `./data/synthetic/runs/*`.
After creating the runs, we perform the Monte-carlo sampling by calling

```shell
python field_similarity.py --ensemble_path ./data/synthetic/runs/ --ensemble_name synthetic --file_type npy --test_split 0.
```
This script can also be called for other file types like "nc", "npy". See the `field_similarity.py > create_run_samples()` how to use your own data on this script.

The samples which were used for the paper's results can be downloaded HERE .

## 3. Train Similaritynet
In order to train similarity net, run
```shell
python train.py --name SimilarityNet --train_data_path data/phantom/samples/train
```
Further training attributes can be seen in train.py, but the current default setup was used in the paper. 

## 4. Calculating and plotting of results
In order to calculate the latent features of an ensemble, run 
```shell
python results.py -m SimilarityNet -ds synthetic
```
where synthetic is the name of the dataset which can be found in `./data/`. Note, that every input path to this script expects the folders to be structured like the phantom or synthetic dataset.

In order to calculate not only the latent feature representation but also the MDS projections for Field Similarity and/or Correlation Similarity, add the parameters `--calculate_mds_field_similarity True --calculate_mds_correlation True` accordingly.

## 5. Plotting
Run `python plot.py -m SimilarityNet -ds synthetic` or `python plot2.py -m SimilarityNet -ds synthetic` to plot the previously calculated results.

## 6. Reconstructing the presented results
In order to recreate the presented results, you can download the obtained Monte-Carlo Samples of the ensembles here (730 Mb): 
``` 
https://drive.google.com/file/d/1V_NwK3xGGRMmxdvcyflxhl3lupv4rEkH/view?usp=sharing
```
Please handle the folder structure like it is handled in the synthetic dataset. In order to create and plot the results, refer to Section 4. and 5. of the documentation, just changing the attribute `-ds` / `--dataset` to the desired ensemble name.

## 7. When using SimilarityNet, please cite the original authors
Exact citation will be added soon.

## 8. Please note
This is the original paper code. A better engineered version is currently WIP and can be found in the respective branch.




