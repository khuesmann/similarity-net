import os

os.system('python train.py --name SimilarityNet --train_data_path data/phantom/samples/train')
os.system('python results.py -m SimilarityNet -ds synthetic')
os.system('python plot.py -m SimilarityNet -ds synthetic')
os.system('python plot2.py -m SimilarityNet -ds synthetic')
