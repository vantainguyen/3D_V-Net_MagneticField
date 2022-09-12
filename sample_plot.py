from utils import sample_plot
import os
import pickle

os.chdir('Sample4Plot') # To do: Replace with working directory containing Sample4Plot folder

with open('sample_train', 'rb') as filename:
  sample_train = pickle.load(filename)

with open('label_train', 'rb') as filename:
  label_train = pickle.load(filename)


sample_plot(sample_train, label_train)