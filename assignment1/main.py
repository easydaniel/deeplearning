import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pickle
import os

size = 28
depth = 255.0

train_folder = [f'notMNIST_large/{path}' for path in os.listdir('notMNIST_large') if os.path.isdir(f'notMNIST_large/{path}')]
test_folder = [f'notMNIST_small/{path}' for path in os.listdir('notMNIST_small') if os.path.isdir(f'notMNIST_small/{path}') ]

def load_letter(folder):
    imgs = os.listdir(folder)
    dataset = np.ndarray(shape=(len(imgs), size, size), dtype=np.float32)
    counter = 0
    print(folder)
    for img in imgs:
        img_file = os.path.join(folder, img)
        try:
            img_data = (ndimage.imread(img_file).astype(float) - depth / 2) / depth
            if img_data.shape != (size, size):
                raise Exception('Unexpected size of image')
            dataset[counter, : , :] = img_data
            counter += 1
        except IOError as e:
            print(f'Error reading {img_file} - {e}')
    dataset = dataset[0:counter, :, :]
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

def maybe_pickle(folders):
    dataset_name = []
    for folder in folders:
        filename = f'{folder}.pickle'
        dataset_name.append(filename)
        if os.path.exists(filename):
            print(f'{filename} already present - Skipping pickling.')
        else:
            print(f'Pickling {filename}')
            dataset = load_letter(folder)
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f'Unable to save data to {filename}: {e}')
    return dataset_name

def make_arrays(rows):
    if rows:
        dataset = np.ndarray(shape=(rows, size, size), dtype=np.float32)
        labels = np.ndarray(rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
    classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size)
    train_dataset, train_labels = make_arrays(train_size)
    vsize_per_class = valid_size // classes
    tsize_per_class = train_size // classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class

    for label, pfile in enumerate(pickle_files):
        try:
            with open(pfile, 'rb') as f:
                letter_set = pickle.load(f)
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

train_datasets = maybe_pickle(train_folder)
test_datasets = maybe_pickle(test_folder)

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)
