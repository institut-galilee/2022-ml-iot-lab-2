import os
# import subprocess
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from cycler import cycler

from config import Configuration as config


class DataReader(object):
    """
    Example:
    ```python
    train = DataReader(what='train')
    validation = DataReader(what='validation')
    test = DataReader(what='test')

    print('Shape of train.X[''Torso''][''Acc_x'']')
    print(train.X['Torso']['Acc_x'].shape)
    print('Shape of train.y')
    print(train.y.shape)

    print('Shape of validation.X[''Torso''][''Acc_x'']')
    print(validation.X['Torso']['Acc_x'].shape)
    print('Shape of validation.y')
    print(validation.y.shape)

    print('Shape of test.X[''Torso''][''Acc_x'']')
    print(test.X['Torso']['Acc_x'].shape)
    # no labels for the test set!
    ```
    """
    def __init__(self, what='train', train_frames=196072):
        self.what = what
        self.train_frames = train_frames

        # before starting anything, check if the right folder where we will
        # store data exists, otherwise create it
        # e.g. generated/0.1/train/ or generation/0.1/validation
        path = os.path.join(
            config.experimentsfolder,
            what)
        if not os.path.exists(path):
            os.makedirs(path)

        if what == 'train' or what == 'validation':
            self._data = self._load_data(what)
            self._labels = self._load_labels(what)
        else:
            self._data = self._load_data(what)

    @property
    def X(self):
        return self._data

    @property
    def y(self):
        return self._labels

    # channels corresponding to the columns of <position>_motion.txt files
    # ordered according to the SHL dataset documentation.
    channels = {
        # [...]
        2: 'Acc_x',
        3: 'Acc_y',
        4: 'Acc_z',
        5: 'Gyr_x',
        6: 'Gyr_y',
        7: 'Gyr_z',
        8: 'Mag_x',
        9: 'Mag_y',
        10: 'Mag_z',
        # 11: 'Ori_w',
        # 12: 'Ori_x',
        # 13: 'Ori_y',
        # 14: 'Ori_z',
        # 15: 'Gra_x',
        # 16: 'Gra_y',
        # 17: 'Gra_z',
        # 18: 'LAcc_x',
        # 19: 'LAcc_y',
        # 20: 'LAcc_z',
        # 21: 'Pressure'
        # [...]
    }

    modalities = [
        'Acc',
        'Gyr',
        'Mag',
        # 'LAc',
        # 'Gra',
        # 'Ori',
        # 'Pre'
    ]

    def channel_to_modality(channel):
        return channel[:3]  # haha

    coarselabel_map = {
        # 0: 'null',
        1: 'still',
        2: 'walk',
        3: 'run',
        4: 'bike',
        5: 'car',
        6: 'bus',
        7: 'train',
        8: 'subway',
    }

    # finelabel_map = {
    #     1: 'till;Stand;Outside',
    #     2: 'Still;Stand;Inside',
    #     3: 'Still;Sit;Outside',
    #     4: 'Still;Sit;Inside',
    #     5: 'Walking;Outside',
    #     6: 'Walking;Inside',
    #     7: 'Run',
    #     8: 'Bike',
    #     9: 'Car;Driver',
    #     10: 'Car;Passenger',
    #     11: 'Bus;Stand',
    #     12: 'Bus;Sit',
    #     13: 'Bus;Up;Stand',
    #     14: 'Bus;Up;Sit',
    #     15: 'Train;Stand',
    #     16: 'Train;Sit',
    #     17: 'Subway;Stand',
    #     18: 'Subway;Sit',
    # }

    smartphone_positions = [
        'Torso',
        # 'Hips',
        # 'Bag',
        # 'Hand'
    ]

    # files = [
    #     'Acc_x.txt', 'Acc_y.txt', 'Acc_z.txt',
    #     'Gyr_x.txt', 'Gyr_y.txt', 'Gyr_z.txt',
    #     'Mag_x.txt', 'Mag_y.txt', 'Mag_z.txt',
    #     'LAcc_x.txt', 'LAcc_y.txt', 'LAcc_z.txt',
    #     'Gra_x.txt', 'Gra_y.txt', 'Gra_z.txt',
    #     'Ori_x.txt', 'Ori_y.txt', 'Ori_z.txt', 'Ori_w.txt',
    #     'Pressure.txt'
    # ]

    # trainfiles = {
    #     'User1': ['220617', '260617', '270617'],
    #     'User2': ['140617', '140717', '180717'],
    #     'User3': ['030717', '070717', '140617'],
    # }

    # testfiles = {
    # }

    num_channels = len(channels)  # 20
    num_modalities = len(modalities)  # 7
    num_coarselabels = len(coarselabel_map)
    # num_finelabels = len(finelabel_map)

    samples = 500
    # train_frames = 196072
    validation_frames = 28789
    test_frames = 57573

    def _load_data(self, what='train'):
        """
        Synopsis

         Returns
        """
        data = {}

        # test data
        if what == 'test':
            for _, channel in self.channels.items():
                # e.g. data/Gra_x.txt
                src = os.path.join(
                    config.datafolder,
                    channel + '.txt')

                key = \
                    what + '_' +\
                    channel

                dest = os.path.join(
                    config.experimentsfolder, key + '.mmap')

                data[channel] = self._mmap_file(
                    src,
                    dest,
                    dtype=np.double,
                    shape=(self.test_frames, self.samples))

            return data

        # common for train and validation
        for position in self.smartphone_positions:
            data[position] = {}
            for _, channel in self.channels.items():
                # e.g. data/train/Torso/Gra_x.txt
                src = os.path.join(
                    config.datafolder,
                    what,
                    position,
                    channel + '.txt')

                key = \
                    what + '_' +\
                    position + '_' +\
                    channel

                dest = os.path.join(
                    config.experimentsfolder,
                    what,
                    key + '.mmap')

                data[position][channel] = self._mmap_file(
                    src,
                    dest,
                    dtype=np.double,
                    shape=(self.train_frames if what == 'train' else self.validation_frames,
                           self.samples))

        return data

    def _load_labels(self, what='train', position='Torso'):
        """
        Synopsis

         Returns

        NB. Each sub-directory (Bag, Hips, Torso, Hand) has a Label.txt file,
        however, they are the same. Indeed, the data collection from the
        different positions is synchronized. Therefore, this function loads only
        one Label.txt among those available at each position.
        By default, this function loads the training Label.txt contained in Torso
        sub-folder (i.e., '/train/Torso/Label.txt') as well as the validation
        Label.txt contained, also, in Torso sub-folder.
        """

        filename = 'Label.txt'

        src = os.path.join(
            config.datafolder,
            what,
            position,
            filename)

        dest = os.path.join(
            config.experimentsfolder,
            what,
            what + '_Label.mmap')

        labels = self._mmap_file(
            src,
            dest,
            dtype=np.integer,
            shape=(self.train_frames if what == 'train' else self.validation_frames,
                   self.samples))

        return labels

    def _mmap_file(self, src, dest, dtype, shape):
        if os.path.exists(dest):
            # just load mmap file contents
            print('%s exists, loading ...' % dest)
            mmap = np.memmap(
                dest,
                mode='r+',  # originally, mode was set to r+, i.e. Open existing file for reading and writing.
                dtype=dtype,
                shape=shape)

            return mmap
        else:
            # build mmap file from scratch
            print('Building from scratch %s ...' % dest)
            print(shape)
            mmap = np.memmap(
                dest,
                mode='w+',
                dtype=dtype,
                shape=shape)

            chunksize = 5000
            offset = 0
            for chunk in pd.read_csv(src, delimiter=' ', chunksize=chunksize, header=None):
                mmap[offset:offset+chunk.shape[0]] = chunk.values
                offset += chunk.shape[0]

            return mmap

    def CHECK_transition_frames(self):
        """
        This function checks for frames that contain a transition between two activities.

        Returns
         a list containing the index of the frames that contain a transition between two activities
        """
        tr_frames = []
        for i, frame in enumerate(self.y):
            if not np.all(frame == frame[0]):
                tr_frames.append(frame)

        print('there are ', len(tr_frames), ' frames containing a transition')
        return tr_frames

    def CHECK_nans(self):
        # checking for nan's
        nans = {}
        for position in self.smartphone_positions:
            for _, channel in self.channels.items():
                for i, a in enumerate(self.X[position][channel]):
                    if np.isnan(a).any():
                        if position not in nans:
                            nans[position] = {}
                        if channel not in nans[position]:
                            nans[position][channel] = []
                        nans[position][channel].append(i)

        print('there are ', len(nans), ' frames containing NaNs')
        return nans

    def replace_nans(self, index=0):
        """
        First, checks if the frame of index `index` actually contains NaNs;
        Second, infere an interpolation function on the values from 0 to 450;
        Lastly, replaces the NaNs using the infered interpolation function;

        NB. this function supposes that the index of the frame that contains NaNs
        is known beforehand by the user.
        """
        from scipy.interpolate import interp1d

        for position in self.smartphone_positions:
            for _, channel in self.channels.items():
                a = self.X[position][channel][index]
                if np.isnan(a).any():
                    print('Imputing NaN in frame ', index,\
                          ' of channel ', channel,\
                          ' located on ', position)
                    interp = interp1d(range(450), a[:450], fill_value='extrapolate')

                    for j in np.where(np.isnan(a))[0]:
                        print('j ', j, ' interp ', interp(j))
                        self.X[position][channel][index, j] = interp(j)
                else:
                    print('No missing value was found. No imputation performed')

    def plot(self, sample_idx, modality, position, save=False):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(self.X[position][modality][sample_idx], 'b-')
        title = 'sample_idx_'\
            +str(sample_idx)+'_'\
            +modality+'_'\
            +position+'__'\
            +self.coarselabel_map[self.y[sample_idx][0]]
        ax.set_title(title)
        if save:
            fig.savefig(title)
        return ax

    def plot_activites(self, portion=None, save=False):
        """
        plot transitions between activities over time
        """
        y = self.y[:, 0]  # FIXME: we are considering the label of the first sample of a given example

        start = 0
        end = 0
        for portion in np.array_split(y, 10):
            print('portion.shape ', portion.shape)
            index = np.arange(portion.shape[0])
            print('index.shape ', index.shape)
            fulfill = np.zeros_like(index) + 1
            start = end + 1
            end += portion.shape[0]

            n = 8  # number of classes
            # color = plt.cm.YlGnBu(np.linspace(0, 1,n))
            color = plt.cm.Paired(np.linspace(0, 1,n))
            plt.rcParams['axes.prop_cycle'] = cycler('color', color)

            fig, ax = plt.subplots(figsize=(15,1))
            for i in range(1, n+1):
                ax.bar(index[portion==i],
                       fulfill[portion==i],
                       label=self.coarselabel_map[i])

            print('start ', str(start))
            print('end ', str(end))

            #fig.get_axes()[0].set_xlabel('Recognition performances')
            plt.yticks([])
            plt.xticks(rotation=70)
            ax.set_xticks([i for i in np.arange(1, index.shape[0], 500)])
            ax.set_xticklabels([i for i in np.arange(start, end, 500)])
            ax.set_xlabel('example index')
            # plt.legend(['still', 'walk', 'run', 'bike', 'car', 'subway', 'train', 'bus'])
            fig.suptitle('Transitions between activities --- examples from '\
                         + str(start) +' to '+ str(end))
            fig.savefig('misc/'+self.what+'_transitions_between_activities_'\
                        +str(start)+'_'+str(end)+'.svg',
                        format='svg',
                        bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    train = DataReader(what='train')
    validation = DataReader(what='validation')
    test = DataReader(what='test')

    print('Shape of train.X[''Torso''][''Acc_x'']')
    print(train.X['Torso']['Acc_x'].shape)
    print('Shape of train.y')
    print(train.y.shape)
    print('unique labels in train: ', np.unique(train.y))
    train.CHECK_transition_frames()
    train.CHECK_nans()
    # train.replace_nans(index=121217)

    print('Shape of validation.X[''Torso''][''Acc_x'']')
    print(validation.X['Torso']['Acc_x'].shape)
    print('Shape of validation.y')
    print(validation.y.shape)
    print('unique labels in validation: ', np.unique(validation.y))
    validation.CHECK_transition_frames()
    validation.CHECK_nans()

    print('Shape of test.X[''Torso''][''Acc_x'']')
    print(test.X['Acc_x'].shape)
    validation.CHECK_nans()
    # no labels for the test set!
