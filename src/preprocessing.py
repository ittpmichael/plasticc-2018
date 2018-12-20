import pandas as pd
import numpy as np

from constants import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class PreprocessTrainingDataset:
    """"Preprocess training set and training metadata set"""

    def __init__(self, training_file, training_metadata_file, balance_class=False):
        self.training_data = pd.read_csv(training_file)
        self.training_metadata = pd.read_csv(training_metadata_file)
        if balance_class:
            self.training_data_b, self.training_metadata_b = self.balance_class()
        self.training_metadata_stats = self.training_metadata.describe()

    def balance_class(self):
        training_data, training_metadata = self.training_data, self.training_metadata
        np.random.seed(129)
        number_each_class = training_metadata['target'].value_counts()
        # highest number of object is in class 90: 2313
        max_len_class_id = training_metadata['target'].value_counts().max()
        for class_id in number_each_class.index:
            # todo: complete this method
            len_class_id = number_each_class[class_id]
            max_object_id = training_metadata['object_id'].max()
            object_each_class = training_metadata['object_id'][
                training_metadata['target'] == class_id].as_matrix()
            training_data_class = training_data[training_data['object_id'].isin(object_each_class)]
            training_metadata_class = training_metadata[training_metadata['object_id'].isin(object_each_class)]
            if class_id == 90:
                continue
            else:
                for i in range(max_len_class_id//len_class_id):
                    training_data_class['object_id'] += max_object_id
                    training_metadata_class['object_id'] += max_object_id
                    training_data = training_data.append(training_data_class)
                    training_metadata = training_metadata.append(training_metadata_class)

        training_data['flux'] += training_data['flux_err']*np.random.uniform(-1.0,1.0)
        training_data, training_metadata = training_data.reset_index(drop=True), training_metadata.reset_index(drop=True)
        training_metadata['object_id2'] = training_metadata.index
        for idx in training_metadata['object_id']:
            training_data['object_id'][training_data['object_id']==idx] = training_metadata['object_id2'][training_metadata['object_id']==idx].values[0]
        return training_data, training_metadata

    def normalized_dataset(self, training_data, training_metadata):
        """as the most stable Sisyphus ever till the last 3 days, staying on my comfortable 5th place. Then I saw the post from CPMP about their single model scoring 0.750 and I gave up because my blend was b

        :return: training_data_normed
        :return: training_metadata_normed
        """
        # create multi-index array
        object_set = training_data.object_id.unique()
        index1 = np.repeat(object_set, 2 * 6)
        index2 = np.tile(np.repeat(list(passbands_dict.keys()), 2), len(object_set))
        index3 = np.tile(['mjd', 'flux'], len(object_set) * 6)
        arrays = [index1, index2, index3]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['object_id', 'passbands', 'col'])

        training_data_normed = pd.DataFrame(np.random.randn(len(object_set) * 2 * 6, 72), index=index)
        # define the max length
        max_passband_len = 72

        mjd_mean, flux_mean, flux_std = np.zeros(len(object_set)), np.zeros(len(object_set)), np.zeros(len(object_set))

        for idx, object_id in enumerate(object_set):
            lightcurve = training_data[training_data['object_id'] == object_id]
            lightcurve['mjd'], mjd_mean_id = self.normalize_col(lightcurve['mjd'])
            lightcurve['flux'], flux_mean_id, flux_std_id = self.normalize_col(lightcurve['flux'], feature_scale=False)

            mjd_mean[idx], flux_mean[idx], flux_std[idx] = mjd_mean_id, flux_mean_id, flux_std_id

            for key in passbands_dict.keys():
                # put ['mjd','flux'] from one passband into numpy array
                each_passband = np.array(lightcurve[['mjd', 'flux']][lightcurve['passband'] == key].as_matrix())

                # get the array length
                passband_len = len(each_passband)

                # because torch.Tensor and np.array accept equal-size array, if the length less than
                # maximum length, fill it up with zero flux data
                if passband_len < max_passband_len:
                    # get delta length
                    passband_len_dif = max_passband_len - passband_len
                    # create new data points to added -> [[max_mjd+1,0],[max_mjd+2,0],...]
                    new_points = np.transpose([np.arange(passband_len_dif) + each_passband[-1, 0],
                                               np.repeat(0, passband_len_dif)])

                    each_passband = np.append(
                        each_passband,
                        new_points,
                        axis=0
                    )
                training_data_normed.loc[(object_id, key, 'mjd')], training_data_normed.loc[
                    (object_id, key, 'flux')] = each_passband[:, 0], each_passband[:, 1]

        training_metadata = training_metadata.assign(mjd_mean=mjd_mean, flux_mean=flux_mean, flux_std=flux_std)
        self.training_metadata_stats = training_metadata.describe()
        training_metadata_normed = self.normalize_metadata(training_metadata)

        return training_data_normed, training_metadata_normed

    def normalize_metadata(self, training_metadata):
        training_metadata['ra']= self.normalize_col_name(training_metadata, 'ra')
        training_metadata['decl'] = self.normalize_col_name(training_metadata, 'decl')
        training_metadata['hostgal_specz'] = self.normalize_col_name(training_metadata, 'hostgal_specz')
        training_metadata['distmod'] = training_metadata['distmod'].fillna(0)
        training_metadata['distmod'] = self.normalize_col_name(training_metadata, 'distmod')
        training_metadata['mwebv'] = self.normalize_col_name(training_metadata, 'mwebv')
        training_metadata['mjd_mean'] = self.normalize_col_name(training_metadata, 'mjd_mean')
        training_metadata['flux_mean'] = self.normalize_col_name(training_metadata, 'flux_mean', feature_scale=False)
        training_metadata['flux_std'] = self.normalize_col_name(training_metadata, 'flux_std', feature_scale=False)

        return training_metadata

    def normalize_col_name(self, training_metadata, col_name, feature_scale=True):
        """
        Normalize a dataframe column into [0,1]
        :param col_name: column name from self.training_metadata
        :type col_name: str
        :param feature_scale: apply feature scaling to the data, otherwise apply standardization,
                              Z-score normalization, to the data
        :type feature_scale: bool
        """
        if feature_scale:
            # return 2 values: a normalized column and mean
            return (training_metadata[col_name] - self.training_metadata_stats[col_name]['min']) / \
                   (self.training_metadata_stats[col_name]['max'] - self.training_metadata_stats[col_name]['min'])
        else:
            # return 3 values: a normalized column, mean and std
            return (training_metadata[col_name] - self.training_metadata_stats[col_name]['mean']) / \
                   self.training_metadata_stats[col_name]['std']

    def normalize_col(self, column, feature_scale=True):
        """
        Normalize a dataframe column into [0,1]
        :param column: column data from pandas DataFrame
        :type column: object
        :param feature_scale: apply feature scaling to the data, otherwise apply standardization,
                              Z-score normalization, to the data
        :type feature_scale: bool
        """
        if feature_scale:
            # return 2 values: a normalized column and mean
            return (column - column.min()) / (column.max() - column.min()), column.mean()
        else:
            # return 3 values: a normalized column, mean and std
            return (column - column.mean()) / (column.std()), column.mean(), column.std()


def main():
    dataset = PreprocessTrainingDataset(training_file=DATA_TRAINING_SET_CSV,
                                        training_metadata_file=DATA_TRAINING_SET_METADATA_CSV,
                                        balance_class=True)

    #training_data_b, training_metadata_b = dataset.training_data_b, dataset.training_metadata_b

    training_data_b_normed, training_metadata_b_normed = dataset.normalized_dataset(dataset.training_data_b,
                                                                                    dataset.training_metadata_b)

    training_data_normed, training_metadata_normed = dataset.normalized_dataset(dataset.training_data,
                                                                               dataset.training_metadata)

    training_data_b_normed.to_csv(DATA_TRAINING_SET_B_NORMED_CSV)
    training_metadata_b_normed.to_csv(DATA_TRAINING_SET_METADATA_B_NORMED_CSV)

    training_data_normed.to_csv(DATA_TRAINING_SET_NORMED_CSV)
    training_metadata_normed.to_csv(DATA_TRAINING_SET_METADATA_NORMED_CSV)

    #return training_data_b_normed, training_metadata_b_normed, training_data_normed, training_metadata_normed

if __name__ == '__main__':
    #training_data_b_normed, training_metadata_b_normed, training_data_normed, training_metadata_normed = main()
    main()
