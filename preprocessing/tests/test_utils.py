import unittest
import pandas as pd
from unittest.mock import MagicMock
import numpy as np
from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value=100)
        base._get_num_train_samples = MagicMock(return_value=80)
        expected_train_batches = 4
        self.assertEqual(base._get_num_train_batches(), expected_train_batches)

    def test__get_num_test_batches(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value=100)
        base._get_num_train_samples = MagicMock(return_value=20)
        expected_test_batches = 1
        self.assertEqual(base._get_num_train_batches(), expected_test_batches)

    def test_get_index_to_label_map(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['label_0', 'label_1'])
        expected_map = {0: 'label_0', 1: 'label_1'}
        self.assertEqual(base.get_index_to_label_map(), expected_map)

    def test_index_to_label_and_label_to_index_are_identity(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['label_0', 'label_1'])
        expected_map = {'label_0': 0, 'label_1': 1}
        expected_map_2 = {0: 'label_0', 1: 'label_1'}
        self.assertEqual(base.get_label_to_index_map(), expected_map)
        self.assertEqual(base.get_index_to_label_map(), expected_map_2)

    def test_to_indexes(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['label_0', 'label_1'])
        indexes_list = [0,1]
        self.assertEqual(base.to_indexes(['label_0', 'label_1']), indexes_list)


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples_is_correct(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6', 'id_7', 'id_8', 'id_9', 'id_10'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b'],
            'tag_id': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'tag_position': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6', 'title_7', 'title_8', 'title_9',
                      'title_10']
        }))
        dataset = utils.LocalTextCategorizationDataset("fake_path", 1, min_samples_per_label=1)
        dataset._get_num_train_samples = MagicMock(return_value=8)
        dataset._get_num_test_samples = MagicMock(return_value=2)

        # The total number of samples should be the sum of train and test samples
        total_samples = dataset._get_num_train_samples() + dataset._get_num_test_samples()
        self.assertEqual(dataset._get_num_samples(), total_samples)

    def test_get_train_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6', 'id_7', 'id_8', 'id_9', 'id_10'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b'],
            'tag_id': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'tag_position': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6', 'title_7', 'title_8', 'title_9',
                      'title_10']
        }))
        dataset = utils.LocalTextCategorizationDataset("fake_path", 2, min_samples_per_label=1)
        x_batch, y_batch = dataset.get_train_batch()
        self.assertEqual(x_batch.shape, (2,))
        self.assertEqual(y_batch.shape, (2, 2))

    def test_get_test_batch_returns_expected_shape(self):
        # TODO: CODE HERE*
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6', 'id_7', 'id_8', 'id_9', 'id_10'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b'],
            'tag_id': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'tag_position': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6', 'title_7', 'title_8', 'title_9',
                      'title_10']
        }))
        dataset = utils.LocalTextCategorizationDataset("fake_path", 2, min_samples_per_label=1)
        x_batch, y_batch = dataset.get_test_batch()
        self.assertEqual(x_batch.shape, (2,))
        self.assertEqual(y_batch.shape, (2, 2))

    def test_get_train_batch_raises_assertion_error(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6', 'id_7', 'id_8', 'id_9', 'id_10'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b'],
            'tag_id': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'tag_position': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6', 'title_7', 'title_8', 'title_9',
                      'title_10']
        }))

        with self.assertRaises(AssertionError):
            dataset = utils.LocalTextCategorizationDataset("fake_path", batch_size=40, min_samples_per_label=1)



