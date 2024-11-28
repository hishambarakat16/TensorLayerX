from tests.utils import CustomTestCase
import tensorlayerx as tlx
import unittest
import os
import numpy as np
from unittest.mock import mock_open, patch
from tensorlayerx.utils.lazy_imports import LazyImport
nltk = LazyImport("nltk")


os.environ["TL_BACKEND"] = "jittor"



class TestTextProcessingFunctions(CustomTestCase):

    # def test_as_bytes(self):
    #     # Test for converting text to bytes
    #     text = "Hello"
    #     result = tlx.text.nlp.as_bytes(text)
    #     self.assertIsInstance(result, bytes)

    #     # Test for converting bytes to bytes (no change)
    #     byte_text = b"Hello"
    #     result = tlx.text.nlp.as_bytes(byte_text)
    #     self.assertEqual(result, byte_text)

    #     # Test for raising error if input is neither text nor bytes
    #     with self.assertRaises(TypeError):
    #         tlx.text.nlp.as_bytes(123)

    # def test_as_text(self):
    #     # Test for converting bytes to text
    #     byte_text = b"Hello"
    #     result = tlx.text.nlp.as_text(byte_text)
    #     self.assertIsInstance(result, str)
    #     self.assertEqual(result, "Hello")

    #     # Test for converting text to text (no change)
    #     text = "Hello"
    #     result = tlx.text.nlp.as_text(text)
    #     self.assertEqual(result, text)

    #     # Test for raising error if input is neither text nor bytes
    #     with self.assertRaises(TypeError):
    #         tlx.text.nlp.as_text(123)

    # def test_generate_skip_gram_batch(self):
    #     # Test for generating skip gram batches
    #     data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #     batch_size = 8
    #     num_skips = 2
    #     skip_window = 1
    #     data_index = 0
    #     batch, labels, data_index = tlx.text.nlp.generate_skip_gram_batch(data, batch_size, num_skips, skip_window, data_index)

    #     self.assertEqual(len(batch), batch_size)
    #     self.assertEqual(len(labels), batch_size)
        
    #     # Convert numpy array to list for comparison
    #     self.assertIsInstance(batch.tolist(), list)
    #     self.assertIsInstance(labels.tolist(), list)

    #     # Test for raising exception if batch_size is not divisible by num_skips
    #     with self.assertRaises(Exception):
    #         tlx.text.nlp.generate_skip_gram_batch(data, batch_size=7, num_skips=2, skip_window=1)

    #     # Test for raising exception if num_skips is greater than twice skip_window
    #     with self.assertRaises(Exception):
    #         tlx.text.nlp.generate_skip_gram_batch(data, batch_size=8, num_skips=5, skip_window=2)


    # def test_sample(self):
    #     # Test for sampling with no temperature
    #     a = [0.1, 0.2, 0.7]
    #     result = tlx.text.nlp.sample(a, temperature=None)
    #     self.assertIsInstance(result, np.int64)  # np.argmax returns np.int64
    #     self.assertEqual(result, 2)  # As 0.7 is the highest probability

    #     # Test for sampling with temperature=1.0 (should not change probabilities)
    #     result = tlx.text.nlp.sample(a, temperature=1.0)
    #     self.assertIsInstance(result, np.int64)

    #     # Test for sampling with temperature > 1.0 (distribution becomes flatter)
    #     result = tlx.text.nlp.sample(a, temperature=1.5)
    #     self.assertIsInstance(result, np.int64)

    #     # Test for raising exception if a is None
    #     with self.assertRaises(Exception):
    #         tlx.text.nlp.sample(None, temperature=1.0)


    # def test_sample_top(self):
    #     # Fix the random seed for reproducibility
    #     np.random.seed(42)  # Or any fixed value

    #     # Test for top-k sampling with a valid input
    #     a = np.array([0.1, 0.2, 0.7, 0.0, 0.0])  # Convert a to a NumPy array
    #     result = tlx.text.nlp.sample_top(a, top_k=2)
    #     self.assertTrue(isinstance(result, (int, np.integer)))  # Check if it's an int or numpy integer
    #     self.assertEqual(result, 2)  # Ensure the correct index is returned


    def process_sentence(sentence, start_word="<S>", end_word="</S>"):
        """Seperate a sentence string into a list of string words, add start_word and end_word."""
        if not isinstance(sentence, str):
            raise TypeError("Input must be a string.")

        if start_word is not None:
            process_sentence = [start_word]
        else:
            process_sentence = []
        process_sentence.extend(nltk.tokenize.word_tokenize(sentence.lower()))

        if end_word is not None:
            process_sentence.append(end_word)
        return process_sentence


#     def test_create_vocab(self):
#         """Test for creating vocabulary from sentences"""
#         sentences = [["hello", "world"], ["hello", "there"]]
#         word_counts_output_file = "vocab.txt"

#         # Call the function to create vocab
#         tlx.text.nlp.create_vocab(sentences, word_counts_output_file, min_word_count=1)

#         # Check the contents of the vocab file
#         with open(word_counts_output_file, 'r') as f:
#             lines = f.readlines()
#         self.assertGreater(len(lines), 0)  # Ensure that some words have been written

#         # Clean up file after test
#         os.remove(word_counts_output_file)


#     def test_simple_read_words(self):
#         """Test for reading words from a simple vocabulary file"""
#         # Create a mock simple vocabulary file
#         vocab_file = "simple_vocab.txt"
#         with open(vocab_file, 'w') as f:
#             f.write("hello 0\nworld 1\n")

#         # Call the function to read words
#         result = tlx.text.nlp.simple_read_words(vocab_file)

#         # Adjust the test to check for a single string with newlines
#         self.assertEqual(result, 'hello 0\nworld 1\n')

#         # Clean up file after test
#         os.remove(vocab_file)


#     def test_read_words(self):
#         """Test for reading words from a file"""
#         # Create a mock vocabulary file
#         vocab_file = "mock_vocab.txt"
#         with open(vocab_file, 'w') as f:
#             f.write("hello 2\nworld 3\n")

#         # Call the function to read words
#         result = tlx.text.nlp.read_words(vocab_file)

#         # Check if the result matches expected output (with <eos> tokens included)
#         self.assertEqual(result, ['hello', '2<eos>world', '3<eos>'])

#         # Clean up file after test
#         os.remove(vocab_file)

#     def test_read_analogies_file(self):
#         """Test for reading analogy file and returning the correct ID format."""
#         mock_file_content = b"""Athens Greece Baghdad Iraq
# Athens Greece Bangkok Thailand
# Athens Greece Beijing China
# """

#         word2id = {
#             'athens': 1,
#             'greece': 2,
#             'baghdad': 3,
#             'iraq': 4,
#             'bangkok': 5,
#             'thailand': 6,
#             'beijing': 7,
#             'china': 8
#         }

#         expected_output = np.array([
#             [1, 2, 3, 4],
#             [1, 2, 5, 6],
#             [1, 2, 7, 8]
#         ], dtype=np.int32)

#         # Mocking open() to return string instead of bytes
#         with patch("builtins.open", mock_open(read_data=mock_file_content)):
#             result = tlx.text.nlp.read_analogies_file(eval_file='questions-words.txt', word2id=word2id)

#         np.testing.assert_array_equal(result, expected_output)

#     def test_build_words_dataset(self):
#         # Sample list of words
#         words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
#         vocabulary_size = 5  # Limit the vocabulary to 5 words
#         expected_data = [1, 2, 3, 4, 0, 0, 1, 0, 0]  # Expected word IDs, 'UNK' replaces rare words

#         # Call the function to generate the data
#         data, count, dictionary, reverse_dictionary = tlx.text.nlp.build_words_dataset(words, vocabulary_size, unk_key='UNK')

#         # Test if the generated data matches the expected data
#         self.assertEqual(data, expected_data)

#         # Test that 'UNK' token is at index 0
#         self.assertEqual(dictionary['UNK'], 0)

#         # Test that reverse dictionary maps the correct ID to the word
#         self.assertEqual(reverse_dictionary[0], 'UNK')

#         # Test if the 'UNK' count is correct
#         self.assertEqual(count[0][1], 4)  # 'the' is replaced by 'UNK' four times

#         # Test that the most common words are assigned correct indices
#         self.assertEqual(dictionary['the'], 1)
#         self.assertEqual(dictionary['quick'], 2)
#         self.assertEqual(dictionary['brown'], 3)
#         self.assertEqual(dictionary['fox'], 4)

#         # Test if vocabulary size is respected
#         self.assertEqual(len(dictionary), vocabulary_size)

#         # Test that the reverse dictionary correctly maps back from ID to word
#         self.assertEqual(reverse_dictionary[1], 'the')
#         self.assertEqual(reverse_dictionary[2], 'quick')
#         self.assertEqual(reverse_dictionary[3], 'brown')
#         self.assertEqual(reverse_dictionary[4], 'fox')


#     def test_build_reverse_dictionary(self):
#         # Test for building reverse dictionary
#         word_to_id = {
#             'hello': 0,
#             'world': 1,
#             'there': 2
#         }
#         expected_reverse_dict = {
#             0: 'hello',
#             1: 'world',
#             2: 'there'
#         }
        
#         result = tlx.text.nlp.build_reverse_dictionary(word_to_id)
#         self.assertEqual(result, expected_reverse_dict)


#     def test_build_words_dataset(self):
#         # Sample list of words
#         words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
#         vocabulary_size = 5
#         expected_data = [1, 2, 3, 4, 5, 6, 1, 7, 8]  # Update expected values based on correct dictionary indices

#         data, count, dictionary, reverse_dictionary = tlx.text.nlp.build_words_dataset(words, vocabulary_size, unk_key='UNK')

#         # Test if the generated data matches the expected data
#         self.assertEqual(data, expected_data)

#         # Test that 'UNK' token is at index 0
#         self.assertEqual(dictionary['UNK'], 0)

#         # Test that reverse dictionary maps the correct ID to the word
#         self.assertEqual(reverse_dictionary[0], 'UNK')

#         # Test if the 'UNK' count is correct
#         self.assertEqual(count[0][1], 4)  # 'UNK' should be counted as 4

if __name__ == '__main__':
    unittest.main()
