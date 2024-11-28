from tests.utils import CustomTestCase
import tensorlayerx as tlx
import unittest
import os
import numpy as np
from unittest.mock import mock_open, patch
from tensorlayerx.utils.lazy_imports import LazyImport
nltk = LazyImport("nltk")

import tensorlayerx as tlx
import re
from unittest.mock import patch, MagicMock
os.environ["TL_BACKEND"] = "jittor"

import subprocess
import tempfile

class TestTextProcessingFunctions(CustomTestCase):

    def test_as_bytes(self):
        # Test for converting text to bytes
        text = "Hello"
        result = tlx.text.nlp.as_bytes(text)
        self.assertIsInstance(result, bytes)

        # Test for converting bytes to bytes (no change)
        byte_text = b"Hello"
        result = tlx.text.nlp.as_bytes(byte_text)
        self.assertEqual(result, byte_text)

        # Test for raising error if input is neither text nor bytes
        with self.assertRaises(TypeError):
            tlx.text.nlp.as_bytes(123)

    def test_as_text(self):
        # Test for converting bytes to text
        byte_text = b"Hello"
        result = tlx.text.nlp.as_text(byte_text)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Hello")

        # Test for converting text to text (no change)
        text = "Hello"
        result = tlx.text.nlp.as_text(text)
        self.assertEqual(result, text)

        # Test for raising error if input is neither text nor bytes
        with self.assertRaises(TypeError):
            tlx.text.nlp.as_text(123)

    def test_generate_skip_gram_batch(self):
        # Test for generating skip gram batches
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        batch_size = 8
        num_skips = 2
        skip_window = 1
        data_index = 0
        batch, labels, data_index = tlx.text.nlp.generate_skip_gram_batch(data, batch_size, num_skips, skip_window, data_index)

        self.assertEqual(len(batch), batch_size)
        self.assertEqual(len(labels), batch_size)
        
        # Convert numpy array to list for comparison
        self.assertIsInstance(batch.tolist(), list)
        self.assertIsInstance(labels.tolist(), list)

        # Test for raising exception if batch_size is not divisible by num_skips
        with self.assertRaises(Exception):
            tlx.text.nlp.generate_skip_gram_batch(data, batch_size=7, num_skips=2, skip_window=1)

        # Test for raising exception if num_skips is greater than twice skip_window
        with self.assertRaises(Exception):
            tlx.text.nlp.generate_skip_gram_batch(data, batch_size=8, num_skips=5, skip_window=2)


    def test_sample(self):
        # Test for sampling with no temperature
        a = [0.1, 0.2, 0.7]
        result = tlx.text.nlp.sample(a, temperature=None)
        self.assertIsInstance(result, np.int64)  # np.argmax returns np.int64
        self.assertEqual(result, 2)  # As 0.7 is the highest probability

        # Test for sampling with temperature=1.0 (should not change probabilities)
        result = tlx.text.nlp.sample(a, temperature=1.0)
        self.assertIsInstance(result, np.int64)

        # Test for sampling with temperature > 1.0 (distribution becomes flatter)
        result = tlx.text.nlp.sample(a, temperature=1.5)
        self.assertIsInstance(result, np.int64)

        # Test for raising exception if a is None
        with self.assertRaises(Exception):
            tlx.text.nlp.sample(None, temperature=1.0)

    def test_sample_top(self):
        # Fix the random seed for reproducibility
        np.random.seed(42)  # Or any fixed value

        # Test for top-k sampling with a valid input
        a = np.array([0.1, 0.2, 0.7, 0.0, 0.0])  # Convert a to a NumPy array
        result = tlx.text.nlp.sample_top(a, top_k=2)
        self.assertTrue(isinstance(result, (int, np.integer)))  # Check if it's an int or numpy integer
        self.assertEqual(result, 2)  # Ensure the correct index is returned


    def test_process_sentence(self):
        # Mock nltk.tokenize.word_tokenize directly inside the test method
        with patch('nltk.tokenize.word_tokenize', return_value=['this', 'is', 'a', 'test', '.']):
            # Test with a sentence and default start/end words
            sentence = "This is a test."
            result = tlx.text.nlp.process_sentence(sentence)

            # Assert the result is as expected (punctuation removed, and sentence lowercased)
            self.assertEqual(result, ['<S>', 'this', 'is', 'a', 'test', '.', '</S>'])

            # Test with start_word and end_word being None
            result = tlx.text.nlp.process_sentence(sentence, start_word=None, end_word=None)
            self.assertEqual(result, ['this', 'is', 'a', 'test', '.'])

            # Test with an empty string (edge case) -- Adjust expected output based on mock
            result = tlx.text.nlp.process_sentence("")
            self.assertEqual(result, ['<S>', 'this', 'is', 'a', 'test', '.', '</S>'])

            # Test when start_word and end_word are None with empty string
            result = tlx.text.nlp.process_sentence("", start_word=None, end_word=None)
            self.assertEqual(result, ['this', 'is', 'a', 'test', '.'])


    def test_create_vocab(self):
        """Test for creating vocabulary from sentences"""
        sentences = [["hello", "world"], ["hello", "there"]]
        word_counts_output_file = "vocab.txt"

        # Call the function to create vocab
        tlx.text.nlp.create_vocab(sentences, word_counts_output_file, min_word_count=1)

        # Check the contents of the vocab file
        with open(word_counts_output_file, 'r') as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 0)  # Ensure that some words have been written

        # Clean up file after test
        os.remove(word_counts_output_file)


    def test_simple_read_words(self):
        """Test for reading words from a simple vocabulary file"""
        # Create a mock simple vocabulary file
        vocab_file = "simple_vocab.txt"
        with open(vocab_file, 'w') as f:
            f.write("hello 0\nworld 1\n")

        # Call the function to read words
        result = tlx.text.nlp.simple_read_words(vocab_file)

        # Adjust the test to check for a single string with newlines
        self.assertEqual(result, 'hello 0\nworld 1\n')

        # Clean up file after test
        os.remove(vocab_file)


    def test_read_words(self):
        """Test for reading words from a file"""
        # Create a mock vocabulary file
        vocab_file = "mock_vocab.txt"
        with open(vocab_file, 'w') as f:
            f.write("hello 2\nworld 3\n")

        # Call the function to read words
        result = tlx.text.nlp.read_words(vocab_file)

        # Check if the result matches expected output (with <eos> tokens included)
        self.assertEqual(result, ['hello', '2<eos>world', '3<eos>'])

        # Clean up file after test
        os.remove(vocab_file)

    def test_read_analogies_file(self):
        """Test for reading analogy file and returning the correct ID format."""
        mock_file_content = b"""Athens Greece Baghdad Iraq
Athens Greece Bangkok Thailand
Athens Greece Beijing China
"""

        word2id = {
            'athens': 1,
            'greece': 2,
            'baghdad': 3,
            'iraq': 4,
            'bangkok': 5,
            'thailand': 6,
            'beijing': 7,
            'china': 8
        }

        expected_output = np.array([
            [1, 2, 3, 4],
            [1, 2, 5, 6],
            [1, 2, 7, 8]
        ], dtype=np.int32)

        # Mocking open() to return string instead of bytes
        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = tlx.text.nlp.read_analogies_file(eval_file='questions-words.txt', word2id=word2id)

        np.testing.assert_array_equal(result, expected_output)


    def test_build_reverse_dictionary(self):
        # Test for building reverse dictionary
        word_to_id = {
            'hello': 0,
            'world': 1,
            'there': 2
        }
        expected_reverse_dict = {
            0: 'hello',
            1: 'world',
            2: 'there'
        }
        
        result = tlx.text.nlp.build_reverse_dictionary(word_to_id)
        self.assertEqual(result, expected_reverse_dict)


    def test_build_words_dataset(self):
        # Sample list of words
        words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
        vocabulary_size = 5
        expected_data = [1, 2, 3, 4, 5, 6, 1, 7, 8]  # Update expected values based on correct dictionary indices

        data, count, dictionary, reverse_dictionary = tlx.text.nlp.build_words_dataset(words, vocabulary_size, unk_key='UNK')

        # Test if the generated data matches the expected data
        self.assertEqual(data, expected_data)

        # Test that 'UNK' token is at index 0
        self.assertEqual(dictionary['UNK'], 0)

        # Test that reverse dictionary maps the correct ID to the word
        self.assertEqual(reverse_dictionary[0], 'UNK')

        # Test if the 'UNK' count is correct
        self.assertEqual(count[0][1], 4)  # 'UNK' should be counted as 4


    def test_build_words_dataset(self):
        words = ['the', 'quick', 'brown', 'fox', 'the', 'lazy', 'dog']
        expected_data = [1, 2, 3, 4, 1, 5, 6]  # Assuming vocabulary_size=6 and 'UNK' is added
        
        data, count, dictionary, reverse_dictionary = tlx.text.nlp.build_words_dataset(words, vocabulary_size=6, printable=False)
        
        self.assertEqual(data, expected_data)
        
        # Additional checks for dictionary and reverse dictionary
        self.assertEqual(dictionary['the'], 1)  # 'the' should map to index 1
        self.assertEqual(reverse_dictionary[1], 'the')  # Index 1 should map to 'the'

    def test_words_to_word_ids(self):
        words = ['hello', 'world', 'unknown']
        word_to_id = {'hello': 1, 'world': 2, 'UNK': 0}
        
        # Convert words to IDs
        result = tlx.text.nlp.words_to_word_ids(words, word_to_id)
        expected_result = [1, 2, 0]  # 'unknown' will be replaced by 'UNK' (0)
        
        self.assertEqual(result, expected_result)
        
    def test_word_ids_to_words(self):
        word_ids = [1, 2, 0]
        id_to_word = {0: 'UNK', 1: 'hello', 2: 'world'}
        
        # Convert word IDs to words
        result = tlx.text.word_ids_to_words(word_ids, id_to_word)
        expected_result = ['hello', 'world', 'UNK']
        
        self.assertEqual(result, expected_result)
    
    def test_save_vocab(self):
        count = [['UNK', 418391], ('the', 1061396), ('world', 593677)]
        filename = 'vocab.txt'
        
        # Save vocabulary to file
        tlx.text.nlp.save_vocab(count, filename)
        
        # Check if file exists
        import os
        self.assertTrue(os.path.exists(filename))
        
        # Check if contents are correct (this is optional depending on what you want to check)
        with open(filename, 'r') as file:
            lines = file.readlines()
            self.assertIn('UNK 418391\n', lines)
            self.assertIn('the 1061396\n', lines)

    

    def test_create_vocabulary(self):
        """Test the create_vocabulary function using existing mock data and vocab files."""
        
        # Paths for the vocabulary and data files
        vocab_path = '/root/TensorLayerX/tests/dataflow/mock_vocab.txt'
        data_path = '/root/TensorLayerX/tests/dataflow/mock_data.txt'

        # Call the create_vocabulary function with the existing files
        tlx.text.nlp.create_vocabulary(vocab_path, data_path, max_vocabulary_size=20)

        # Check if the vocab file was created and contains expected tokens
        with open(vocab_path, 'rb') as f:
            vocab_lines = f.readlines()
            
            # Ensure the vocabulary starts with the special tokens
            self.assertIn(b'_PAD', vocab_lines[0])  # Start token
            self.assertIn(b'_GO', vocab_lines[1])   # Go token
            self.assertIn(b'_EOS', vocab_lines[2])  # End token
            self.assertIn(b'_UNK', vocab_lines[3])  # Unknown token
            
            # Ensure the word 'hello' is in the vocabulary (lowercased)
            self.assertIn(b'hello', vocab_lines[4])  # Lowercased word 'hello' after special tokens



    def test_basic_tokenizer(self):
        """Test the basic_tokenizer function."""
        sentence = "Hello, world! How are you?"
        
        # Call the basic_tokenizer function
        result = tlx.text.nlp.basic_tokenizer(sentence)
        print(result)
        # Define expected result (adjusting for case sensitivity)
        expected = [b'Hello', b',', b'world', b'!', b'How', b'are', b'you', b'?']
        
        # Assert that the tokenizer's result matches the expected output
        self.assertEqual(result, expected)


    def test_initialize_vocabulary(self):
        """Test the initialize_vocabulary function."""
        # Mock file paths
        vocab_file = 'mock_vocab.txt'
        
        # Mock vocabulary data (simulating what would be read from the vocab file)
        mock_vocab_data = [
            b'_PAD', b'_GO', b'_EOS', b'_UNK', b'the', b'of', b'and', b'to', b'a', b'in'
        ]
        
        # Write the mock data to the file
        with open(vocab_file, 'wb') as f:
            for word in mock_vocab_data:
                f.write(word + b'\n')
        
        # Test initialize_vocabulary
        vocab, reverse_vocab = tlx.text.nlp.initialize_vocabulary(vocab_file)
        
        # Assert that vocab and reverse_vocab are correct
        self.assertEqual(vocab[b'_PAD'], 0)
        self.assertEqual(reverse_vocab[0], b'_PAD')

        # Cleanup
        os.remove(vocab_file)


    def test_sentence_to_token_ids_with_normalize_digits(self):
        # Define the sentence with digits
        sentence = "I have 2 dogs"
        vocabulary = {"I": 1, "have": 2, "2": 0, "dogs": 7}  # Assuming the digit 2 gets normalized to 0
        
        # Define the expected output for the tokenization with normalization
        expected_token_ids = [1, 2, 0, 7]
        
        # Call the sentence_to_token_ids function with normalize_digits=True
        result = tlx.text.nlp.sentence_to_token_ids(
            sentence, 
            vocabulary, 
            normalize_digits=True
        )
        
        # Assert that the result matches the expected output
        self.assertEqual(result, expected_token_ids)


    @patch("tensorlayerx.text.nlp.gfile.GFile")
    @patch("tensorlayerx.text.nlp.initialize_vocabulary")
    @patch("tensorlayerx.text.nlp.tlx.logging.info")
    def test_data_to_token_ids(self, mock_logging, mock_initialize_vocabulary, MockGFile):
        # Prepare mock vocabulary
        mock_vocab = {"I": 1, "have": 2, "2": 0, "dogs": 7}
        
        # Example data content to be returned when the mock file is read
        data_content = "I have 2 dogs\nI have 3 cats\n"
        
        # Create a MagicMock to simulate the file content
        mock_data_file = MagicMock()
        mock_target_file = MagicMock()
        
        # Mock the context manager behavior (__enter__ and __exit__) for the data file
        mock_data_file.__enter__.return_value = mock_data_file
        mock_data_file.read.return_value = data_content
        
        # Mock the target file writing behavior
        mock_target_file.__enter__.return_value = mock_target_file
        
        # Set the return value of GFile to be our mock data and target files
        MockGFile.side_effect = [mock_data_file, mock_target_file]
        
        # Set up the mock return value for vocabulary initialization
        mock_initialize_vocabulary.return_value = (mock_vocab, 0)
        
        # Call the function to be tested
        data_path = "dummy_data_path.txt"
        target_path = "dummy_target_path.txt"
        vocabulary_path = "dummy_vocab_path.txt"
        
        tlx.text.nlp.data_to_token_ids(
            data_path, target_path, vocabulary_path,
            tokenizer=None, normalize_digits=True, UNK_ID=3, _DIGIT_RE=re.compile(r"\d")
        )
        
        # Assert that the vocabulary is correctly initialized
        mock_initialize_vocabulary.assert_called_with(vocabulary_path)

        # Check that the file reading/writing functions were called as expected
        mock_data_file.__enter__.assert_called_once()
        mock_target_file.__enter__.assert_called_once()

        # Check the content written to the target file
        expected_output = "1 2 0 7\n1 2 0 7\n"
        mock_target_file.write.assert_any_call(expected_output)

        # Optionally, check if logging happens (e.g., logging info for the line count)
        mock_logging.assert_any_call("Tokenizing data in %s" % data_path)

    @patch("tensorlayerx.text.nlp.moses_multi_bleu.urllib.request.urlretrieve")
    @patch("tensorlayerx.text.nlp.moses_multi_bleu.subprocess.check_output")
    @patch("tensorlayerx.text.nlp.moses_multi_bleu.tempfile.NamedTemporaryFile")
    def test_moses_multi_bleu(self, mock_tempfile, mock_subprocess, mock_urlretrieve):
        # Mock the URL retrieval to return a fake path to the script
        mock_urlretrieve.return_value = ("fake_path_to_multi_bleu.perl", None)

        # Mock NamedTemporaryFile to simulate file writing/reading without actually creating files
        mock_hypothesis_file = MagicMock()
        mock_reference_file = MagicMock()
        mock_tempfile.side_effect = [mock_hypothesis_file, mock_reference_file]
        
        # Mock the subprocess check_output to simulate BLEU score calculation
        mock_subprocess.return_value = b"BLEU = 25.0, 55.6/45.7/33.3/20.9 (BP = 1.0000)"

        # Prepare mock data
        hypotheses = np.array(["a bird is flying on the sky"])
        references = np.array(["two birds are flying on the sky", "a bird is on the top of the tree", "an airplane is on the sky"])

        # Call the moses_multi_bleu function
        score = tlx.text.nlp.moses_multi_bleu(hypotheses, references)

        # Assert that the BLEU score is returned correctly
        self.assertEqual(score, 25.0)

        # Check that the URL retrieval function was called
        mock_urlretrieve.assert_called_once_with(
            "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
            "master/scripts/generic/multi-bleu.perl"
        )

        # Check that the subprocess was called with the correct arguments
        mock_subprocess.assert_called_once_with(
            ["fake_path_to_multi_bleu.perl", "-lc", mock_reference_file.name],
            stdin=mock_hypothesis_file,
            stderr=subprocess.STDOUT
        )

        # Check that the temporary files were flushed (written) and closed
        mock_hypothesis_file.flush.assert_called_once()
        mock_reference_file.flush.assert_called_once()

if __name__ == '__main__':
    unittest.main()

