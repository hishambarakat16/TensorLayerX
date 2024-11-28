from tests.utils import CustomTestCase
import tensorlayerx as tlx
import tensorflow as tf
import unittest
import os

# Test for SimpleVocabulary class
class TestSimpleVocabulary(CustomTestCase):
    """Test the SimpleVocabulary class from tlx.text.nlp"""

    def test_simple_vocabulary(self):
        """Test SimpleVocabulary functionality."""
        # Sample vocabulary and unk_id
        vocab = {'apple': 0, 'banana': 1, 'cherry': 2}
        unk_id = 999
        simple_vocab = tlx.text.nlp.SimpleVocabulary(vocab, unk_id)

        # Test known word
        self.assertEqual(simple_vocab.word_to_id('apple'), 0)
        self.assertEqual(simple_vocab.word_to_id('banana'), 1)
        self.assertEqual(simple_vocab.word_to_id('cherry'), 2)

        # Test unknown word
        self.assertEqual(simple_vocab.word_to_id('grape'), unk_id)

        print("Test for SimpleVocabulary passed!")

# Test for Vocabulary class
class TestVocabulary(CustomTestCase):
    """Test the Vocabulary class from tlx.text.nlp"""

    def test_vocabulary(self):
        """Test Vocabulary functionality."""
        # Create a temporary vocab file
        vocab_data = """apple 0
                        banana 1
                        cherry 2
                        <S> 3
                        </S> 4
                        <UNK> 5
                        <PAD> 6"""
        vocab_file = "/tmp/test_vocab.txt"
        with open(vocab_file, 'w') as f:
            f.write(vocab_data)

        # Initialize the Vocabulary class
        vocab = tlx.text.nlp.Vocabulary(vocab_file)

        # Test known words
        self.assertEqual(vocab.word_to_id('apple'), 0)
        self.assertEqual(vocab.word_to_id('banana'), 1)
        self.assertEqual(vocab.word_to_id('cherry'), 2)

        # Test special tokens
        self.assertEqual(vocab.word_to_id('<S>'), 3)
        self.assertEqual(vocab.word_to_id('</S>'), 4)
        self.assertEqual(vocab.word_to_id('<UNK>'), 5)
        self.assertEqual(vocab.word_to_id('<PAD>'), 6)

        # Test unknown word
        self.assertEqual(vocab.word_to_id('grape'), 5)

        # Clean up the temporary vocab file
        os.remove(vocab_file)

        print("Test for Vocabulary passed!")

# Run the tests
if __name__ == '__main__':
    unittest.main()

