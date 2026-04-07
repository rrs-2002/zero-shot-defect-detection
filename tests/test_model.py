import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.winclip import WinCLIP
from src.config import Config

class TestWinCLIP(unittest.TestCase):
    def test_model_loading(self):
        """Test if WinCLIP model loads without error."""
        print("\nTesting Model Loading...")
        try:
            model = WinCLIP()
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.processor)
            print("Model loaded successfully.")
        except Exception as e:
            self.fail(f"Model loading failed: {e}")

    def test_text_encoding(self):
        """Test text encoding shape."""
        print("\nTesting Text Encoding...")
        model = WinCLIP()
        embeddings = model.encode_text("bottle")
        # Should be [2, Feature_Dim]
        self.assertEqual(embeddings.shape[0], 2)
        print(f"Text embeddings shape: {embeddings.shape}")

if __name__ == '__main__':
    unittest.main()
