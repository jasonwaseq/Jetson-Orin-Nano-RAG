import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path so we can import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rag import _clean_text, Chunk, pages_to_chunks

class TestRagUtils(unittest.TestCase):
    def test_clean_text(self):
        """Test text cleaning utility."""
        self.assertEqual(_clean_text(" This   is  a test. "), "This is a test.")
        self.assertEqual(_clean_text("Null\x00byte"), "Null byte")
        self.assertEqual(_clean_text("Line\nBreak"), "Line Break")

    def test_chunking_logic(self):
        """Test basic chunking behavior."""
        # Create a string of known length
        # "1234567890" * 10 = 100 chars
        text = "1234567890" * 10 
        pages = [(1, text)]
        
        # chunk_chars=50, overlap=10
        # Chunk 1: 0-50 (50 chars)
        # Chunk 2: 40-90 (50 chars)
        # Chunk 3: 80-130 (remaing 20 chars) -> min_chunk_chars=10 -> accepted
        
        chunks = pages_to_chunks(
            "doc1", 
            pages, 
            chunk_chars=50, 
            overlap_chars=10, 
            min_chunk_chars=10
        )
        
        self.assertTrue(len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}")
        self.assertEqual(chunks[0].text, text[0:50])
        self.assertEqual(chunks[1].text, text[40:90])
        
    def test_chunking_min_chars(self):
        """Test usage of min_chunk_chars."""
        text = "short"
        pages = [(1, text)]
        chunks = pages_to_chunks("doc1", pages, min_chunk_chars=10)
        self.assertEqual(len(chunks), 0, "Should handle texts shorter than min_chunk_chars")

    def test_chunking_identifiers(self):
        """Test that chunks have correct metadata."""
        text = "HelloWorld"
        pages = [(1, text)]
        chunks = pages_to_chunks("test_doc.pdf", pages, min_chunk_chars=5)
        
        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertEqual(c.doc_id, "test_doc.pdf")
        self.assertEqual(c.page, 1)
        self.assertEqual(c.text, "HelloWorld")
        self.assertTrue(len(c.chunk_id) > 0)

if __name__ == "__main__":
    unittest.main()
