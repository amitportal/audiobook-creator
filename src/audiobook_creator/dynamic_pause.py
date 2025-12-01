"""
Dynamic Pause Calculator

Uses sentence embeddings to compute semantic similarity and determine pause lengths.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class DynamicPauseCalculator:
    """Calculate pauses between text chunks based on semantic similarity"""
    
    def __init__(self):
        """Initialize the pause calculator with sentence embeddings model"""
        self.model = None
        self._model_loaded = False
    
    def _load_model(self):
        """Lazy load the sentence transformer model"""
        if self._model_loaded:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence embedding model for dynamic pauses...")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._model_loaded = True
            logger.info("[OK] Sentence embedding model loaded")
        except ImportError:
            logger.warning("sentence-transformers not available, using fixed pauses")
            self._model_loaded = False
        except Exception as e:
            logger.warning(f"Failed to load sentence embedding model: {e}, using fixed pauses")
            self._model_loaded = False
    
    def calculate_pause(
        self,
        text1: str,
        text2: str,
        default_pause: float = 0.3
    ) -> float:
        """
        Calculate pause duration based on semantic similarity
        
        Args:
            text1: First text chunk
            text2: Second text chunk  
            default_pause: Default pause if model not available
            
        Returns:
            Pause duration in seconds
        """
        # Try to load model if not already loaded
        if not self._model_loaded:
            self._load_model()
        
        # If model still not available, use default
        if not self._model_loaded or self.model is None:
            return default_pause
        
        try:
            from sentence_transformers import util
            
            # Get embeddings (take first 200 chars for efficiency)
            text1_short = text1[:200]
            text2_short = text2[:200]
            
            embeddings = self.model.encode([text1_short, text2_short], convert_to_tensor=True)
            
            # Compute similarity
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            
            # Continuous non-linear pause mapping
            min_pause, max_pause, gamma = 0.2, 1.0, 0.9
            scaled = (1 - similarity**2) ** gamma
            pause = min_pause + (max_pause - min_pause) * scaled
            
            logger.debug(f"Similarity: {similarity:.3f} -> Pause: {pause:.3f}s")
            return pause
            
        except Exception as e:
            logger.debug(f"Error calculating similarity: {e}, using default pause")
            return default_pause