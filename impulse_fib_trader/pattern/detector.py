import pandas as pd
from typing import List, Dict
import json
import logging
from pattern.impulse import ImpulseDetector
from pattern.pullback import PullbackMeasurer
from pattern.structure import StructureValidator

logger = logging.getLogger(__name__)

class PatternDetector:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.impulse_detector = ImpulseDetector(self.config)
        self.pullback_measurer = PullbackMeasurer(self.config)
        self.structure_validator = StructureValidator(self.config)

    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Runs the full detection pipeline.
        """
        logger.info("Detecting impulses...")
        impulses = self.impulse_detector.detect(df)
        logger.info(f"Found {len(impulses)} potential impulses.")
        
        patterns = []
        for imp in impulses:
            # For each impulse, try to find a valid pullback
            pullback = self.pullback_measurer.measure(imp, df)
            if pullback:
                # If pullback is valid, validate structure break
                structure = self.structure_validator.validate(imp, pullback, df)
                if structure:
                    # Determine success (just a simple check for Phase 2)
                    # Success = price moved further in impulse direction
                    success = self._evaluate_success(imp, structure, df)
                    
                    patterns.append({
                        'impulse': imp,
                        'pullback': pullback,
                        'structure': structure,
                        'success': success,
                        'timestamp': df.iloc[imp['start_idx']]['timestamp']
                    })
                    
        logger.info(f"Found {len(patterns)} valid patterns.")
        return patterns

    def _evaluate_success(self, impulse: Dict, structure: Dict, df: pd.DataFrame) -> bool:
        """
        Simple evaluation if the pattern resulted in a continuation.
        In Phase 2, we just look forward 20 bars.
        """
        entry_idx = structure['entry_idx']
        impulse_range = impulse['range']
        
        target_bars = 20
        end_idx = min(entry_idx + target_bars, len(df) - 1)
        
        future_prices = df.iloc[entry_idx + 1 : end_idx + 1]
        if future_prices.empty:
            return False
            
        if impulse['type'] == 'bullish':
            # Reach 1.0 RR (another impulse range)
            max_future = future_prices['high'].max()
            return max_future > structure['entry_price'] + 0.5 * impulse_range
        else:
            min_future = future_prices['low'].min()
            return min_future < structure['entry_price'] - 0.5 * impulse_range
