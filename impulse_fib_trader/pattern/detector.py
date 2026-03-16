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
        Runs the full detection pipeline (for Breakout mode).
        """
        logger.info("Detecting impulses...")
        impulses = self.impulse_detector.detect(df)
        
        patterns = []
        for imp in impulses:
            pullback = self.pullback_measurer.measure(imp, df)
            if pullback:
                structure = self.structure_validator.validate(imp, pullback, df)
                if structure:
                    success = self._evaluate_success(imp, structure, df)
                    patterns.append({
                        'symbol': 'UNKNOWN', # To be filled by scanner
                        'impulse': imp,
                        'pullback': pullback,
                        'structure': structure,
                        'success': success,
                        'timestamp': df.iloc[imp['start_idx']]['timestamp']
                    })
        return patterns

    def detect_pending_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detects patterns that are in the pullback phase (for Limit mode).
        """
        impulses = self.impulse_detector.detect(df)
        pending = []
        
        for imp in impulses:
            # Если импульс закончился совсем недавно (последние 12 свечей)
            if imp['end_idx'] >= len(df) - 12:
                pullback = self.pullback_measurer.measure(imp, df)
                # Если цена сейчас в зоне 0.5 - 0.705
                if pullback and pullback['end_idx'] >= len(df) - 2:
                    # Рассчитываем идеальный вход 0.618
                    fib_level = 0.618
                    if imp['type'] == 'bullish':
                        entry_price = imp['high'] - (imp['range'] * fib_level)
                    else:
                        entry_price = imp['low'] + (imp['range'] * fib_level)
                        
                    pending.append({
                        'impulse': imp,
                        'pullback': pullback,
                        'limit_entry_price': entry_price,
                        'timestamp': df.iloc[imp['start_idx']]['timestamp']
                    })
        return pending

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
