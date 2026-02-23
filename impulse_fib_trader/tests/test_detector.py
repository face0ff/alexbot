import pytest
import pandas as pd
import numpy as np
from pattern.detector import PatternDetector
import os

def test_pattern_detector_init():
    config_path = 'config/pattern_spec.json'
    if os.path.exists(config_path):
        detector = PatternDetector(config_path)
        assert detector is not None
        assert detector.impulse_detector is not None

def test_impulse_logic():
    # Create dummy data with an impulse
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=20, freq='1h'),
        'open':  [100]*5 + [100, 105, 110, 115, 120] + [120]*10,
        'high':  [101]*5 + [106, 111, 116, 121, 125] + [125]*10,
        'low':   [99]*5  + [99, 104, 109, 114, 119] + [119]*10,
        'close': [100]*5 + [105, 110, 115, 120, 125] + [125]*10,
        'volume':[1000]*20,
        'atr':   [5]*20
    }
    df = pd.DataFrame(data)
    
    from pattern.impulse import ImpulseDetector
    import json
    with open('config/pattern_spec.json', 'r') as f:
        config = json.load(f)
        
    detector = ImpulseDetector(config)
    impulses = detector.detect(df)
    
    assert len(impulses) > 0
    assert impulses[0]['type'] == 'bullish'
