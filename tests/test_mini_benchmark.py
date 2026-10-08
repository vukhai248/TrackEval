import os
import pytest
import trackeval


def test_mot17_mini_evaluation():
    """End-to-end evaluation test using MOT17-mini sample dataset."""
    code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    eval_config = {
        'USE_PARALLEL': False,
        'PRINT_RESULTS': False,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': False,
    }
    dataset_config = {
        'BENCHMARK': 'MOT17',
        'SPLIT_TO_EVAL': 'mini',
        'TRACKERS_TO_EVAL': ['DemoTracker'],
        'GT_FOLDER': os.path.join(code_path, 'data', 'gt', 'mot_challenge'),
        'TRACKERS_FOLDER': os.path.join(code_path, 'data', 'trackers', 'mot_challenge'),
    }
    
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity()
    ]
    
    raw_results, messages = evaluator.evaluate(dataset_list, metrics_list)
    
    assert 'MotChallenge2DBox' in raw_results
    assert 'DemoTracker' in raw_results['MotChallenge2DBox']
    tracker_res = raw_results['MotChallenge2DBox']['DemoTracker']
    assert 'MOT17-02-mini' in tracker_res
    
    seq_res = tracker_res['MOT17-02-mini']['pedestrian']
    import numpy as np
    # Check HOTA (array over alpha thresholds, mean is final score in [0, 1])
    assert np.mean(seq_res['HOTA']['HOTA']) * 100 > 90.0
    # Check CLEAR MOTA
    assert seq_res['CLEAR']['MOTA'] * 100 > 90.0
    # Check Identity IDF1
    assert seq_res['Identity']['IDF1'] * 100 > 90.0
    # Check Count
    assert seq_res['Count']['Dets'] == 20
    assert seq_res['Count']['GT_Dets'] == 20
