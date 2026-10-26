"""API tests for the Flask app (CPU, uses the trained checkpoints and demo layouts).

    python -m pytest tests -q
"""

import io
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, 'checkpoints', 'norm.json')),
                                reason='checkpoints/ not found')


@pytest.fixture(scope='module')
def client():
    from server import create_app
    return create_app().test_client()


@pytest.fixture(scope='module')
def demo_result(client):
    r = client.post('/api/predict', json={'demo': 0})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def npz_bytes(**arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def test_index(client):
    r = client.get('/')
    assert r.status_code == 200 and b'RouteVision' in r.data


def test_info(client):
    d = client.get('/api/info').get_json()
    names = [m['name'] for m in d['methods']]
    assert {'RUDY (EDA estimate)', 'CNN', 'Multimodal'} <= set(names)
    assert len(d['demos']) > 0 and all(x['group'] for x in d['demos'])
    assert all(m['threshold'] is not None for m in d['methods'])


def test_overall_stats(client):
    o = client.get('/api/info').get_json()['overall']
    assert o['n_layouts'] == 96 and o['n_patches'] == 6144
    rows = {r['method']: r for r in o['rows']}
    assert 'wins_vs_egr' not in rows['eGR (EDA estimate)']
    for m, r in rows.items():
        if m != 'eGR (EDA estimate)':
            assert r['wins_vs_egr'] + r['ties_vs_egr'] + r['losses_vs_egr'] == 96
    import pandas as pd
    csv = pd.read_csv(os.path.join(ROOT, 'outputs', 'test_results.csv')).set_index('method')
    for m in ('Multimodal', 'Ensemble'):
        if m in rows:
            assert rows[m]['ROC-AUC'] == pytest.approx(csv.loc[m, 'ROC-AUC'])


def test_predict_demo(demo_result):
    d = demo_result
    assert len(d['channels']) == 4
    assert np.array(d['truth']).shape == (8, 8)
    for m, s in d['scores'].items():
        s = np.array(s)
        assert s.shape == (8, 8) and np.isfinite(s).all(), m
    assert {row['method'] for row in d['comparison']} == set(d['scores'])
    assert d['source'] == 'demo' and d['demo_index'] == 0


def test_predict_cached(client, demo_result):
    d = client.post('/api/predict', json={'demo': 0}).get_json()
    assert d['elapsed_ms'] == 0 and d['scores'] == demo_result['scores']


def test_upload_matches_demo(client, demo_result):
    """Uploading the same layout as an .npz gives the same scores as the built-in demo."""
    z = np.load(os.path.join(ROOT, 'demo', 'demo_layouts.npz'))
    raw = npz_bytes(X=z['X'][0], params=z['params'][0], label=z['label'][0], egr=z['egr'][0])
    r = client.post('/api/predict', data={'file': (io.BytesIO(raw), 'layout.npz')}, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_json()
    d = r.get_json()
    for m in demo_result['scores']:
        np.testing.assert_allclose(d['scores'][m], demo_result['scores'][m], atol=1e-4)


def test_upload_without_label(client):
    z = np.load(os.path.join(ROOT, 'demo', 'demo_layouts.npz'))
    raw = npz_bytes(X=z['X'][1], params=z['params'][1])
    d = client.post('/api/predict', data={'file': (io.BytesIO(raw), 'x.npz')}, content_type='multipart/form-data').get_json()
    assert d['truth'] is None and d['comparison'] is None
    assert 'eGR (EDA estimate)' not in d['scores'] and 'CNN' in d['scores']


@pytest.mark.parametrize('payload, msg', [
    ({'X': np.zeros((3, 256, 256), np.float16), 'params': np.zeros(6)}, 'X'),
    ({'X': np.zeros((4, 256, 256), np.float16)}, 'params'),
    ({'X': np.zeros((4, 256, 256), np.float16), 'params': np.zeros(5)}, 'params'),
])
def test_upload_rejects_bad_layout(client, payload, msg):
    r = client.post('/api/predict', data={'file': (io.BytesIO(npz_bytes(**payload)), 'bad.npz')},
                    content_type='multipart/form-data')
    assert r.status_code == 400 and msg in r.get_json()['error']


def test_upload_rejects_non_npz(client):
    r = client.post('/api/predict', data={'file': (io.BytesIO(b'not a zip'), 'bad.npz')}, content_type='multipart/form-data')
    assert r.status_code == 400


@pytest.mark.parametrize('body', [{}, {'demo': -1}, {'demo': 999}, {'demo': '0'}, {'demo': True}])
def test_predict_rejects_bad_demo(client, body):
    assert client.post('/api/predict', json=body).status_code == 400


def test_gradcam(client, demo_result):
    r = client.post('/api/gradcam', json={'layout_id': demo_result['layout_id'], 'method': 'CNN'})
    assert r.status_code == 200 and r.get_json()['overlay'].startswith('data:image/png;base64,')
    assert client.post('/api/gradcam', json={'layout_id': demo_result['layout_id'], 'method': 'ViT'}).status_code == 400
    assert client.post('/api/gradcam', json={'layout_id': 'nope', 'method': 'CNN'}).status_code == 404


def test_uncertainty_is_member_disagreement(client, demo_result):
    info = client.get('/api/info').get_json()
    u = info['uncertainty']
    if u is None:
        pytest.skip('outputs/test_scores.npz has no ensemble members')
    unc = np.array(demo_result['uncertainty'])
    members = np.stack([np.array(demo_result['scores'][m]) for m in u['members']])
    np.testing.assert_allclose(unc, members.std(0), atol=1e-3)
    # the cut-off flags the top 20 % of the test design's patches (from unlabelled scores only)
    z = np.load(os.path.join(ROOT, 'outputs', 'test_scores.npz'))
    d = np.std(np.stack([z[m] for m in u['members']]), 0)
    assert (d >= u['cutoff']).mean() == pytest.approx(u['top'], abs=0.01)


def test_unet_maps(client, demo_result):
    names = [m['name'] for m in client.get('/api/info').get_json()['methods']]
    unet = [n for n in names if n.startswith('U-Net')]
    if not unet:
        pytest.skip('checkpoints/UNet.pt not exported')
    d = demo_result
    assert set(d['maps']) >= {unet[0], 'eGR (EDA estimate)', 'RUDY (EDA estimate)', 'truth'}
    assert all(v.startswith('data:image/png;base64,') for v in d['maps'].values())
    m = d['map_metrics'][unet[0]]
    assert -1 <= m['Pearson'] <= 1 and m['SSIM'] is not None
    # the hotspot score of every patch is the mean of the predicted map over that patch
    assert np.array(d['scores'][unet[0]]).shape == (8, 8)
    # Seg-Grad-CAM of the U-Net
    r = client.post('/api/gradcam', json={'layout_id': d['layout_id'], 'method': unet[0]})
    assert r.status_code == 200 and r.get_json()['heat'].startswith('data:image/png;base64,')


def test_ensemble_is_mean_of_members():
    """checkpoints from before the ensemble was added have no ensemble.json, so set it up by hand."""
    from inference import Engine, load_demos
    eng = Engine()
    eng.ensemble = ['OpenCV + RandomForest', 'Multimodal', 'ViT']
    eng.thr['Ensemble'] = 0.5
    assert 'Ensemble' in eng.available()
    _, layout, _ = load_demos()[0]
    p = eng.predict_all(eng.store(layout))
    np.testing.assert_allclose(p['Ensemble'], np.mean([p[m] for m in eng.ensemble], 0), atol=1e-6)
