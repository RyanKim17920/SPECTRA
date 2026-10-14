"""Pick ~300 TCGA slides (75/type) for the site-shortcut test: 4 multi-site types, true-20x only
(excludes the 258 obj=20 & mpp<0.35 slides of mag_census.json), one slide per patient, pyramidal .svs only, only sites with
>=5 eligible slides, the largest sites first, >=6 slides per site (so within-type site kNN can use every site). Seed 0."""
import collections, json, sys
import numpy as np
import openslide
R = '/data/ryan.kim/ashby/results/tests/clustering'
TYPES = sys.argv[1].split(',') if len(sys.argv) > 1 else ['LUAD', 'LUSC', 'KIRC', 'KIRP']
N = 75; OUT = '/data/ryan.kim/ashby/results/spectra_conch/tcga/selection.json'
man = json.load(open(f'{R}/manifest.json')); mag = json.load(open(f'{R}/mag_census.json'))
bad = {s for s, (o, m) in mag.items() if o == '20' and m not in (None, 'ERR') and float(m) < 0.35}
assert len(bad) == 258, len(bad)
rng = np.random.default_rng(0); sel = []
for t in TYPES:
    rs = [r for r in man if r['type'] == t and r['svs'] and r['slide'] not in bad and mag.get(r['slide'], [None, None])[1] not in (None, 'ERR')]
    rs = [r for r in rs if openslide.OpenSlide(r['svs']).level_count > 1]  # single-level svs cannot be thumbnailed cheaply
    seen = set(); rs2 = []
    for r in rng.permutation(len(rs)):
        r = rs[r]
        if r['patient'] not in seen: seen.add(r['patient']); rs2.append(r)
    by = collections.defaultdict(list)
    for r in rs2: by[r['tss']].append(r)
    sites = sorted([s for s in by if len(by[s]) >= 5], key=lambda s: -len(by[s]))
    k = max(6, -(-N // len(sites)))  # >=6/site so within-type site kNN (needs >=5/site) sees every site
    pick = []
    for s in sites:
        pick += by[s][:k]
        if len(pick) >= N: break
    pick = pick[:N]
    c = collections.Counter(r['tss'] for r in pick)
    print(t, len(pick), 'sites', len(c), dict(c)); sel += pick
json.dump(sel, open(OUT, 'w'), indent=0); print(len(sel), OUT)
