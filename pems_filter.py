"""Cut PeMS Clearinghouse downloads (District 11, 2019) down to the 716 LargeST-SD sensors.

    python3 pems_filter.py --sd-meta sd_meta.csv --out pems_out \
        --hour ~/Downloads/pems/stationHour --five-min ~/Downloads/pems/5minStation \
        --incidents ~/Downloads/pems/chp --station-meta ~/Downloads/pems/metadata

Outputs (in --out); station arrays are (bins, 716) in sd_meta.csv row order, on the same naive local
wall-clock grid as sd_his_2019.h5, with flow always in LargeST's unit (vehicles per 5 min, window mean):
  pems_sd_2019_hour.npz    hourly bins (8760):   from Station Hour files (hourly Total Flow / 12)
  pems_sd_2019_15min.npz   15-min bins (35040):  from Station 5-Minute files; flow rounded like LargeST
      flow    float32 (hour) / uint16 (15-min)   mean flow, veh/5min (0 where no record)
      occ     float16  mean Avg Occupancy (fraction 0-1), NaN where not reported
      speed   float16  mean Avg Speed (mph), NaN where not reported
      pct_obs uint8    mean % Observed, 255 = no record
      nrec    uint8    records averaged into the bin (0 = missing; >1 in the repeated DST hour)
      ids     int64    sensor order
  d11_station_meta_sd.csv                        PeMS metadata rows for the 716 IDs (postmiles etc.)
  chp_incidents_d11_2019.csv, chp_incident_details_d11_2019.csv   District-11 CHP incidents + detail text

Only numpy + pandas are required. Accepts .txt, .txt.gz and single-file .zip downloads.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

YEAR_START = pd.Timestamp('2019-01-01')
YEAR_MIN = 365 * 24 * 60
CORE = ['timestamp', 'station', 'district', 'fwy', 'dir', 'lane_type', 'length',
        'samples', 'pct_obs', 'flow', 'occ', 'speed']
INCIDENT_COLS = ['incident_id', 'cc_code', 'incident_number', 'timestamp', 'description', 'location', 'area',
                 'zoom_map', 'tb_xy', 'lat', 'lng', 'district', 'county_fips', 'city_fips', 'fwy', 'dir',
                 'state_pm', 'abs_pm', 'severity', 'duration']
DETAIL_COLS = ['incident_id', 'detail_id', 'timestamp', 'description']


def data_files(folder, pattern='*'):
    files = sorted(p for p in Path(folder).expanduser().rglob(pattern)
                   if p.is_file() and (p.suffix in {'.txt', '.gz', '.zip'}))
    if not files:
        raise SystemExit(f'no .txt/.gz/.zip files under {folder}')
    return files


def read_station_chunks(path, chunk=400_000):
    """Station files have ragged per-lane columns: read whole lines in chunks, split off the 12 core fields."""
    for lines in pd.read_csv(path, header=None, sep='\x01', names=['raw'], dtype=str, chunksize=chunk):
        df = lines['raw'].str.split(',', n=12, expand=True).iloc[:, :12]
        df.columns = CORE
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
        yield df[df['timestamp'].notna()]


def filter_station(folder, ids, rec_min, bin_min):
    """Average station records (rec_min apart) into bins of bin_min minutes for the given sensor IDs."""
    n, nbins = len(ids), YEAR_MIN // bin_min
    col_of = pd.Index(ids)
    sums = {v: np.zeros(nbins * n, np.float32) for v in ('flow', 'occ', 'speed', 'pct_obs')}
    cnts = {v: np.zeros(nbins * n, np.uint8) for v in sums}
    seen = np.zeros(n, bool)
    files = data_files(folder)
    for i, path in enumerate(files, 1):
        rows, tmin, tmax = 0, None, None
        for df in read_station_chunks(path):
            col = col_of.get_indexer(pd.to_numeric(df['station'], errors='coerce').fillna(-1).astype(np.int64))
            rec = ((df['timestamp'] - YEAR_START) // pd.Timedelta(minutes=rec_min)).to_numpy()
            b = rec * rec_min // bin_min
            keep = (col >= 0) & (b >= 0) & (b < nbins)
            if not keep.any():
                continue
            df, col, b = df[keep], col[keep], b[keep]
            seen[np.unique(col)] = True
            b0, b1 = b.min(), b.max()
            key, size, span = (b - b0) * n + col, (b1 - b0 + 1) * n, slice(b0 * n, (b1 + 1) * n)
            for v in sums:
                x = pd.to_numeric(df[v], errors='coerce').to_numpy(np.float64)
                if v == 'flow':
                    x = x * 5 / rec_min                     # -> vehicles per 5 min, LargeST's unit
                ok = ~np.isnan(x)
                sums[v][span] += np.bincount(key[ok], weights=x[ok], minlength=size).astype(np.float32)
                cnts[v][span] += np.bincount(key[ok], minlength=size).astype(np.uint8)
            rows += len(df)
            tmin = df['timestamp'].min() if tmin is None else min(tmin, df['timestamp'].min())
            tmax = df['timestamp'].max() if tmax is None else max(tmax, df['timestamp'].max())
        print(f'[{i}/{len(files)}] {path.name}: {rows:,} SD rows' + (f', {tmin:%Y-%m-%d %H:%M} → {tmax:%Y-%m-%d %H:%M}' if rows else ''),
              flush=True)
    if not seen.any():
        raise SystemExit('none of the 716 SD sensor IDs appear in these files — wrong district/file type?')

    def mean(v):
        with np.errstate(invalid='ignore', divide='ignore'):
            return (sums[v] / cnts[v]).reshape(nbins, n)

    flow = np.nan_to_num(mean('flow'))
    nrec = cnts['flow'].reshape(nbins, n)
    out = {'ids': np.asarray(ids, np.int64),
           'flow': np.round(flow).astype(np.uint16) if bin_min == 15 else flow.astype(np.float32),
           'occ': mean('occ').astype(np.float16), 'speed': mean('speed').astype(np.float16),
           'pct_obs': np.where(cnts['pct_obs'].reshape(nbins, n) > 0, np.nan_to_num(mean('pct_obs')), 255).astype(np.uint8),
           'nrec': nrec}
    has = nrec > 0
    per_day = 24 * 60 // bin_min
    print(f'\n{bin_min}-min bins with data: {has.mean():.4f}   days with any data: '
          f'{int(has.reshape(365, per_day, n).any(axis=(1, 2)).sum())}/365   sensors never seen: {int((~seen).sum())}')
    print(f'occupancy reported in {np.isfinite(out["occ"])[has].mean():.3f} of records, range '
          f'{np.nanmin(out["occ"]):.3f}–{np.nanmax(out["occ"]):.3f}   speed reported in {np.isfinite(out["speed"])[has].mean():.3f}   '
          f'max flow {float(flow.max()):.0f} veh/5min')
    if (~seen).any():
        print('never seen IDs (first 20):', out['ids'][~seen][:20].tolist())
    return out


def read_details(path):
    """Detail text can spill onto following lines: join continuation lines back onto their record."""
    lines = pd.read_csv(path, header=None, sep='\x01', names=['raw'], dtype=str, skip_blank_lines=True)['raw']
    parts = lines.str.extract(r'^(\d+),(\d+),(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d),(.*)$')
    rec = parts[0].notna().cumsum()
    parts[3] = parts[3].where(parts[0].notna(), lines)          # continuation line = more description text
    parts = parts[rec > 0]
    df = parts.groupby(rec[rec > 0]).agg({0: 'first', 1: 'first', 2: 'first', 3: lambda s: ' '.join(s.dropna())})
    df.columns = DETAIL_COLS
    return df, int((parts[0].isna()).sum())


def filter_incidents(folder, out_dir):
    inc, det, joined = [], [], 0
    for path in data_files(folder):
        if '_det_' in path.name.lower() or 'detail' in path.name.lower():
            df, cont = read_details(path)
            det.append(df)
            joined += cont
            print(f'{path.name}: {len(df):,} detail records ({cont:,} continuation lines joined)', flush=True)
        else:
            df = pd.read_csv(path, header=None, names=INCIDENT_COLS, dtype=str)
            districts = pd.to_numeric(df['district'], errors='coerce')
            if not (districts == 11).any():
                raise SystemExit(f'{path.name}: no district==11 rows in column 12; values seen: '
                                 f'{districts.value_counts().head(10).to_dict()} — send this message to Claude')
            inc.append(df[districts == 11])
            print(f'{path.name}: {len(df):,} incidents statewide, {int((districts == 11).sum()):,} in D11', flush=True)
    inc = pd.concat(inc, ignore_index=True)
    inc.to_csv(out_dir / 'chp_incidents_d11_2019.csv', index=False)
    print(f'\nD11 incidents: {len(inc):,}  ({inc["timestamp"].min()} → {inc["timestamp"].max()})')
    if det:
        det = pd.concat(det, ignore_index=True)
        det = det[det['incident_id'].isin(inc['incident_id'])]
        det.to_csv(out_dir / 'chp_incident_details_d11_2019.csv', index=False)
        print(f'D11 incident detail lines: {len(det):,}')


def filter_station_meta(path, ids, out_dir):
    """One metadata file, or a folder of snapshots: combined, keeping each sensor's most recent row."""
    p = Path(path).expanduser()
    files = sorted(p.glob('*meta*.txt')) if p.is_dir() else [p]
    meta = pd.concat([pd.read_csv(f, sep='\t').assign(snapshot=f.stem[-10:]) for f in files], ignore_index=True)
    meta = meta.sort_values('snapshot').drop_duplicates('ID', keep='last')
    meta = meta[meta['ID'].isin(ids)]
    meta.to_csv(out_dir / 'd11_station_meta_sd.csv', index=False)
    print(f'station metadata: {len(meta)} of {len(ids)} SD sensors found; types {meta["Type"].value_counts().to_dict()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sd-meta', required=True, help='sd_meta.csv from the LargeST SD pipeline')
    ap.add_argument('--out', required=True)
    ap.add_argument('--five-min', help='folder with d11_text_station_5min_2019_*.txt[.gz]')
    ap.add_argument('--hour', help='folder with d11_text_station_hour_2019_*.txt[.gz]')
    ap.add_argument('--incidents', help='folder with 2019 CHP incident (and detail) files')
    ap.add_argument('--station-meta', help='a d11_text_meta_2019_*.txt file, or a folder of them')
    a = ap.parse_args()

    out_dir = Path(a.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(Path(a.sd_meta).expanduser())['ID'].astype(np.int64).tolist()
    assert len(ids) == 716, f'expected 716 SD sensors, got {len(ids)}'
    if a.station_meta:
        filter_station_meta(a.station_meta, ids, out_dir)
    if a.hour:
        np.savez_compressed(out_dir / 'pems_sd_2019_hour.npz', **filter_station(a.hour, ids, rec_min=60, bin_min=60))
        print('wrote', out_dir / 'pems_sd_2019_hour.npz')
    if a.five_min:
        np.savez_compressed(out_dir / 'pems_sd_2019_15min.npz', **filter_station(a.five_min, ids, rec_min=5, bin_min=15))
        print('wrote', out_dir / 'pems_sd_2019_15min.npz')
    if a.incidents:
        filter_incidents(a.incidents, out_dir)
