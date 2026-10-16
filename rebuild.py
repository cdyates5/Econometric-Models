#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US GRID Models -- Real GDP & Inflation
======================================
Econometric reconstruction of 42 Macro's GRID forecasting process for the United States, rebuilt
from the structure recovered from the user's 42 Macro workbooks (Aug-2023 model file with the cached
'42 Macro Regime Forecasting Tool.xlsx' links, Nov-2024 GRID data file) and the Sep-2026 screenshot:

  * Stationary Mean Reversion Model (SMRM): growth rates revert from their run-rate to a trailing
    "stationary mean" at an estimated AR decay; projected levels give YoY paths with exact base effects.
  * Agent-Based Nowcast Model (ABNM): 27 single-indicator "agents" per family (growth / inflation),
    each a first-difference regression of the change in the target's growth rate on the 3-month change
    in its feature at the feature's real-time ragged edge; agents are weighted by relative R^2 and
    retrained every month on a rolling window, separately for every horizon.
  * Median projection = 50/50 split of SMRM and ABNM ("split the difference").
  * Bottom-up GRID regime = sign of 3-month changes in Real GDP YoY and inflation YoY
    (G = growth up & inflation down, R = both up, I = growth down & inflation up, D = both down).

Scorecard definitions replicate 42 Macro's exactly (verified against the Nov-2024 data file):
hit rate, average absolute forecast error and directional accuracy over trailing 10 years, Z-scores
against each series' own 2015-19 mean and standard deviation.

Regenerates the single self-contained HTML dashboard from an empty directory:

    FRED_API_KEY=<key> python3 rebuild.py --out us_grid_models.html            # full build (needs ref/ref42.json)
    FRED_API_KEY=<key> python3 rebuild.py --public --out site/index.html       # no 42 Macro / Bloomberg data

GitHub Actions use (see README.md): --fingerprint and --archive-dir record a forecast snapshot only when the
underlying data change, and the script writes changed=true|false to $GITHUB_OUTPUT. Exit code 2 = failed sanity
gate (nothing written), so a broken build never replaces the live page.

Environment: FRED_API_KEY (required for reliable FRED access), BLS_API_KEY (optional; BLS API v2, 500 queries/day),
REF42_JSON (optional; the 42 Macro reference data as a JSON string instead of ref/ref42.json).

Sources: FRED (JSON API with key; fredgraph.csv via curl fallback), BLS public API v1 (five CPI
sub-indices; FRED fallbacks), Philadelphia Fed Survey of Professional Forecasters median levels
(consensus benchmark in place of Bloomberg), plus an embedded snapshot of 42 Macro's US track record
(Nov-2024 workbook) and the Sep-2026 screenshot (hand-transcribed, 0.1pp resolution).
"""
import os, sys, re, json, time, argparse, subprocess, datetime as dt, warnings
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')
np.seterr(all='ignore')

# ============================================================================ configuration
BASE = pd.Period('1985-01', 'M')
EVAL_START = '2005-01'          # first evaluated forecast origin / dashboard axis start
TABLE_START = '2024-08'         # GRID table start (matches the Sep-2026 screenshot)
W_INF, W_GDPQ = 180, 60         # rolling training windows: inflation origins (months); GDP origins per intra-quarter position (quarters)
H_INF, H_GDP = 18, 6            # horizons: months / quarters
MIN_INF, MIN_GDP = 60, 24       # minimum training pairs before an agent may vote
MU_WIN_M, RHO_WIN_M = 120, 180  # SMRM windows (months)
MU_WIN_Q, RHO_WIN_Q = 40, 60    # SMRM windows (quarters)
MU_STAT = os.environ.get('GRID_MU', 'mean')  # stationary mean estimator: trailing mean (COVID quarters excluded for GDP) or median
UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
FRED_KEY = os.environ.get('FRED_API_KEY', '').strip()
BLS_KEY = os.environ.get('BLS_API_KEY', '').strip()
SESSION = requests.Session()
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


# ============================================================================ feature map
# (key, 42 Macro feature in the 2023 workbook, its weight %, our proxy, source, transform)
# source: list of FRED ids (tuples = alternates, first available wins) -> composite of 3m changes if >1;
#         'CTRL' retail control group; 'PPI' spliced PPI; {'bls': id, 'fb': fred_id} BLS with FRED fallback
GROWTH = [
    ('ip', 'US Industrial Production YoY', 6, 'Industrial production YoY (INDPRO)', ['INDPRO'], 'yoy'),
    ('exports', 'US Exports YoY', 4, 'Exports of goods & services YoY (BOPTEXP)', ['BOPTEXP'], 'yoy'),
    ('retail_ctrl', 'Retail sales control group YoY', 4, 'Retail control group YoY (RSXFS less autos, building materials, gasoline)', 'CTRL', 'yoy'),
    ('ism_mfg', 'ISM Manufacturing PMI', 8, 'Regional Fed manufacturing composite (Philadelphia, Empire, Dallas)', ['GACDFSA066MSFRBPHI', 'GACDISA066MSFRBNY', 'BACTSAMFRBDAL'], 'mean'),
    ('ism_svc', 'ISM Services PMI', 7, 'Regional Fed services composite (NY Fed BLS, Philadelphia NBOS, Dallas TSSOS)', [('BACDISA066MSFRBNY', 'BACDINA066MNFRBNY'), 'GABNDIF066MSFRBPHI', 'TSSOSBACTSAMFRBDAL'], 'mean'),
    ('oecd_bci', 'OECD business confidence (manufacturing)', 3, 'Regional Fed 6-month-ahead activity expectations (Philadelphia, Empire)', ['GAFDFSA066MSFRBPHI', 'GAFDISA066MSFRBNY'], 'mean'),
    ('oecd_cci', 'OECD consumer confidence', 2, 'University of Michigan consumer sentiment (UMCSENT)', ['UMCSENT'], 'level'),
    ('nfp', 'Nonfarm payrolls', 7, 'Nonfarm payrolls YoY (PAYEMS)', ['PAYEMS'], 'yoy'),
    ('priv', 'Private payrolls', 7, 'Private payrolls YoY (USPRIV)', ['USPRIV'], 'yoy'),
    ('hours', 'Avg weekly hours, prod. & nonsup.', 0, 'Avg weekly hours, prod. & nonsup. (AWHNONAG)', ['AWHNONAG'], 'level'),
    ('awe', 'Average weekly earnings', 1, 'Avg weekly earnings, prod. & nonsup. YoY (CES0500000030)', [('CES0500000030', 'CES0500000011')], 'yoy'),
    ('jolts', 'JOLTS job openings', 2, 'Job openings YoY (JTSJOL)', ['JTSJOL'], 'yoy'),
    ('diffusion', 'BLS employment diffusion, 1-month', 4, 'Employment diffusion index, 1-month span, across states (SMS…021)', ['SMS00000000000000021'], 'level'),
    ('rpce', 'Real PCE', 5, 'Real PCE YoY (DPCERA3M086SBEA)', [('DPCERA3M086SBEA', 'PCEC96')], 'yoy'),
    ('rpce_g', 'Real PCE goods', 4, 'Real PCE goods YoY (DGDSRA3M086SBEA)', [('DGDSRA3M086SBEA', 'DGDSRX1')], 'yoy'),
    ('rpce_s', 'Real PCE services', 5, 'Real PCE services YoY (DSERRA3M086SBEA)', [('DSERRA3M086SBEA', 'PCESC96')], 'yoy'),
    ('rdpi', 'Real disposable income', 1, 'Real disposable personal income YoY (DSPIC96)', ['DSPIC96'], 'yoy'),
    ('retail', 'Retail & food services sales', 3, 'Retail & food services sales YoY (RSAFS)', ['RSAFS'], 'yoy'),
    ('mba', 'MBA purchase index', 1, 'New home sales YoY, 3m avg (HSN1F)', ['HSN1F'], 'yoy3'),
    ('autos', 'Auto sales SAAR', 1, 'Total vehicle sales SAAR (TOTALSA)', ['TOTALSA'], 'level'),
    ('capu', 'Capacity utilization', 5, 'Capacity utilization (TCU)', ['TCU'], 'level'),
    ('mfg_orders', 'Manufacturers new orders', 5, 'Manufacturers new orders YoY (AMTMNO)', ['AMTMNO'], 'yoy'),
    ('core_capex', 'Core capital goods orders', 4, 'Nondefense capital goods ex-aircraft orders YoY (NEWORDER)', ['NEWORDER'], 'yoy'),
    ('durables', 'Durable goods orders', 4, 'Durable goods orders YoY (DGORDER)', ['DGORDER'], 'yoy'),
    ('imports', 'US Imports', 3, 'Imports of goods & services YoY (BOPTIMP)', ['BOPTIMP'], 'yoy'),
    ('rail', 'AAR rail carloads', 2, 'Rail freight carloads YoY, 3m avg (RAILFRTCARLOADSD11)', ['RAILFRTCARLOADSD11'], 'yoy3'),
    ('construction', 'Construction spending', 3, 'Total construction spending YoY (TTLCONS)', ['TTLCONS'], 'yoy'),
]
INFLATION = [
    ('core_cpi', 'Core CPI YoY', 8, 'Core CPI YoY (CPILFESL)', ['CPILFESL'], 'yoy'),
    ('ppi', 'PPI final demand YoY', 8, 'PPI final demand YoY (PPIFIS; finished goods WPSFD49207 before 2010)', 'PPI', 'yoy'),
    ('crb_all', 'CRB spot, all commodities', 3, 'IMF all-commodity price index YoY (PALLFNFINDEXM)', ['PALLFNFINDEXM'], 'yoy'),
    ('crb_food', 'CRB spot, foodstuffs', 3, 'IMF food price index YoY (PFOODINDEXM)', ['PFOODINDEXM'], 'yoy'),
    ('crb_raw', 'CRB spot, raw industrials', 1, 'IMF industrial materials index YoY (PINDUINDEXM)', ['PINDUINDEXM'], 'yoy'),
    ('cpi_food', 'CPI food', 0, 'CPI food YoY (CPIUFDSL)', ['CPIUFDSL'], 'yoy'),
    ('cpi_energy', 'CPI energy', 9, 'CPI energy YoY (CPIENGSL)', ['CPIENGSL'], 'yoy'),
    ('cpi_coregoods', 'CPI commodities less food & energy', 5, 'CPI core goods YoY (CUSR0000SACL1E)', ['CUSR0000SACL1E'], 'yoy'),
    ('cpi_hhf', 'CPI household furnishings & supplies', 1, 'CPI household furnishings & operations YoY (CUSR0000SAH3)', ['CUSR0000SAH3'], 'yoy'),
    ('cpi_apparel', 'CPI apparel', 7, 'CPI apparel YoY (CPIAPPSL)', ['CPIAPPSL'], 'yoy'),
    ('cpi_trans_goods', 'CPI transportation commodities ex motor fuel', 3, 'BLS CUSR0000SATCLTB YoY (fallback: CPI new & used vehicles)', {'bls': 'CUSR0000SATCLTB', 'fb': 'CUSR0000SETA'}, 'yoy'),
    ('cpi_med_goods', 'CPI medical care commodities', 2, 'CPI medical care commodities YoY (CUSR0000SAM1)', ['CUSR0000SAM1'], 'yoy'),
    ('cpi_rec_goods', 'CPI recreation commodities', 6, 'BLS CUSR0000SARC YoY (fallback: size-class A index)', {'bls': 'CUSR0000SARC', 'fb': 'CUURA000SARC'}, 'yoy'),
    ('cpi_edu_goods', 'CPI education & communication commodities', 0, 'BLS CUSR0000SAEC YoY (fallback: size-class A index)', {'bls': 'CUSR0000SAEC', 'fb': 'CUURA000SAEC'}, 'yoy'),
    ('cpi_alcohol', 'CPI alcoholic beverages', 2, 'CPI alcoholic beverages YoY (CUSR0000SAF116)', ['CUSR0000SAF116'], 'yoy'),
    ('cpi_svc_xe', 'CPI services less energy services', 5, 'CPI services less energy YoY (CUSR0000SASLE)', ['CUSR0000SASLE'], 'yoy'),
    ('cpi_shelter', 'CPI shelter', 1, 'CPI shelter YoY (CUSR0000SAH1)', ['CUSR0000SAH1'], 'yoy'),
    ('cpi_water', 'CPI water, sewer & trash', 0, 'CPI water, sewer & trash YoY (CUSR0000SEHG)', ['CUSR0000SEHG'], 'yoy'),
    ('cpi_med_svc', 'CPI medical care services', 0, 'CPI medical care services YoY (CUSR0000SAM2)', ['CUSR0000SAM2'], 'yoy'),
    ('cpi_trans_svc', 'CPI transportation services', 5, 'CPI transportation services YoY (CUSR0000SAS4)', ['CUSR0000SAS4'], 'yoy'),
    ('cpi_rec_svc', 'CPI recreation services', 0, 'BLS CUSR0000SARS YoY (fallback: size-class A index)', {'bls': 'CUSR0000SARS', 'fb': 'CUURA000SARS'}, 'yoy'),
    ('cpi_edu_svc', 'CPI education & communication services', 0, 'BLS CUSR0000SAES YoY (fallback: size-class A index)', {'bls': 'CUSR0000SAES', 'fb': 'CUURA000SAES'}, 'yoy'),
    ('pce', 'PCE price index', 12, 'PCE price index YoY (PCEPI)', ['PCEPI'], 'yoy'),
    ('core_pce', 'Core PCE price index', 9, 'Core PCE price index YoY (PCEPILFE)', ['PCEPILFE'], 'yoy'),
    ('nfib', 'NFIB small business higher prices', 6, 'Regional Fed prices-received composite (Philadelphia, Empire, NY Fed BLS, Dallas TSSOS)', ['PRCDFSA066MSFRBPHI', 'PRCDISA066MSFRBNY', 'PRCDINA066MNFRBNY', 'TSSOSSELLSAMFRBDAL'], 'mean'),
    ('ism_mfg_prices', 'ISM manufacturing prices', 2, 'Regional Fed manufacturing prices-paid composite (Philadelphia, Empire, Dallas)', ['PPCDFSA066MSFRBPHI', 'PPCDISA066MSFRBNY', 'PRMSAMFRBDAL'], 'mean'),
    ('ism_svc_prices', 'ISM services prices', 1, 'Regional Fed services prices-paid composite (NY Fed BLS, Philadelphia NBOS)', ['PPCDINA066MNFRBNY', 'PPBNDIF066MSFRBPHI'], 'mean'),
]
TARGET_IDS = ['GDPC1', 'CPIAUCNS', 'CPIAUCSL', 'PCEPILFE']
CTRL_IDS = ['RSXFS', 'RSMVPD', 'RSBMGESD', 'RSGASS']
PPI_IDS = ['PPIFIS', 'WPSFD49207']
BLS_IDS = ['CUSR0000SATCLTB', 'CUSR0000SARC', 'CUSR0000SAEC', 'CUSR0000SARS', 'CUSR0000SAES']

REF42 = None  # 42 Macro reference data is loaded at run time from ref/ref42.json or $REF42_JSON (never embedded)
PUBLIC = False  # --public: build without any 42 Macro / Bloomberg data
TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>US GRID Models: Real GDP &amp; Inflation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{
  --paper:#F7F1E6; --card:#FDFAF3; --ink:#221C14; --ink2:#5E5446; --rule:#E2D6C2; --tint:#F0E6D3; --fc:#EFE2CB;
  --orange:#D2622A; --teal:#0E756C; --ochre:#A67A22; --plum:#7A4A66; --slate:#4E6577;
  --g:#0E756C; --r:#A67A22; --i:#D2622A; --d:#4E6577; --onreg:#FFFFFF;
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px);
  padding-bottom:env(safe-area-inset-bottom,0px);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#16130F; --card:#1F1A14; --ink:#EEE6D8; --ink2:#B5A893; --rule:#3A3127; --tint:#28211A; --fc:#2C2419;
    --orange:#E8834D; --teal:#37A797; --ochre:#CFA14A; --plum:#C08AAB; --slate:#93A8BD;
    --g:#2E9C8D; --r:#BF9340; --i:#DF7643; --d:#7F95AB; --onreg:#14110D;
  }
}
:root[data-theme="dark"]{
  --paper:#16130F; --card:#1F1A14; --ink:#EEE6D8; --ink2:#B5A893; --rule:#3A3127; --tint:#28211A; --fc:#2C2419;
  --orange:#E8834D; --teal:#37A797; --ochre:#CFA14A; --plum:#C08AAB; --slate:#93A8BD;
  --g:#2E9C8D; --r:#BF9340; --i:#DF7643; --d:#7F95AB; --onreg:#14110D;
}
html{scroll-padding-top:env(safe-area-inset-top,0px);}
*,*::before,*::after{box-sizing:inherit;}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;-webkit-font-smoothing:antialiased;}
.wrap{max-width:1320px;margin:0 auto;padding:22px clamp(14px,3vw,34px) 64px;}
h1,h2,h3{font-family:'Space Grotesk','IBM Plex Sans',system-ui,sans-serif;letter-spacing:-0.01em;margin:0;}
h1{font-size:clamp(28px,3.6vw,42px);font-weight:700;line-height:1.08;}
h2{font-size:21px;font-weight:600;margin:0 0 6px;}
h3{font-size:16px;font-weight:600;margin:0 0 4px;}
p{margin:0 0 10px;max-width:78ch;}
.num,.mono{font-family:'IBM Plex Mono',ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums;}
.muted{color:var(--ink2);}
.small{font-size:13px;}
header.top{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;border-bottom:2px solid var(--ink);padding-bottom:14px;}
.top .lede{margin:6px 0 0;color:var(--ink2);max-width:70ch;}
.stamp{display:flex;flex-wrap:wrap;gap:4px 22px;margin-top:10px;font-size:13px;color:var(--ink2);}
.stamp span b{font-family:'IBM Plex Mono',monospace;font-weight:500;color:var(--ink);}
.ctrls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
.seg{display:inline-flex;border:1px solid var(--ink2);border-radius:4px;overflow:hidden;}
.seg button{all:unset;cursor:pointer;padding:5px 11px;font-size:13px;color:var(--ink);background:transparent;}
.seg button+button{border-left:1px solid var(--ink2);}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--paper);}
.seg button:focus-visible,.tabs button:focus-visible,.iconbtn:focus-visible,select:focus-visible{outline:2px solid var(--orange);outline-offset:2px;}
.iconbtn{all:unset;cursor:pointer;font-size:13px;padding:5px 10px;border:1px solid var(--ink2);border-radius:4px;}
select{font:13px 'IBM Plex Sans',sans-serif;color:var(--ink);background:var(--card);border:1px solid var(--ink2);border-radius:4px;padding:4px 6px;}
/* hero: the regime tape */
.hero{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(0,1fr);gap:26px;margin:22px 0 10px;align-items:start;}
@media (max-width:980px){.hero{grid-template-columns:1fr;}}
.call{font-size:clamp(16px,1.45vw,19px);line-height:1.5;margin:0 0 14px;max-width:none;}
.call b{font-weight:600;}
.tapehead{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px;}
.tape{display:grid;grid-template-columns:repeat(var(--n,36),minmax(0,1fr));gap:2px;}
.tile{height:58px;border-radius:2px;position:relative;display:flex;align-items:flex-end;justify-content:center;padding-bottom:4px;font:500 11px 'IBM Plex Mono',monospace;color:var(--onreg);}
.tile.fcst{background-image:repeating-linear-gradient(135deg,rgba(255,255,255,.18) 0 3px,transparent 3px 7px);}
.tile .p{position:absolute;top:4px;left:0;right:0;text-align:center;font-size:9.5px;opacity:.95;}
.tile.empty{background:var(--tint);color:var(--ink2);}
.tapeyears{display:grid;grid-template-columns:repeat(var(--n,36),minmax(0,1fr));gap:2px;margin-top:4px;font-size:11px;color:var(--ink2);}
.tapeyears div{white-space:nowrap;overflow:visible;}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:10px;font-size:12.5px;color:var(--ink2);}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px;}
.outlook table{width:100%;border-collapse:collapse;font-size:13px;}
.outlook th{font-weight:500;color:var(--ink2);text-align:right;padding:4px 6px;border-bottom:1px solid var(--rule);}
.outlook th:first-child,.outlook td:first-child{text-align:left;}
.outlook td{text-align:right;padding:4px 6px;border-bottom:1px solid var(--rule);font-family:'IBM Plex Mono',monospace;}
.outlook td:first-child{font-family:'IBM Plex Sans',sans-serif;}
.outlook tr.grp td{font-family:'Space Grotesk',sans-serif;font-weight:600;padding-top:10px;color:var(--ink);border-bottom:1px solid var(--ink2);}
.outlook tr.mdl td{color:var(--plum);font-weight:500;}
.findings{margin:18px 0 6px;padding:0;list-style:none;columns:2 420px;column-gap:34px;}
.findings li{break-inside:avoid;margin:0 0 12px;padding-left:12px;border-left:3px solid var(--rule);font-size:14px;line-height:1.5;}
.findings li.k-call,.findings li.k-agree{border-left-color:var(--plum);}
.findings li.k-neg{border-left-color:var(--orange);}
/* tabs */
nav.tabs{display:flex;gap:2px;flex-wrap:wrap;margin:22px 0 0;border-bottom:1px solid var(--ink2);}
.tabs button{all:unset;cursor:pointer;padding:9px 14px 8px;font:500 14px 'Space Grotesk',sans-serif;color:var(--ink2);border:1px solid transparent;border-bottom:none;border-radius:4px 4px 0 0;margin-bottom:-1px;}
.tabs button[aria-selected="true"]{color:var(--ink);background:var(--card);border-color:var(--ink2);}
.panel{display:none;background:var(--card);border:1px solid var(--ink2);border-top:none;padding:18px clamp(12px,2vw,22px) 22px;}
.panel.on{display:block;}
.phead{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;margin-bottom:12px;}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,540px),1fr));gap:18px 22px;}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:18px 22px;}
.box{border-top:1px solid var(--rule);padding-top:12px;}
.chart{position:relative;height:330px;}
.chart.tall{height:420px;}
.chart.short{height:250px;}
.note{font-size:12.5px;color:var(--ink2);margin-top:6px;max-width:none;}
/* GRID table */
.gwrap{overflow-x:auto;border:1px solid var(--rule);}
table.grid{border-collapse:separate;border-spacing:0;font:12px/1.2 'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums;}
table.grid th,table.grid td{padding:4px 5px;text-align:center;white-space:nowrap;border-bottom:1px solid var(--rule);min-width:46px;}
table.grid thead th{position:sticky;top:0;background:var(--card);font:500 11.5px 'IBM Plex Sans',sans-serif;color:var(--ink2);z-index:2;}
table.grid th:first-child,table.grid td:first-child{position:sticky;left:0;z-index:3;background:var(--card);text-align:left;font:13px 'IBM Plex Sans',sans-serif;min-width:250px;max-width:280px;border-right:1px solid var(--ink2);}
table.grid td.fc,table.grid th.fc{background:var(--fc);}
table.grid td.fc0,table.grid th.fc0{border-left:2px solid var(--ink);}
table.grid tr.sec td{font:600 13.5px 'Space Grotesk',sans-serif;color:var(--ink);background:var(--tint);border-bottom:1px solid var(--ink2);padding-top:7px;}
table.grid tr.sec td:first-child{background:var(--tint);}
table.grid tr.b td:first-child{font-weight:600;}
table.grid tr.b td{font-weight:500;}
table.grid td.rg[data-r]{color:var(--onreg);font-weight:500;}
table.grid td.rg[data-r="G"]{background:var(--g);} table.grid td.rg[data-r="R"]{background:var(--r);}
table.grid td.rg[data-r="I"]{background:var(--i);} table.grid td.rg[data-r="D"]{background:var(--d);}
.lab-model{color:var(--plum);font-weight:500;} .lab-m42{color:var(--slate);font-weight:500;} .lab-bb{color:var(--orange);font-weight:500;} .lab-spf{color:var(--ochre);font-weight:500;}
td.sub{padding-left:22px !important;color:var(--ink2);}
/* generic tables */
table.t{border-collapse:collapse;width:100%;font-size:13px;}
table.t th{font-weight:500;color:var(--ink2);text-align:right;padding:6px 8px;border-bottom:1px solid var(--ink2);vertical-align:bottom;}
table.t td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--rule);font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums;}
table.t th:first-child,table.t td:first-child{text-align:left;font-family:'IBM Plex Sans',sans-serif;}
table.t td.best{font-weight:600;color:var(--ink);background:var(--tint);}
table.t td.txt{text-align:left;font-family:'IBM Plex Sans',sans-serif;}
.twrap{overflow-x:auto;}
.stat{display:flex;gap:22px;flex-wrap:wrap;margin:4px 0 8px;}
.stat div{font-size:12.5px;color:var(--ink2);}
.stat b{display:block;font:500 18px 'IBM Plex Mono',monospace;color:var(--ink);}
.method h3{margin-top:18px;}
.method ul{margin:4px 0 10px;padding-left:20px;max-width:80ch;}
.method li{margin-bottom:4px;}
.kbd{font-family:'IBM Plex Mono',monospace;font-size:12.5px;background:var(--tint);padding:1px 5px;border-radius:3px;}
[hidden]{display:none !important;}
.rightcol{display:flex;flex-direction:column;gap:8px;align-items:flex-end;}
@media (max-width:760px){.rightcol{align-items:flex-start;}}
.dlrow{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end;}
.seg button[disabled],.iconbtn[disabled]{opacity:.55;cursor:progress;}
#dlstatus{min-height:1.2em;}
footer{margin-top:26px;font-size:12px;color:var(--ink2);}
@media (prefers-reduced-motion: reduce){*{transition:none !important;animation:none !important;}}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div>
      <h1>US GRID Models</h1>
      <p class="lede">Real GDP and inflation forecasts rebuilt from 42 Macro's GRID process: a stationary mean-reversion model and a 27-agent nowcast ensemble per variable, split down the middle, then mapped to the four growth and inflation regimes.</p>
      <div class="stamp" id="stamp"></div>
    </div>
    <div class="rightcol">
      <div class="ctrls">
        <span class="small muted">Regime inflation measure</span>
        <div class="seg" role="group" aria-label="Inflation measure for the regime">
          <button type="button" data-spec="pce" aria-pressed="true">Core PCE</button>
          <button type="button" data-spec="cpi" aria-pressed="false">Headline CPI</button>
        </div>
        <button type="button" class="iconbtn" id="themebtn" aria-label="Toggle colour theme">Theme: auto</button>
      </div>
      <div class="dlrow dlui" hidden>
        <span class="small muted">Download the data</span>
        <div class="seg" role="group" aria-label="Download the data">
          <button type="button" data-dl="xlsx">Excel workbook</button>
          <button type="button" data-dl="csv">Monthly CSV</button>
          <button type="button" data-dl="json">JSON</button>
        </div>
      </div>
      <span class="small muted dlui" id="dlstatus" aria-live="polite" hidden></span>
    </div>
  </header>

  <section class="hero" aria-label="Current call">
    <div>
      <p class="call" id="call"></p>
      <div class="tapehead">
        <h3>Bottom-up regime, actual and projected</h3>
        <span class="small muted" id="tapenote"></span>
      </div>
      <div class="tape" id="tape"></div>
      <div class="tapeyears" id="tapeyears"></div>
      <div class="legend" id="legend"></div>
    </div>
    <div class="outlook">
      <h3 id="outlook-h">Outlook versus 42 Macro and consensus</h3>
      <div class="twrap"><table id="outlook"></table></div>
      <p class="note" id="outlook-n"></p>
    </div>
  </section>

  <ul class="findings" id="findings"></ul>

  <nav class="tabs" role="tablist" aria-label="Dashboard sections">
    <button type="button" role="tab" aria-selected="true" data-tab="table">GRID table</button>
    <button type="button" role="tab" aria-selected="false" data-tab="fc">Forecasts</button>
    <button type="button" role="tab" aria-selected="false" data-tab="reg">Regimes</button>
    <button type="button" role="tab" aria-selected="false" data-tab="ag">Nowcast agents</button>
    <button type="button" role="tab" aria-selected="false" data-tab="sm">Mean reversion</button>
    <button type="button" role="tab" aria-selected="false" data-tab="tr">Track record</button>
    <button type="button" role="tab" aria-selected="false" data-tab="me">Method</button>
  </nav>

  <section class="panel on" id="p-table" role="tabpanel">
    <div class="phead">
      <div>
        <h2>The GRID table</h2>
        <p class="muted small">Same layout and scorecard definitions as 42 Macro's table. Shaded columns are projections; history rows show what the model would have forecast one release ahead. Values in % YoY; errors in basis points over a trailing ten years.</p>
      </div>
      <button type="button" class="iconbtn dlui" data-dl="table" hidden>Download this table (CSV)</button>
    </div>
    <div class="gwrap"><table class="grid" id="gridtbl"></table></div>
    <p class="note">Regime letters: G Goldilocks (growth up, inflation down), R Reflation (both up), I Inflation (growth down, inflation up), D Deflation (both down), from 3-month changes in Real GDP YoY and the selected inflation measure. SPF directional accuracy is not shown for inflation because quarterly averages repeat across months.</p>
  </section>

  <section class="panel" id="p-fc" role="tabpanel">
    <div class="phead">
      <div><h2>Forecast paths</h2><p class="muted small">Solid plum: the model's one-release-ahead forecasts in real time. Dashed plum: the current projection, with its two sub-models dotted.</p></div>
      <div class="seg" role="group" aria-label="Chart range">
        <button type="button" data-range="2008-01" aria-pressed="false">Since 2008</button>
        <button type="button" data-range="2018-01" aria-pressed="false">Since 2018</button>
        <button type="button" data-range="2022-01" aria-pressed="true">Since 2022</button>
      </div>
    </div>
    <div class="grid2">
      <div class="box"><h3>Real GDP, % YoY</h3><div class="chart tall"><canvas id="ch-gdp" aria-label="Real GDP forecast chart" role="img"></canvas></div><p class="note" id="clip-gdp"></p></div>
      <div class="box"><h3>Core PCE, % YoY</h3><div class="chart tall"><canvas id="ch-pce" aria-label="Core PCE forecast chart" role="img"></canvas></div></div>
      <div class="box"><h3>Headline CPI, % YoY</h3><div class="chart tall"><canvas id="ch-cpi" aria-label="Headline CPI forecast chart" role="img"></canvas></div></div>
      <div class="box"><h3>Reading the paths</h3><div id="fcnotes" class="small"></div></div>
    </div>
  </section>

  <section class="panel" id="p-reg" role="tabpanel">
    <div class="phead"><div><h2>Regime probabilities</h2><p class="muted small">Direction probabilities are calibrated on the model's own out-of-sample record (logistic fit by horizon bucket) and combined as independent growth and inflation calls, as in 42 Macro's 2023 workbook.</p></div></div>
    <div class="grid2">
      <div class="box"><h3>Probability of each regime</h3><div class="chart"><canvas id="ch-prob" role="img" aria-label="Regime probabilities"></canvas></div></div>
      <div class="box"><h3>Probability growth and inflation are accelerating</h3><div class="chart"><canvas id="ch-pdir" role="img" aria-label="Direction probabilities"></canvas></div></div>
      <div class="box"><h3>Rolling 10-year hit rate (42 Macro definition)</h3><div class="chart"><canvas id="ch-hit" role="img" aria-label="Rolling hit rates"></canvas></div><p class="note" id="hitnote"></p></div>
      <div class="box"><h3>Forward regime calls by horizon</h3><div class="chart"><canvas id="ch-fwd" role="img" aria-label="Forward hit rate by horizon"></canvas></div><p class="note">Share of months where the regime projected at each origin matches the regime that occurred, versus assuming today's regime persists. Since 2005.</p></div>
      <div class="box"><h3>Where the model goes wrong</h3><div class="twrap" id="conf"></div><p class="note">Rows: actual regime. Columns: Econometric Model regime (42 Macro definition), monthly since 2005.</p></div>
      <div class="box"><h3>Calibration by horizon</h3><div class="twrap" id="calib"></div></div>
    </div>
  </section>

  <section class="panel" id="p-ag" role="tabpanel">
    <div class="phead">
      <div><h2>Agent-based nowcast</h2><p class="muted small">Each agent is a first-difference regression of the change in the target's growth rate on the 3-month change in one feature, retrained monthly on a rolling window. Weights are each agent's share of total R², the rule stated in 42 Macro's workbook.</p></div>
      <div class="ctrls">
        <div class="seg" role="group" aria-label="Agent target">
          <button type="button" data-agt="gdp" aria-pressed="false">Real GDP</button>
          <button type="button" data-agt="pce" aria-pressed="true">Core PCE</button>
          <button type="button" data-agt="cpi" aria-pressed="false">Headline CPI</button>
        </div>
        <label class="small muted" for="agh">Horizon</label> <select id="agh"></select>
      </div>
    </div>
    <div class="grid2">
      <div class="box"><h3 id="agw-h">Current weights versus 42 Macro's Aug-2023 weights</h3><div class="chart tall"><canvas id="ch-agw" role="img" aria-label="Agent weights"></canvas></div></div>
      <div class="box"><h3>Weights over time</h3><div class="chart tall"><canvas id="ch-agh" role="img" aria-label="Agent weight history"></canvas></div><p class="note" id="aghnote"></p></div>
    </div>
    <div class="box" style="margin-top:16px"><h3>Agent detail</h3><div class="twrap" id="agtbl"></div></div>
  </section>

  <section class="panel" id="p-sm" role="tabpanel">
    <div class="phead"><div><h2>Stationary mean reversion</h2><p class="muted small">Annualized growth reverts from its recent run-rate to the trailing 10-year mean at the AR decay estimated over the prior 15 years. Integrating the path back to levels gives YoY forecasts with exact base effects.</p></div></div>
    <div class="grid3">
      <div class="box"><h3>Real GDP, q/q annualized</h3><div class="stat" id="st-gdp"></div><div class="chart short"><canvas id="ch-sm-gdp" role="img" aria-label="GDP mean reversion"></canvas></div></div>
      <div class="box"><h3>Core PCE, 3-month annualized</h3><div class="stat" id="st-pce"></div><div class="chart short"><canvas id="ch-sm-pce" role="img" aria-label="Core PCE mean reversion"></canvas></div></div>
      <div class="box"><h3>Headline CPI, 3-month annualized</h3><div class="stat" id="st-cpi"></div><div class="chart short"><canvas id="ch-sm-cpi" role="img" aria-label="CPI mean reversion"></canvas></div></div>
    </div>
  </section>

  <section class="panel" id="p-tr" role="tabpanel">
    <div class="phead"><div><h2>Track record</h2><p class="muted small" id="tr-intro">Everything is scored against today's data vintage, so 42 Macro's and Bloomberg's numbers are rescored too. Best value in each column is shaded.</p></div>
      <div class="seg" role="group" aria-label="COVID treatment"><button type="button" data-cov="0" aria-pressed="true">Include COVID</button><button type="button" data-cov="1" aria-pressed="false">Exclude 2020–21</button></div>
    </div>
    <div class="box"><h3 id="scA-h"></h3><div class="twrap" id="scA"></div><p class="note" id="scA-n"></p></div>
    <div class="box" style="margin-top:14px"><h3 id="scB-h"></h3><div class="twrap" id="scB"></div><p class="note" id="scB-n"></p></div>
    <div class="box" style="margin-top:14px"><h3>Full backtest since 2005</h3><div class="twrap" id="scC"></div></div>
    <div class="grid3" style="margin-top:16px">
      <div class="box"><h3>Real GDP error by horizon</h3><div class="chart short"><canvas id="ch-hz-gdp" role="img" aria-label="GDP error by horizon"></canvas></div><p class="note">Quarters ahead from a mid-quarter origin, the SPF survey's timing.</p></div>
      <div class="box"><h3>Core PCE error by horizon</h3><div class="chart short"><canvas id="ch-hz-pce" role="img" aria-label="Core PCE error by horizon"></canvas></div><p class="note">Months ahead of the latest print.</p></div>
      <div class="box"><h3>Headline CPI error by horizon</h3><div class="chart short"><canvas id="ch-hz-cpi" role="img" aria-label="CPI error by horizon"></canvas></div><p class="note">Months ahead of the latest print.</p></div>
    </div>
  </section>

  <section class="panel method" id="p-me" role="tabpanel">
    <h2>How this was built</h2>
    <div class="privonly">
    <h3>What the workbooks revealed</h3>
    <p>The Aug-2023 file still carries Excel's cached copy of its links into Darius Dale's master workbook, <span class="kbd">42 Macro Regime Forecasting Tool.xlsx</span>: about 1.2 million cells, including the US histories of all 54 nowcast features with their weights, regime and probability histories, and the in-house framework backtests. That cache documents the process:</p>
    <ul>
      <li>A Stationary Mean Reversion Model and an "Agent-Based Nowcast Model" project growth and inflation twelve months out. The median projection "splits the difference" between them.</li>
      <li>Nowcast feature weights are "dynamically retrained monthly according to the relative strength of the first difference regressions between the independent and the dependent variables."</li>
      <li>Regime probabilities multiply independent growth and inflation direction probabilities. Jan-22 Goldilocks: 0.371 × 0.121 = 0.045, an exact match.</li>
      <li>Targets changed over time: the OECD leading indicator and headline CPI (2023), Real GDP and headline CPI (Nov-2024), Real GDP and core PCE (Sep-2026).</li>
    </ul>
    </div>
    <div class="pubonly" hidden>
    <h3>Structure</h3>
    <ul>
      <li>Two sub-models project growth and inflation: a stationary mean-reversion model and an agent-based nowcast ensemble whose weights are retrained monthly from first-difference regressions. The median projection is their 50/50 average.</li>
      <li>Regime probabilities multiply independent growth and inflation direction probabilities.</li>
      <li>This public build carries no 42 Macro or Bloomberg forecasts, hit rates or model weights; only the method is reconstructed.</li>
    </ul>
    </div>
    <h3>Scorecard definitions</h3>
    <ul>
      <li>Actual regime: sign of the 3-month change in Real GDP YoY (quarterly values repeated across months) and in inflation YoY, with a zero change counted as down. Matched 210 of 210 months.</li>
      <li>Forecast regime: the same rule applied to the forecaster's own forecast series. Matched 222 of 222 months.</li>
      <li>Hit rate and average absolute forecast error: trailing 120-month means. Exact matches.</li>
      <li>Directional accuracy: sign agreement of 1-month changes for CPI (exact); 3-month changes for GDP (within about 2 points).</li>
      <li>Z-scores: each series against its own 2015–19 mean and standard deviation.</li>
    </ul>
    <h3>This reconstruction</h3>
    <ul id="spec"></ul>
    <h3>Substitutions for proprietary inputs</h3>
    <ul>
      <li>ISM manufacturing and services: regional Fed survey composites built from averaged 3-month changes, so a survey joining the panel does not create a jump. The free ISM feed stopped updating in Jan-2026.</li>
      <li>ISM prices and NFIB higher prices: regional Fed prices-paid and prices-received composites.</li>
      <li>CRB spot indexes: IMF all-commodity, food and industrial-materials indexes.</li>
      <li>MBA purchase index: new home sales. BLS employment diffusion (industries): the 1-month diffusion index across states. OECD confidence (ended on FRED Jan-2024): regional Fed 6-month-ahead expectations and Michigan sentiment.</li>
      <li>Five CPI sub-indices not published monthly on FRED: BLS public API, with size-class A indexes (new and used vehicles for transportation goods) when the API is unavailable. <span id="blsnote"></span></li>
      <li>Bloomberg consensus: the Philadelphia Fed SPF median, converted to YoY by chaining its forecast growth onto actual growth for earlier quarters.<span class="privonly"> Bloomberg history comes from the Nov-2024 file and the Sep-2026 screenshot.</span></li>
    </ul>
    <h3>Caveats</h3>
    <ul>
      <li>Backtests use today's data vintage, not real-time vintages. GDP is revised heavily, so real-time GDP errors would be larger; headline CPI is not revised.</li>
      <li>The ragged edge of each feature's release lag is fixed at today's pattern and applied historically.</li>
      <li>October 2025 CPI was never collected because of the government shutdown; the hole is linearly interpolated.</li>
      <li>42 Macro's history before its 2020 launch is almost certainly backfilled model output, not published forecasts.</li>
      <li>The Sep-2026 numbers were read from a screenshot at 0.1pp resolution.</li>
    </ul>
    <h3>Run log</h3>
    <pre class="small mono" id="runlog" style="white-space:pre-wrap;color:var(--ink2);max-height:220px;overflow:auto;background:var(--tint);padding:10px;border-radius:4px;"></pre>
  </section>

  <footer id="foot"></footer>
</div>

<script id="payload" type="application/json">__GRID_PAYLOAD__</script>
<script>
(function () {
  'use strict';
  var D = JSON.parse(document.getElementById('payload').textContent);
  var PUB = !!(D.meta && D.meta.public);
  var AX = D.axis, IX = {};
  AX.forEach(function (m, i) { IX[m] = i; });
  var RN = { G: 'Goldilocks', R: 'Reflation', I: 'Inflation', D: 'Deflation' };
  var RD = { G: 'growth up, inflation down', R: 'growth and inflation up', I: 'growth down, inflation up', D: 'growth and inflation down' };
  var MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var TNAME = { gdp: 'Real GDP', pce: 'Core PCE', cpi: 'Headline CPI' };
  var st = { spec: 'pce', range: '2022-01', agT: 'pce', agH: 3, cov: false, tab: 'table' };
  var charts = {};
  try { var saved = localStorage.getItem('gridTheme'); if (saved === 'light' || saved === 'dark') document.documentElement.setAttribute('data-theme', saved); } catch (e) { }

  function cv(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
  function C() {
    return { ink: cv('--ink'), ink2: cv('--ink2'), rule: cv('--rule'), card: cv('--card'), paper: cv('--paper'), tint: cv('--tint'),
      orange: cv('--orange'), teal: cv('--teal'), ochre: cv('--ochre'), plum: cv('--plum'), slate: cv('--slate'),
      G: cv('--g'), R: cv('--r'), I: cv('--i'), D: cv('--d') };
  }
  function rgba(hex, a) {
    var h = (hex || '#888888').replace('#', '');
    if (h.length === 3) h = h.split('').map(function (c) { return c + c; }).join('');
    var n = parseInt(h, 16);
    return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
  }
  function ok(x) { return x !== null && x !== undefined && isFinite(x); }
  function ml(m) { var p = m.split('-'); return MON[+p[1] - 1] + '-' + p[0].slice(2); }
  function f(x, d) { return ok(x) ? (+x).toFixed(d === undefined ? 1 : d) : ''; }
  function pc(x, d) { return ok(x) ? (100 * x).toFixed(d === undefined ? 0 : d) + '%' : ''; }
  function bp(x) { return ok(x) ? Math.round(x) + 'bp' : ''; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function el(id) { return document.getElementById(id); }

  // ------------------------------------------------------------------ chart helpers
  var vline = {
    id: 'vline',
    afterDatasetsDraw: function (ch, args, o) {
      if (!o || o.index === undefined || o.index === null || o.index < 0) return;
      var xs = ch.scales.x; if (!xs) return;
      var x = xs.getPixelForValue(o.index), a = ch.chartArea, ctx = ch.ctx;
      if (!isFinite(x) || x < a.left || x > a.right) return;
      ctx.save(); ctx.strokeStyle = o.color || '#888888'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, a.top); ctx.lineTo(x, a.bottom); ctx.stroke();
      if (o.label) { ctx.setLineDash([]); ctx.fillStyle = o.color || '#888888'; ctx.font = "11px 'IBM Plex Sans', sans-serif"; ctx.fillText(o.label, x + 4, a.top + 11); }
      ctx.restore();
    }
  };
  var hline = {
    id: 'hline',
    afterDatasetsDraw: function (ch, args, o) {
      if (!o || !o.lines) return;
      var ys = ch.scales.y, a = ch.chartArea, ctx = ch.ctx;
      o.lines.forEach(function (L) {
        var y = ys.getPixelForValue(L.v); if (!isFinite(y) || y < a.top || y > a.bottom) return;
        ctx.save(); ctx.strokeStyle = L.color || '#888888'; ctx.lineWidth = 1; ctx.setLineDash(L.dash || [4, 4]);
        ctx.beginPath(); ctx.moveTo(a.left, y); ctx.lineTo(a.right, y); ctx.stroke();
        if (L.label) { ctx.setLineDash([]); ctx.fillStyle = L.color || '#888888'; ctx.font = "11px 'IBM Plex Sans', sans-serif"; ctx.fillText(L.label, a.left + 4, y - 4); }
        ctx.restore();
      });
    }
  };
  if (window.Chart) { Chart.register(vline, hline); }

  function setDefaults() {
    if (!window.Chart) return;
    var c = C();
    Chart.defaults.font.family = "'IBM Plex Sans', system-ui, sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.color = c.ink2;
    Chart.defaults.borderColor = rgba(c.rule, 1);
  }
  function mk(id, cfg) {
    if (!window.Chart) return;
    if (charts[id]) { charts[id].destroy(); delete charts[id]; }
    var cnv = el(id); if (!cnv) return;
    charts[id] = new Chart(cnv, cfg);
  }
  function ds(label, data, color, o) {
    var d = { label: label, data: data, borderColor: color, backgroundColor: color, pointRadius: 0, pointHoverRadius: 3, borderWidth: 1.6, spanGaps: false, tension: 0 };
    if (o) Object.keys(o).forEach(function (k) { d[k] = o[k]; });
    return d;
  }
  function hasData(d) { return d.data && d.data.some(ok); }
  function lineCfg(labels, datasets, o) {
    o = o || {}; var c = C();
    return {
      type: 'line', data: { labels: labels, datasets: datasets.filter(hasData) },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 16, boxHeight: 2, padding: 10 } },
          tooltip: { callbacks: { label: function (t) { return t.dataset.label + ': ' + (ok(t.parsed.y) ? (+t.parsed.y).toFixed(o.dec === undefined ? 2 : o.dec) + (o.unit || '') : 'n/a'); } } },
          vline: o.vline || {}, hline: o.hline || {}
        },
        scales: {
          x: { ticks: { maxTicksLimit: o.xt || 9, autoSkip: true, maxRotation: 0 }, grid: { display: false } },
          y: { min: o.ymin, max: o.ymax, ticks: { callback: function (v) { return (+v).toFixed(o.ydec === undefined ? 1 : o.ydec) + (o.yunit || ''); } }, grid: { color: rgba(c.rule, 0.7) } }
        }
      }
    };
  }

  // ------------------------------------------------------------------ header, call, tape, outlook, findings
  function renderStamp() {
    var m = D.meta;
    el('stamp').innerHTML = '<span>Real GDP to <b>' + m.asof.gdp + '</b></span><span>Headline CPI to <b>' + m.asof.cpi + '</b></span><span>Core PCE to <b>' + m.asof.pce +
      '</b></span><span>Consensus <b>SPF ' + (m.spf_survey || 'n/a') + '</b></span><span>Built <b>' + m.generated_utc + '</b></span>';
    el('outlook-n').innerHTML = 'Quarterly averages, % YoY. ' + (PUB ? '' : '42 Macro and Bloomberg are transcribed from the Sep-2026 screenshot (0.1pp); they are static and drop out once the projection passes their horizon. ') +
      'SPF is the Philadelphia Fed median from the ' + esc(m.spf_survey || 'latest') + ' survey.';
    el('outlook-h').textContent = PUB ? 'Outlook versus the professional consensus' : 'Outlook versus 42 Macro and consensus';
    el('foot').textContent = 'Data: FRED, BLS, Philadelphia Fed SPF' + (PUB ? '' : '; 42 Macro reference values from the Nov-2024 workbook and Sep-2026 screenshot') + '. Regenerated by rebuild.py.';
  }
  function renderCall() {
    var calls = D.calls[st.spec], last = D.last_reg[st.spec];
    var lastR = D.REG[st.spec].act[IX[last]];
    var s = 'Latest confirmed regime: <b>' + RN[lastR] + '</b> (' + ml(last) + ', ' + (st.spec === 'pce' ? 'core PCE' : 'headline CPI') + ' spec). Projected: ';
    s += calls.map(function (c) { return '<b>' + RN[c.r] + '</b> ' + c.a + (c.b !== c.a ? ' to ' + c.b : '') + ' <span class="muted">(' + Math.round(c.p * 100) + '%)</span>'; }).join(', then ');
    el('call').innerHTML = s + '.';
  }
  function renderTape() {
    var c = C(), spec = st.spec, P = D.probs[spec];
    var p0 = IX[P.months[0]] !== undefined ? IX[P.months[0]] : 0;
    var start = Math.max(0, p0 - 18), end = AX.length - 1, n = end - start + 1;
    var pm = {}; P.months.forEach(function (m, k) { pm[m] = k; });
    var html = '', yrs = '';
    for (var i = start; i <= end; i++) {
      var m = AX[i], ra = D.REG[spec].act[i], k = pm[m];
      if (ra && k === undefined) {
        html += '<div class="tile" style="background:' + c[ra] + '" title="' + ml(m) + ': ' + RN[ra] + ' (actual)">' + ra + '</div>';
      } else if (k !== undefined) {
        var r = P.modal[k], p = P[r][k];
        var tip = ml(m) + ' projected ' + RN[r] + ' | G ' + Math.round(P.G[k] * 100) + '%, R ' + Math.round(P.R[k] * 100) + '%, I ' + Math.round(P.I[k] * 100) + '%, D ' + Math.round(P.D[k] * 100) + '%';
        html += '<div class="tile fcst" style="background:' + rgba(c[r], 0.38 + 0.62 * p) + '" title="' + tip + '"><span class="p">' + Math.round(p * 100) + '</span>' + r + '</div>';
      } else {
        html += '<div class="tile empty" title="' + ml(m) + '">\u00b7</div>';
      }
      var mo = +m.split('-')[1];
      yrs += '<div>' + (mo === 1 ? m.slice(0, 4) : (mo === 7 ? 'Jul' : '')) + '</div>';
    }
    var t = el('tape'); t.style.setProperty('--n', n); t.innerHTML = html;
    var y = el('tapeyears'); y.style.setProperty('--n', n); y.innerHTML = yrs;
    el('legend').innerHTML = ['G', 'R', 'I', 'D'].map(function (r) { return '<span><i style="background:' + c[r] + '"></i>' + RN[r] + ': ' + RD[r] + '</span>'; }).join('') +
      '<span><i style="background:' + rgba(c.ink2, 0.45) + ';background-image:repeating-linear-gradient(135deg,rgba(255,255,255,.35) 0 3px,transparent 3px 7px)"></i>Hatched: projection; number = probability %</span>';
    el('tapenote').textContent = 'Projected tiles use the forecast path; shading follows the calibrated probability.';
  }
  function renderOutlook() {
    var O = D.outlook.slice(0, 5);
    var h = '<thead><tr><th></th>' + O.map(function (o) { return '<th>' + o.q + '</th>'; }).join('') + '</tr></thead><tbody>';
    function grp(t) { h += '<tr class="grp"><td colspan="' + (O.length + 1) + '">' + t + '</td></tr>'; }
    function row(lab, key, cls) {
      if (!O.some(function (o) { return ok(o[key]); })) return;
      h += '<tr class="' + (cls || '') + '"><td>' + lab + '</td>' + O.map(function (o) { return '<td>' + f(o[key], 1) + '</td>'; }).join('') + '</tr>';
    }
    grp('Real GDP'); row('Econometric Model', 'gdp_med', 'mdl'); row('42 Macro', 'gdp_42'); row('Bloomberg', 'gdp_bb'); row('SPF', 'gdp_spf');
    grp('Core PCE'); row('Econometric Model', 'pce_med', 'mdl'); row('42 Macro', 'pce_42'); row('Bloomberg', 'pce_bb'); row('SPF', 'pce_spf');
    grp('Headline CPI'); row('Econometric Model', 'cpi_med', 'mdl'); row('SPF', 'cpi_spf');
    el('outlook').innerHTML = h + '</tbody>';
  }
  function renderFindings() {
    el('findings').innerHTML = D.findings.map(function (x) { return '<li class="k-' + x.k + '">' + x.t + '</li>'; }).join('');
  }

  // ------------------------------------------------------------------ GRID table
  function gridSpec(spec) {
    var S = D.S, R = [];
    var fcG = IX[D.meta.fc_start.gdp];
    var fcT = { gdp: fcG, cpi: IX[D.meta.fc_start.cpi], pce: IX[D.meta.fc_start.pce] };
    var lastReg = IX[D.last_reg[spec]];
    var secName = '', parent = '';
    function sec(t) { secName = t; R.push({ type: 'sec', label: t }); }
    function reg(lab, arr, lc, upto) { parent = lab; R.push({ type: 'reg', label: lab, arr: arr, lc: lc || '', upto: upto, fc: fcG, xl: secName + ' | ' + lab }); }
    function val(lab, arr, kind, o) {
      o = o || {}; if (!o.sub) parent = lab;
      R.push({ type: 'val', label: lab, arr: arr, kind: kind, o: o, fc: o.fc === undefined ? fcG : o.fc, upto: o.upto, xl: secName + ' | ' + (o.sub ? parent + ' | ' : '') + lab });
    }
    var m42lab = spec === 'pce' ? '42 Macro (Sep-26 screenshot)' : '42 Macro (Nov-24 file)';
    sec('Regime');
    reg('Bottom-up macro regime (actual)', D.REG[spec].act, '', lastReg);
    reg('Econometric Model', D.REG[spec].model, 'lab-model');
    val('Hit rate (t10y; luck = 25%)', D.HIT[spec].model, 'hit', { upto: lastReg, sub: true });
    if (D.REG[spec].m42) {
      reg(m42lab, D.REG[spec].m42, 'lab-m42');
      val('Hit rate, as reported', D.HIT[spec].m42, 'hit', { sub: true });
    }
    reg('SPF consensus', D.REG[spec].spf, 'lab-spf');
    val('Hit rate (t10y)', D.HIT[spec].spf, 'hit', { upto: lastReg, sub: true });
    function block(t, title) {
      var s = S[t], infl = t !== 'gdp', fc = fcT[t], la = fc - 1, lag = t === 'gdp' ? '3-month' : '1-month';
      sec(title);
      val(title, s.act, 'v', { bold: true, fc: fc, upto: la });
      val('Z-score (vs 2015\u201319)', s.z_act, 'z', { infl: infl, fc: fc, upto: la, sub: true });
      val('Econometric Model', s.mf, 'v', { lc: 'lab-model', fc: fc });
      val('Z-score', s.z_mf, 'z', { infl: infl, fc: fc, sub: true });
      val('Directional accuracy (t10y; ' + lag + ')', s.da, 'da', { fc: fc, upto: la, sub: true });
      val('Avg absolute forecast error (t10y)', s.aafe, 'ae', { fc: fc, upto: la, sub: true });
      val('SPF median', s.spf, 'v', { lc: 'lab-spf', fc: fc });
      val('Z-score', s.z_spf, 'z', { infl: infl, fc: fc, sub: true });
      if (!infl) val('Directional accuracy (t10y)', s.da_spf, 'da', { fc: fc, upto: la, sub: true });
      val('Avg absolute forecast error (t10y)', s.aafe_spf, 'ae', { fc: fc, upto: la, sub: true });
      if (s.m42s) val('42 Macro (Sep-26 screenshot)', s.m42s, 'v', { lc: 'lab-m42', fc: fc });
      if (s.bbs) val('Bloomberg (Sep-26 screenshot)', s.bbs, 'v', { lc: 'lab-bb', fc: fc });
      if (s.m42 && t !== 'gdp') val('42 Macro (Nov-24 file)', s.m42, 'v', { lc: 'lab-m42', fc: fc });
      if (s.bb && t !== 'gdp') val('Bloomberg (Nov-24 file)', s.bb, 'v', { lc: 'lab-bb', fc: fc });
    }
    block('gdp', 'Real GDP YoY');
    if (spec === 'pce') { block('pce', 'Core PCE YoY'); block('cpi', 'Headline CPI YoY'); }
    else { block('cpi', 'Headline CPI YoY'); block('pce', 'Core PCE YoY'); }
    var cols = []; for (var i = IX[D.table_start]; i <= AX.length - 1; i++) cols.push(i);
    return { rows: R, fcG: fcG, cols: cols };
  }
  function cellVal(r, i) {
    var v = r.arr ? r.arr[i] : null;
    if (r.upto !== undefined && i > r.upto) v = null;
    return (v === undefined) ? null : v;
  }
  function renderTable() {
    var c = C(), G = gridSpec(st.spec), cols = G.cols, fcG = G.fcG;
    function cl(i, fc) { return (i >= fc ? 'fc' : '') + (i === fcG ? ' fc0' : ''); }
    function shade(kind, v, infl, rowmax) {
      if (!ok(v)) return '';
      if (kind === 'z') {
        var hot = infl ? (v > 0) : (v < 0), col = hot ? c.orange : c.teal;
        return 'background:' + rgba(col, Math.min(Math.abs(v) / (infl ? 6 : 2.5), 1) * 0.55);
      }
      if (kind === 'da' || kind === 'hit') return 'background:' + rgba(c.slate, Math.max(0, Math.min(1, (v - (kind === 'hit' ? 0.25 : 0.5)) / 0.4)) * 0.5);
      if (kind === 'ae') return 'background:' + rgba(c.orange, Math.max(0, Math.min(1, v / (rowmax || 1))) * 0.35);
      return '';
    }
    var h = '<thead><tr><th>United States</th>' + cols.map(function (i) { return '<th class="' + cl(i, fcG) + '">' + ml(AX[i]) + '</th>'; }).join('') + '</tr></thead><tbody>';
    G.rows.forEach(function (r) {
      if (r.type === 'sec') {
        h += '<tr class="sec"><td>' + r.label + '</td>' + cols.map(function (i) { return '<td class="' + cl(i, fcG) + '"></td>'; }).join('') + '</tr>';
        return;
      }
      if (r.type === 'reg') {
        h += '<tr><td class="' + r.lc + '">' + r.label + '</td>' + cols.map(function (i) {
          var x = cellVal(r, i);
          return '<td class="rg ' + cl(i, fcG) + '"' + (x ? ' data-r="' + x + '" title="' + RN[x] + '"' : '') + '>' + (x || '') + '</td>';
        }).join('') + '</tr>';
        return;
      }
      var o = r.o, vals = cols.map(function (i) { return cellVal(r, i); });
      var rowmax = Math.max.apply(null, vals.filter(ok).concat([1]));
      h += '<tr class="' + (o.bold ? 'b' : '') + '"><td class="' + (o.lc || '') + (o.sub ? ' sub' : '') + '">' + r.label + '</td>' + cols.map(function (i, k) {
        var v = vals[k], txt = '';
        if (ok(v)) txt = (r.kind === 'v' || r.kind === 'z') ? f(v, 1) : (r.kind === 'ae' ? bp(v) : pc(v));
        return '<td class="' + cl(i, r.fc) + '" style="' + shade(r.kind, v, o.infl, rowmax) + '">' + txt + '</td>';
      }).join('') + '</tr>';
    });
    el('gridtbl').innerHTML = h + '</tbody>';
    var w = el('gridtbl').parentElement, th = el('gridtbl').querySelector('th.fc0');
    if (th && !w.dataset.scrolled) { w.scrollLeft = Math.max(0, th.offsetLeft - 520); w.dataset.scrolled = '1'; }
  }

  // ------------------------------------------------------------------ forecasts
  function anchored(arr, act, fc) {
    var a = arr.slice(); if (fc > 0 && ok(act[fc - 1]) && !ok(a[fc - 1])) a[fc - 1] = act[fc - 1]; return a;
  }
  function merged(a, b) { return AX.map(function (m, i) { return (b && ok(b[i])) ? b[i] : (a && ok(a[i]) ? a[i] : null); }); }
  function fcChart(t) {
    var S = D.S[t], c = C(), i0 = IX[st.range] || 0, fc = IX[D.meta.fc_start[t]];
    var lab = AX.slice(i0).map(ml);
    function sl(a) { return a ? a.slice(i0) : []; }
    var m42 = merged(S.m42, S.m42s), bb = merged(S.bb, S.bbs);
    var sets = [
      ds('Actual', sl(S.act), c.ink, { borderWidth: 2.2 }),
      ds('Econometric Model, one release ahead', sl(S.f1), rgba(c.plum, 0.6), { borderWidth: 1.2 }),
      ds('Econometric Model projection', sl(anchored(S.p_med, S.act, fc)), c.plum, { borderWidth: 2.6, borderDash: [7, 4] }),
      ds('Mean-reversion model', sl(anchored(S.p_smrm, S.act, fc)), rgba(c.plum, 0.55), { borderWidth: 1.2, borderDash: [2, 3] }),
      ds('Agent ensemble', sl(anchored(S.p_abnm, S.act, fc)), rgba(c.plum, 0.55), { borderWidth: 1.2, borderDash: [9, 3, 2, 3] }),
      ds('42 Macro', sl(m42), c.slate, { borderWidth: 1.6 }),
      ds('Bloomberg consensus', sl(bb), c.orange, { borderWidth: 1.3 }),
      ds('SPF median', sl(S.spf), c.ochre, { borderWidth: 1.3 })
    ];
    var clip = t === 'gdp' && st.range < '2021-07';
    mk('ch-' + t, lineCfg(lab, sets, { vline: { index: fc - i0, color: c.ink2, label: 'projection' }, yunit: '%', dec: 2, xt: 10, ymin: clip ? -4 : undefined, ymax: clip ? 8 : undefined }));
    var nt = el('clip-' + t); if (nt) nt.textContent = clip ? 'Axis clipped at \u22124% and 8%: the 2020\u201321 swing runs off the chart.' : '';
  }
  function spread(t) {
    var S = D.S[t], fc = IX[D.meta.fc_start[t]], last = null;
    for (var i = fc; i < AX.length; i++) if (ok(S.p_smrm[i]) && ok(S.p_abnm[i])) last = i;
    return last === null ? null : { m: AX[last], d: S.p_smrm[last] - S.p_abnm[last] };
  }
  function rel(d) { return Math.abs(d) < 0.05 ? 'in line with' : (d > 0 ? 'above' : 'below'); }
  function halfLife(s, unit) { return ok(s.hl) ? f(s.hl, 1) + ' ' + unit : 'under one ' + unit.replace(/s$/, ''); }
  function renderFcNotes() {
    var sm = D.smrm, O = D.outlook, g3 = O[Math.min(3, O.length - 1)], sp = spread('pce'), sc = spread('cpi');
    var h = '<p><b>Real GDP.</b> ' + O[0].q + ' is a nowcast built from monthly data through ' + D.meta.live.gdp_T + '. Mean reversion pulls growth toward its trailing mean of ' + f(sm.gdp.mu, 2) +
      '% annualized (half-life ' + halfLife(sm.gdp, 'quarters') + '); by ' + g3.q + ' it sits ' + rel(g3.gdp_smrm - g3.gdp_abnm) + ' the agent ensemble (' + f(g3.gdp_smrm, 1) + '% vs ' + f(g3.gdp_abnm, 1) + '%).</p>';
    h += '<p><b>Core PCE.</b> The 3-month run-rate is ' + f(sm.pce.x0, 2) + '% annualized against a stationary mean of ' + f(sm.pce.mu, 2) + '% (half-life ' + halfLife(sm.pce, 'months') + ').' +
      (sp ? ' By ' + ml(sp.m) + ' mean reversion sits ' + f(Math.abs(sp.d), 2) + 'pp ' + (sp.d >= 0 ? 'above' : 'below') + ' the agents.' : '') + '</p>';
    h += '<p><b>Headline CPI.</b> The 3-month run-rate is ' + f(sm.cpi.x0, 2) + '% annualized against a stationary mean of ' + f(sm.cpi.mu, 2) + '%. Year-on-year readings move with the base: large monthly prints from a year earlier drop out one at a time.' +
      (sc ? ' By ' + ml(sc.m) + ' the sub-models are ' + f(Math.abs(sc.d), 2) + 'pp apart.' : '') + '</p>';
    if (!PUB) h += '<p class="muted">42 Macro and Bloomberg lines join the Nov-2024 workbook history to the Sep-2026 screenshot and do not update; the headline CPI comparison stops in Oct-2025 because 42 Macro switched to core PCE.</p>';
    el('fcnotes').innerHTML = h;
  }

  // ------------------------------------------------------------------ regimes
  function renderRegimes() {
    var c = C(), P = D.probs[st.spec];
    var lab = P.months.map(ml);
    mk('ch-prob', {
      type: 'bar',
      data: { labels: lab, datasets: ['G', 'R', 'I', 'D'].map(function (r) { return { label: RN[r], data: P[r].map(function (x) { return 100 * x; }), backgroundColor: c[r], borderWidth: 0, stack: 's' }; }) },
      options: { responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: 'index', intersect: false },
        plugins: { legend: { position: 'bottom', labels: { boxWidth: 12 } }, tooltip: { callbacks: { label: function (t) { return t.dataset.label + ': ' + Math.round(t.parsed.y) + '%'; } } } },
        scales: { x: { stacked: true, grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 9 } }, y: { stacked: true, min: 0, max: 100, ticks: { callback: function (v) { return v + '%'; } }, grid: { color: rgba(c.rule, 0.7) } } } }
    });
    mk('ch-pdir', lineCfg(lab, [
      ds('Growth accelerating', P.pg.map(function (x) { return 100 * x; }), c.teal, { borderWidth: 2 }),
      ds('Inflation accelerating', P.pi.map(function (x) { return 100 * x; }), c.orange, { borderWidth: 2 })
    ], { yunit: '%', ydec: 0, dec: 0, unit: '%', ymin: 0, ymax: 100, hline: { lines: [{ v: 50, color: c.ink2, label: '50%' }] } }));
    var H = D.HIT[st.spec], i0 = IX['2014-12'] || 0, labh = AX.slice(i0).map(ml);
    function sl(a) { return a ? a.slice(i0).map(function (x) { return ok(x) ? 100 * x : null; }) : []; }
    var lr = IX[D.last_reg[st.spec]];
    function cut(a) { return a ? a.map(function (x, i) { return i <= lr ? x : null; }) : a; }
    mk('ch-hit', lineCfg(labh, [
      ds('Econometric Model', sl(cut(H.model)), c.plum, { borderWidth: 2.2 }),
      ds('SPF consensus', sl(cut(H.spf)), c.ochre, { borderWidth: 1.5 }),
      ds(st.spec === 'pce' ? '42 Macro (screenshot)' : '42 Macro (Nov-24 file)', sl(H.m42), c.slate, { borderWidth: 1.8 }),
      ds(st.spec === 'pce' ? 'Bloomberg (screenshot)' : 'Bloomberg (Nov-24 file)', sl(H.bb), c.orange, { borderWidth: 1.5 })
    ], { yunit: '%', ydec: 0, dec: 1, unit: '%', ymin: 20, ymax: 80, hline: { lines: [{ v: 25, color: c.ink2, label: 'luck 25%' }] } }));
    el('hitnote').textContent = PUB ? 'Hit rate of the Econometric Model and the SPF consensus against the actual regime, trailing 120 months.' : (st.spec === 'pce'
      ? '42 Macro and Bloomberg hit rates are as reported in the Sep-2026 screenshot (their vintage, core PCE spec).'
      : '42 Macro and Bloomberg hit rates are as reported in the Nov-2024 workbook (their vintage, headline CPI spec).');
    var F = D.fwd[st.spec];
    mk('ch-fwd', lineCfg(F.j.map(function (j) { return j + 'm'; }), [
      ds('Econometric Model projection', F.model.map(function (x) { return ok(x) ? 100 * x : null; }), c.plum, { borderWidth: 2.4, pointRadius: 2 }),
      ds('Persistence (today\u2019s regime)', F.persist.map(function (x) { return ok(x) ? 100 * x : null; }), c.ink2, { borderWidth: 1.5, pointRadius: 2 })
    ], { yunit: '%', ydec: 0, dec: 0, unit: '%', ymin: 0, ymax: 80, xt: 18, hline: { lines: [{ v: 25, color: c.orange, label: 'luck 25%' }] } }));
    var M = D.conf[st.spec], L = ['G', 'R', 'I', 'D'];
    var h = '<table class="t"><thead><tr><th>Actual \\ model</th>' + L.map(function (r) { return '<th>' + RN[r] + '</th>'; }).join('') + '<th>Hit</th></tr></thead><tbody>';
    M.forEach(function (row, a) {
      var tot = row.reduce(function (s, x) { return s + x; }, 0);
      h += '<tr><td>' + RN[L[a]] + '</td>' + row.map(function (x, b) { return '<td' + (a === b ? ' class="best"' : '') + '>' + x + '</td>'; }).join('') + '<td>' + pc(tot ? row[a] / tot : null) + '</td></tr>';
    });
    el('conf').innerHTML = h + '</tbody></table>';
    var K = D.calib[st.spec];
    var hc = '<table class="t"><thead><tr><th>Months ahead</th><th>Growth sign right</th><th>Inflation sign right</th><th>Obs.</th></tr></thead><tbody>';
    K.g.forEach(function (g, k) { hc += '<tr><td>' + g.lo + '\u2013' + g.hi + '</td><td>' + pc(g.acc) + '</td><td>' + pc(K.i[k].acc) + '</td><td>' + g.n + '</td></tr>'; });
    el('calib').innerHTML = hc + '</tbody></table><p class="note">Out-of-sample sign accuracy of the projected 3-month change. Probabilities shrink toward 50% where accuracy is low.</p>';
  }

  // ------------------------------------------------------------------ agents
  function agentOpts() {
    var A = D.agents[st.agT], s = el('agh'), H = A.H, unit = st.agT === 'gdp' ? 'Q' : 'M';
    if (st.agH > H) st.agH = A.hsel;
    s.innerHTML = ''; for (var h = 1; h <= H; h++) { var o = document.createElement('option'); o.value = h; o.textContent = h + unit + ' ahead'; if (h === st.agH) o.selected = true; s.appendChild(o); }
  }
  function shortName(a) { return a.f42.length > 34 ? a.f42.slice(0, 33) + '\u2026' : a.f42; }
  function renderAgents() {
    var c = C(), A = D.agents[st.agT], hi = st.agH - 1;
    var L = A.list.slice().sort(function (a, b) { return (b.w[hi] || 0) - (a.w[hi] || 0); });
    mk('ch-agw', {
      type: 'bar',
      data: { labels: L.map(shortName), datasets: [
        { label: 'This model, ' + st.agH + (st.agT === 'gdp' ? 'Q' : 'M') + ' ahead', data: L.map(function (a) { return ok(a.w[hi]) ? 100 * a.w[hi] : 0; }), backgroundColor: c.plum, borderWidth: 0, barPercentage: 0.9, categoryPercentage: 0.8 },
        { label: '42 Macro, Aug-2023', data: L.map(function (a) { return a.w42; }), backgroundColor: rgba(c.slate, 0.45), borderWidth: 0, barPercentage: 0.9, categoryPercentage: 0.8 }
      ].filter(function (d) { return d.data.some(ok); }) },
      options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { position: 'bottom', labels: { boxWidth: 12 } }, tooltip: { callbacks: { label: function (t) { return t.dataset.label + ': ' + (+t.parsed.x).toFixed(1) + '%'; } } } },
        scales: { x: { ticks: { callback: function (v) { return v + '%'; } }, grid: { color: rgba(c.rule, 0.7) } }, y: { ticks: { autoSkip: false, font: { size: 10.5 } }, grid: { display: false } } } }
    });
    var HW = A.list.some(function (a) { return ok(a.w42); });
    el('agw-h').textContent = HW ? 'Current weights versus 42 Macro\u2019s Aug-2023 weights' : 'Current agent weights';
    var t0 = IX[A.whist.start] || 0, last = L.slice(0, 7).map(function (a) { return a.key; });
    var len = A.whist.w[A.list[0].key].length, lab = AX.slice(t0, t0 + len).map(ml);
    var pal = [c.plum, c.teal, c.ochre, c.orange, c.slate, rgba(c.plum, 0.55), rgba(c.teal, 0.55)];
    var other = new Array(len).fill(0), sets = [];
    A.list.forEach(function (a) { if (last.indexOf(a.key) < 0) { A.whist.w[a.key].forEach(function (x, k) { if (ok(x)) other[k] += x; }); } });
    last.forEach(function (key, k) {
      var a = A.list.filter(function (z) { return z.key === key; })[0];
      sets.push({ label: shortName(a), data: A.whist.w[key].map(function (x) { return ok(x) ? 100 * x : 0; }), backgroundColor: rgba(pal[k].indexOf('rgba') === 0 ? c.plum : pal[k], 0.85), borderColor: 'transparent', fill: true, pointRadius: 0, borderWidth: 0, stack: 'w' });
      if (pal[k].indexOf('rgba') === 0) sets[sets.length - 1].backgroundColor = pal[k];
    });
    sets.push({ label: 'All other agents', data: other.map(function (x) { return 100 * x; }), backgroundColor: rgba(c.ink2, 0.25), borderColor: 'transparent', fill: true, pointRadius: 0, borderWidth: 0, stack: 'w' });
    mk('ch-agh', { type: 'line', data: { labels: lab, datasets: sets },
      options: { responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: 'index', intersect: false },
        plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 10.5 } } }, tooltip: { callbacks: { label: function (t) { return t.dataset.label + ': ' + (+t.parsed.y).toFixed(1) + '%'; } } } },
        scales: { x: { grid: { display: false }, ticks: { maxRotation: 0, maxTicksLimit: 8 } }, y: { stacked: true, min: 0, max: 100, ticks: { callback: function (v) { return v + '%'; } }, grid: { color: rgba(c.rule, 0.7) } } } } });
    el('aghnote').textContent = 'Weights at the ' + A.hsel + (st.agT === 'gdp' ? '-quarter' : '-month') + ' horizon, re-estimated every month; top seven agents by current weight.';
    var h = '<table class="t"><thead><tr><th>42 Macro feature</th>' + (HW ? '<th>42 Macro weight</th>' : '') + '<th>Proxy used here</th><th>Latest</th><th>Lag vs origin</th><th>3m change</th><th>R\u00b2</th><th>Weight</th><th>Signal</th></tr></thead><tbody>';
    L.forEach(function (a) {
      var w = a.w[hi], y = a.yhat[hi];
      h += '<tr><td>' + esc(a.f42) + '</td>' + (HW ? '<td>' + (ok(a.w42) ? a.w42 + '%' : '') + '</td>' : '') + '<td class="txt small">' + esc(a.proxy) + (a.note ? ' <span class="muted">(' + esc(a.note) + ')</span>' : '') + '</td><td>' + ml(a.last) + '</td><td>' + (a.delta > 0 ? '+' : '') + a.delta + 'm</td><td>' + f(a.x, 2) +
        '</td><td>' + f(a.r2[hi], 3) + '</td><td' + ((w || 0) >= 0.08 ? ' class="best"' : '') + '>' + pc(w, 1) + '</td><td>' + (ok(y) ? (y > 0 ? '+' : '') + f(y, 2) + 'pp' : 'inactive') + '</td></tr>';
    });
    el('agtbl').innerHTML = h + '</tbody></table><p class="note">Signal: the agent\u2019s forecast of the change in average annualized growth over the horizon relative to the trailing year, in percentage points. Lag: months between the feature\u2019s latest print and the forecast origin (' + (st.agT === 'gdp' ? D.meta.live.gdp_T : D.meta.live[st.agT + '_T']) + ').</p>';
  }

  // ------------------------------------------------------------------ mean reversion
  function renderSmrm() {
    var c = C();
    ['gdp', 'pce', 'cpi'].forEach(function (t) {
      var s = D.smrm[t], unit = s.unit === 'quarter' ? 'quarters' : 'months';
      el('st-' + t).innerHTML = '<div>Stationary mean<b>' + f(s.mu, 2) + '%</b></div><div>Run-rate now<b>' + f(s.x0, 2) + '%</b></div><div>Persistence<b>' + f(s.rho, 2) + '</b></div><div>Half-life<b>' + (ok(s.hl) ? f(s.hl, 1) + ' ' + unit : 'immediate') + '</b></div>';
      var lab = s.hist_x.concat(s.path_x).map(function (x) { return x.indexOf('-') === 4 ? ml(x) : x; });
      var nh = s.hist.length;
      var hist = s.hist.concat(new Array(s.path.length).fill(null));
      var path = new Array(nh - 1).fill(null).concat([s.hist[nh - 1]]).concat(s.path);
      var mu = lab.map(function () { return s.mu; });
      var sets = [ds(t === 'gdp' ? 'q/q annualized' : '3-month annualized', hist, c.ink, { borderWidth: 1.8 }), ds('Projection', path, c.plum, { borderWidth: 2.4, borderDash: [6, 4] }), ds('Stationary mean', mu, c.ochre, { borderWidth: 1.2, borderDash: [2, 3] })];
      if (s.hist_raw) sets.splice(1, 0, ds('Monthly annualized', s.hist_raw.concat(new Array(s.path.length).fill(null)), rgba(c.ink2, 0.35), { borderWidth: 1 }));
      mk('ch-sm-' + t, lineCfg(lab, sets, { yunit: '%', dec: 2, ymin: t === 'gdp' ? -6 : -4, ymax: t === 'gdp' ? 10 : 12, vline: { index: nh - 0.5, color: c.ink2 } }));
    });
  }

  // ------------------------------------------------------------------ track record
  function bestIdx(rows, key, lower) {
    var b = null, bi = -1;
    rows.forEach(function (r, i) { var v = r[key]; if (ok(v) && (b === null || (lower ? v < b : v > b))) { b = v; bi = i; } });
    return bi;
  }
  function scoreTable(id, rows, cols) {
    var h = '<table class="t"><thead><tr><th>Forecaster</th>' + cols.map(function (c) { return '<th>' + c[1] + '</th>'; }).join('') + '</tr></thead><tbody>';
    var best = cols.map(function (c) { return bestIdx(rows, c[0], c[2] === 'ae'); });
    rows.forEach(function (r, i) {
      h += '<tr><td>' + esc(r.name) + '</td>' + cols.map(function (c, k) {
        var v = r[c[0]], txt = c[2] === 'ae' ? (ok(v) ? f(v, 0) + 'bp' : 'n/a') : (ok(v) ? pc(v) : 'n/a');
        return '<td' + (best[k] === i ? ' class="best"' : '') + '>' + txt + '</td>';
      }).join('') + '</tr>';
    });
    el(id).innerHTML = h + '</tbody></table>';
  }
  function renderTrack() {
    var A = D.score.A, B = D.score.B, Cc = D.score.C, c = C();
    el('tr-intro').textContent = PUB ? 'Everything is scored against today\u2019s data vintage. Best value in each column is shaded.' : 'Everything is scored against today\u2019s data vintage, so 42 Macro\u2019s and Bloomberg\u2019s numbers are rescored too. Best value in each column is shaded.';
    el('scA-h').textContent = (PUB ? 'Window ' : '42 Macro\u2019s own window: ') + A.window + ' (Real GDP and headline CPI)';
    scoreTable('scA', A.rows, [['hit', 'Regime hit rate', 'h'], ['gdp_aafe', 'GDP error', 'ae'], ['gdp_da', 'GDP direction', 'h'], ['cpi_aafe', 'CPI error', 'ae'], ['cpi_da', 'CPI direction', 'h']]);
    var R = A.reported;
    if (!R) el('scA-n').textContent = 'SPF CPI direction is n/a because quarterly values repeat across months.';
    else el('scA-n').textContent = 'As reported by 42 Macro on its own vintage at Sep-24: hit rate ' + pc(R.hit_42, 1) + ' (Bloomberg ' + pc(R.hit_bb, 1) + '), GDP error ' + f(R.gdp_aafe_42, 0) + 'bp (' + f(R.gdp_aafe_bb, 0) + 'bp), CPI error ' + f(R.cpi_aafe_42, 0) + 'bp (' + f(R.cpi_aafe_bb, 0) + 'bp). SPF CPI direction is n/a because quarterly values repeat across months.';
    el('scB-h').textContent = (PUB ? 'Window ' : 'Sep-2026 screenshot window: ') + B.window + ' (Real GDP and core PCE)';
    el('scB-n').textContent = 'Short window: about two years of monthly data, so differences of a few points are noise.' + (PUB ? '' : ' Directional accuracy is blank for the screenshot series because 0.1pp rounding creates ties.');
    scoreTable('scB', B.rows, [['hit', 'Regime hit rate', 'h'], ['gdp_aafe', 'GDP error', 'ae'], ['gdp_da', 'GDP direction', 'h'], ['pce_aafe', 'Core PCE error', 'ae'], ['pce_da', 'Core PCE direction', 'h']]);
    var x = st.cov;
    var h = '<table class="t"><thead><tr><th>Target</th><th>Forecaster</th><th>Error</th><th>Direction</th></tr></thead><tbody>';
    ['gdp', 'cpi', 'pce'].forEach(function (t) {
      var rows = Cc.rows.filter(function (r) { return r.target === t; });
      var be = bestIdx(rows, x ? 'aafe_x' : 'aafe', true), bd = bestIdx(rows, x ? 'da_x' : 'da', false);
      rows.forEach(function (r, i) {
        var e = x ? r.aafe_x : r.aafe, d = x ? r.da_x : r.da;
        h += '<tr><td>' + (i === 0 ? TNAME[t] : '') + '</td><td class="txt">' + r.name + '</td><td' + (i === be ? ' class="best"' : '') + '>' + (ok(e) ? f(e, 0) + 'bp' : 'n/a') + '</td><td' + (i === bd ? ' class="best"' : '') + '>' + (ok(d) ? pc(d) : 'n/a') + '</td></tr>';
      });
    });
    ['cpi', 'pce'].forEach(function (s) {
      var rows = Cc.rows.filter(function (r) { return r.target === 'hit_' + s; }), bh = bestIdx(rows, 'hit', false);
      rows.forEach(function (r, i) {
        h += '<tr><td>' + (i === 0 ? 'Regime, ' + (s === 'pce' ? 'core PCE' : 'headline CPI') + ' spec' : '') + '</td><td class="txt">' + r.name + '</td><td' + (i === bh ? ' class="best"' : '') + '>' + pc(r.hit) + ' hit</td><td>' + pc(r.hit_t10) + ' last 10y</td></tr>';
      });
    });
    el('scC').innerHTML = h + '</tbody></table><p class="note">Errors are average absolute misses of the one-release-ahead forecast (bp). The COVID toggle drops 2020\u201321 from the error and direction columns; regime hit rates always include it.</p>';
    var G = D.hor.gdp, sfx = x ? 'aafex_' : 'aafe_';
    mk('ch-hz-gdp', lineCfg(G.h.map(function (h) { return h + 'Q'; }), [
      ds('Econometric Model', G[sfx + 'med_k2'], c.plum, { borderWidth: 2.4, pointRadius: 2 }),
      ds('Mean reversion', G[sfx + 'smrm_k2'], rgba(c.plum, 0.55), { borderDash: [2, 3], pointRadius: 2 }),
      ds('Agents', G[sfx + 'abnm_k2'], rgba(c.plum, 0.55), { borderDash: [9, 3, 2, 3], pointRadius: 2 }),
      ds('SPF median', G[sfx + 'spf'], c.ochre, { pointRadius: 2 }),
      ds('Naive', G[sfx + 'naive_k2'], c.ink2, { pointRadius: 2 })
    ], { ydec: 0, dec: 0, unit: 'bp', xt: 6, ymin: 0 }));
    ['pce', 'cpi'].forEach(function (t) {
      var Hh = D.hor[t];
      mk('ch-hz-' + t, lineCfg(Hh.h.map(function (h) { return h + 'm'; }), [
        ds('Econometric Model', Hh.aafe_med, c.plum, { borderWidth: 2.4 }),
        ds('Mean reversion', Hh.aafe_smrm, rgba(c.plum, 0.55), { borderDash: [2, 3] }),
        ds('Agents', Hh.aafe_abnm, rgba(c.plum, 0.55), { borderDash: [9, 3, 2, 3] }),
        ds('Naive', Hh.aafe_naive, c.ink2)
      ], { ydec: 0, dec: 0, unit: 'bp', xt: 9, ymin: 0 }));
    });
  }

  // ------------------------------------------------------------------ method
  function renderMethod() {
    var m = D.meta, w = m.windows;
    document.querySelectorAll('.privonly').forEach(function (e) { e.hidden = PUB; });
    document.querySelectorAll('.pubonly').forEach(function (e) { e.hidden = !PUB; });
    el('spec').innerHTML = [
      'Targets: Real GDP YoY (quarterly), headline CPI YoY (NSA, modelled on the SA index with trailing seasonal factors) and core PCE YoY.',
      'Mean-reversion model: monthly (quarterly for GDP) annualized log growth reverts from its 3-month (2-quarter) run-rate to the trailing 10-year mean, excluding 2020Q1\u20132021Q2 for GDP, at an AR decay re-estimated each month over the prior 15 years.',
      'Agent ensemble: ' + m.n_agents.gdp + ' growth and ' + m.n_agents.infl + ' inflation agents. For each horizon (1\u2013' + w.H_GDP + ' quarters, 1\u2013' + w.H_INF + ' months) an agent regresses future average growth minus the trailing year on its feature\u2019s 3-month change, at the lag the feature has today. Rolling windows: ' + w.W_INF + ' months (inflation), ' + w.W_GDPQ + ' quarters per intra-quarter position (GDP). Agents vote after ' + w.MIN_INF + ' / ' + w.MIN_GDP + ' observations, weighted by their share of total R\u00b2.',
      'GDP agents are estimated separately for each month within the quarter, so a forecast made in the first month of a quarter never borrows information from the third. COVID quarters are excluded from GDP training.',
      'Median projection: 50/50 average of the two sub-models, as in 42 Macro\u2019s workbook.',
      'History rows are pseudo-out-of-sample: every past forecast uses only data that would have been available, re-estimated at each origin.'
    ].map(function (s) { return '<li>' + s + '</li>'; }).join('');
    el('blsnote').textContent = m.bls && m.bls.length ? 'This build used the BLS API for: ' + m.bls.join(', ') + '.' : 'This build used the FRED fallbacks (BLS API limit reached).';
    el('runlog').textContent = (m.log || []).join('\n');
  }

  // ------------------------------------------------------------------ data export
  var UNITS = { v: '% YoY', z: 'z-score vs 2015-19', da: 'share (t10y)', hit: 'share (t10y)', ae: 'bp (t10y)' };
  var SPECN = { pce: 'Core PCE', cpi: 'Headline CPI' };
  function tableAoa(spec) {
    var G = gridSpec(spec);
    var out = [['Row (regime on ' + SPECN[spec] + ')', 'Units'].concat(G.cols.map(function (i) { return AX[i]; }))];
    out.push(['Column status (GDP and regime rows)', ''].concat(G.cols.map(function (i) { return i >= G.fcG ? 'projection' : 'actual / backtest'; })));
    G.rows.forEach(function (r) {
      if (r.type === 'sec') return;
      out.push([r.xl, r.type === 'reg' ? 'regime G/R/I/D' : UNITS[r.kind]].concat(G.cols.map(function (i) { return cellVal(r, i); })));
    });
    return out;
  }
  var SNAME = { act: 'actual', f1: 'model_one_release_ahead', mf: 'model_table_series', p_med: 'model_projection_median', p_smrm: 'model_projection_mean_reversion',
    p_abnm: 'model_projection_agents', spf: 'spf_median', m42: '42macro_nov24_file', bb: 'bloomberg_nov24_file', m42s: '42macro_sep26_screenshot', bbs: 'bloomberg_sep26_screenshot',
    z_act: 'z_actual', z_mf: 'z_model', z_spf: 'z_spf', da: 'model_directional_accuracy_t10y', aafe: 'model_aafe_t10y_bp', da_spf: 'spf_directional_accuracy_t10y', aafe_spf: 'spf_aafe_t10y_bp' };
  function monthlyAoa() {
    var cols = [];
    ['gdp', 'pce', 'cpi'].forEach(function (t) { Object.keys(SNAME).forEach(function (k) { if (D.S[t][k]) cols.push([t + '_' + SNAME[k], D.S[t][k]]); }); });
    ['pce', 'cpi'].forEach(function (s) {
      [['act', 'actual'], ['model', 'econometric_model'], ['spf', 'spf'], ['m42', '42macro'], ['bb', 'bloomberg'], ['act42', '42macro_reported_actual']].forEach(function (k) { if (D.REG[s][k[0]]) cols.push(['regime_' + s + 'spec_' + k[1], D.REG[s][k[0]]]); });
      [['model', 'econometric_model'], ['spf', 'spf'], ['m42', '42macro_reported'], ['bb', 'bloomberg_reported']].forEach(function (k) { if (D.HIT[s][k[0]]) cols.push(['hit_rate_t10y_' + s + 'spec_' + k[1], D.HIT[s][k[0]]]); });
    });
    var out = [['month'].concat(cols.map(function (c) { return c[0]; }))];
    AX.forEach(function (m, i) { out.push([m].concat(cols.map(function (c) { var v = c[1][i]; return v === undefined ? null : v; }))); });
    return out;
  }
  function probsAoa() {
    var out = [['Inflation measure', 'Month', 'P(growth accelerating)', 'P(inflation accelerating)', 'P(Goldilocks)', 'P(Reflation)', 'P(Inflation)', 'P(Deflation)', 'Projected regime', 'Highest-probability regime']];
    ['pce', 'cpi'].forEach(function (s) {
      var P = D.probs[s];
      P.months.forEach(function (m, k) { out.push([SPECN[s], m, P.pg[k], P.pi[k], P.G[k], P.R[k], P.I[k], P.D[k], P.modal[k], P.best ? P.best[k] : null]); });
    });
    return out;
  }
  function outlookAoa() {
    var K = [['q', 'Quarter'], ['gdp_med', 'Real GDP: Econometric Model'], ['gdp_smrm', 'Real GDP: mean reversion'], ['gdp_abnm', 'Real GDP: agents'], ['gdp_42', 'Real GDP: 42 Macro (Sep-26)'],
      ['gdp_bb', 'Real GDP: Bloomberg (Sep-26)'], ['gdp_spf', 'Real GDP: SPF'], ['pce_med', 'Core PCE: Econometric Model'], ['pce_42', 'Core PCE: 42 Macro (Sep-26)'], ['pce_bb', 'Core PCE: Bloomberg (Sep-26)'],
      ['pce_spf', 'Core PCE: SPF'], ['cpi_med', 'Headline CPI: Econometric Model'], ['cpi_spf', 'Headline CPI: SPF']];
    K = K.filter(function (k) { return k[0] === 'q' || D.outlook.some(function (o) { return ok(o[k[0]]); }); });
    var out = [K.map(function (k) { return k[1]; })];
    D.outlook.forEach(function (o) { out.push(K.map(function (k) { var v = o[k[0]]; return v === undefined ? null : v; })); });
    return out;
  }
  function agentAoa(t) {
    var A = D.agents[t], H = A.H, u = t === 'gdp' ? 'Q' : 'M', h;
    var HW = A.list.some(function (a) { return ok(a.w42); });
    var hdr = ['42 Macro feature', HW ? '42 Macro weight Aug-23 (%)' : '(42 Macro weight withheld)', 'Proxy used here', 'Source series', 'Transform', 'Latest observation', 'Lag vs origin (months)', 'Latest 3m change', 'Latest level'];
    for (h = 1; h <= H; h++) hdr.push('Weight ' + h + u);
    for (h = 1; h <= H; h++) hdr.push('R2 ' + h + u);
    for (h = 1; h <= H; h++) hdr.push('Signal ' + h + u + ' (pp)');
    var out = [hdr];
    A.list.forEach(function (a) {
      out.push([a.f42, a.w42, a.proxy + (a.note ? ' (' + a.note + ')' : ''), (a.ids || []).join(' '), a.tr, a.last, a.delta, a.x, a.lvl].concat(a.w, a.r2, a.yhat));
    });
    return out;
  }
  function whistAoa(t) {
    var A = D.agents[t], keys = A.list.map(function (a) { return a.key; });
    var t0 = IX[A.whist.start] || 0, len = A.whist.w[keys[0]].length;
    var out = [['Month (weights at ' + A.hsel + (t === 'gdp' ? 'Q' : 'M') + ' horizon)'].concat(A.list.map(function (a) { return a.f42; }))];
    for (var k = 0; k < len; k++) out.push([AX[t0 + k]].concat(keys.map(function (key) { var v = A.whist.w[key][k]; return v === undefined ? null : v; })));
    return out;
  }
  function smrmAoa() {
    var out = [['Target', 'Stationary mean (% annualized)', 'Run-rate now (% annualized)', 'Persistence per period', 'Half-life (periods)', 'Period']];
    ['gdp', 'pce', 'cpi'].forEach(function (t) { var s = D.smrm[t]; out.push([TNAME[t], s.mu, s.x0, s.rho, s.hl, s.unit]); });
    out.push([]);
    out.push(['Target', 'Period', 'Series', 'Value (% annualized)']);
    ['gdp', 'pce', 'cpi'].forEach(function (t) {
      var s = D.smrm[t];
      s.hist_x.forEach(function (x, k) {
        out.push([TNAME[t], x, t === 'gdp' ? 'q/q annualized' : '3-month annualized', s.hist[k]]);
        if (s.hist_raw) out.push([TNAME[t], x, 'monthly annualized', s.hist_raw[k]]);
      });
      s.path_x.forEach(function (x, k) { out.push([TNAME[t], x, 'projection', s.path[k]]); });
    });
    return out;
  }
  function trackAoa() {
    var Sc = D.score, out = [], R = Sc.A.reported;
    out.push(['42 Macro window: ' + Sc.A.window + ' (Real GDP and headline CPI), rescored on current-vintage data']);
    out.push(['Forecaster', 'Regime hit rate', 'GDP AAFE (bp)', 'GDP directional accuracy', 'CPI AAFE (bp)', 'CPI directional accuracy']);
    Sc.A.rows.forEach(function (r) { out.push([r.name, r.hit, r.gdp_aafe, r.gdp_da, r.cpi_aafe, r.cpi_da]); });
    if (R) {
      out.push(['42 Macro as reported (own vintage, Sep-24)', R.hit_42, R.gdp_aafe_42, R.gdp_da_42, R.cpi_aafe_42, R.cpi_da_42]);
      out.push(['Bloomberg as reported (Sep-24)', R.hit_bb, R.gdp_aafe_bb, R.gdp_da_bb, R.cpi_aafe_bb, R.cpi_da_bb]);
    }
    out.push([]);
    out.push(['Sep-2026 screenshot window: ' + Sc.B.window + ' (Real GDP and core PCE)']);
    out.push(['Forecaster', 'Regime hit rate', 'GDP AAFE (bp)', 'GDP directional accuracy', 'Core PCE AAFE (bp)', 'Core PCE directional accuracy']);
    Sc.B.rows.forEach(function (r) { out.push([r.name, r.hit, r.gdp_aafe, r.gdp_da, r.pce_aafe, r.pce_da]); });
    out.push([]);
    out.push(['Full backtest since 2005']);
    out.push(['Target', 'Forecaster', 'AAFE (bp)', 'AAFE ex-2020-21 (bp)', 'Directional accuracy', 'Directional accuracy ex-2020-21', 'Regime hit rate', 'Regime hit rate, last 10y']);
    Sc.C.rows.forEach(function (r) { out.push([r.target, r.name, r.aafe, r.aafe_x, r.da, r.da_x, r.hit, r.hit_t10]); });
    out.push([]);
    ['gdp', 'pce', 'cpi'].forEach(function (t) {
      var Hh = D.hor[t], keys = Object.keys(Hh).filter(function (k) { return k !== 'h' && Array.isArray(Hh[k]); });
      out.push(['Error by horizon: ' + TNAME[t] + (t === 'gdp' ? ' (quarters ahead; _k2 mid-quarter origin, _k3 end-quarter; aafex = ex-COVID; bp)' : ' (months ahead; bp)')]);
      out.push(['Horizon'].concat(keys));
      Hh.h.forEach(function (hh, k) { out.push([hh].concat(keys.map(function (key) { var v = Hh[key][k]; return v === undefined ? null : v; }))); });
      out.push([]);
    });
    ['pce', 'cpi'].forEach(function (s) {
      var F = D.fwd[s];
      out.push(['Forward regime hit rate by months ahead (' + SPECN[s] + ' spec)']);
      out.push(['Months ahead', 'Econometric Model', 'Persistence', 'Observations']);
      F.j.forEach(function (j, k) { out.push([j, F.model[k], F.persist[k], F.n[k]]); });
      out.push([]);
    });
    ['pce', 'cpi'].forEach(function (s) {
      var M = D.conf[s], L = ['G', 'R', 'I', 'D'];
      out.push(['Confusion matrix since 2005, actual (rows) vs Econometric Model (columns), ' + SPECN[s] + ' spec']);
      out.push(['Actual / model'].concat(L.map(function (r) { return RN[r]; })));
      M.forEach(function (row, a) { out.push([RN[L[a]]].concat(row)); });
      out.push([]);
    });
    ['pce', 'cpi'].forEach(function (s) {
      var K = D.calib[s];
      out.push(['Probability calibration (' + SPECN[s] + ' spec)']);
      out.push(['Months ahead', 'Growth sign accuracy', 'Growth logit a', 'Growth logit b', 'Inflation sign accuracy', 'Inflation logit a', 'Inflation logit b', 'Observations']);
      K.g.forEach(function (g, k) { var i = K.i[k]; out.push([g.lo + '-' + g.hi, g.acc, g.a, g.b, i.acc, i.a, i.b, g.n]); });
      out.push([]);
    });
    return out;
  }
  function readmeAoa() {
    var m = D.meta;
    return [
      ['US GRID Models: Real GDP and inflation, reconstructed from 42 Macro\u2019s GRID process'],
      ['Generated', m.generated_utc],
      ['Data to', 'Real GDP ' + m.asof.gdp + '; headline CPI ' + m.asof.cpi + '; core PCE ' + m.asof.pce + '; SPF survey ' + (m.spf_survey || 'n/a')],
      ['Econometric Model', 'This reconstruction (50/50 of the stationary mean-reversion model and the agent ensemble), not 42 Macro\u2019s published forecasts.'],
      [],
      ['Sheet', 'Contents'],
      ['GRID table (core PCE / headline CPI)', 'The dashboard table, monthly, regime computed on each inflation measure. % YoY values; z-scores vs each series\u2019 2015-19 mean and s.d.; directional accuracy and hit rates as shares over a trailing 10 years; errors in bp.'],
      ['Monthly series', 'Every monthly series behind the charts and table since ' + AX[0] + '. model_one_release_ahead = pseudo-out-of-sample history; model_table_series = that history joined to the current projection.'],
      ['Regime probabilities', 'Calibrated direction probabilities and the four regime probabilities for each projected month.'],
      ['Outlook', PUB ? 'Quarterly comparison with the SPF median.' : 'Quarterly comparison with 42 Macro, Bloomberg (Sep-26 screenshot, 0.1pp) and the SPF median.'],
      ['Agents (Real GDP / Core PCE / Headline CPI)', 'Every agent at the live origin: 42 Macro feature and weight, the proxy used, release lag, and weight, R2 and signal for each horizon.'],
      ['Agent weights (\u2026)', 'Monthly history of agent weights at the default horizon.'],
      ['Mean reversion', 'Stationary mean, run-rate, persistence and half-life, with the growth history and projected path.'],
      ['Track record', 'Scorecards on 42 Macro\u2019s window, the Sep-26 window and the full backtest; error by horizon; forward regime hit rates; confusion matrices; calibration.'],
      [],
      ['Sources', 'FRED; BLS; Philadelphia Fed Survey of Professional Forecasters (median levels)' + (PUB ? '. Public build: no 42 Macro or Bloomberg data.' : '; 42 Macro and Bloomberg values from the Nov-2024 workbook and Sep-2026 screenshot.')],
      ['Caveat', 'Backtests use today\u2019s data vintage, not real-time vintages.']
    ];
  }
  function findingsAoa() {
    return [['Findings']].concat(D.findings.map(function (x) { return [x.t.replace(/<[^>]+>/g, '')]; }));
  }
  function csvOf(aoa) {
    return '\uFEFF' + aoa.map(function (row) {
      return row.map(function (v) {
        if (v === null || v === undefined) return '';
        if (typeof v === 'number') return isFinite(v) ? String(v) : '';
        var s = String(v);
        return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
      }).join(',');
    }).join('\r\n');
  }
  var XLSX_URL = 'https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js', xlsxP = null;
  function loadXLSX() {
    if (window.XLSX) return Promise.resolve(window.XLSX);
    if (!xlsxP) {
      xlsxP = new Promise(function (res, rej) {
        var sc = document.createElement('script');
        sc.src = XLSX_URL;
        sc.onload = function () { if (window.XLSX) res(window.XLSX); else { xlsxP = null; rej(new Error('XLSX missing')); } };
        sc.onerror = function () { xlsxP = null; rej(new Error('XLSX load failed')); };
        document.head.appendChild(sc);
      });
    }
    return xlsxP;
  }
  function buildWorkbook(X) {
    var wb = X.utils.book_new();
    function add(name, aoa, widths) {
      var ws = X.utils.aoa_to_sheet(aoa);
      if (widths) ws['!cols'] = widths.map(function (w) { return { wch: w }; });
      X.utils.book_append_sheet(wb, ws, name.slice(0, 31));
    }
    add('README', readmeAoa(), [40, 120]);
    add('Findings', findingsAoa(), [160]);
    add('GRID table (core PCE)', tableAoa('pce'), [64, 18]);
    add('GRID table (headline CPI)', tableAoa('cpi'), [64, 18]);
    add('Monthly series', monthlyAoa(), [10]);
    add('Regime probabilities', probsAoa(), [14, 10, 12, 12, 12, 12, 12, 12, 12, 12]);
    add('Outlook', outlookAoa(), [10, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]);
    ['gdp', 'pce', 'cpi'].forEach(function (t) { add('Agents ' + TNAME[t], agentAoa(t), [40, 12, 70, 34, 10, 12, 10, 10, 10]); });
    ['gdp', 'pce', 'cpi'].forEach(function (t) { add('Agent weights ' + TNAME[t], whistAoa(t), [12]); });
    add('Mean reversion', smrmAoa(), [16, 14, 20, 14, 14, 10]);
    add('Track record', trackAoa(), [46, 16, 16, 16, 16, 16, 16, 16]);
    return X.write(wb, { bookType: 'xlsx', type: 'array', compression: true });
  }
  // downloads: the viewer's `downloads` capability when framed by claude.ai; a plain browser save in standalone copies
  var DL = { mode: null, ns: null };
  function dlStatus(msg) { var s = el('dlstatus'); if (s) { s.textContent = msg; s.hidden = !DL.mode; } }
  function showDl(on) { document.querySelectorAll('.dlui').forEach(function (e) { if (e.id !== 'dlstatus') e.hidden = !on; }); if (!on) { var s = el('dlstatus'); if (s) s.hidden = true; } }
  function initDownloads() {
    if (!window.claude) { DL.mode = 'blob'; showDl(true); return; }
    if (typeof window.claude.use !== 'function') { showDl(false); return; }
    window.claude.use('downloads').then(function (ns) {
      if (ns) { DL.mode = 'cap'; DL.ns = ns; showDl(true); } else { showDl(false); }
    }, function () { showDl(false); });
  }
  function saveFile(filename, data, mime) {
    if (DL.mode === 'cap' && DL.ns) {
      return DL.ns.save({ filename: filename, data: data }).then(function (r) {
        dlStatus(r && r.status === 'delivered' ? 'Sent.' : 'Saved ' + filename + '.');
      }, function (e) {
        var code = e && e.code;
        if (code === 'declined') dlStatus('Download cancelled.');
        else if (code === 'rate_limited') dlStatus('A save prompt is already open.');
        else if (code === 'rejected_extension' || code === 'extension_not_enabled') dlStatus('That format is not available in this view.');
        else if (code === 'too_large') dlStatus('That file is too large for this view; try the CSV.');
        else if (code === 'bad_request' || code === 'transform_error') dlStatus('Could not prepare the file.');
        else { dlStatus(''); DL.mode = null; showDl(false); }
      });
    }
    var blob = data instanceof Blob ? data : new Blob([data], { type: mime });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = filename; a.style.display = 'none';
    document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
    dlStatus('Saved ' + filename + '.');
    return Promise.resolve();
  }
  var STAMP = (D.meta.generated_utc || '').slice(0, 10) || 'latest';
  function doDownload(kind, btn) {
    if (!DL.mode) return;
    if (btn) btn.disabled = true;
    var p;
    try {
      if (kind === 'csv') p = saveFile('us_grid_models_monthly_' + STAMP + '.csv', csvOf(monthlyAoa()), 'text/csv;charset=utf-8');
      else if (kind === 'table') p = saveFile('us_grid_table_' + (st.spec === 'pce' ? 'core_pce' : 'headline_cpi') + '_' + STAMP + '.csv', csvOf(tableAoa(st.spec)), 'text/csv;charset=utf-8');
      else if (kind === 'json') p = saveFile('us_grid_models_' + STAMP + '.json', JSON.stringify(D), 'application/json');
      else if (kind === 'xlsx') {
        dlStatus('Preparing the workbook\u2026');
        p = loadXLSX().then(function (X) {
          var buf = buildWorkbook(X);
          return saveFile('us_grid_models_' + STAMP + '.xlsx', new Blob([buf], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
        }, function () { dlStatus('The spreadsheet library did not load; the CSV and JSON downloads still work.'); });
      }
    } catch (e) { dlStatus('Could not prepare the file.'); if (window.console) console.warn(e); }
    Promise.resolve(p).then(function () { if (btn) btn.disabled = false; }, function (e) { dlStatus('Could not prepare the file.'); if (window.console) console.warn(e); if (btn) btn.disabled = false; });
  }

  // ------------------------------------------------------------------ orchestration
  var renderers = {
    table: renderTable,
    fc: function () { fcChart('gdp'); fcChart('pce'); fcChart('cpi'); renderFcNotes(); },
    reg: renderRegimes,
    ag: function () { agentOpts(); renderAgents(); },
    sm: renderSmrm,
    tr: renderTrack,
    me: renderMethod
  };
  function safe(fn) { try { fn(); } catch (e) { if (window.console) console.warn('render issue', e); } }
  function renderAll() {
    setDefaults(); safe(renderStamp); safe(renderCall); safe(renderTape); safe(renderOutlook); safe(renderFindings);
    safe(renderers[st.tab]);
  }
  function showTab(t) {
    st.tab = t;
    document.querySelectorAll('.tabs button').forEach(function (b) { b.setAttribute('aria-selected', b.dataset.tab === t ? 'true' : 'false'); });
    document.querySelectorAll('.panel').forEach(function (p) { p.classList.toggle('on', p.id === 'p-' + t); });
    safe(renderers[t]);
  }
  document.querySelectorAll('.tabs button').forEach(function (b) { b.addEventListener('click', function () { showTab(b.dataset.tab); }); });
  document.querySelectorAll('[data-spec]').forEach(function (b) {
    b.addEventListener('click', function () {
      st.spec = b.dataset.spec;
      document.querySelectorAll('[data-spec]').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
      renderAll();
    });
  });
  document.querySelectorAll('[data-range]').forEach(function (b) {
    b.addEventListener('click', function () {
      st.range = b.dataset.range;
      document.querySelectorAll('[data-range]').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
      safe(renderers.fc);
    });
  });
  document.querySelectorAll('[data-agt]').forEach(function (b) {
    b.addEventListener('click', function () {
      st.agT = b.dataset.agt; st.agH = D.agents[st.agT].hsel;
      document.querySelectorAll('[data-agt]').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
      safe(renderers.ag);
    });
  });
  document.querySelectorAll('[data-cov]').forEach(function (b) {
    b.addEventListener('click', function () {
      st.cov = b.dataset.cov === '1';
      document.querySelectorAll('[data-cov]').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
      safe(renderTrack);
    });
  });
  el('agh').addEventListener('change', function () { st.agH = +el('agh').value; safe(renderAgents); });
  var tb = el('themebtn');
  function themeLabel() { var t = document.documentElement.getAttribute('data-theme'); tb.textContent = 'Theme: ' + (t || 'auto'); }
  tb.addEventListener('click', function () {
    var t = document.documentElement.getAttribute('data-theme'), nx = t === null ? 'light' : (t === 'light' ? 'dark' : null);
    if (nx) document.documentElement.setAttribute('data-theme', nx); else document.documentElement.removeAttribute('data-theme');
    try { if (nx) localStorage.setItem('gridTheme', nx); else localStorage.removeItem('gridTheme'); } catch (e) { }
    themeLabel(); renderAll();
  });
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var onmq = function () { if (!document.documentElement.getAttribute('data-theme')) renderAll(); };
    if (mq.addEventListener) mq.addEventListener('change', onmq); else if (mq.addListener) mq.addListener(onmq);
  }
  document.querySelectorAll('[data-dl]').forEach(function (b) { b.addEventListener('click', function () { doDownload(b.dataset.dl, b); }); });
  themeLabel();
  renderAll();
  initDownloads();
})();
</script>
</body>
</html>
'''


# ============================================================================ index helpers
def mi(p):
    if isinstance(p, str):
        p = pd.Period(p, 'M')
    return (p.year - 1985) * 12 + p.month - 1


def mp(i):
    return BASE + int(i)


def qi(p):
    if isinstance(p, str):
        p = pd.Period(p, 'Q')
    return (p.year - 1985) * 4 + p.quarter - 1


def qp(i):
    return pd.Period('1985Q1', 'Q') + int(i)


def mlab(i):
    p = mp(i)
    return f'{p.year}-{p.month:02d}'


def mshort(i):
    p = mp(i)
    return p.strftime('%b-%y')


def qlab(i):
    p = qp(i)
    return f'Q{p.quarter}-{str(p.year)[2:]}'


# ============================================================================ array helpers
def shift(a, k):
    b = np.full(len(a), np.nan)
    if k == 0:
        return a.astype(float).copy()
    if k > 0:
        b[k:] = a[:-k]
    else:
        b[:k] = a[-k:]
    return b


def yoy(a, k=12):
    return 100.0 * (a / shift(a, k) - 1.0)


def ma(a, n=3):
    return pd.Series(a).rolling(n, min_periods=n).mean().values


def d3(a):
    return a - shift(a, 3)


def take(a, idx):
    out = np.full(len(idx), np.nan)
    ok = (idx >= 0) & (idx < len(a))
    out[ok] = a[idx[ok]]
    return out


def last_valid(a):
    idx = np.where(np.isfinite(a))[0]
    return int(idx[-1]) if len(idx) else -1


def fill_gaps(a, maxgap=2):
    """Linear interpolation of interior gaps up to `maxgap` months (e.g. the Oct-2025 shutdown CPI hole)."""
    a = a.astype(float).copy()
    idx = np.where(np.isfinite(a))[0]
    for k in range(len(idx) - 1):
        i0, i1 = idx[k], idx[k + 1]
        if 0 < i1 - i0 - 1 <= maxgap:
            a[i0 + 1:i1] = np.interp(np.arange(i0 + 1, i1), [i0, i1], [a[i0], a[i1]])
    return a


def roll(arr, W=120, minp=60):
    return pd.Series(arr).rolling(W, min_periods=minp).mean().values


def rolling_ols(x, y, W, lag, min_n):
    """Univariate OLS of y on x for every origin T using pairs T' in [T-lag-W+1, T-lag] (outcomes known at T)."""
    N = len(x)
    v = np.isfinite(x) & np.isfinite(y)
    xv = np.where(v, x, 0.0)
    yv = np.where(v, y, 0.0)

    def cs(z):
        return np.concatenate([[0.0], np.cumsum(z)])
    C, SX, SY, SXX, SYY, SXY = cs(v.astype(float)), cs(xv), cs(yv), cs(xv * xv), cs(yv * yv), cs(xv * yv)
    T = np.arange(N)
    hi = np.clip(T - lag + 1, 0, N)
    lo = np.clip(T - lag + 1 - W, 0, N)
    n = C[hi] - C[lo]
    mx = (SX[hi] - SX[lo]) / n
    my = (SY[hi] - SY[lo]) / n
    vxx = (SXX[hi] - SXX[lo]) / n - mx * mx
    vyy = (SYY[hi] - SYY[lo]) / n - my * my
    vxy = (SXY[hi] - SXY[lo]) / n - mx * my
    b = vxy / vxx
    a = my - b * mx
    r2 = np.clip((vxy * vxy) / (vxx * vyy), 0, 1)
    bad = (n < min_n) | ~(vxx > 1e-10) | ~(vyy > 1e-10)
    for z in (a, b, r2):
        z[bad] = np.nan
    return a, b, r2, n


# ============================================================================ data layer
def fetch_fred(sid, cache, offline=False):
    fp = os.path.join(cache, f'fred_{sid}.json')
    if offline and os.path.exists(fp):
        d = json.load(open(fp))
        return pd.Series(d['v'], index=pd.to_datetime(d['d']))
    obs, err = None, ''
    if FRED_KEY:
        for attempt in range(4):
            try:
                r = SESSION.get('https://api.stlouisfed.org/fred/series/observations',
                                params={'series_id': sid, 'api_key': FRED_KEY, 'file_type': 'json'}, timeout=60)
                if r.status_code == 200:
                    obs = [(o['date'], o['value']) for o in r.json()['observations']]
                    break
                err = f'HTTP {r.status_code} {r.text[:100]}'
                if r.status_code == 400:
                    break
            except Exception as e:
                err = str(e)
            time.sleep(1.5 * (attempt + 1))
    if obs is None and (not FRED_KEY or 'does not exist' not in err):
        try:  # curl's default user agent works for fredgraph.csv where browser agents get 503
            out = subprocess.run(['curl', '-s', '-L', '--max-time', '60', f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}'],
                                 capture_output=True, text=True, timeout=90).stdout
            lines = out.strip().splitlines()
            if len(lines) > 1 and ',' in lines[0] and '<' not in lines[0]:
                obs = [tuple(l.split(',')[:2]) for l in lines[1:]]
        except Exception as e:
            err += f' | csv: {e}'
    if obs is None:
        if os.path.exists(fp):
            log(f'  ! {sid}: live fetch failed ({err}); using cached copy')
            d = json.load(open(fp))
            return pd.Series(d['v'], index=pd.to_datetime(d['d']))
        raise RuntimeError(f'{sid}: {err}')
    pairs = [(a, float(b)) for a, b in obs if b not in ('.', '', 'NA', '#N/A')]
    json.dump({'d': [a for a, _ in pairs], 'v': [b for _, b in pairs]}, open(fp, 'w'))
    return pd.Series([b for _, b in pairs], index=pd.to_datetime([a for a, _ in pairs]))


def fetch_bls(ids, cache, offline=False):
    fp = os.path.join(cache, 'bls_cpi.json')
    if offline and os.path.exists(fp):
        d = json.load(open(fp))
        return {k: pd.Series(v['v'], index=pd.to_datetime(v['d'])) for k, v in d.items()}
    yr = dt.date.today().year
    if BLS_KEY:   # v2: 20-year spans, 500 queries/day per key
        chunks, url = [(1997, 2016), (2017, yr)], 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
    else:         # v1: 10-year spans, 25 queries/day per IP (shared on CI runners)
        chunks, url = [(1997, 2006), (2007, 2016), (2017, yr)], 'https://api.bls.gov/publicAPI/v1/timeseries/data/'
    store = {k: {} for k in ids}
    try:
        for a, b in chunks:
            body = {'seriesid': ids, 'startyear': str(a), 'endyear': str(b)}
            if BLS_KEY:
                body['registrationkey'] = BLS_KEY
            r = SESSION.post(url, json=body, timeout=60)
            js = r.json()
            if js.get('status') != 'REQUEST_SUCCEEDED':
                raise RuntimeError(f"BLS {js.get('status')}: {js.get('message')}")
            for s in js['Results']['series']:
                for d in s['data']:
                    if d['period'].startswith('M') and d['period'] != 'M13' and d['value'] not in ('-', ''):
                        store[s['seriesID']][f"{d['year']}-{d['period'][1:]}-01"] = float(d['value'])
            time.sleep(0.6)
        out = {k: {'d': sorted(v), 'v': [v[x] for x in sorted(v)]} for k, v in store.items() if v}
        json.dump(out, open(fp, 'w'))
        return {k: pd.Series(v['v'], index=pd.to_datetime(v['d'])) for k, v in out.items()}
    except Exception as e:
        log(f'  ! BLS API unavailable ({e}); falling back to cached/FRED proxies')
        if os.path.exists(fp):
            d = json.load(open(fp))
            return {k: pd.Series(v['v'], index=pd.to_datetime(v['d'])) for k, v in d.items()}
        return {}


def fetch_spf(cache, offline=False):
    fp = os.path.join(cache, 'spf_medianLevel.xlsx')
    url = ('https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/'
           'survey-of-professional-forecasters/historical-data/medianLevel.xlsx')
    if not (offline and os.path.exists(fp)):
        try:
            r = SESSION.get(url, headers={'User-Agent': UA}, timeout=90)
            if r.status_code == 200 and r.content[:2] == b'PK':
                open(fp, 'wb').write(r.content)
            else:
                log(f'  ! SPF download HTTP {r.status_code}')
        except Exception as e:
            log(f'  ! SPF download failed: {e}')
    if not os.path.exists(fp):
        return None
    x = pd.ExcelFile(fp)
    out = {}
    for sh in ['RGDP', 'CPI', 'COREPCE']:
        d = x.parse(sh)
        for c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        out[sh] = d
    return out


def to_marr(s, N):
    a = np.full(N, np.nan)
    if s is None or len(s) == 0:
        return a
    g = s.groupby(s.index.to_period('M')).mean()
    for p, v in g.items():
        i = mi(p)
        if 0 <= i < N and np.isfinite(v):
            a[i] = v
    return fill_gaps(a)


def to_qarr(s, NQ):
    a = np.full(NQ, np.nan)
    for d, v in s.items():
        i = qi(pd.Period(d, 'Q'))
        if 0 <= i < NQ:
            a[i] = v
    return a


def build_features(specs, raw, bls, N):
    feats = []
    for key, f42, w42, label, src, tr in specs:
        comps, used, note = [], [], ''
        if src == 'CTRL':
            if all(k in raw for k in CTRL_IDS):
                comps = [raw['RSXFS'] - raw['RSMVPD'] - raw['RSBMGESD'] - raw['RSGASS']]
                used = CTRL_IDS
        elif src == 'PPI':
            y1 = yoy(raw['PPIFIS']) if 'PPIFIS' in raw else np.full(N, np.nan)
            y0 = yoy(raw['WPSFD49207']) if 'WPSFD49207' in raw else np.full(N, np.nan)
            v = np.where(np.isfinite(y1), y1, y0)
            used = [k for k in PPI_IDS if k in raw]
            feats.append(dict(key=key, f42=f42, w42=w42, label=label, tr=tr, ids=used, v=v, d=d3(v),
                              last=last_valid(v), note='YoY spliced: PPIFIS from Nov-2010, WPSFD49207 before'))
            continue
        elif isinstance(src, dict):
            s = bls.get(src['bls'])
            if s is not None and len(s) > 120:
                comps, used = [to_marr(s, N)], [src['bls'] + ' (BLS)']
            elif src['fb'] in raw:
                comps, used, note = [raw[src['fb']]], [src['fb']], 'BLS API unavailable: FRED fallback'
        else:
            for c in src:
                alts = c if isinstance(c, tuple) else (c,)
                for a_ in alts:
                    if a_ in raw and np.isfinite(raw[a_]).sum() > 24:
                        comps.append(raw[a_])
                        used.append(a_)
                        break
        if not comps:
            log(f'  ! feature {key}: no source available, agent dropped')
            continue
        if tr == 'mean':
            v = np.nanmean(np.vstack(comps), axis=0) if len(comps) > 1 else comps[0]
            d = np.nanmean(np.vstack([d3(c) for c in comps]), axis=0) if len(comps) > 1 else d3(comps[0])
            lv = max(last_valid(c) for c in comps)
        else:
            base = comps[0]
            v = yoy(base) if tr == 'yoy' else (yoy(ma(base, 3)) if tr == 'yoy3' else base)
            d = d3(v)
            lv = last_valid(v)
        feats.append(dict(key=key, f42=f42, w42=w42, label=label, tr=tr, ids=used, v=v, d=d, last=lv, note=note))
    return feats


# ============================================================================ models
def regime(dg, di):
    out = np.array([None] * len(dg), dtype=object)
    ok = np.isfinite(dg) & np.isfinite(di)
    ug, ui = dg > 0, di > 0
    out[ok & ug & ~ui] = 'G'
    out[ok & ug & ui] = 'R'
    out[ok & ~ug & ui] = 'I'
    out[ok & ~ug & ~ui] = 'D'
    return out


def run_inflation(name, lp, lnsa, feats, N, eval_from):
    """lp: log SA price index (model space); lnsa: log NSA index for reported YoY (CPI) or None (core PCE)."""
    m_last = last_valid(lp)
    T_live = m_last + 1
    T = np.arange(N)
    lrep = lnsa if lnsa is not None else lp
    a_rep = 100 * (np.exp(lrep - shift(lrep, 12)) - 1)
    # seasonal log-factors known at each origin (36m trailing mean by calendar month)
    if lnsa is not None:
        lsf = lnsa - lp
        cal = np.array([mp(i).month for i in range(N)])
        SF = np.full((N, 13), 0.0)
        for c in range(1, 13):
            s = pd.Series(np.where(cal == c, lsf, np.nan)).rolling(36, min_periods=1).mean().values
            SF[:, c] = shift(s, 1)          # window ending T-1
        SF = np.nan_to_num(SF)
    Y12 = 100 * (shift(lp, 1) - shift(lp, 13))
    keys = [f['key'] for f in feats]
    deltas = {f['key']: f['last'] - T_live for f in feats}
    X = {f['key']: shift(f['d'], -deltas[f['key']]) for f in feats}
    nA = len(feats)
    L_ab = np.full((N, H_INF + 1), np.nan)
    L_sm = np.full((N, H_INF + 1), np.nan)
    live = {k: {'r2': [], 'w': [], 'yhat': [], 'b': [], 'n': []} for k in keys}
    whist = None
    hsel = 3
    for h in range(1, H_INF + 1):
        y = (1200.0 / h) * (shift(lp, 1 - h) - shift(lp, 1)) - Y12
        P = np.full((nA, N), np.nan)
        R = np.full((nA, N), np.nan)
        for ai, k in enumerate(keys):
            a_, b_, r2_, n_ = rolling_ols(X[k], y, W_INF, h, MIN_INF)
            yh = a_ + b_ * X[k]
            P[ai] = yh
            R[ai] = np.where(np.isfinite(yh), r2_, np.nan)
            live[k]['b'].append(b_[T_live]); live[k]['n'].append(int(n_[T_live]))
            live[k]['yhat'].append(yh[T_live]); live[k]['r2'].append(R[ai, T_live])
        wsum = np.nansum(R, axis=0)
        Wt = R / np.where(wsum > 0, wsum, np.nan)
        comb = np.nansum(Wt * P, axis=0)
        comb[~(wsum > 0)] = np.nan
        L_ab[:, h] = shift(lp, 1) + h * (Y12 + comb) / 1200.0
        for ai, k in enumerate(keys):
            live[k]['w'].append(Wt[ai, T_live])
        if h == hsel:
            whist = Wt.copy()
    # ---- SMRM
    g = 1200 * (lp - shift(lp, 1))
    g3 = ma(g, 3)
    sm_hist = {}
    t0 = mi('1995-01')
    for To in range(t0, T_live + 1):
        m0 = To - 1
        seg = g[m0 - MU_WIN_M + 1:m0 + 1]
        mu = np.nanmean(seg) if MU_STAT == 'mean' else np.nanmedian(seg)
        x0 = np.nanmean(g[m0 - 2:m0 + 1])
        s_idx = np.arange(max(m0 - RHO_WIN_M + 1, 3), m0 - 3 + 1)
        u, v = g3[s_idx] - mu, g3[s_idx + 3] - mu
        ok = np.isfinite(u) & np.isfinite(v)
        rho3 = np.clip(np.sum(u[ok] * v[ok]) / np.sum(u[ok] ** 2), 0.0, 0.97) if ok.sum() > 24 else 0.5
        rho = rho3 ** (1 / 3)
        k = np.arange(1, H_INF + 1)
        gp = mu + (x0 - mu) * rho ** k
        L_sm[To, 1:] = lp[m0] + np.cumsum(gp) / 1200.0
        sm_hist[To] = (mu, rho, x0, gp)

    def to_yoy(L):
        Y = np.full((N, H_INF + 1), np.nan)
        for h in range(1, H_INF + 1):
            m = T - 1 + h
            num = L[:, h].copy()
            if h <= 12:
                den = take(lrep, m - 12)
            else:
                den = L[:, h - 12].copy()
            if lnsa is not None:
                calm = np.array([mp(i).month for i in m])
                num = num + SF[T, calm]
                if h > 12:
                    den = den + SF[T, calm]
            Y[:, h] = 100 * (np.exp(num - den) - 1)
        return Y
    Y_ab, Y_sm = to_yoy(L_ab), to_yoy(L_sm)
    Y_md = 0.5 * (Y_ab + Y_sm)
    mu, rho, x0, gp = sm_hist[T_live]
    # horizon metrics (origins from eval_from with realized outcome)
    hor = {'h': list(range(1, H_INF + 1))}
    for nm, Y in [('med', Y_md), ('smrm', Y_sm), ('abnm', Y_ab)]:
        ae, da, n = [], [], []
        for h in range(1, H_INF + 1):
            To = np.arange(eval_from, T_live)
            act = take(a_rep, To - 1 + h)
            base = take(a_rep, To - 1)
            f = Y[To, h]
            ok = np.isfinite(act) & np.isfinite(f) & np.isfinite(base)
            ae.append(float(np.mean(np.abs(f[ok] - act[ok])) * 100) if ok.sum() else None)
            dd = ok & (act != base)
            da.append(float(np.mean(np.sign(f[dd] - base[dd]) == np.sign(act[dd] - base[dd]))) if dd.sum() else None)
            n.append(int(ok.sum()))
        hor['aafe_' + nm], hor['da_' + nm], hor['n'] = ae, da, n
    nv = []
    for h in range(1, H_INF + 1):
        To = np.arange(eval_from, T_live)
        act, base = take(a_rep, To - 1 + h), take(a_rep, To - 1)
        ok = np.isfinite(act) & np.isfinite(base)
        nv.append(float(np.mean(np.abs(base[ok] - act[ok])) * 100))
    hor['aafe_naive'] = nv
    return dict(name=name, m_last=m_last, T_live=T_live, a_rep=a_rep, lp=lp, lrep=lrep, g=g, g3=g3,
                Y_md=Y_md, Y_sm=Y_sm, Y_ab=Y_ab, deltas=deltas, live=live, whist=whist, hsel=hsel,
                smrm=dict(mu=mu, rho=rho, x0=x0, path=gp), sm_hist=sm_hist, keys=keys, hor=hor)


def run_gdp(lq, feats, N, NQ, eval_from):
    q_last = last_valid(lq)
    qend = 3 * q_last + 2
    max_feat = max(f['last'] for f in feats)
    T_live = max(qend + 1, min(qend + 3, max_feat))
    k_live = T_live - qend
    T = np.arange(N)
    Q = (T - 3) // 3
    pos = T - (3 * Q + 2)
    Y4 = 100 * (take(lq, Q) - take(lq, Q - 4))
    aq = 100 * (np.exp(lq - shift(lq, 4)) - 1)
    keys = [f['key'] for f in feats]
    deltas = {f['key']: f['last'] - T_live for f in feats}
    X = {f['key']: shift(f['d'], -deltas[f['key']]) for f in feats}
    nA = len(feats)
    qc0, qc1 = qi('2020Q1'), qi('2021Q2')
    covT = (T >= mi('2020-01')) & (T <= mi('2021-12'))
    L_ab = np.full((N, H_GDP + 1), np.nan)
    L_sm = np.full((N, H_GDP + 1), np.nan)
    live = {k: {'r2': [], 'w': [], 'yhat': [], 'b': [], 'n': []} for k in keys}
    whist, hsel = None, 1
    for h in range(1, H_GDP + 1):
        y = (400.0 / h) * (take(lq, Q + h) - take(lq, Q)) - Y4
        overlap = ~((Q + h < qc0) | (Q - 4 > qc1))
        ytr = np.where(overlap | covT, np.nan, y)
        P = np.full((nA, N), np.nan)
        R = np.full((nA, N), np.nan)
        for ai, k in enumerate(keys):
            a_ = np.full(N, np.nan); b_ = np.full(N, np.nan); r2_ = np.full(N, np.nan); n_ = np.zeros(N)
            for r in range(3):
                idx = np.arange(r, N, 3)
                aa, bb, rr, nn = rolling_ols(X[k][idx], ytr[idx], W_GDPQ, h, MIN_GDP)
                a_[idx], b_[idx], r2_[idx], n_[idx] = aa, bb, rr, nn
            yh = a_ + b_ * X[k]
            P[ai] = yh
            R[ai] = np.where(np.isfinite(yh), r2_, np.nan)
            live[k]['b'].append(b_[T_live]); live[k]['n'].append(int(n_[T_live]))
            live[k]['yhat'].append(yh[T_live]); live[k]['r2'].append(R[ai, T_live])
        wsum = np.nansum(R, axis=0)
        Wt = R / np.where(wsum > 0, wsum, np.nan)
        comb = np.nansum(Wt * P, axis=0)
        comb[~(wsum > 0)] = np.nan
        L_ab[:, h] = take(lq, Q) + h * (Y4 + comb) / 400.0
        for ai, k in enumerate(keys):
            live[k]['w'].append(Wt[ai, T_live])
        if h == hsel:
            whist = Wt.copy()
    # ---- SMRM (quarterly; identical for the three intra-quarter positions)
    gq = 400 * (lq - shift(lq, 1))
    covq = np.zeros(NQ, bool)
    covq[qc0:qc1 + 1] = True
    cache = {}
    sm_live = None
    for To in range(mi('1995-01'), T_live + 1):
        Qo = Q[To]
        if Qo not in cache:
            qq = np.arange(Qo - MU_WIN_Q + 1, Qo + 1)
            seg = gq[qq][~covq[qq]]
            mu = np.nanmean(seg) if MU_STAT == 'mean' else np.nanmedian(gq[qq])
            x0 = np.nanmean(gq[Qo - 1:Qo + 1])
            qs = np.arange(max(Qo - RHO_WIN_Q, 2), Qo)
            g2 = 0.5 * (gq[qs] + gq[qs - 1])
            u, v = g2 - mu, gq[qs + 1] - mu
            ok = np.isfinite(u) & np.isfinite(v) & ~covq[qs] & ~covq[qs - 1] & ~covq[qs + 1]
            rho = np.clip(np.sum(u[ok] * v[ok]) / np.sum(u[ok] ** 2), 0.0, 0.9) if ok.sum() > 12 else 0.4
            k = np.arange(1, H_GDP + 1)
            gp = mu + (x0 - mu) * rho ** k
            cache[Qo] = (mu, rho, x0, gp)
        mu, rho, x0, gp = cache[Qo]
        L_sm[To, 1:] = lq[Qo] + np.cumsum(gp) / 400.0
    sm_live = cache[Q[T_live]]

    def to_yoy(L):
        Y = np.full((N, H_GDP + 1), np.nan)
        for h in range(1, H_GDP + 1):
            den = take(lq, Q + h - 4) if h <= 4 else L[:, h - 4]
            Y[:, h] = 100 * (np.exp(L[:, h] - den) - 1)
        return Y
    Y_ab, Y_sm = to_yoy(L_ab), to_yoy(L_sm)
    Y_md = 0.5 * (Y_ab + Y_sm)
    am = take(aq, T // 3)                 # monthly step series of actual quarterly YoY
    hor = {'h': list(range(1, H_GDP + 1))}
    for posk in (2, 3):
        for nm, Y in [('med', Y_md), ('smrm', Y_sm), ('abnm', Y_ab)]:
            ae, da, n, aex = [], [], [], []
            for h in range(1, H_GDP + 1):
                To = np.arange(eval_from, T_live)
                To = To[pos[To] == posk]
                act, base, f = take(aq, Q[To] + h), take(aq, Q[To]), Y[To, h]
                ok = np.isfinite(act) & np.isfinite(f) & np.isfinite(base)
                okx = ok & ~((Q[To] + h >= qc0) & (Q[To] + h <= qc1 + 2))
                ae.append(float(np.mean(np.abs(f[ok] - act[ok])) * 100))
                aex.append(float(np.mean(np.abs(f[okx] - act[okx])) * 100))
                da.append(float(np.mean(np.sign(f[ok] - base[ok]) == np.sign(act[ok] - base[ok]))))
                n.append(int(ok.sum()))
            sfx = f'_k{posk}'
            hor['aafe_' + nm + sfx], hor['aafex_' + nm + sfx], hor['da_' + nm + sfx], hor['n' + sfx] = ae, aex, da, n
        nv, nvx = [], []
        for h in range(1, H_GDP + 1):
            To = np.arange(eval_from, T_live)
            To = To[pos[To] == posk]
            act, base = take(aq, Q[To] + h), take(aq, Q[To])
            ok = np.isfinite(act) & np.isfinite(base)
            okx = ok & ~((Q[To] + h >= qc0) & (Q[To] + h <= qc1 + 2))
            nv.append(float(np.mean(np.abs(base[ok] - act[ok])) * 100))
            nvx.append(float(np.mean(np.abs(base[okx] - act[okx])) * 100))
        hor[f'aafe_naive_k{posk}'], hor[f'aafex_naive_k{posk}'] = nv, nvx
    return dict(q_last=q_last, T_live=T_live, k_live=k_live, Q=Q, pos=pos, aq=aq, am=am, lq=lq, gq=gq,
                Y_md=Y_md, Y_sm=Y_sm, Y_ab=Y_ab, deltas=deltas, live=live, whist=whist, hsel=hsel,
                smrm=dict(mu=sm_live[0], rho=sm_live[1], x0=sm_live[2], path=sm_live[3]), keys=keys, hor=hor)


# ============================================================================ benchmarks
def spf_bench(spf, lq, cpi_sa, pce_sa, N, NQ):
    """SPF median: YoY by splicing SPF forecast growth onto actual (latest-vintage) growth of earlier quarters."""
    out = {}
    glq = lq - shift(lq, 1)

    def qavg_log(a):
        qa = np.full(NQ, np.nan)
        for q in range(NQ):
            seg = a[3 * q:3 * q + 3] if 3 * q + 3 <= len(a) else []
            if len(seg) == 3 and np.all(np.isfinite(seg)):
                qa[q] = np.log(np.mean(seg))
        return qa
    for key, sheet, act_g in [('gdp', 'RGDP', glq), ('cpi', 'CPI', None), ('pce', 'COREPCE', None)]:
        if spf is None or sheet not in spf:
            continue
        if key == 'cpi':
            ql = qavg_log(cpi_sa); act_g = ql - shift(ql, 1)
        if key == 'pce':
            ql = qavg_log(pce_sa); act_g = ql - shift(ql, 1)
        d = spf[sheet]
        surveys = {}
        for _, r in d.iterrows():
            if not (np.isfinite(r['YEAR']) and np.isfinite(r['QUARTER'])):
                continue
            s = qi(pd.Period(year=int(r['YEAR']), quarter=int(r['QUARTER']), freq='Q'))
            if key == 'gdp':
                lv = [r.get(f'RGDP{j}') for j in range(1, 7)]
                if not all(np.isfinite(lv)):
                    continue
                gf = {s + j: np.log(lv[j + 1] / lv[j]) for j in range(0, 5)}
            else:
                rt = [r.get(f'{sheet}{j}') for j in range(2, 7)]
                if not all(np.isfinite(rt)):
                    continue
                gf = {s + j: np.log(1 + rt[j] / 100.0) / 4.0 for j in range(0, 5)}
            yo = {}
            for j in range(0, 5):
                tot, ok = 0.0, True
                for q in range(s + j - 3, s + j + 1):
                    gv = gf.get(q) if q >= s else act_g[q] if 0 <= q < NQ else np.nan
                    if gv is None or not np.isfinite(gv):
                        ok = False
                        break
                    tot += gv
                yo[s + j] = 100 * (np.exp(tot) - 1) if ok else np.nan
            surveys[s] = yo
        f1 = np.full(N, np.nan)
        for s, yo in surveys.items():
            for k in range(3):
                m = 3 * s + k
                if 0 <= m < N:
                    f1[m] = yo[s]
        s_last = max(surveys)
        path = np.full(N, np.nan)
        for q, v in surveys[s_last].items():
            for k in range(3):
                if 3 * q + k < N:
                    path[3 * q + k] = v
        mf = f1.copy()
        sel = np.arange(N) >= 3 * (s_last + 1)
        mf[sel] = path[sel]
        hz = {}
        for j in range(5):
            errs = []
            for s, yo in surveys.items():
                if s < qi(EVAL_START[:4] + 'Q1'):
                    continue
                q = s + j
                if key == 'gdp':
                    act = 100 * (np.exp(lq[q] - lq[q - 4]) - 1) if q < NQ and np.isfinite(lq[q]) else np.nan
                else:
                    ql_ = qavg_log(cpi_sa if key == 'cpi' else pce_sa)
                    act = 100 * (np.exp(ql_[q] - ql_[q - 4]) - 1) if q < NQ and np.isfinite(ql_[q]) else np.nan
                if np.isfinite(act) and np.isfinite(yo[q]):
                    errs.append((abs(yo[q] - act), q))
            hz[j] = errs
        out[key] = dict(f1=f1, path=path, mf=mf, s_last=s_last, surveys=surveys, hz=hz)
    return out


def ref_arrays(ref, N):
    """Map embedded 42 Macro reference lists onto the model's month index."""
    out = {}
    for blk in ('nov24', 'sep26'):
        d = ref[blk]
        s0 = mi(d['start'])
        o = {}
        for k, v in d.items():
            if not isinstance(v, list):
                continue
            if k.startswith('reg_') or (v and any(isinstance(x, str) for x in v if x is not None)):
                arr = np.array([None] * N, dtype=object)
                for j, x in enumerate(v):
                    if 0 <= s0 + j < N:
                        arr[s0 + j] = x
            else:
                arr = np.full(N, np.nan)
                for j, x in enumerate(v):
                    if x is not None and 0 <= s0 + j < N:
                        arr[s0 + j] = x
            o[k] = arr
        out[blk] = o
    return out


REF_KEYS = {
    'nov24': ['reg_act', 'reg_42', 'reg_bb', 'hit_42', 'hit_bb', 'gdp_act', 'gdp_42', 'gdp_bb', 'cpi_act', 'cpi_42', 'cpi_bb',
              'gdp_42_da', 'gdp_42_aae', 'gdp_bb_da', 'gdp_bb_aae', 'cpi_42_da', 'cpi_42_aae', 'cpi_bb_da', 'cpi_bb_aae'],
    'sep26': ['reg_act', 'reg_42', 'reg_bb', 'hit_42', 'hit_bb', 'gdp_act', 'gdp_42', 'gdp_bb', 'gdp_42_da', 'gdp_42_aae',
              'gdp_bb_da', 'gdp_bb_aae', 'pce_act', 'pce_42', 'pce_bb'],
}


def load_ref(path):
    """42 Macro reference data: --public -> empty; else $REF42_JSON, --ref path, ref/ref42.json or ref42.json beside the script."""
    if PUBLIC:
        return {'nov24': dict({'start': '2007-01'}, **{k: [] for k in REF_KEYS['nov24']}),
                'sep26': dict({'start': '2024-08'}, **{k: [] for k in REF_KEYS['sep26']}), 'bt2023': None}
    if REF42 is not None:
        return REF42
    env = os.environ.get('REF42_JSON', '').strip()
    if env:
        return json.loads(env)
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [path, os.path.join(here, 'ref', 'ref42.json'), os.path.join(here, 'ref42.json')]:
        if p and os.path.exists(p):
            return json.load(open(p))
    raise SystemExit('42 Macro reference data not found: add ref/ref42.json, set $REF42_JSON, or run with --public')


# ============================================================================ scoring helpers
def da_arr(f, a, lag):
    df_, da_ = f - shift(f, lag), a - shift(a, lag)
    ok = np.isfinite(df_) & np.isfinite(da_)
    out = np.full(len(f), np.nan)
    out[ok] = (np.sign(df_[ok]) == np.sign(da_[ok])).astype(float)
    return out


def ae_arr(f, a):
    return np.abs(f - a) * 100


def z_arr(arr, ref=None):
    ref = arr if ref is None else ref
    seg = ref[mi('2015-01'):mi('2019-12') + 1]
    m, s = np.nanmean(seg), np.nanstd(seg, ddof=1)
    return (arr - m) / s, float(m), float(s)


def reg_arr(fg, fi):
    return regime(fg - shift(fg, 3), fi - shift(fi, 3))


def hit_arr(rf, ra):
    out = np.full(len(rf), np.nan)
    for i in range(len(rf)):
        if rf[i] is not None and ra[i] is not None:
            out[i] = 1.0 if rf[i] == ra[i] else 0.0
    return out


def logit_fit(x, y):
    X = np.c_[np.ones_like(x), x]
    w = np.zeros(2)
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - 1e-3 * w
        H = -(X.T * (p * (1 - p))) @ X - 1e-3 * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


# ============================================================================ main build
def build(args):
    cache = args.cache_dir
    os.makedirs(cache, exist_ok=True)
    today = dt.date.today()
    N = mi(pd.Period(today, 'M')) + 40
    NQ = N // 3 + 2
    # ---------------- fetch
    ids = set(TARGET_IDS + CTRL_IDS + PPI_IDS)
    for spec in GROWTH + INFLATION:
        src = spec[4]
        if isinstance(src, list):
            for c in src:
                ids.update(c if isinstance(c, tuple) else (c,))
        elif isinstance(src, dict):
            ids.add(src['fb'])
    log(f'Fetching {len(ids)} FRED series' + (' (offline cache)' if args.offline else ''))
    rawser = {}
    for sid in sorted(ids):
        try:
            rawser[sid] = fetch_fred(sid, cache, args.offline)
            if not args.offline:
                time.sleep(0.55)  # FRED allows 120 requests/minute
        except Exception as e:
            log(f'  ! {sid}: {e}')
    bls = fetch_bls(BLS_IDS, cache, args.offline)
    log(f'BLS series: {sorted(bls)}')
    spf = fetch_spf(cache, args.offline)
    raw = {k: to_marr(v, N) for k, v in rawser.items() if k != 'GDPC1'}
    lq = np.log(to_qarr(rawser['GDPC1'], NQ))
    # ---------------- features & targets
    gfe = build_features(GROWTH, raw, bls, N)
    ife = build_features(INFLATION, raw, bls, N)
    lp_cpi, ln_cpi, lp_pce = np.log(raw['CPIAUCSL']), np.log(raw['CPIAUCNS']), np.log(raw['PCEPILFE'])
    ev = mi(EVAL_START)
    log('Running GDP model'); G = run_gdp(lq, gfe, N, NQ, ev)
    log('Running headline CPI model'); C = run_inflation('cpi', lp_cpi, ln_cpi, ife, N, ev)
    log('Running core PCE model'); P = run_inflation('pce', lp_pce, None, ife, N, ev)
    ref = load_ref(args.ref)
    R42 = ref_arrays(ref, N)
    SPF = spf_bench(spf, lq, raw['CPIAUCSL'], raw['PCEPILFE'], N, NQ)
    # ---------------- monthly model series ("one release ahead" history + live path)
    Tm = np.arange(N)
    series = {}
    gdp_f1 = G['Y_md'][Tm, 1]
    gdp_act = G['am'].copy()
    gdp_act[Tm // 3 > G['q_last']] = np.nan
    gdp_path = {nm: np.full(N, np.nan) for nm in ('med', 'smrm', 'abnm')}
    for nm, Y in [('med', G['Y_md']), ('smrm', G['Y_sm']), ('abnm', G['Y_ab'])]:
        for h in range(1, H_GDP + 1):
            q = G['q_last'] + h
            for k in range(3):
                gdp_path[nm][3 * q + k] = Y[G['T_live'], h]
    gdp_mf = gdp_f1.copy()
    gdp_mf[Tm > G['T_live']] = gdp_path['med'][Tm > G['T_live']]
    gdp_mf[Tm <= mi('2004-12')] = np.nan
    series['gdp'] = dict(act=gdp_act, f1=np.where(Tm <= G['T_live'], gdp_f1, np.nan), mf=gdp_mf, **{'p_' + k: v for k, v in gdp_path.items()})
    infl = {'cpi': C, 'pce': P}
    for key, M in infl.items():
        act = M['a_rep'].copy()
        act[Tm > M['m_last']] = np.nan
        f1 = M['Y_md'][Tm, 1]
        paths = {nm: np.full(N, np.nan) for nm in ('med', 'smrm', 'abnm')}
        for nm, Y in [('med', M['Y_md']), ('smrm', M['Y_sm']), ('abnm', M['Y_ab'])]:
            for h in range(1, H_INF + 1):
                paths[nm][M['m_last'] + h] = Y[M['T_live'], h]
        mf = f1.copy()
        mf[Tm > M['T_live']] = paths['med'][Tm > M['T_live']]
        mf[Tm <= mi('2004-12')] = np.nan
        series[key] = dict(act=act, f1=np.where(Tm <= M['T_live'], f1, np.nan), mf=mf, **{'p_' + k: v for k, v in paths.items()})
    # ---------------- 42 Macro-style scorecard rows (rolling 10y)
    lagDA = {'gdp': 3, 'cpi': 1, 'pce': 1}
    stats = {}
    for key in ('gdp', 'cpi', 'pce'):
        s = series[key]
        s['z_act'], ma_, sa_ = z_arr(s['act'])
        s['z_mf'], mm_, sm_ = z_arr(s['mf'])
        s['da'] = roll(da_arr(s['mf'], s['act'], lagDA[key]))
        s['aafe'] = roll(ae_arr(s['mf'], s['act']))
        sp = SPF.get(key)
        if sp:
            s['spf'] = sp['mf']
            s['z_spf'] = z_arr(sp['mf'])[0]
            s['da_spf'] = roll(da_arr(sp['mf'], s['act'], lagDA[key]))
            s['aafe_spf'] = roll(ae_arr(sp['mf'], s['act']))
        stats[key] = dict(z_act=(ma_, sa_), z_mf=(mm_, sm_))
    # reference overlays
    n24, s26 = R42['nov24'], R42['sep26']
    series['gdp'].update(m42=n24['gdp_42'], bb=n24['gdp_bb'], m42s=s26['gdp_42'], bbs=s26['gdp_bb'])
    series['cpi'].update(m42=n24['cpi_42'], bb=n24['cpi_bb'])
    series['pce'].update(m42s=s26['pce_42'], bbs=s26['pce_bb'])
    # ---------------- regimes
    regs, hits = {}, {}
    for spec in ('cpi', 'pce'):
        ra = reg_arr(series['gdp']['act'], series[spec]['act'])
        rm = reg_arr(series['gdp']['mf'], series[spec]['mf'])
        rs = reg_arr(series['gdp'].get('spf', np.full(N, np.nan)), series[spec].get('spf', np.full(N, np.nan)))
        regs[spec] = dict(act=ra, model=rm, spf=rs)
        hits[spec] = dict(model=roll(hit_arr(rm, ra)), spf=roll(hit_arr(rs, ra)))
    regs['cpi'].update(m42=n24['reg_42'], bb=n24['reg_bb'], act42=n24['reg_act'])
    hits['cpi'].update(m42=n24['hit_42'], bb=n24['hit_bb'])
    regs['pce'].update(m42=s26['reg_42'], bb=s26['reg_bb'], act42=s26['reg_act'])
    hits['pce'].update(m42=s26['hit_42'], bb=s26['hit_bb'])
    # ---------------- forward (actual-anchored) regime test + calibrated probabilities
    fwd, probs, calib, conf = {}, {}, {}, {}
    for spec, M in infl.items():
        Tlo = max(ev, mi('2005-01'))
        rows = []
        for To in range(Tlo, min(M['T_live'], G['T_live']) + 1):
            Qo = G['Q'][To]
            for j in range(0, 18):
                t = To + j

                def gval(mo):
                    q = mo // 3
                    if q <= Qo:
                        return G['aq'][q]
                    h = q - Qo
                    return G['Y_md'][To, h] if h <= H_GDP else np.nan

                def ival(mo):
                    if mo <= To - 1:
                        return M['a_rep'][mo]
                    h = mo - (To - 1)
                    return M['Y_md'][To, h] if h <= H_INF else np.nan
                dg = gval(t) - gval(t - 3)
                di = ival(t) - ival(t - 3)
                adg = series['gdp']['act'][t] - series['gdp']['act'][t - 3] if t < N else np.nan
                adi = series[spec]['act'][t] - series[spec]['act'][t - 3] if t < N else np.nan
                ms = min(3 * Qo + 2, To - 1)
                pers = regs[spec]['act'][ms]
                rows.append((To, j, dg, di, adg, adi, pers))
        rows = np.array(rows, dtype=object)
        J = rows[:, 1].astype(int)
        dg, di = rows[:, 2].astype(float), rows[:, 3].astype(float)
        adg, adi = rows[:, 4].astype(float), rows[:, 5].astype(float)
        pr_reg = regime(dg, di)
        ac_reg = regime(adg, adi)
        out_j = {'j': list(range(18)), 'model': [], 'persist': [], 'n': []}
        for j in range(18):
            sel = (J == j)
            okr = np.array([pr_reg[i] is not None and ac_reg[i] is not None for i in range(len(J))]) & sel
            okp = np.array([rows[i, 6] is not None and ac_reg[i] is not None for i in range(len(J))]) & sel
            out_j['model'].append(float(np.mean([pr_reg[i] == ac_reg[i] for i in np.where(okr)[0]])) if okr.sum() else None)
            out_j['persist'].append(float(np.mean([rows[i, 6] == ac_reg[i] for i in np.where(okp)[0]])) if okp.sum() else None)
            out_j['n'].append(int(okr.sum()))
        fwd[spec] = out_j
        buckets = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 17)]
        cal = {}
        for nm, pd_, ad_ in (('g', dg, adg), ('i', di, adi)):
            cal[nm] = []
            for lo, hi in buckets:
                sel = (J >= lo) & (J <= hi) & np.isfinite(pd_) & np.isfinite(ad_) & (ad_ != 0)
                sc = float(np.nanstd(pd_[sel])) or 1.0
                w = logit_fit(pd_[sel] / sc, (ad_[sel] > 0).astype(float))
                acc = float(np.mean(np.sign(pd_[sel]) == np.sign(ad_[sel])))
                cal[nm].append(dict(lo=lo, hi=hi, a=float(w[0]), b=float(w[1]), sc=sc, acc=acc, n=int(sel.sum())))
        calib[spec] = cal
        # live forward paths (actual-anchored)
        g_fwd = np.where(np.isfinite(series['gdp']['act']), series['gdp']['act'], gdp_path['med'])
        i_fwd = np.where(np.isfinite(series[spec]['act']), series[spec]['act'], series[spec]['p_med'])
        start = min(3 * (G['q_last'] + 1), M['m_last'] + 1)
        end = min(3 * (G['q_last'] + H_GDP) + 2, M['m_last'] + H_INF)
        months, pg, pi, PG, PR, PI, PD, modal = [], [], [], [], [], [], [], []
        for t in range(start, end + 1):
            j = t - M['T_live']
            bi = next(b for b, (lo, hi) in enumerate(buckets) if max(j, 0) <= hi)
            dgl = g_fwd[t] - g_fwd[t - 3]
            dil = i_fwd[t] - i_fwd[t - 3]
            cg, ci = cal['g'][bi], cal['i'][bi]
            p_g = 1 / (1 + np.exp(-(cg['a'] + cg['b'] * dgl / cg['sc'])))
            p_i = 1 / (1 + np.exp(-(ci['a'] + ci['b'] * dil / ci['sc'])))
            if np.isfinite(series['gdp']['act'][t]):
                p_g = 1.0 if dgl > 0 else 0.0
            if np.isfinite(series[spec]['act'][t]):
                p_i = 1.0 if dil > 0 else 0.0
            months.append(mlab(t)); pg.append(p_g); pi.append(p_i)
            PG.append(p_g * (1 - p_i)); PR.append(p_g * p_i); PI.append((1 - p_g) * p_i); PD.append((1 - p_g) * (1 - p_i))
            modal.append(regime(np.array([dgl]), np.array([dil]))[0])
        probs[spec] = dict(months=months, pg=pg, pi=pi, G=PG, R=PR, I=PI, D=PD, modal=modal)
        # confusion matrix (42 Macro definition, model vs actual, from 2005)
        cm = np.zeros((4, 4), int)
        lab = 'GRID'
        for t in range(ev, N):
            a_, m_ = regs[spec]['act'][t], regs[spec]['model'][t]
            if a_ is not None and m_ is not None:
                cm[lab.index(a_), lab.index(m_)] += 1
        conf[spec] = cm.tolist()
    # ---------------- scorecards on common windows
    def score(f, a, lag, w0, w1):
        sl = slice(mi(w0), mi(w1) + 1)
        e = ae_arr(f, a)[sl]
        d = da_arr(f, a, lag)[sl]
        return (float(np.nanmean(e)) if np.isfinite(e).any() else None,
                float(np.nanmean(d)) if np.isfinite(d).any() else None, int(np.isfinite(e).sum()))

    def hit_window(rf, ra, w0, w1):
        h = hit_arr(rf, ra)[mi(w0):mi(w1) + 1]
        return float(np.nanmean(h)) if np.isfinite(h).any() else None
    naive = {}
    naive['gdp'] = take(G['aq'], Tm // 3 - 1)
    for key in ('cpi', 'pce'):
        naive[key] = shift(series[key]['act'], 1)
        naive[key][Tm > infl[key]['m_last'] + 1] = np.nan
    score_tbl = {}
    # A) Nov-2024 workbook window: GDP + headline CPI (42 Macro & Bloomberg monthly forecast series)
    wA = ('2014-12', '2024-09')
    rowsA = []
    reg_naive_cpi = reg_arr(naive['gdp'], naive['cpi'])
    for nm, fg, fc, rr in [('Econometric model (this build)', series['gdp']['mf'], series['cpi']['mf'], regs['cpi']['model']),
                           ('42 Macro (Nov-24 file)', n24['gdp_42'], n24['cpi_42'], n24['reg_42']),
                           ('Bloomberg consensus (Nov-24 file)', n24['gdp_bb'], n24['cpi_bb'], n24['reg_bb']),
                           ('SPF median (current quarter)', series['gdp'].get('spf'), series['cpi'].get('spf'), regs['cpi']['spf']),
                           ('Naive (last print)', naive['gdp'], naive['cpi'], reg_naive_cpi)]:
        g_ = score(fg, series['gdp']['act'], 3, *wA)
        c_ = score(fc, series['cpi']['act'], 1, *wA)
        rowsA.append(dict(name=nm, gdp_aafe=g_[0], gdp_da=g_[1], cpi_aafe=c_[0], cpi_da=c_[1],
                          hit=hit_window(rr, regs['cpi']['act'], *wA), n=g_[2]))
    score_tbl['A'] = dict(window=f'{mshort(mi(wA[0]))} to {mshort(mi(wA[1]))}', rows=rowsA,
                          reported=dict(gdp_aafe_42=float(np.nanmean(n24['gdp_42_aae'][mi('2024-09'):mi('2024-09') + 1])),
                                        gdp_aafe_bb=float(n24['gdp_bb_aae'][mi('2024-09')]),
                                        cpi_aafe_42=float(n24['cpi_42_aae'][mi('2024-09')]), cpi_aafe_bb=float(n24['cpi_bb_aae'][mi('2024-09')]),
                                        gdp_da_42=float(n24['gdp_42_da'][mi('2024-09')]), gdp_da_bb=float(n24['gdp_bb_da'][mi('2024-09')]),
                                        cpi_da_42=float(n24['cpi_42_da'][mi('2024-09')]), cpi_da_bb=float(n24['cpi_bb_da'][mi('2024-09')]),
                                        hit_42=float(n24['hit_42'][mi('2024-09')]), hit_bb=float(n24['hit_bb'][mi('2024-09')])))
    # B) Sep-2026 screenshot window: GDP + core PCE
    gl = min(3 * G['q_last'] + 2, mi('2026-06'))
    wB = ('2024-08', mlab(min(P['m_last'], mi('2026-08'))))
    reg_naive_pce = reg_arr(naive['gdp'], naive['pce'])
    rowsB = []
    for nm, fg, fc, rr in [('Econometric model (this build)', series['gdp']['mf'], series['pce']['mf'], regs['pce']['model']),
                           ('42 Macro (Sep-26 screenshot)', s26['gdp_42'], s26['pce_42'], s26['reg_42']),
                           ('Bloomberg consensus (Sep-26 screenshot)', s26['gdp_bb'], s26['pce_bb'], s26['reg_bb']),
                           ('SPF median (current quarter)', series['gdp'].get('spf'), series['pce'].get('spf'), regs['pce']['spf']),
                           ('Naive (last print)', naive['gdp'], naive['pce'], reg_naive_pce)]:
        g_ = score(fg, series['gdp']['act'], 3, wB[0], mlab(gl))
        c_ = score(fc, series['pce']['act'], 1, *wB)
        rowsB.append(dict(name=nm, gdp_aafe=g_[0], gdp_da=g_[1], pce_aafe=c_[0], pce_da=c_[1],
                          hit=hit_window(rr, regs['pce']['act'], wB[0], mlab(gl)), n=c_[2]))
    score_tbl['B'] = dict(window=f'{mshort(mi(wB[0]))} to {mshort(mi(wB[1]))} (GDP to {mshort(gl)})', rows=rowsB)
    # C) full backtest, model vs SPF vs naive, with and without COVID
    rowsC = []
    lastm = {'gdp': 3 * G['q_last'] + 2, 'cpi': C['m_last'], 'pce': P['m_last']}
    for key in ('gdp', 'cpi', 'pce'):
        a = series[key]['act']
        for nm, f in [('Econometric model', series[key]['mf']), ('SPF median', series[key].get('spf')), ('Naive', naive[key])]:
            if f is None:
                continue
            e = ae_arr(f, a)
            d = da_arr(f, a, lagDA[key])
            sl = slice(ev, lastm[key] + 1)
            cov = np.zeros(N, bool)
            cov[mi('2020-01'):mi('2021-12') + 1] = True
            ex = np.arange(N)[sl]
            ex = ex[~cov[ex]]
            rowsC.append(dict(target=key, name=nm, aafe=float(np.nanmean(e[sl])), aafe_x=float(np.nanmean(e[ex])),
                              da=float(np.nanmean(d[sl])), da_x=float(np.nanmean(d[ex]))))
    for spec in ('cpi', 'pce'):
        for nm, rr in [('Econometric model', regs[spec]['model']), ('SPF median', regs[spec]['spf']),
                       ('Naive', reg_naive_cpi if spec == 'cpi' else reg_naive_pce)]:
            h = hit_arr(rr, regs[spec]['act'])
            rowsC.append(dict(target='hit_' + spec, name=nm, hit=float(np.nanmean(h[ev:])),
                              hit_t10=float(np.nanmean(h[max(ev, lastm['gdp'] - 119):lastm['gdp'] + 1]))))
    score_tbl['C'] = dict(window=f'{mshort(ev)} to latest', rows=rowsC)
    # ---------------- outlook comparison (quarterly)
    outlook = []
    for h in range(1, H_GDP + 1):
        q = G['q_last'] + h
        ms = [3 * q, 3 * q + 1, 3 * q + 2]

        def qm(arr):
            v = [arr[m] for m in ms if m < N and np.isfinite(arr[m])]
            return float(np.mean(v)) if len(v) == 3 else None

        def qm_part(arr):
            v = [arr[m] for m in ms if m < N and np.isfinite(arr[m])]
            return float(np.mean(v)) if v else None
        pce_mix = np.where(np.isfinite(series['pce']['act']), series['pce']['act'], series['pce']['p_med'])
        cpi_mix = np.where(np.isfinite(series['cpi']['act']), series['cpi']['act'], series['cpi']['p_med'])
        outlook.append(dict(q=qlab(q), gdp_med=G['Y_md'][G['T_live'], h], gdp_smrm=G['Y_sm'][G['T_live'], h], gdp_abnm=G['Y_ab'][G['T_live'], h],
                            gdp_42=qm_part(s26['gdp_42']), gdp_bb=qm_part(s26['gdp_bb']), gdp_spf=qm_part(SPF['gdp']['path']) if 'gdp' in SPF else None,
                            pce_med=qm(pce_mix), pce_42=qm(s26['pce_42']), pce_bb=qm(s26['pce_bb']),
                            pce_spf=qm_part(SPF['pce']['path']) if 'pce' in SPF else None,
                            cpi_med=qm(cpi_mix), cpi_spf=qm_part(SPF['cpi']['path']) if 'cpi' in SPF else None))
    # ---------------- agents payload
    def agent_block(M, feats, H):
        lst = []
        for f in feats:
            k = f['key']
            lv = M['live'][k]
            lst.append(dict(key=k, f42=f['f42'], w42=f['w42'], proxy=f['label'], ids=f['ids'], tr=f['tr'], note=f['note'],
                            last=mlab(f['last']), delta=int(M['deltas'][k]), x=float(f['d'][f['last']]) if np.isfinite(f['d'][f['last']]) else None,
                            lvl=float(f['v'][f['last']]) if np.isfinite(f['v'][f['last']]) else None,
                            r2=lv['r2'], w=lv['w'], yhat=lv['yhat'], b=lv['b'], n=lv['n']))
        wh = M['whist']
        t0 = mi('2005-01')
        hist = {f['key']: [None if not np.isfinite(x) else round(float(x), 4) for x in wh[i, t0:M['T_live'] + 1]] for i, f in enumerate(feats)}
        return dict(H=H, hsel=M['hsel'], list=lst, whist=dict(start=mlab(t0), w=hist))
    agents = dict(gdp=agent_block(G, gfe, H_GDP), cpi=agent_block(C, ife, H_INF), pce=agent_block(P, ife, H_INF))
    # ---------------- SMRM payload
    smrm = {}
    t0 = mi('2014-01')
    smrm['gdp'] = dict(mu=G['smrm']['mu'], rho=G['smrm']['rho'], x0=G['smrm']['x0'], unit='quarter',
                       hl=float(np.log(0.5) / np.log(G['smrm']['rho'])) if 0 < G['smrm']['rho'] < 1 else None,
                       hist_x=[qlab(q) for q in range(t0 // 3, G['q_last'] + 1)], hist=[G['gq'][q] for q in range(t0 // 3, G['q_last'] + 1)],
                       path_x=[qlab(G['q_last'] + h) for h in range(1, H_GDP + 1)], path=list(G['smrm']['path']))
    for key, M in infl.items():
        smrm[key] = dict(mu=M['smrm']['mu'], rho=M['smrm']['rho'], x0=M['smrm']['x0'], unit='month',
                         hl=float(np.log(0.5) / np.log(M['smrm']['rho'])) if 0 < M['smrm']['rho'] < 1 else None,
                         hist_x=[mlab(m) for m in range(t0, M['m_last'] + 1)], hist=[M['g3'][m] for m in range(t0, M['m_last'] + 1)],
                         hist_raw=[M['g'][m] for m in range(t0, M['m_last'] + 1)],
                         path_x=[mlab(M['m_last'] + h) for h in range(1, H_INF + 1)], path=list(M['smrm']['path']))
    # ---------------- horizon metrics incl. SPF (GDP at intra-quarter position 2 ~ mid-quarter survey timing)
    hor = dict(gdp=G['hor'], cpi=C['hor'], pce=P['hor'])
    if 'gdp' in SPF:
        hor['gdp']['aafe_spf'] = [float(np.mean([e for e, q in SPF['gdp']['hz'][j]]) * 100) if SPF['gdp']['hz'][j] else None for j in range(5)] + [None]
        hor['gdp']['aafex_spf'] = [float(np.mean([e for e, q in SPF['gdp']['hz'][j] if not (qi('2020Q1') <= q <= qi('2021Q4'))]) * 100) for j in range(5)] + [None]
    # ---------------- comparability flags: quarterly SPF inflation and 0.1pp-rounded screenshot series cannot be
    # scored on 1-month directional accuracy (ties), so those cells are blanked rather than shown misleadingly low
    for row in score_tbl['A']['rows']:
        if row['name'].startswith('SPF'):
            row['cpi_da'] = None
    for row in score_tbl['B']['rows']:
        if 'screenshot' in row['name']:
            row['gdp_da'] = None; row['pce_da'] = None
        if row['name'].startswith('SPF'):
            row['pce_da'] = None
    for row in score_tbl['C']['rows']:
        if row['name'].startswith('SPF') and row['target'] in ('cpi', 'pce'):
            row['da'] = None; row['da_x'] = None
    # ---------------- findings (generated from the numbers, not hard-coded)
    RN = {'G': 'Goldilocks', 'R': 'Reflation', 'I': 'Inflation', 'D': 'Deflation'}

    def mf_(lbl):
        return pd.Period(lbl, 'M').strftime('%b-%y')

    def runs(months, rg):
        out = []
        for m, r in zip(months, rg):
            if out and out[-1][0] == r:
                out[-1][2] = m
            else:
                out.append([r, m, m])
        return out
    calls = {}
    for spec in ('pce', 'cpi'):
        pr = probs[spec]
        best = [max('GRID', key=lambda r: pr[r][i]) for i in range(len(pr['months']))]
        pr['best'] = best
        pr['pbest'] = [max(pr[r][i] for r in 'GRID') for i in range(len(pr['months']))]
        cl = []
        for r, a, b in runs(pr['months'], pr['modal']):
            ks = [k_ for k_, m in enumerate(pr['months']) if mi(a) <= mi(m) <= mi(b)]
            cl.append(dict(r=r, a=mf_(a), b=mf_(b), p=float(np.mean([pr[r][k_] for k_ in ks]))))
        calls[spec] = cl
    agree = {}
    fcp = P['m_last'] + 1
    pr = probs['pce']
    mm = [mi(x) for x in pr['months']]
    tot = fw = fm = d42 = 0
    for k_, t in enumerate(mm):
        r42 = s26['reg_42'][t]
        if t >= fcp and r42 is not None:
            tot += 1
            fw += pr['best'][k_] == r42
            fm += pr['modal'][k_] == r42
            d42 += regs['pce']['model'][t] == r42
    agree = dict(n=tot, best=fw, modal=fm, def42=d42,
                 first=mshort(fcp), last=mshort(max([t for t in mm if s26['reg_42'][t] is not None] or [fcp])))
    la = max(i for i in range(N) if regs['pce']['act'][i] is not None)
    lac = max(i for i in range(N) if regs['cpi']['act'][i] is not None)
    o = outlook
    hz = G['hor']
    fwd3 = fwd['pce']['model'][3]; per3 = fwd['pce']['persist'][3]
    j_fade = next((j for j in range(18) if fwd['pce']['model'][j] is not None and fwd['pce']['model'][j] < 0.30), None)
    A_ = {r_['name'].split(' (')[0]: r_ for r_ in score_tbl['A']['rows']}
    C_ = {(r_['target'], r_['name']): r_ for r_ in score_tbl['C']['rows']}
    ax_end = min(3 * (G['q_last'] + H_GDP) + 2, C['m_last'] + H_INF, P['m_last'] + H_INF)
    pce_path = [(t, series['pce']['p_med'][t]) for t in range(P['m_last'] + 1, min(P['m_last'] + H_INF, ax_end) + 1)]
    pk = max(pce_path, key=lambda z: z[1])
    cpi_path = [(t, series['cpi']['p_med'][t]) for t in range(C['m_last'] + 1, min(C['m_last'] + H_INF, ax_end) + 1)]
    cmin, cmax = min(cpi_path, key=lambda z: z[1]), max(cpi_path, key=lambda z: z[1])
    pce_dec = series['pce']['p_med'][mi('2026-12')] if mi('2026-12') > P['m_last'] else np.nan
    findings = []
    findings.append(dict(k='call', t=(f"Latest confirmed bottom-up regime is <b>{RN[regs['pce']['act'][la]]}</b> ({mshort(la)}, Core PCE spec). "
        f"The Econometric Model's projected path from here: " + ', then '.join(f"{RN[c['r']]} ({c['a']}{'–' + c['b'] if c['b'] != c['a'] else ''}, p≈{c['p'] * 100:.0f}%)" for c in calls['pce']) + '.')))
    if agree['n'] and not PUBLIC:
        findings.append(dict(k='agree', t=(f"Against 42 Macro's own Sep-26 projection, the projected regime matches in <b>{agree['modal']} of {agree['n']}</b> overlapping months "
            f"({agree['first']} to {agree['last']}); {agree['def42']} of {agree['n']} on 42 Macro's literal definition (3-month change in the forecast series). "
            f"Calibrated probabilities are less decisive: the highest-probability regime matches {agree['best']} of {agree['n']}.")))
    # GDP: name the quarter leaving the YoY base when it explains the move
    q_nx = G['q_last'] + 2
    gq_base, mu_g = G['gq'][q_nx - 4], G['smrm']['mu']
    why = ''
    if np.isfinite(gq_base) and gq_base < mu_g - 1.0:
        why = f" as the weak {qlab(q_nx - 4)} quarter ({gq_base:.1f}% annualized) leaves the comparison"
    elif np.isfinite(gq_base) and gq_base > mu_g + 1.0:
        why = f" as the strong {qlab(q_nx - 4)} quarter ({gq_base:.1f}% annualized) leaves the comparison"
    d01 = o[1]['gdp_med'] - o[0]['gdp_med']
    move = 'rises to' if d01 > 0.05 else ('falls to' if d01 < -0.05 else 'holds at')
    cmp_ = []
    if not PUBLIC and o[0].get('gdp_42') is not None:
        cmp_.append(f"42 Macro {o[0]['gdp_42']:.1f}")
    if not PUBLIC and o[0].get('gdp_bb') is not None:
        cmp_.append(f"Bloomberg {o[0]['gdp_bb']:.1f}")
    if o[0].get('gdp_spf') is not None:
        cmp_.append(f"SPF {o[0]['gdp_spf']:.1f}")
    findings.append(dict(k='gdp', t=(f"Real GDP YoY: {o[0]['q']} nowcast <b>{o[0]['gdp_med']:.1f}%</b> (mean reversion {o[0]['gdp_smrm']:.1f}, agents {o[0]['gdp_abnm']:.1f})"
        + (' versus ' + ', '.join(cmp_) if cmp_ else '') + f". {o[1]['q']} {move} {o[1]['gdp_med']:.1f}%{why}; the path then settles near {o[3]['gdp_med']:.1f}%.")))
    # Core PCE: horizons relative to the latest print
    pp = series['pce']['p_med']
    m1 = P['m_last'] + 1
    pts = [(P['m_last'] + h, pp[P['m_last'] + h]) for h in (6, 12) if P['m_last'] + h <= ax_end and np.isfinite(pp[P['m_last'] + h])]
    txt = f"Core PCE YoY: {pp[m1]:.2f}% in {mshort(m1)}, peaking at {pk[1]:.2f}% ({mshort(pk[0])})" + ''.join(f", {v:.2f}% by {mshort(m)}" for m, v in pts) + '.'
    if not PUBLIC:
        ov = [tt for tt in range(m1, ax_end + 1) if np.isfinite(s26['pce_42'][tt]) and np.isfinite(pp[tt])]
        if ov:
            tt = ov[-1]
            dd = pp[tt] - s26['pce_42'][tt]
            txt += (f" For {mshort(tt)}, 42 Macro sits at {s26['pce_42'][tt]:.1f}%" + (f" and Bloomberg at {s26['pce_bb'][tt]:.1f}%" if np.isfinite(s26['pce_bb'][tt]) else '')
                    + f": the model runs {abs(dd) * 100:.0f}bp {'below' if dd < 0 else 'above'} 42 Macro.")
    findings.append(dict(k='pce', t=txt))
    # Headline CPI: flag the largest recent monthly print that will leave the YoY base
    gC, muC = C['g'], C['smrm']['mu']
    txt = (f"Headline CPI YoY: 3-month run-rate {C['smrm']['x0']:.1f}% annualized against a {muC:.1f}% stationary mean; over the projection the YoY rate ranges from "
           f"{cmin[1]:.1f}% ({mshort(cmin[0])}) to {cmax[1]:.1f}% ({mshort(cmax[0])})")
    win = [m for m in range(C['m_last'] - 11, C['m_last'] + 1) if np.isfinite(gC[m])]
    if win:
        big = max(win, key=lambda m: abs(gC[m] - muC))
        if abs(gC[big] - muC) > 4 and big + 12 <= ax_end:
            txt += f"; the {mshort(big)} print ({gC[big]:.1f}% annualized) leaves the base in {mshort(big + 12)}, a mechanical step {'down' if gC[big] > muC else 'up'} that month"
    findings.append(dict(k='cpi', t=txt + '.'))
    if not PUBLIC:
        findings.append(dict(k='track', t=(f"On 42 Macro's own scorecard ({score_tbl['A']['window']}, headline-CPI spec, rescored on current-vintage data) the reconstruction hits the regime "
            f"<b>{A_['Econometric model']['hit'] * 100:.0f}%</b> of the time versus {A_['42 Macro']['hit'] * 100:.0f}% for 42 Macro and {A_['Bloomberg consensus']['hit'] * 100:.0f}% for Bloomberg. "
            f"CPI error {A_['Econometric model']['cpi_aafe']:.0f}bp versus {A_['42 Macro']['cpi_aafe']:.0f}bp and {A_['Bloomberg consensus']['cpi_aafe']:.0f}bp; GDP error {A_['Econometric model']['gdp_aafe']:.0f}bp versus {A_['42 Macro']['gdp_aafe']:.0f}bp and {A_['Bloomberg consensus']['gdp_aafe']:.0f}bp.")))
    g_m, g_s = C_[('gdp', 'Econometric model')], C_[('gdp', 'SPF median')]
    hp_m, hp_s = C_[('hit_pce', 'Econometric model')], C_[('hit_pce', 'SPF median')]
    hit_part = f" and on the Core PCE-spec regime hit rate ({hp_s['hit'] * 100:.0f}% vs {hp_m['hit'] * 100:.0f}%)" if hp_s['hit'] > hp_m['hit'] else ''
    if g_s['aafe'] < g_m['aafe']:
        findings.append(dict(k='neg', t=(f"Negative result: the free SPF median beats the model on GDP ({g_s['aafe']:.0f}bp vs {g_m['aafe']:.0f}bp since 2005; {g_s['aafe_x']:.0f} vs {g_m['aafe_x']:.0f}bp ex-COVID)"
            + hit_part + ('. Most of 42 Macro\'s edge over Bloomberg is reproducible with public data; an edge over the professional consensus is not.' if not PUBLIC
                          else '. The professional consensus remains the benchmark to beat.'))))
    else:
        findings.append(dict(k='neg', t=(f"On GDP the model now edges the SPF median ({g_m['aafe']:.0f}bp vs {g_s['aafe']:.0f}bp since 2005; {g_m['aafe_x']:.0f} vs {g_s['aafe_x']:.0f}bp ex-COVID)"
            + (f", though the SPF still leads on the Core PCE-spec regime hit rate ({hp_s['hit'] * 100:.0f}% vs {hp_m['hit'] * 100:.0f}%)" if hit_part else '') + '.')))
    findings.append(dict(k='fwd', t=(f"True forward test: regime called 3 months ahead is right <b>{fwd3 * 100:.0f}%</b> of the time versus {per3 * 100:.0f}% for persistence and 25% for luck"
        + (f"; skill decays to luck by roughly {j_fade} months out." if j_fade else '.'))))
    gs_, ga_, ps_, pa_ = hz['aafe_smrm_k2'][3], hz['aafe_abnm_k2'][3], P['hor']['aafe_smrm'][11], P['hor']['aafe_abnm'][11]
    if gs_ < ga_ and ps_ < pa_:
        findings.append(dict(k='sub', t=(f"Sub-models: the agent ensemble only earns its weight in the nowcast window. Beyond one quarter the stationary mean-reversion model is more accurate "
            f"(GDP 4Q-ahead error {gs_:.0f}bp vs {ga_:.0f}bp for the agents; core PCE 12M-ahead {ps_:.0f}bp vs {pa_:.0f}bp), so the 50/50 median is a conservative choice rather than an optimal one.")))
    else:
        findings.append(dict(k='sub', t=(f"Sub-models: GDP 4Q-ahead error {gs_:.0f}bp for mean reversion vs {ga_:.0f}bp for the agents; core PCE 12M-ahead {ps_:.0f}bp vs {pa_:.0f}bp.")))
    bt = ref.get('bt2023') or {}
    if bt and not PUBLIC:
        findings.append(dict(k='bt', t=(f"Context on 42 Macro's own claims: the Aug-2023 workbook's in-house backtest shows a {bt['median']['full'] * 100:.0f}% regime hit rate for the median framework "
            f"(OECD-CLI growth definition). The live record after the switch to Real GDP is {score_tbl['A']['reported']['hit_42'] * 100:.1f}% (Nov-24) and {s26['hit_42'][mi('2026-06')] * 100:.0f}% (Sep-26).")))
    # ---------------- axis & payload
    ax0 = mi(EVAL_START)
    ax1 = min(3 * (G['q_last'] + H_GDP) + 2, C['m_last'] + H_INF, P['m_last'] + H_INF, N - 1)

    def cut(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.dtype == object:
            return [None if x is None else x for x in arr[ax0:ax1 + 1]]
        return [None if not np.isfinite(x) else round(float(x), 4) for x in arr[ax0:ax1 + 1]]
    S = {}
    for key, s in series.items():
        S[key] = {k: cut(v) for k, v in s.items()}
    REG = {spec: {k: cut(v) for k, v in d.items()} for spec, d in regs.items()}
    HIT = {spec: {k: cut(v) for k, v in d.items()} for spec, d in hits.items()}
    meta = dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        asof=dict(gdp=qlab(G['q_last']), cpi=mshort(C['m_last']), pce=mshort(P['m_last'])),
        live=dict(gdp_T=mshort(G['T_live']), gdp_k=int(G['k_live']), cpi_T=mshort(C['T_live']), pce_T=mshort(P['T_live'])),
        fc_start=dict(gdp=mlab(3 * (G['q_last'] + 1)), cpi=mlab(C['m_last'] + 1), pce=mlab(P['m_last'] + 1)),
        spf_survey=qlab(SPF['gdp']['s_last']) if 'gdp' in SPF else None,
        windows=dict(W_INF=W_INF, W_GDPQ=W_GDPQ, H_INF=H_INF, H_GDP=H_GDP, MIN_INF=MIN_INF, MIN_GDP=MIN_GDP),
        zstats={k: v for k, v in stats.items()},
        n_agents=dict(gdp=len(gfe), infl=len(ife)),
        bls=sorted(bls.keys()),
        bt2023=None if PUBLIC else ref.get('bt2023'),
        public=PUBLIC,
        asof_iso=dict(gdp=str(qp(G['q_last'])), cpi=mlab(C['m_last']), pce=mlab(P['m_last'])),
        log=LOG[-40:],
    )
    tstart = max(mi(TABLE_START), min(3 * (G['q_last'] + 1), C['m_last'] + 1, P['m_last'] + 1) - 25)
    payload = dict(meta=meta, axis=[mlab(i) for i in range(ax0, ax1 + 1)], table_start=mlab(tstart), S=S, REG=REG, HIT=HIT,
                   probs=probs, calib=calib, fwd=fwd, conf=conf, score=score_tbl, outlook=outlook, agents=agents, calls=calls, agree=agree, findings=findings,
                   last_reg=dict(pce=mlab(la), cpi=mlab(lac)),
                   smrm=smrm, hor=hor)
    if PUBLIC:
        strip_proprietary(payload)
    return payload


def strip_proprietary(p):
    """--public: remove every 42 Macro and Bloomberg number so the page can be hosted publicly."""
    for t in p['S'].values():
        for k in ('m42', 'bb', 'm42s', 'bbs'):
            t.pop(k, None)
    for d in list(p['REG'].values()) + list(p['HIT'].values()):
        for k in ('m42', 'bb', 'act42'):
            d.pop(k, None)
    for o in p['outlook']:
        for k in ('gdp_42', 'gdp_bb', 'pce_42', 'pce_bb'):
            o.pop(k, None)
    for blk in ('A', 'B'):
        p['score'][blk]['rows'] = [r for r in p['score'][blk]['rows'] if not r['name'].startswith(('42 Macro', 'Bloomberg'))]
        p['score'][blk].pop('reported', None)
    for t in p['agents'].values():
        for a in t['list']:
            a['w42'] = None
    p['agree'] = dict(n=0)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def sanity(p, min_agents):
    """Hard checks a build must pass before it may replace the live page."""
    errs = []
    na = p['meta']['n_agents']
    if na['gdp'] < min_agents or na['infl'] < min_agents:
        errs.append(f"only {na['gdp']} growth / {na['infl']} inflation agents built (minimum {min_agents})")
    o = p.get('outlook') or []
    if not o or o[0].get('gdp_med') is None or not (-15 < o[0]['gdp_med'] < 20):
        errs.append('GDP nowcast missing or out of range')
    for k, lo, hi in (('gdp', -15, 20), ('pce', -3, 15), ('cpi', -5, 20)):
        v = [x for x in p['S'][k]['p_med'] if x is not None]
        if not v:
            errs.append(f'{k} projection missing')
        elif not all(lo < x < hi for x in v):
            errs.append(f'{k} projection out of range')
    if not p['calls'].get('pce'):
        errs.append('regime projection missing')
    return errs


def staleness(p):
    """Soft warnings: data older than expected (FRED outage served from cache, or a delayed release)."""
    today = pd.Period(dt.date.today(), 'M')
    w = []
    lim = {'cpi': 3, 'pce': 3, 'gdp': 7}
    for k, v in p['meta']['asof_iso'].items():
        per = pd.Period(v, 'Q').asfreq('M', 'end') if k == 'gdp' else pd.Period(v, 'M')
        age = (today - per).n
        if age > lim[k]:
            w.append(f'{k} data end {v} is {age} months old')
    if not p['meta'].get('spf_survey'):
        w.append('SPF consensus unavailable')
    if not p['meta'].get('bls'):
        w.append('BLS API unavailable: five CPI sub-indices used FRED fallbacks')
    return w


def fingerprint(p):
    """Hash of the public-data content only, so switching between --public and full builds is not a data change."""
    import hashlib
    q = json.loads(json.dumps(p))
    strip_proprietary(q)
    q.pop('findings', None)
    for k in ('generated_utc', 'log', 'public', 'bt2023'):
        q['meta'].pop(k, None)
    return hashlib.sha256(json.dumps(q, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def snapshot(p):
    """Compact real-time record of what the model said on this date (for an honest out-of-sample track record later)."""
    def path(k):
        S = p['S'][k]
        return dict(months=[m for m, v in zip(p['axis'], S['p_med']) if v is not None],
                    med=[v for v in S['p_med'] if v is not None],
                    smrm=[v for v, w in zip(S['p_smrm'], S['p_med']) if w is not None],
                    abnm=[v for v, w in zip(S['p_abnm'], S['p_med']) if w is not None])
    keep = ('q', 'gdp_med', 'gdp_smrm', 'gdp_abnm', 'gdp_spf', 'pce_med', 'pce_spf', 'cpi_med', 'cpi_spf')
    m = p['meta']
    return dict(generated_utc=m['generated_utc'], asof=m['asof_iso'], spf_survey=m.get('spf_survey'), live=m['live'],
                outlook=[{k: o.get(k) for k in keep} for o in p['outlook']],
                paths=dict(pce=path('pce'), cpi=path('cpi')), calls=p['calls'],
                probs={s: {k: v for k, v in d.items()} for s, d in p['probs'].items()},
                last_regime={s: [mm, p['REG'][s]['act'][p['axis'].index(mm)]] for s, mm in p['last_reg'].items()},
                smrm={k: {kk: v.get(kk) for kk in ('mu', 'rho', 'x0', 'hl')} for k, v in p['smrm'].items()})


LOG_COLS = ['generated_utc', 'gdp_data_to', 'cpi_data_to', 'pce_data_to', 'spf_survey', 'regime_last_confirmed', 'regime_last_month',
            'gdp_q1', 'gdp_q1_yoy', 'gdp_q2', 'gdp_q2_yoy', 'gdp_q3_yoy', 'gdp_q4_yoy',
            'pce_first_month', 'pce_m1', 'pce_m3', 'pce_m6', 'pce_m12', 'cpi_first_month', 'cpi_m1', 'cpi_m3', 'cpi_m6', 'cpi_m12',
            'call1', 'call1_to', 'call1_p', 'call2', 'call2_to', 'call2_p', 'call3', 'call3_to', 'call3_p']


def log_row(snap):
    o = snap['outlook']

    def h(k, n):
        P = snap['paths'][k]
        return P['med'][n - 1] if len(P['med']) >= n else None
    c = snap['calls']['pce'] + [None, None, None]
    row = [snap['generated_utc'], snap['asof']['gdp'], snap['asof']['cpi'], snap['asof']['pce'], snap['spf_survey'],
           snap['last_regime']['pce'][1], snap['last_regime']['pce'][0],
           o[0]['q'], o[0]['gdp_med'], o[1]['q'], o[1]['gdp_med'], o[2]['gdp_med'], o[3]['gdp_med'],
           snap['paths']['pce']['months'][0], h('pce', 1), h('pce', 3), h('pce', 6), h('pce', 12),
           snap['paths']['cpi']['months'][0], h('cpi', 1), h('cpi', 3), h('cpi', 6), h('cpi', 12)]
    for x in c[:3]:
        row += [x['r'], x['b'], round(x['p'], 3)] if x else ['', '', '']
    return ['' if v is None else (round(v, 4) if isinstance(v, float) else v) for v in row]


def gh_output(**kw):
    fp = os.environ.get('GITHUB_OUTPUT')
    if fp:
        with open(fp, 'a') as f:
            for k, v in kw.items():
                f.write(f'{k}={v}\n')


def gh_summary(lines):
    fp = os.environ.get('GITHUB_STEP_SUMMARY')
    if fp:
        with open(fp, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')


def main():
    global PUBLIC
    ap = argparse.ArgumentParser(description='US GRID Models: rebuild the dashboard from public data.')
    ap.add_argument('--out', default='us_grid_models.html')
    ap.add_argument('--cache-dir', default='.grid_cache')
    ap.add_argument('--offline', action='store_true', help='use cached downloads only')
    ap.add_argument('--payload', default='', help='optional path to also write the JSON payload')
    ap.add_argument('--public', action='store_true', help='strip all 42 Macro and Bloomberg data (for public hosting)')
    ap.add_argument('--ref', default='', help='path to the 42 Macro reference JSON (default ref/ref42.json)')
    ap.add_argument('--fingerprint', default='', help='file holding the last published data fingerprint')
    ap.add_argument('--archive-dir', default='', help='write a forecast snapshot + log row here when the data changed')
    ap.add_argument('--force', action='store_true', help='treat the build as changed even if the data are identical')
    ap.add_argument('--min-agents', type=int, default=22, help='sanity gate: minimum agents per family')
    args = ap.parse_args()
    PUBLIC = args.public
    payload = clean(build(args))
    errs = sanity(payload, args.min_agents)
    warns = staleness(payload)
    for w in warns:
        log(f'  ! warning: {w}')
    if errs:
        for e in errs:
            log(f'  X sanity check failed: {e}')
        gh_output(changed='false', ok='false')
        gh_summary(['### US GRID Models: build FAILED the sanity gate', ''] + [f'- {e}' for e in errs])
        sys.exit(2)
    fpv, mode = fingerprint(payload), ('public' if PUBLIC else 'full')
    prev = (open(args.fingerprint).read().split() + ['', ''])[:2] if args.fingerprint and os.path.exists(args.fingerprint) else ['', '']
    data_changed = args.force or fpv != prev[0]          # new data -> archive a forecast snapshot
    changed = data_changed or mode != prev[1]            # page differs -> redeploy
    js = json.dumps(payload, separators=(',', ':'), ensure_ascii=True).replace('</', '<\\/')
    if args.payload:
        open(args.payload, 'w').write(js)
    tpl = TEMPLATE
    if tpl is None:
        tp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template.html')
        if not os.path.exists(tp):
            log('No template available; payload only.')
            return
        tpl = open(tp, encoding='utf-8').read()
    html, n = re.subn(r'__GRID_PAYLOAD__', lambda m: js, tpl, count=1)
    if n != 1:
        raise SystemExit('payload placeholder not found in template')
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, 'w', encoding='utf-8').write(html)
    log(f'Wrote {args.out} ({len(html) / 1024:.0f} KB)')
    m = payload['meta']
    if data_changed and args.archive_dir:
        snap = snapshot(payload)
        fdir = os.path.join(args.archive_dir, 'forecasts')
        os.makedirs(fdir, exist_ok=True)
        day = m['generated_utc'][:10]
        json.dump(snap, open(os.path.join(fdir, f'{day}.json'), 'w'), separators=(',', ':'))
        lp = os.path.join(args.archive_dir, 'forecast_log.csv')
        import csv
        new = not os.path.exists(lp)
        with open(lp, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(LOG_COLS)
            w.writerow(log_row(snap))
        log(f'Archived forecast snapshot {day}')
    if args.fingerprint and changed:
        os.makedirs(os.path.dirname(args.fingerprint) or '.', exist_ok=True)
        open(args.fingerprint, 'w').write(f'{fpv} {mode}\n')
    log('Data changed since last publish' if data_changed else ('Build mode changed (page republished, nothing archived)' if changed else 'No data change since last publish'))
    gh_output(changed='true' if changed else 'false', ok='true', asof=f"GDP {m['asof']['gdp']}, CPI {m['asof']['cpi']}, core PCE {m['asof']['pce']}")
    calls = '; '.join(f"{c['r']} {c['a']}-{c['b']} ({c['p'] * 100:.0f}%)" for c in payload['calls']['pce'])
    o = payload['outlook']
    gh_summary(['### US GRID Models' + (' (public build)' if PUBLIC else ''), '',
                f"- Data to: GDP {m['asof']['gdp']}, headline CPI {m['asof']['cpi']}, core PCE {m['asof']['pce']}; SPF {m.get('spf_survey')}",
                f"- Changed since last publish: **{'yes' if changed else 'no'}**",
                f"- GDP YoY: {o[0]['q']} {o[0]['gdp_med']:.1f}%, {o[1]['q']} {o[1]['gdp_med']:.1f}%",
                f"- Regime path (core PCE spec): {calls}"] + [f'- Warning: {w}' for w in warns])


if __name__ == '__main__':
    main()
