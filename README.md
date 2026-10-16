# US GRID Models: auto-refreshing dashboard

Real GDP and inflation forecasts rebuilt from 42 Macro's GRID process (a stationary mean-reversion model plus a 27-agent nowcast ensemble per variable, split 50/50, mapped to Goldilocks / Reflation / Inflation / Deflation). A GitHub Actions workflow rebuilds the page twice every weekday from FRED, BLS and the Philadelphia Fed SPF, and publishes it to GitHub Pages only when the underlying data have changed.

```
schedule ─► rebuild.py ─► sanity gate ─► data changed? ─┬─ yes ─► deploy to Pages + commit forecast snapshot
                                                        └─ no  ─► stop (live page untouched)
```

## What refreshes and what doesn't

| Refreshes on every new release | Static |
|---|---|
| All actual data (GDP, CPI, core PCE, the 54 agent features) | 42 Macro forecasts, regimes and hit rates (Nov-2024 workbook, Sep-2026 screenshot) |
| Both sub-models, re-estimated at every origin | Bloomberg consensus (same two sources) |
| Regime probabilities, calibration, forward tests | 42 Macro's Aug-2023 agent weights |
| SPF consensus (new survey each quarter) | |
| Track-record tables, findings text, GRID table window | |

The static series stay on the page and drop out naturally once the projection moves past their horizon (Aug-2027 for the screenshot). The findings are regenerated from the numbers on every build, so the narrative never goes stale.

## Step 1: choose public or private

The 42 Macro and Bloomberg numbers come from paid subscriptions. Don't put them on a public website. `rebuild.py` never contains them; they live in a separate file, `ref42.json`, that you add only to a private setup.

| Setup | 42 Macro / Bloomberg data | Where you view it |
|---|---|---|
| **A. Public repo** (any GitHub plan) | Stripped automatically | Public Pages site, e.g. `https://YOUR-USER.github.io/us-grid-models/` |
| **B. Private repo, Enterprise Cloud** | Included | Pages site with access restricted to your organisation |
| **C. Private repo, any other plan** | Included | Download `us-grid-dashboard` from each workflow run (Actions tab) and open it locally |

Outside Enterprise Cloud a Pages site is public even when the repo is private (and Free plans can't publish Pages from a private repo at all), so setup C turns Pages off.

A public build still names 42 Macro's 54 features and describes the method in the Method tab. If that is also sensitive, use B or C.

## Step 2: create the repo and add the files

1. Create a new repository, e.g. `us-grid-models` (public for A, private for B or C).
2. Upload these files, keeping the folder structure. `.github/workflows/refresh.yml` must sit exactly there.

```
rebuild.py
requirements.txt
.gitignore
README.md
.github/workflows/refresh.yml
ref/ref42.json            <- supplied separately; private setups B and C only. Never commit it to a public repo.
```

With the web uploader, drag the unzipped folder contents in. Make sure `.github` and `.gitignore` come along; macOS Finder hides dot-files (press Cmd+Shift+. to show them).

For a private repo you can store the reference data as a secret instead of committing `ref/ref42.json`: create a secret named `REF42_JSON` and paste the whole file contents into it (it fits under the 48 KB secret limit).

## Step 3: secrets and settings

**Settings > Secrets and variables > Actions > Secrets**

| Secret | Required | Notes |
|---|---|---|
| `FRED_API_KEY` | Yes | Free at fred.stlouisfed.org (My Account > API Keys). If a key has been pasted into chats or code, generate a fresh one. |
| `BLS_API_KEY` | Recommended | Free at data.bls.gov/registrationEngine. Without it the BLS API allows 25 calls a day per IP, and GitHub runners share IPs; the build then falls back to the last cached BLS data. |
| `REF42_JSON` | Optional | Private setups only, instead of committing `ref/ref42.json`. |

**Settings > Secrets and variables > Actions > Variables** (all optional)

| Variable | Value | Effect |
|---|---|---|
| `GRID_PUBLIC` | `true` | Force a public build even when reference data is present. A build with no reference data is always public. |
| `GRID_PAGES` | `false` | Skip the Pages deploy (setup C). |

**Settings > Actions > General > Workflow permissions:** choose **Read and write permissions** and save. The workflow commits forecast snapshots to `archive/`.

**Settings > Pages > Build and deployment > Source:** choose **GitHub Actions** (setups A and B). For B, also set the site's visibility to private on the same page.

## Step 4: first run

Actions tab > **Refresh US GRID dashboard** > **Run workflow**. The first run takes about three minutes (around 80 data downloads at FRED's rate limit). When it finishes:

- the `deploy` job shows the site URL (A and B);
- the run's **Summary** shows the data dates, whether anything changed, and the regime path;
- `archive/` appears in the repo with the first forecast snapshot.

A manual run always republishes the page. Tick **force** to also archive a snapshot when nothing changed.

## Schedule

Weekdays at 16:17 and 22:47 UTC: roughly 3am and 10am in Melbourne during daylight saving, an hour earlier otherwise. The first run catches the 08:30 and 10:00 New York releases once FRED has posted them; the second catches late postings. Edit the two `cron` lines in `refresh.yml` to change it (times are always UTC).

Most weekdays bring at least one relevant release, so the first run usually publishes and the second usually stops without touching the live page.

## The forecast archive

Every time the data change, the workflow commits:

- `archive/forecast_log.csv`: one row per data change with the GDP nowcasts, core PCE and CPI paths at fixed horizons, and the projected regime calls with their probabilities. It opens straight in Excel.
- `archive/forecasts/YYYY-MM-DD.json`: the full forecast snapshot for that day (paths, both sub-models, regime probabilities).
- `archive/fingerprint.txt`: a hash of the data behind the last published page.

The dashboard's own track record is scored on today's revised data, which flatters GDP. This archive is a genuine real-time record: in a year you can score what the model actually said against what happened.

## Monitoring

- **Failure emails:** GitHub emails you when a scheduled run fails. Check your notification settings if you don't get them.
- **Sanity gate:** `rebuild.py` exits with code 2 if fewer than 22 of 27 agents build in either family, or a projection is missing or out of range. The run then fails and the live page stays as it was.
- **Warnings** (shown in the run Summary, never blocking): data older than expected, SPF unavailable, or BLS falling back to FRED proxies.
- **Cache:** raw downloads are kept between runs in the Actions cache. If FRED, BLS or the Philadelphia Fed is down, the build uses the last good copy instead of failing.

## Running locally

```bash
pip install -r requirements.txt
export FRED_API_KEY=your_key            # optional: export BLS_API_KEY=...
python rebuild.py --out us_grid_models.html             # full build, needs ref/ref42.json
python rebuild.py --public --out us_grid_models.html    # public build
python rebuild.py --offline --out us_grid_models.html   # reuse cached downloads only
```

Other flags: `--fingerprint FILE`, `--archive-dir DIR`, `--force`, `--min-agents N`, `--payload FILE` (write the raw JSON), `--ref FILE`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Repository secret FRED_API_KEY is not set` | Add the secret (Step 3). Secrets are case-sensitive. |
| `deploy` fails with "Get Pages site failed" or a 404 | Pages source isn't set to **GitHub Actions**, or the plan doesn't allow Pages on a private repo. Fix the setting, or set `GRID_PAGES=false`. |
| `deploy` fails with "Branch is not allowed to deploy to github-pages" | The workflow must run from the default branch. Under Settings > Environments > github-pages, allow your branch. |
| Archive commit fails with a 403 | Workflow permissions are read-only. Switch to read and write (Step 3). |
| Run Summary warns that BLS fell back to FRED | The shared runner IP hit the BLS daily limit. Add `BLS_API_KEY`. |
| FRED HTTP 429 | Rate limit. The script retries with backoff; a re-run normally clears it. |
| `X sanity check failed` | A source is down or a series was renamed. The run log names which feature dropped out; the live page is unaffected. |
| Page shows old numbers after a successful run | Nothing changed, so nothing was deployed. Check the run Summary, or run manually to force a republish. |
| Scheduled runs stopped | GitHub pauses schedules in public repos after 60 days without activity. The archive commits normally prevent this; if it happens, re-enable the workflow in the Actions tab. |
| Downloads: Excel button does nothing | The Excel export loads SheetJS from cdnjs. A network or content blocker can stop it; CSV and JSON still work. |

## Data sources

FRED (St. Louis Fed) for targets and most features; BLS public API for five CPI sub-indices; Philadelphia Fed Survey of Professional Forecasters median levels as the consensus benchmark. Substitutions for proprietary inputs (ISM, NFIB, CRB, MBA, OECD confidence) are listed in the dashboard's Method tab.
