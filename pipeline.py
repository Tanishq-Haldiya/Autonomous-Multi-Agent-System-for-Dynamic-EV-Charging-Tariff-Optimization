"""
Autonomous Multi-Agent System for Dynamic EV Charging Tariff Optimization

Pipeline:
  1. Preprocessing: UrbanEV district time series (247 grids) joined to information.csv on `grid`
     (stations.csv is station-level, a different entity); occupancy rate = occupied piles / total piles,
     charging-time utilization = duration / pile-hours; real timestamps from time.csv; ACN times GMT -> Pacific.
  2. EDA: intraday/weekly profiles, peak/shoulder/off-peak volatility, district ranking, ACN idle time.
  3. Demand Prediction Agent: 1-hour-ahead occupancy, kWh and congestion probability; lag features only,
     chronological train/valid/test split, Optuna-tuned LightGBM/XGBoost/RandomForest vs naive baselines.
  4. Price response: elasticity estimated from time-of-use price variation (descriptive), then stress-tested.
  5. Tariff Pricing Agent: surge/discount rule + revenue optimiser; revenue = kWh x tariff; Erlang-C wait proxy.
  6. Monitoring & Learning Agent: daily episodes, randomized price tests, Bayesian elasticity updates, regret.

Run:  python pipeline.py        (outputs -> outputs/)
"""
import json
import os
import warnings

warnings.filterwarnings('ignore')   # set before library imports: joblib warns about physical-core detection on Windows

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (average_precision_score, brier_score_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
URBAN_DIR = 'data/UrbanEV_SZ_districts'
ACN_PATH = 'data/ACN/acndata_sessions.json.xlsx'
OUT = 'outputs'
FIG = os.path.join(OUT, 'figures')

BASE_PRICE = 15.0          # Rs/kWh fixed baseline mandated by the problem statement
SURGE_T, DISC_T = 0.80, 0.30
SURGE_GRID = [1.1, 1.2, 1.3, 1.4, 1.5]
DISC_GRID = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
ELASTICITY_SCENARIOS = [-0.2, -0.5, -0.8, -1.2]

# Chronological split on day index (0 = 2022-06-19). Days 0-6 are consumed by the 168h lag warm-up.
TRAIN_DAYS = range(7, 19)      # 12 days  (Jun 26 - Jul 07)
VALID_DAYS = range(19, 23)     # 4 days   (Jul 08 - Jul 11)  -> Optuna tuning + model selection
TEST_DAYS = range(23, 30)      # 7 days   (Jul 12 - Jul 18)  -> untouched until final evaluation

os.makedirs(FIG, exist_ok=True)
plt.style.use('ggplot')
results = {}


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=130)
    plt.close()


# ----------------------------------------------------------------------------------------------------------------
# 1. DATA PREPROCESSING
# ----------------------------------------------------------------------------------------------------------------
print('1. Loading UrbanEV (district level)...')
info = pd.read_csv(f'{URBAN_DIR}/information.csv')
t = pd.read_csv(f'{URBAN_DIR}/time.csv', encoding='utf-8-sig')
ts5 = pd.to_datetime(t[['year', 'month', 'day', 'hour', 'minute']])

raw = {}
for name in ['occupancy', 'duration', 'volume', 'price']:
    df = pd.read_csv(f'{URBAN_DIR}/{name}.csv').drop(columns='timestamp')
    raw[name] = df
grids = raw['occupancy'].columns.astype(int)
for name, df in raw.items():
    assert (df.columns.astype(int) == grids).all(), f'{name} columns not aligned'
info = info.set_index('grid').loc[grids]

missing = {k: int(v.isna().sum().sum()) for k, v in raw.items()}
# Documented missing-value policy: forward-fill within district (sensor dropout), then 0 (no activity observed).
mats = {k: v.ffill().fillna(0).values.astype(float) for k, v in raw.items()}

T5, N = mats['occupancy'].shape
H = T5 // 12
cnt = info['count'].values.astype(float)
over_capacity = int((mats['occupancy'] > cnt).sum())


def hourly(a):
    return a.reshape(H, 12, N)


occ_rate = np.clip(hourly(mats['occupancy']).mean(1) / cnt, 0, 1)             # occupied piles / total piles
charge_util = np.clip(hourly(mats['duration']).sum(1) / cnt, 0, 1)            # charging hours / available pile-hours
energy = hourly(mats['volume']).sum(1)                                         # kWh delivered in the hour
price = hourly(mats['price']).mean(1)                                          # observed tariff (CNY/kWh)
saturation = (hourly(mats['occupancy']) >= 0.95 * cnt).mean(1)                 # queue-length proxy
hour_ts = ts5.values.reshape(H, 12)[:, 0]

panel = pd.DataFrame({
    'time': np.repeat(hour_ts, N),
    'grid': np.tile(grids, H),
    'occ_rate': occ_rate.ravel(),
    'charge_util': charge_util.ravel(),
    'energy': energy.ravel(),
    'price': price.ravel(),
    'saturation': saturation.ravel(),
})
static = info[['count', 'fast_count', 'slow_count', 'area', 'lon', 'la', 'CBD', 'dynamic_pricing']].copy()
static['fast_share'] = static['fast_count'] / static['count']
static['pile_density'] = static['count'] / static['area']
panel = panel.merge(static, left_on='grid', right_index=True, how='left')
panel['hour'] = panel['time'].dt.hour
panel['dow'] = panel['time'].dt.dayofweek
panel['is_weekend'] = (panel['dow'] >= 5).astype(int)
panel['day_idx'] = (panel['time'] - panel['time'].min()).dt.days
panel['occupied_pile_hours'] = panel['occ_rate'] * panel['count']
panel['revenue_fixed'] = panel['energy'] * BASE_PRICE
panel['congested'] = (panel['occ_rate'] > SURGE_T).astype(int)

results['data'] = {
    'districts': int(N), 'piles': int(cnt.sum()), 'five_min_intervals': int(T5), 'hours': int(H),
    'five_min_records': int(T5 * N), 'hourly_records': int(len(panel)),
    'period': f'{ts5.min():%Y-%m-%d} to {ts5.max():%Y-%m-%d}',
    'missing_values': missing, 'occupancy_above_capacity_5min_cells': over_capacity,
    'mean_occupancy_rate': round(float(panel.occ_rate.mean()), 4),
    'mean_charging_utilization': round(float(panel.charge_util.mean()), 4),
    'share_hours_above_80pct': round(float((panel.occ_rate > SURGE_T).mean()), 4),
    'share_hours_below_30pct': round(float((panel.occ_rate < DISC_T).mean()), 4),
}
print('   ', results['data'])

print('1b. Loading ACN (session level)...')
acn = pd.read_excel(ACN_PATH)
acn = acn[acn['connectionTime'].notna()].drop_duplicates('sessionID').copy()
acn_raw_rows = len(acn)
for c in ['connectionTime', 'disconnectTime', 'doneChargingTime']:
    acn[c] = pd.to_datetime(acn[c], utc=True, errors='coerce').dt.tz_convert('America/Los_Angeles')
acn['connected_h'] = (acn['disconnectTime'] - acn['connectionTime']).dt.total_seconds() / 3600
acn['charging_h'] = ((acn['doneChargingTime'] - acn['connectionTime']).dt.total_seconds() / 3600)
acn['charging_h'] = acn['charging_h'].clip(lower=0)
acn['charging_h'] = np.minimum(acn['charging_h'], acn['connected_h'])
acn = acn[(acn['connected_h'] > 0) & (acn['connected_h'] <= 48) & (acn['kWhDelivered'] > 0)].copy()
acn['charging_h'] = acn['charging_h'].fillna(acn['connected_h'])   # missing doneChargingTime -> assume charged whole stay
acn['idle_h'] = acn['connected_h'] - acn['charging_h']
acn['session_utilization'] = acn['charging_h'] / acn['connected_h']
acn['revenue_per_session'] = acn['kWhDelivered'] * BASE_PRICE
acn['arrival_hour'] = acn['connectionTime'].dt.hour
acn['is_weekend'] = acn['connectionTime'].dt.dayofweek >= 5
results['acn'] = {
    'raw_sessions': int(acn_raw_rows), 'clean_sessions': int(len(acn)),
    'sites': acn['siteID'].value_counts().to_dict(),
    'median_connected_h': round(float(acn.connected_h.median()), 2),
    'median_charging_h': round(float(acn.charging_h.median()), 2),
    'mean_session_utilization': round(float(acn.session_utilization.mean()), 3),
    'share_of_plugged_time_idle': round(float(acn.idle_h.sum() / acn.connected_h.sum()), 3),
    'mean_kwh_per_session': round(float(acn.kWhDelivered.mean()), 2),
    'mean_revenue_per_session_rs': round(float(acn.revenue_per_session.mean()), 2),
    'weekend_share_of_sessions': round(float(acn.is_weekend.mean()), 3),
    'peak_arrival_hour': int(acn.arrival_hour.value_counts().idxmax()),
}
print('   ', results['acn'])

# ----------------------------------------------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------------------------------------------------
print('2. EDA...')
prof = panel.groupby(['hour', 'is_weekend'])['occ_rate'].mean().unstack()
dyn_price = panel[panel.dynamic_pricing == 1].groupby('hour')['price'].mean()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(prof.index, prof[0], marker='o', label='Weekday occupancy')
ax.plot(prof.index, prof[1], marker='o', label='Weekend occupancy')
ax.axhline(DISC_T, ls='--', c='grey', lw=1)
ax.set_xlabel('Hour of day (Shenzhen local)'); ax.set_ylabel('Mean occupancy rate')
ax2 = ax.twinx(); ax2.plot(dyn_price.index, dyn_price.values, c='black', ls=':', label='Mean price, dynamic-pricing districts')
ax2.set_ylabel('Observed price (CNY/kWh)'); ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)
ax.set_title('Occupancy peaks overnight, when time-of-use prices are lowest')
save_fig('01_hourly_profile_vs_price.png')

heat = panel.groupby(['dow', 'hour'])['occ_rate'].mean().unstack()
plt.figure(figsize=(12, 4))
plt.imshow(heat.values, aspect='auto', cmap='viridis')
plt.colorbar(label='Mean occupancy rate')
plt.yticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']); plt.xticks(range(24))
plt.xlabel('Hour of day'); plt.title('Occupancy by day of week and hour')
plt.grid(False)
save_fig('02_heatmap_dow_hour.png')

# Peak / shoulder / off-peak defined from the data (system hourly profile quartiles)
sys_prof = panel.groupby('hour')['occ_rate'].mean()
q25, q75 = sys_prof.quantile([0.25, 0.75])
period_of_hour = sys_prof.apply(lambda v: 'peak' if v >= q75 else ('off-peak' if v <= q25 else 'shoulder'))
panel['period'] = panel['hour'].map(period_of_hour)
vol = (panel.groupby(['grid', 'period'])['occ_rate']
       .agg(mean='mean', std='std')
       .assign(cv=lambda d: d['std'] / d['mean'].replace(0, np.nan))
       .groupby('period').agg(mean_occupancy=('mean', 'mean'), mean_within_district_std=('std', 'mean'),
                              mean_coef_of_variation=('cv', 'mean')))
vol['hours_of_day'] = period_of_hour.groupby(period_of_hour).apply(lambda s: ','.join(map(str, s.index)))
vol.to_csv(os.path.join(OUT, 'volatility_by_period.csv'))
results['periods'] = {p: list(map(int, period_of_hour[period_of_hour == p].index)) for p in ['peak', 'shoulder', 'off-peak']}

rank = panel.groupby('grid').agg(
    count=('count', 'first'), CBD=('CBD', 'first'), dynamic_pricing=('dynamic_pricing', 'first'),
    mean_occupancy=('occ_rate', 'mean'), share_hours_over_80=('congested', 'mean'),
    share_hours_under_30=('occ_rate', lambda s: (s < DISC_T).mean()),
    mean_saturation=('saturation', 'mean'), kwh_per_day=('energy', lambda s: s.sum() / 30))
rank['typical_busiest_hour'] = panel.groupby(['grid', 'hour'])['occ_rate'].mean().unstack().idxmax(axis=1)
rank['status'] = np.select([rank.share_hours_over_80 >= 0.10, rank.mean_occupancy < DISC_T],
                           ['overloaded (>=10% of hours above 80%)', 'underutilized (mean < 30%)'], 'balanced')
rank.sort_values('mean_occupancy', ascending=False).to_csv(os.path.join(OUT, 'district_utilization_ranking.csv'))
results['district_status_counts'] = rank['status'].value_counts().to_dict()

plt.figure(figsize=(9, 4.5))
plt.hist(rank.mean_occupancy, bins=40, color='steelblue')
for x in (DISC_T, SURGE_T):
    plt.axvline(x, ls='--', c='black')
plt.xlabel('District mean occupancy rate (30 days)'); plt.ylabel('Districts')
plt.title('Most districts are underutilized; very few are persistently congested')
save_fig('03_district_occupancy_distribution.png')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
arr = acn.groupby(['arrival_hour', 'is_weekend']).size().unstack(fill_value=0)
arr = arr / arr.sum()
axes[0].plot(arr.index, arr[False], marker='o', label='Weekday'); axes[0].plot(arr.index, arr[True], marker='o', label='Weekend')
axes[0].set_xlabel('Arrival hour (Pacific time)'); axes[0].set_ylabel('Share of sessions'); axes[0].legend()
axes[0].set_title('ACN: workplace arrivals cluster at 7-9 AM')
idle = acn.groupby('arrival_hour').apply(lambda d: d.idle_h.sum() / d.connected_h.sum())
axes[1].bar(idle.index, idle.values, color='indianred')
axes[1].set_xlabel('Arrival hour'); axes[1].set_ylabel('Idle share of plugged-in time')
axes[1].set_title('ACN: most plugged-in time is idle (blocking)')
save_fig('04_acn_arrivals_and_idle.png')

# ----------------------------------------------------------------------------------------------------------------
# 3. DEMAND PREDICTION AGENT
# ----------------------------------------------------------------------------------------------------------------
print('3. Demand Prediction Agent...')
panel = panel.sort_values(['grid', 'time']).reset_index(drop=True)
g = panel.groupby('grid')
for L in [1, 2, 3, 24, 168]:
    panel[f'occ_lag{L}'] = g['occ_rate'].shift(L)
for L in [1, 24]:
    panel[f'energy_lag{L}'] = g['energy'].shift(L)
panel['sat_lag1'] = g['saturation'].shift(1)
panel['occ_roll24_mean'] = g['occ_rate'].transform(lambda s: s.shift(1).rolling(24).mean())
panel['occ_roll24_std'] = g['occ_rate'].transform(lambda s: s.shift(1).rolling(24).std())
panel['occ_diff1'] = panel['occ_lag1'] - panel['occ_lag2']

FEATURES = ['occ_lag1', 'occ_lag2', 'occ_lag3', 'occ_lag24', 'occ_lag168', 'occ_roll24_mean', 'occ_roll24_std',
            'occ_diff1', 'energy_lag1', 'energy_lag24', 'sat_lag1', 'price', 'hour', 'dow', 'is_weekend',
            'count', 'fast_share', 'area', 'pile_density', 'CBD', 'dynamic_pricing', 'lon', 'la']
ml = panel.dropna(subset=FEATURES).copy()
tr = ml[ml.day_idx.isin(TRAIN_DAYS)]
va = ml[ml.day_idx.isin(VALID_DAYS)]
te = ml[ml.day_idx.isin(TEST_DAYS)].copy()
trva = pd.concat([tr, va])
results['split'] = {'train_rows': len(tr), 'valid_rows': len(va), 'test_rows': len(te),
                    'train_period': f'{tr.time.min():%Y-%m-%d} to {tr.time.max():%Y-%m-%d}',
                    'valid_period': f'{va.time.min():%Y-%m-%d} to {va.time.max():%Y-%m-%d}',
                    'test_period': f'{te.time.min():%Y-%m-%d} to {te.time.max():%Y-%m-%d}'}


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def metrics(y, p):
    return {'RMSE': round(rmse(y, p), 4), 'MAE': round(float(mean_absolute_error(y, p)), 4), 'R2': round(float(r2_score(y, p)), 4)}


def make_model(name, params):
    if name == 'LightGBM':
        return LGBMRegressor(**params, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1)
    if name == 'XGBoost':
        return XGBRegressor(**params, tree_method='hist', random_state=SEED, n_jobs=-1)
    return RandomForestRegressor(**params, max_samples=0.7, random_state=SEED, n_jobs=-1)


def space(name, trial):
    if name == 'LightGBM':
        return {'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 255),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True)}
    if name == 'XGBoost':
        return {'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)}
    return {'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 6, 24),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', [0.3, 0.5, 0.8, 1.0])}


def tune(name, target, n_trials):
    def objective(trial):
        m = make_model(name, space(name, trial))
        m.fit(tr[FEATURES], tr[target])
        return rmse(va[target], m.predict(va[FEATURES]))
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study


# Naive baselines (evaluated on test) - any model must beat these to be worth deploying
district_hour_mean = tr.groupby(['grid', 'hour'])['occ_rate'].mean().rename('dh_mean')
te = te.merge(district_hour_mean, left_on=['grid', 'hour'], right_index=True, how='left')
comparison = [
    {'model': 'Baseline: persistence (last hour)', 'valid_RMSE': rmse(va.occ_rate, va.occ_lag1), **metrics(te.occ_rate, te.occ_lag1)},
    {'model': 'Baseline: same hour yesterday', 'valid_RMSE': rmse(va.occ_rate, va.occ_lag24), **metrics(te.occ_rate, te.occ_lag24)},
    {'model': 'Baseline: district x hour mean', 'valid_RMSE': np.nan, **metrics(te.occ_rate, te.dh_mean.fillna(tr.occ_rate.mean()))},
]

trials = {'LightGBM': 30, 'XGBoost': 20, 'RandomForest': 8}
fitted, best_params = {}, {}
for name, n in trials.items():
    print(f'   tuning {name} ({n} trials)...')
    study = tune(name, 'occ_rate', n)
    best_params[name] = study.best_params
    model = make_model(name, study.best_params).fit(trva[FEATURES], trva['occ_rate'])  # refit on train+valid
    fitted[name] = model
    comparison.append({'model': f'{name} (Optuna-tuned)', 'valid_RMSE': round(study.best_value, 4),
                       **metrics(te.occ_rate, np.clip(model.predict(te[FEATURES]), 0, 1))})
comp = pd.DataFrame(comparison)
comp.to_csv(os.path.join(OUT, 'model_comparison.csv'), index=False)
print(comp.to_string(index=False))

ml_rows = comp[comp.model.str.contains('Optuna')]
best_name = ml_rows.loc[ml_rows.valid_RMSE.idxmin(), 'model'].split(' ')[0]      # selected on VALIDATION
best = fitted[best_name]
te['pred_occ'] = np.clip(best.predict(te[FEATURES]), 0, 1)

# Error by period and on high-occupancy hours (where pricing decisions actually bite)
by_period = {p: metrics(d.occ_rate, d.pred_occ) for p, d in te.groupby('period')}
hi = te[te.occ_rate > 0.6]
by_period['actual_occ_above_60pct'] = metrics(hi.occ_rate, hi.pred_occ) | {'n': int(len(hi))}

# Decision accuracy of the threshold rule (what the pricing agent consumes)
act_band = np.select([te.occ_rate > SURGE_T, te.occ_rate < DISC_T], ['surge', 'discount'], 'base')
pred_band = np.select([te.pred_occ > SURGE_T, te.pred_occ < DISC_T], ['surge', 'discount'], 'base')
band_acc = float((act_band == pred_band).mean())
band_table = pd.crosstab(pd.Series(act_band, name='actual'), pd.Series(pred_band, name='predicted'))
band_table.to_csv(os.path.join(OUT, 'pricing_band_confusion.csv'))

# Expected charging load (kWh) and congestion probability - the other two outputs asked of the agent
lgb_p = best_params['LightGBM']
energy_model = make_model('LightGBM', lgb_p).fit(trva[FEATURES], trva['energy'])
te['pred_energy'] = np.clip(energy_model.predict(te[FEATURES]), 0, None)
clf = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
clf.fit(trva[FEATURES], trva['congested'])
te['congestion_prob'] = clf.predict_proba(te[FEATURES])[:, 1]

imp = pd.Series(fitted['LightGBM'].feature_importances_, index=FEATURES).sort_values(ascending=False)
imp.to_csv(os.path.join(OUT, 'feature_importance_lightgbm.csv'), header=['split_importance'])

results['demand_agent'] = {
    'selected_model': best_name, 'selected_on': 'validation RMSE (Jul 08-11); test untouched',
    'best_params': best_params,
    'test_metrics_occupancy': comp.set_index('model').loc[f'{best_name} (Optuna-tuned)', ['RMSE', 'MAE', 'R2']].to_dict(),
    'test_metrics_by_period': by_period,
    'pricing_band_accuracy': round(band_acc, 4),
    'energy_kwh_test_metrics': metrics(te.energy, te.pred_energy),
    'congestion_classifier': {
        'test_positive_rate': round(float(te.congested.mean()), 4),
        'ROC_AUC': round(float(roc_auc_score(te.congested, te.congestion_prob)), 4),
        'PR_AUC': round(float(average_precision_score(te.congested, te.congestion_prob)), 4),
        'Brier': round(float(brier_score_loss(te.congested, te.congestion_prob)), 5)},
    'top_features': imp.head(8).index.tolist(),
}
print('   ', {k: results['demand_agent'][k] for k in ['selected_model', 'test_metrics_occupancy', 'pricing_band_accuracy']})

plt.figure(figsize=(9, 4.5))
c = comp.sort_values('RMSE')
plt.barh(c.model, c.RMSE, color=['seagreen' if 'Optuna' in m else 'grey' for m in c.model])
plt.xlabel('Test RMSE (occupancy rate, 1 hour ahead)'); plt.title('Tuned models vs naive baselines (lower is better)')
save_fig('05_model_comparison.png')

busy = te.groupby('grid').occ_rate.mean().idxmax()
d = te[te.grid == busy].sort_values('time')
plt.figure(figsize=(12, 4))
plt.plot(d.time, d.occ_rate, label='Actual'); plt.plot(d.time, d.pred_occ, label=f'Predicted ({best_name})')
plt.axhline(SURGE_T, ls='--', c='red', lw=1, label='Surge threshold'); plt.axhline(DISC_T, ls='--', c='blue', lw=1, label='Discount threshold')
plt.ylabel('Occupancy rate'); plt.title(f'Test week, busiest district (grid {busy})'); plt.legend(fontsize=8)
save_fig('06_prediction_vs_actual_busiest_district.png')

# ----------------------------------------------------------------------------------------------------------------
# 4. PRICE RESPONSE (ELASTICITY) - descriptive estimate from observed time-of-use pricing
# ----------------------------------------------------------------------------------------------------------------
print('4. Estimating price response from observed time-of-use variation...')
est = panel[panel.day_idx <= max(VALID_DAYS)].copy()          # never uses the test week
est['how'] = est['dow'] * 24 + est['hour']
cell = est.groupby(['grid', 'how']).agg(occ=('occ_rate', 'mean'), energy=('energy', 'mean'),
                                         price=('price', 'mean'), CBD=('CBD', 'first')).reset_index()


def fe_elasticity(df, ycol, group_cols):
    """log-log slope with district FE + (group x hour-of-week) FE via exact within-transformation
    (the panel is balanced within each group); standard errors clustered by district."""
    y = np.log(df[ycol] + 0.01)
    x = np.log(df['price'])

    def within(v):
        return (v - v.groupby(df['grid']).transform('mean') - v.groupby([df[c] for c in group_cols] + [df['how']]).transform('mean')
                + v.groupby([df[c] for c in group_cols]).transform('mean') if group_cols else
                v - v.groupby(df['grid']).transform('mean') - v.groupby(df['how']).transform('mean') + v.mean())
    yt, xt = within(y), within(x)
    beta = float((xt * yt).sum() / (xt ** 2).sum())
    e = yt - beta * xt
    scores = (xt * e).groupby(df['grid']).sum()
    G = scores.size
    se = float(np.sqrt((scores ** 2).sum() * G / (G - 1)) / (xt ** 2).sum())
    return {'elasticity': round(beta, 3), 'cluster_se': round(se, 3),
            'ci95': [round(beta - 1.96 * se, 3), round(beta + 1.96 * se, 3)]}


elast = {
    'occupancy ~ price | district FE + CBD x hour-of-week FE': fe_elasticity(cell, 'occ', ['CBD']),
    'occupancy ~ price | district FE + hour-of-week FE': fe_elasticity(cell, 'occ', []),
    'energy ~ price | district FE + CBD x hour-of-week FE': fe_elasticity(cell, 'energy', ['CBD']),
}
results['elasticity_estimates'] = elast
primary = elast['occupancy ~ price | district FE + CBD x hour-of-week FE']['elasticity']
eps_prior = float(np.clip(primary, -1.5, -0.1)) if primary < 0 else -0.3
results['elasticity_prior_used'] = eps_prior
print('   ', elast)

# ----------------------------------------------------------------------------------------------------------------
# 5. TARIFF PRICING AGENT - rule + optimiser, evaluated by simulation on the test week
# ----------------------------------------------------------------------------------------------------------------
print('5. Tariff Pricing Agent...')
kwh_per_pile_hour = (trva.groupby('grid').energy.sum() / trva.groupby('grid').occupied_pile_hours.sum()).replace([np.inf], np.nan)
te['kwh_per_pile_hour'] = te.grid.map(kwh_per_pile_hour).fillna(kwh_per_pile_hour.median())
te = te.sort_values(['grid', 'time']).reset_index(drop=True)
MAX_C = int(te['count'].max())


def erlang_c_wait(rho, c):
    """Mean queueing delay (in units of mean charging-session length) for an M/M/c queue at utilisation rho."""
    rho = np.clip(np.asarray(rho, float), 1e-6, 0.98)
    c = np.asarray(c, int)
    a = rho * c
    B, b_at_c = np.ones_like(a), np.zeros_like(a)
    for k in range(1, MAX_C + 1):                     # Erlang-B recursion, read off at k == c
        B = a * B / (k + a * B)
        hit = c == k
        b_at_c[hit] = B[hit]
    C = b_at_c / (1 - rho * (1 - b_at_c))            # Erlang-C probability of waiting
    return C / (c * (1 - rho))


def policy(pred, surge, disc):
    return np.where(pred > SURGE_T, surge, np.where(pred < DISC_T, disc, 1.0))


def simulate(df, mult, eps, shift_share=0.0, noise=None):
    """Constant-elasticity demand response on the realised (actual) demand, capped at pile capacity.
    shift_share = fraction of demand deterred by surge that re-appears in the same district's discounted hours that day."""
    occ0, e0 = df.occ_rate.values, df.energy.values
    m = mult ** eps
    if noise is not None:
        m = m * noise
    latent = occ0 * m
    occ1 = np.minimum(latent, 1.0)
    scale = np.divide(occ1, occ0, out=m.copy(), where=occ0 > 0)
    e1 = e0 * scale
    if shift_share > 0:
        keys = [df.grid.values, df.day_idx.values]
        lost = pd.Series(np.where(mult > 1, e0 * (1 - scale), 0.0) * shift_share)
        recv = pd.Series((mult < 1).astype(int))
        pool = lost.groupby(keys).transform('sum').values
        n_recv = recv.groupby(keys).transform('sum').values
        add = np.where((mult < 1) & (n_recv > 0), pool / np.maximum(n_recv, 1), 0.0)
        e1 = e1 + add
        occ1 = np.minimum(occ1 + add / (df.kwh_per_pile_hour.values * df['count'].values), 1.0)
    return occ1, e1


def evaluate(df, mult, eps, shift_share=0.0, noise=None):
    occ1, e1 = simulate(df, mult, eps, shift_share, noise)
    tariff = BASE_PRICE * mult
    occ0, e0, c = df.occ_rate.values, df.energy.values, df['count'].values
    rev0, rev1 = (e0 * BASE_PRICE).sum(), (e1 * tariff).sum()
    peak, off = occ0 > SURGE_T, occ0 < DISC_T
    w0, w1 = erlang_c_wait(occ0, c), erlang_c_wait(occ1, c)
    load0, load1 = occ0 * c, occ1 * c
    surge_h, disc_h = mult > 1, mult < 1
    return {
        'revenue_fixed_rs': round(rev0), 'revenue_dynamic_rs': round(rev1),
        'revenue_gain_pct': round(100 * (rev1 - rev0) / rev0, 2),
        'mean_occupancy_before': round(occ0.mean(), 4), 'mean_occupancy_after': round(occ1.mean(), 4),
        'charging_util_before': round(df.charge_util.mean(), 4),
        'charging_util_after': round(float(np.clip(df.charge_util.values * np.divide(occ1, occ0, out=np.ones_like(occ0), where=occ0 > 0), 0, 1).mean()), 4),
        'peak_hour_occupancy_before': round(occ0[peak].mean(), 4) if peak.any() else None,
        'peak_hour_occupancy_after': round(occ1[peak].mean(), 4) if peak.any() else None,
        'congested_district_hours_before': int(peak.sum()), 'congested_district_hours_after': int((occ1 > SURGE_T).sum()),
        'offpeak_uplift_pile_hours': round(float((load1 - load0)[off].sum()), 1),
        'offpeak_uplift_pct': round(100 * float((load1 - load0)[off].sum() / load0[off].sum()), 2),
        'mean_wait_before': float((w0 * load0).sum() / load0.sum()),
        'mean_wait_after': float((w1 * load1).sum() / load1.sum()),
        'pricing_efficiency_rs_per_kwh': round(rev1 / e1.sum(), 3),
        'response_in_surge_hours_pct': round(100 * (e1[surge_h].sum() / e0[surge_h].sum() - 1), 2) if surge_h.any() else 0.0,
        'response_in_discount_hours_pct': round(100 * (e1[disc_h].sum() / e0[disc_h].sum() - 1), 2) if disc_h.any() else 0.0,
        'share_hours_surged': round(float(surge_h.mean()), 4), 'share_hours_discounted': round(float(disc_h.mean()), 4),
        '_occ1': occ1, '_e1': e1,
    }


def public(r):
    r = {k: v for k, v in r.items() if not k.startswith('_')}
    if r['mean_wait_before'] > 0:
        r['wait_reduction_pct'] = round(100 * (1 - r['mean_wait_after'] / r['mean_wait_before']), 2)
    r['mean_wait_before'], r['mean_wait_after'] = f"{r['mean_wait_before']:.3e}", f"{r['mean_wait_after']:.3e}"
    return r


def optimise(df, eps_hat):
    """Choose surge/discount multipliers that maximise *expected* revenue using forecasts only (no actuals)."""
    best_sd, best_rev = None, -np.inf
    for s in SURGE_GRID:
        for d_ in DISC_GRID:
            mult = policy(df.pred_occ.values, s, d_)
            exp_occ = np.minimum(df.pred_occ.values * mult ** eps_hat, 1.0)
            exp_scale = np.divide(exp_occ, df.pred_occ.values, out=mult ** eps_hat, where=df.pred_occ.values > 0)
            rev = (df.pred_energy.values * exp_scale * BASE_PRICE * mult).sum()
            if rev > best_rev:
                best_sd, best_rev = (s, d_), rev
    return best_sd


rule_mult = policy(te.pred_occ.values, 1.5, 0.7)                  # problem-statement rule
opt_sd = optimise(te[te.day_idx == min(TEST_DAYS)], eps_prior)     # optimiser under the data-driven prior
opt_mult = policy(te.pred_occ.values, *opt_sd)
oracle_mult = policy(te.occ_rate.values, 1.5, 0.7)                # rule with perfect foresight (upper bound on targeting)

rows = []
for eps in ELASTICITY_SCENARIOS:
    eps_sd = optimise(te, eps)                                     # multipliers chosen from forecasts, given this elasticity
    for shift in (0.0, 0.5):
        for pol, sd, mult in [('Rule 1.5x / 0.7x (forecast-driven)', (1.5, 0.7), rule_mult),
                              ('Optimised for this elasticity (forecast-driven)', eps_sd, policy(te.pred_occ.values, *eps_sd)),
                              ('Rule 1.5x / 0.7x (perfect foresight)', (1.5, 0.7), oracle_mult)]:
            rows.append({'policy': pol, 'elasticity': eps, 'shift_share': shift, 'surge_mult': sd[0], 'disc_mult': sd[1],
                         **public(evaluate(te, mult, eps, shift))})
scen = pd.DataFrame(rows)
scen.to_csv(os.path.join(OUT, 'pricing_scenarios.csv'), index=False)

headline = public(evaluate(te, rule_mult, -0.5))
headline_opt = public(evaluate(te, opt_mult, -0.5))
results['tariff_agent'] = {'rule_eps_-0.5': headline, f'optimised_{opt_sd}_eps_-0.5': headline_opt,
                           'optimised_multipliers': opt_sd}
print('    rule @ -0.5:', {k: headline[k] for k in ['revenue_gain_pct', 'offpeak_uplift_pct', 'peak_hour_occupancy_before', 'peak_hour_occupancy_after']})

out_df = te[['time', 'grid', 'occ_rate', 'pred_occ', 'congestion_prob', 'energy', 'pred_energy']].copy()
out_df['tariff_rule_rs'] = BASE_PRICE * rule_mult
out_df['tariff_optimised_rs'] = BASE_PRICE * opt_mult
r = evaluate(te, rule_mult, -0.5)
out_df['occ_after_rule_eps-0.5'], out_df['energy_after_rule_eps-0.5'] = r['_occ1'], r['_e1']
out_df.to_csv(os.path.join(OUT, 'test_week_predictions_and_tariffs.csv'), index=False)

fig, ax = plt.subplots(figsize=(9, 4.5))
for pol, grp in scen[scen.shift_share == 0].groupby('policy'):
    ax.plot(grp.elasticity, grp.revenue_gain_pct, marker='o', label=pol)
ax.axhline(0, c='black', lw=1); ax.set_xlabel('Assumed price elasticity of demand'); ax.set_ylabel('Revenue gain vs Rs 15 flat (%)')
ax.set_title('Revenue impact depends on elasticity: deep discounts only pay off when |e| > 1'); ax.legend(fontsize=8)
save_fig('07_revenue_vs_elasticity.png')

# ----------------------------------------------------------------------------------------------------------------
# 6. MONITORING & LEARNING AGENT - closed loop over daily episodes
# ----------------------------------------------------------------------------------------------------------------
print('6. Monitoring & Learning Agent...')


def learning_loop(eps_true, prior=eps_prior, prior_sd=0.5, noise_sd=0.05, explore_frac=0.2, jitter=0.10, seed=SEED):
    """Each test day = one episode. The agent prices with its current elasticity belief, observes simulated
    outcomes (hidden eps_true + noise), and updates its belief. Learning uses only randomised +/-jitter price tests
    on base-price hours, so the estimate is not biased by the forecast-driven selection of surge/discount hours."""
    rng = np.random.default_rng(seed)
    mu, prec = prior, 1 / prior_sd ** 2
    naive_x, naive_y, log = [], [], []
    for day in TEST_DAYS:
        d = te[te.day_idx == day]
        s, dd = optimise(d, mu)
        mult = policy(d.pred_occ.values, s, dd)
        explore = (mult == 1.0) & (rng.random(len(d)) < explore_frac)
        mult = np.where(explore, 1 + jitter * rng.choice([-1, 1], len(d)), mult)
        noise = np.exp(rng.normal(0, noise_sd, len(d)))
        r = evaluate(d, mult, eps_true, noise=noise)
        oracle = evaluate(d, policy(d.pred_occ.values, *optimise(d, eps_true)), eps_true, noise=noise)
        # Belief update: y = log(outcome / forecast) = a + eps * log(price multiplier) + error
        ok = (d.pred_energy.values > 1) & (r['_e1'] > 0)
        x = np.log(mult)
        y = np.log(np.where(ok, r['_e1'], 1) / np.where(ok, d.pred_energy.values, 1))
        sel = explore & ok
        X = np.column_stack([np.ones(sel.sum()), x[sel]])
        coef, *_ = np.linalg.lstsq(X, y[sel], rcond=None)
        resid = y[sel] - X @ coef
        var_b = resid.var(ddof=2) * np.linalg.inv(X.T @ X)[1, 1]
        mu = (mu * prec + coef[1] / var_b) / (prec + 1 / var_b)
        prec = prec + 1 / var_b
        pr = (mult != 1.0) & ~explore & ok                   # naive learner: uses the rule's own priced hours
        naive_x.extend(x[pr]); naive_y.extend(y[pr])
        nb = np.polyfit(naive_x, naive_y, 1)[0] if len(naive_x) > 10 else np.nan
        pub = public(r)
        log.append({'episode': day - min(TEST_DAYS) + 1, 'date': f'{d.time.min():%Y-%m-%d}', 'surge_mult': s, 'disc_mult': dd,
                    'eps_belief_after_update': round(mu, 3), 'eps_belief_sd': round(prec ** -0.5, 3),
                    'naive_eps_estimate': round(float(nb), 3), 'revenue_gain_pct': pub['revenue_gain_pct'],
                    'oracle_revenue_gain_pct': public(oracle)['revenue_gain_pct'],
                    'regret_rs': round(oracle['revenue_dynamic_rs'] - r['revenue_dynamic_rs']),
                    'pricing_efficiency_rs_per_kwh': pub['pricing_efficiency_rs_per_kwh'],
                    'wait_reduction_pct': pub.get('wait_reduction_pct'), 'offpeak_uplift_pct': pub['offpeak_uplift_pct'],
                    'customer_response_surge_pct': pub['response_in_surge_hours_pct'],
                    'customer_response_discount_pct': pub['response_in_discount_hours_pct'],
                    'exploration_hours': int(explore.sum())})
    return pd.DataFrame(log)


loops = {}
for eps_true in (-0.5, -1.2):
    lg = learning_loop(eps_true)
    lg.insert(0, 'eps_true', eps_true)
    loops[eps_true] = lg
loop_df = pd.concat(loops.values())
loop_df.to_csv(os.path.join(OUT, 'learning_loop_episodes.csv'), index=False)
results['learning_agent'] = {str(k): {'final_belief': float(v.eps_belief_after_update.iloc[-1]),
                                      'final_naive_estimate': float(v.naive_eps_estimate.iloc[-1]),
                                      'policy_day1': [float(v.surge_mult.iloc[0]), float(v.disc_mult.iloc[0])],
                                      'policy_day7': [float(v.surge_mult.iloc[-1]), float(v.disc_mult.iloc[-1])],
                                      'mean_regret_rs_days1_2': float(v.regret_rs.iloc[:2].mean()),
                                      'mean_regret_rs_days6_7': float(v.regret_rs.iloc[-2:].mean())}
                             for k, v in loops.items()}
print('   ', results['learning_agent'])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for eps_true, lg in loops.items():
    l = axes[0].plot(lg.episode, lg.eps_belief_after_update, marker='o', label=f'Belief (true = {eps_true})')
    axes[0].fill_between(lg.episode, lg.eps_belief_after_update - 1.96 * lg.eps_belief_sd,
                         lg.eps_belief_after_update + 1.96 * lg.eps_belief_sd, alpha=0.15, color=l[0].get_color())
    axes[0].axhline(eps_true, ls='--', color=l[0].get_color(), lw=1)
    axes[0].plot(lg.episode, lg.naive_eps_estimate, ls=':', color=l[0].get_color(), label=f'Naive estimate (true = {eps_true})')
    axes[1].plot(lg.episode, lg.revenue_gain_pct, marker='o', color=l[0].get_color(), label=f'Agent (true = {eps_true})')
    axes[1].plot(lg.episode, lg.oracle_revenue_gain_pct, ls='--', color=l[0].get_color(), label=f'Oracle (true = {eps_true})')
axes[0].set_xlabel('Episode (day)'); axes[0].set_ylabel('Elasticity'); axes[0].set_title('Learning agent converges on true elasticity'); axes[0].legend(fontsize=7)
axes[1].set_xlabel('Episode (day)'); axes[1].set_ylabel('Revenue gain (%)'); axes[1].set_title('Revenue gain approaches the oracle'); axes[1].legend(fontsize=7)
save_fig('08_learning_loop.png')

with open(os.path.join(OUT, 'metrics.json'), 'w') as f:
    json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('Done. Outputs in', OUT)
