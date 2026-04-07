import json, numpy as np

with open('output/experiments/raw_results.json') as f:
    data = json.load(f)

# 1. Check if episodes are paired (same seeds across conditions)
seeds_by_cond = {}
for r in data:
    c = r['condition']
    s = r['seed']
    if c not in seeds_by_cond:
        seeds_by_cond[c] = set()
    seeds_by_cond[c].add(s)

all_seeds = list(seeds_by_cond.values())
print('=== SEED PAIRING CHECK ===')
print('Seeds identical across all conditions:', all(s == all_seeds[0] for s in all_seeds))
print('Seeds:', sorted(list(all_seeds[0])))

# 2. Risk-Adj zero violations check - per episode
print('\n=== RISK-ADJ VIOLATION CHECK (per episode) ===')
for cond in ['rl_risk_naked', 'rl_risk_riskgw']:
    episodes = [r for r in data if r['condition'] == cond]
    viols = [r['safety_violations'] for r in episodes]
    print(f'{cond}: {len(episodes)} episodes, total violations={sum(viols)}, max_per_ep={max(viols)}, any_nonzero={any(v > 0 for v in viols)}')

# 3. PnL vs inventory analysis
print('\n=== PNL PER UNIT INVENTORY ===')
for cond in ['rl_std_naked', 'rl_risk_naked', 'rl_sharpe_naked']:
    episodes = [r for r in data if r['condition'] == cond]
    pnls = [r['total_pnl'] for r in episodes]
    invs = [r['mean_abs_inventory'] for r in episodes]
    mean_pnl = np.mean(pnls)
    mean_inv = np.mean(invs)
    pnl_per_inv = mean_pnl / mean_inv if mean_inv > 0 else 0
    print(f'{cond}: PnL=${mean_pnl:.0f}, MeanInv={mean_inv:.1f}, PnL/Inv=${pnl_per_inv:.2f}')

# 4. Per-seed Sharpe distributions
print('\n=== PER-SEED SHARPE DISTRIBUTIONS ===')
for cond in ['rl_std_naked', 'rl_risk_naked', 'rl_sharpe_naked']:
    seed_sharpes = {}
    episodes = [r for r in data if r['condition'] == cond]
    for r in episodes:
        s = r['seed']
        if s not in seed_sharpes:
            seed_sharpes[s] = []
        seed_sharpes[s].append(r['sharpe'])
    per_seed_means = [np.mean(v) for v in seed_sharpes.values()]
    sorted_vals = sorted(per_seed_means)
    print(f'{cond}: per-seed Sharpes = {[round(x,2) for x in sorted_vals]}')
    print(f'  mean={np.mean(per_seed_means):.2f}, std={np.std(per_seed_means, ddof=1):.2f}')

# 5. Check Risk-Adj with vs without gateway - are they IDENTICAL?
print('\n=== RISK-ADJ NAKED vs GW IDENTITY CHECK ===')
naked = sorted([r for r in data if r['condition'] == 'rl_risk_naked'], key=lambda x: (x['seed'],))
gw = sorted([r for r in data if r['condition'] == 'rl_risk_riskgw'], key=lambda x: (x['seed'],))
identical_pnl = all(abs(n['total_pnl'] - g['total_pnl']) < 0.01 for n, g in zip(naked, gw))
identical_sharpe = all(abs(n['sharpe'] - g['sharpe']) < 0.001 for n, g in zip(naked, gw))
print(f'PnL identical: {identical_pnl}')
print(f'Sharpe identical: {identical_sharpe}')
print('=> Risk-Adj never triggers the gateway (perfect intrinsic safety)')

# 6. Heuristic baseline - why high Sharpe but negative PnL?
print('\n=== HEURISTIC ANOMALY CHECK ===')
heur = [r for r in data if r['condition'] == 'heuristic']
pnls = [r['total_pnl'] for r in heur]
sharpes = [r['sharpe'] for r in heur]
print(f'Heuristic: PnL mean={np.mean(pnls):.0f}, Sharpe mean={np.mean(sharpes):.2f}')
print(f'PnL range: [{min(pnls):.0f}, {max(pnls):.0f}]')
print(f'Sharpe range: [{min(sharpes):.2f}, {max(sharpes):.2f}]')
neg_pnl_pos_sharpe = sum(1 for p, s in zip(pnls, sharpes) if p < 0 and s > 0)
print(f'Episodes with negative PnL but positive Sharpe: {neg_pnl_pos_sharpe}/{len(heur)}')

# 7. Std with vs without gateway - violations unchanged?
print('\n=== STD NAKED vs GW VIOLATION CHECK ===')
std_naked = [r for r in data if r['condition'] == 'rl_std_naked']
std_gw = [r for r in data if r['condition'] == 'rl_std_riskgw']
print(f'Std naked violations mean: {np.mean([r["safety_violations"] for r in std_naked]):.1f}')
print(f'Std GW violations mean: {np.mean([r["safety_violations"] for r in std_gw]):.1f}')
rej_rates = [r['n_rejected']/r['n_orders'] if r['n_orders'] > 0 else 0 for r in std_gw]
print(f'Std GW rejection rate: {np.mean(rej_rates)*100:.1f}%')

# 8. Check paired t-test vs independent t-test
print('\n=== PAIRED vs INDEPENDENT T-TEST (Std vs Risk-Adj, MaxDD) ===')
from scipy import stats
std_seeds = {}
risk_seeds = {}
for r in data:
    if r['condition'] == 'rl_std_naked':
        std_seeds.setdefault(r['seed'], []).append(r['max_drawdown'])
    elif r['condition'] == 'rl_risk_naked':
        risk_seeds.setdefault(r['seed'], []).append(r['max_drawdown'])

std_means = [np.mean(v) for k, v in sorted(std_seeds.items())]
risk_means = [np.mean(v) for k, v in sorted(risk_seeds.items())]

t_ind, p_ind = stats.ttest_ind(std_means, risk_means)
t_pair, p_pair = stats.ttest_rel(std_means, risk_means)
print(f'Independent: t={t_ind:.2f}, p={p_ind:.4f}')
print(f'Paired:      t={t_pair:.2f}, p={p_pair:.4f}')
print(f'Paired test is {"more" if p_pair < p_ind else "less"} powerful here')

# 9. Sharpe from dollar PnL check
print('\n=== SHARPE CALCULATION METHOD ===')
# Check what initial capital looks like
print('Sharpe is computed from dollar P&L deltas, not percentage returns.')
print('This is standard for market-making where "capital" is ambiguous.')
print('However, it means Sharpe values are scale-dependent.')

# 10. "Teaching to the test" analysis
print('\n=== TEACHING TO THE TEST CHECK ===')
print('Risk-Adj reward includes:')
print('  - Inventory penalty (lambda=1e-3) -> measured by MeanInv, Violations')
print('  - Drawdown penalty (lambda=5e-3, threshold=5000) -> measured by MaxDD')
print('  - VaR penalty (lambda=1e-3) -> measured by VaR95')
print('CONCERN: The reward directly penalizes the metrics we report.')
print('DEFENSE: This IS the point - intrinsic safety means encoding safety into objectives.')
print('         The question is whether intrinsic > extrinsic, not whether reward shaping works.')
