"""Embed the latest backtest numbers into docs/learning_report.html (python -m evaluation.build_learning_report)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    r = json.load(open(os.path.join(ROOT, 'reports', 'evaluation_results.json')))
    series = json.load(open(os.path.join(ROOT, 'reports', 'chart_series.json')))[-360:]
    keys = ['sentiment_7pillar', 'mtf_confluence', 'tf_1h_bias', 'baseline_persistence', 'baseline_always_long',
            'sanity_oracle_future_sign', 'sanity_random_coin']
    sig = {}
    for h in ('h1', 'h4', 'h24'):
        sig[h] = {}
        for k in keys:
            d = r['signals'][h][k]
            sig[h][k] = {x: d.get(x) for x in ('coverage', 'hit_rate', 'nonoverlap_hit_rate', 'binomial_p_vs_50pct',
                                                 'pesaran_timmermann_p', 'pct_bull_calls', 'base_rate_up')}
            sig[h][k]['mcc'] = d['three_class']['mcc']
            sig[h][k]['bal_acc'] = d['three_class']['balanced_accuracy']
            if 'probability' in d:
                sig[h][k]['brier_skill'] = d['probability']['brier_skill_score']
                sig[h][k]['auc'] = d['probability']['roc_auc']
        sig[h]['confusion'] = r['signals'][h]['sentiment_7pillar']['three_class']['confusion_matrix']
        sig[h]['reliability'] = r['signals'][h]['sentiment_7pillar']['probability']['reliability']
    ml = {}
    for h in ('h1', 'h4', 'h24'):
        sec = r['ml'][h]
        ml[h] = {k: {'hit_rate': sec[k]['hit_rate'], 'auc': sec[k]['probability']['roc_auc'],
                     'pct_bull': sec[k]['pct_bull_calls'], 'pt_p': sec[k]['pesaran_timmermann_p'],
                     'p50': sec[k]['binomial_p_vs_50pct'], 'base': sec[k]['base_rate_up']}
                 for k in ('rule_based_sentiment_same_window', 'logistic_regression', 'gradient_boosting')}
        ml[h]['price'] = sec['price_forecast']
    data = {'meta': r['meta'], 'signals': sig, 'trades': r['trades'], 'bands': r['bands'], 'regimes': r['regimes'],
            'strategy': r['strategy'], 'sweeps': r['sweeps'], 'ml': ml,
            'coef': r['ml']['logit_standardised_coefficients_h4'],
            'series': [{'t': s['time'][:16], 'p': s['price'], 's': s['sent_score']} for s in series]}
    imp_path = os.path.join(ROOT, 'reports', 'improvement_results.json')
    if os.path.exists(imp_path):
        im = json.load(open(imp_path))
        g = im['trades']['T4_geometry_grid']
        data['improve'] = {
            'protocol': im['protocol'], 'n_variants': im['n_variants'], 'chosen_geom': g['chosen_on_dev'],
            'grid': {k: {'dev': v['dev'], 'lock': v['lockbox']} for k, v in g['grid'].items()},
            'trades': {k: {'dev': im['trades'][k]['dev'], 'lock': im['trades'][k]['lockbox']}
                       for k in ('T0_engine_as_is', 'T1_drop_range_rotation', 'T5_meta_labeling', 'T6_geometry_plus_meta')},
            'conf4': im['direction']['h4']['E_best_dev_model_top30pct_confident'],
            'base4': im['direction']['h4']['A_engine_sentiment'],
            'calib4': im['calibration']['h4'],
        }
    tpl = open(os.path.join(ROOT, 'evaluation', 'learning_report_template.html')).read()
    out = tpl.replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))
    open(os.path.join(ROOT, 'docs', 'learning_report.html'), 'w').write(out)
    print('wrote docs/learning_report.html', len(out) // 1024, 'KB')


if __name__ == '__main__':
    main()
