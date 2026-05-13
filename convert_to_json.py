"""
Data converter for website

Output:
{
    'risk-strat': {
        'model-key': {
            'name: 'Model Name',
            'type': 'classic' or 'deep' or 'foundation',
            'small: {
                'metric-key': {
                    'elo': {
                        'value': 0.0,
                        'ci': [0.0, 0.0],
                    },
                    'improvability': {
                        'value': 0.0,
                        'ci': [0.0, 0.0],
                    },
                    'rank': {'value': 0},
                    'mrr': {'value': 0},
                },
                ...
            },
            'large': {
                ...
            },
        },
        ...
    }   
    'surv-curve': {
        ...
    }
}
"""
import json
from collections import OrderedDict

from plot import load_leaderboard


model_metadata = OrderedDict([
    ('coxnet', {'name': 'Coxnet', 'type': 'classic'}),
    ('coxnet-tuned', {'name': 'Coxnet (Tuned)', 'type': 'classic'}),
    ('rsf', {'name': 'RSF', 'type': 'classic'}),
    ('rsf-tuned', {'name': 'RSF (Tuned)', 'type': 'classic'}),
    ('gbse', {'name': 'GBSE', 'type': 'classic'}),
    ('gbse-tuned', {'name': 'GBSE (Tuned)', 'type': 'classic'}),
    ('ssvm', {'name': 'SSVM', 'type': 'classic'}),
    ('ssvm-tuned', {'name': 'SSVM (Tuned)', 'type': 'classic'}),
    ('deepsurv', {'name': 'DeepSurv', 'type': 'deep'}),
    ('deephit', {'name': 'DeepHit', 'type': 'deep'}),
    ('deepweisurv1', {'name': 'DeepWeiSurv (p=1)', 'type': 'deep'}),
    ('deepweisurv2', {'name': 'DeepWeiSurv (p=2)', 'type': 'deep'}),
    ('rankdeepsurv', {'name': 'RankDeepSurv', 'type': 'deep'}),
    ('tabpfn', {'name': 'TabPFN*', 'type': 'foundation'}),
    ('popsicl', {'name': 'PopSICL', 'type': 'foundation'}),
])


def convert_to_json():
    output = {
        'risk-strat': {},
        'surv-curve': {}
    }

    for key, metadata in model_metadata.items():
        output['risk-strat'][key] = {
            'name': metadata['name'],
            'type': metadata['type'],
            'small': {},
            'large': {}
        }

    for size in ['small', 'large']:
        data = load_leaderboard(size)
        for key in model_metadata.keys():
            if key in data.index:
                row = data.loc[key]
                output['risk-strat'][key][size]['harrell-c'] = {
                    'elo': {
                        'value': float(row['elo']),
                        'ci': [float(row['elo'] - row['elo-']), float(row['elo'] + row['elo+'])]
                    },
                    'improvability': {
                        'value': float(row['improvability']),
                        'ci': [float(row['improvability'] - row['improvability-']), float(row['improvability'] + row['improvability+'])]
                    },
                    'rank': {'value': float(row['rank'])},
                    'mrr': {'value': float(row['mrr'])},
                }
            else:
                output['risk-strat'][key][size] = None

    with open('leaderboard.json', 'w') as f:
        json.dump(output, f, indent=4)



if __name__ == '__main__':
    convert_to_json()
