from pathlib import Path


def main():
    results_path = Path('results')
    if not results_path.exists():
        print('Results folder not found')
        return
    summary_path = Path('summary.csv')
    if not results_path.exists():
        print('Summary sheet not found')
        return
    num_datasets = sum(1 for _ in open(summary_path)) - 1  # for header
    print('Number of datasets: {}'.format(num_datasets))
    for model_path in results_path.iterdir():
        num_datasets_finished = sum(1 for _ in model_path.iterdir())
        bar = '█' * num_datasets_finished + '░' * (num_datasets - num_datasets_finished)
        print(model_path.name.ljust(28), bar)


if __name__ == '__main__':
    main()
