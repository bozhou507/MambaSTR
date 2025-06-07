from prettytable import PrettyTable, TableStyle
from typing import List, Dict
import os


def model_load(model, checkpoint_path):
    import torch
    state_dict = torch.load(checkpoint_path)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=True)


def get_test_results(runner, checkpoints):
    results: List[Dict[str, float]] = []
    for i, checkpoint in enumerate(checkpoints):
        if os.path.exists(checkpoint):
            # model_load(runner.model, checkpoint)
            runner.load_checkpoint(checkpoint, strict=True)
            result = runner.test()
            result = {"id": os.path.basename(checkpoint), **result}
            # print(json.dumps(result, indent=4, ensure_ascii=False))
            results.append(result)
        else:
            import warnings
            warnings.warn(f"Checkpoint {checkpoint} does not exist.")
    if not checkpoints:
        import warnings
        warnings.warn('No checkpoint are specified.')
        result = runner.test()
        result = {"id": 0, **result}
        results.append(result)
    return results


def get_result_table(results: List[Dict[str, float]]) -> PrettyTable:
    table = PrettyTable()
    table.align = 'l'
    table.set_style(TableStyle.MARKDOWN)
    if results:
        table.field_names = list(map(lambda s: s.split('/', 1)[0], results[0].keys()))
        for r in results:
            table.add_row(list(map(
                lambda s: f"{s:.4f}" if isinstance(s, float) else str(s),
                list(r.values()))))
    return table


def list_checkpoints(checkpoints_dir, recursion=False):
    from utils import osutil
    return sorted(filter(lambda s: s.endswith(".pth"), sorted(osutil.list_files(checkpoints_dir, recursion=recursion))))


def parse_args(config):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=config.model_name, choices=config.available_models, type=str, help='model name')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='checkpoint path')
    parser.add_argument('--cuda_visible_devices', default=None, type=str)
    parser.add_argument('--save_failure_cases', action='store_true')
    args = parser.parse_args()
    if args.cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    if hasattr(config, 'update_config'):
        config.update_config(config, args.model_name)
    args.train_workdir = config.train_workdir
    args.test_workdir = config.test_workdir
    args.train_batch_size = config.train_batch_size
    args.val_batch_size = config.val_batch_size
    args.test_batch_size = config.test_batch_size
    args.train_lmdb_paths = config.train_lmdb_paths
    args.val_lmdb_paths = config.val_lmdb_paths
    args.test_lmdb_paths = config.test_lmdb_paths
    args.num_train_workers = config.num_train_workers
    args.num_val_workers = config.num_val_workers
    args.num_test_workers = config.num_test_workers
    return args


def test_model(model, args):
    import json
    from utils.ocr_lmdb_runner import get_ocr_lmdb_runner
    model_name = args.model_name
    if args.checkpoint_path:
        checkpoints = [args.checkpoint_path]
    else:
        checkpoints = []
    work_dir = os.path.join(args.test_workdir, model_name)
    runner = get_ocr_lmdb_runner(
        model=model,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
        # train_lmdb_paths=args.train_lmdb_paths,
        # val_lmdb_paths=args.val_lmdb_paths,
        test_lmdb_paths=args.test_lmdb_paths,
        num_train_workers = args.num_train_workers,
        num_val_workers = args.num_val_workers,
        num_test_workers = args.num_test_workers,
        work_dir=work_dir,
        save_failure_cases=args.save_failure_cases
    )
    runner.logger.info('\n' + 'Collected checkpoints: ' + json.dumps(checkpoints, indent=4))

    results = get_test_results(runner, checkpoints)
    return get_result_table(results)


def main():
    import config.mambastr as config
    args = parse_args(config)

    from models.mambastr import get_model
    model = get_model(args.model_name, args)

    metrics_table = test_model(model, args)
    table_str = str(metrics_table)
    print(table_str)


if __name__ == '__main__':
    main()