import os


def parse_args(config):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=config.model_name, choices=config.available_models, type=str, help='model name')
    parser.add_argument('--load_from', default=None, type=str, help='specify a checkpoint path to load')
    parser.add_argument('--resume', action='store_true', help='resume a checkpoint')
    parser.add_argument('--max_epochs', default=20, type=int, help='max epochs, default 20')
    parser.add_argument('--lr', default=1e-3, type=float, help='maximum learning rate, default 1e-3')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='min_lr, default 1e-5')
    parser.add_argument('--cuda_visible_devices', default=None, type=str)
    args = parser.parse_args()
    args.min_lr_ratio = args.min_lr / args.lr
    if args.min_lr_ratio > 1:
        args.min_lr_ratio = 1
    print(args)
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


def train_model(model, args):
    from utils.ocr_lmdb_runner import get_ocr_lmdb_runner
    work_dir = args.train_workdir
    runner = get_ocr_lmdb_runner(
        model=model,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        # test_batch_size=args.test_batch_size,
        train_lmdb_paths=args.train_lmdb_paths,
        val_lmdb_paths=args.val_lmdb_paths,
        # test_lmdb_paths=args.test_lmdb_paths,
        num_train_workers=args.num_train_workers,
        num_val_workers = args.num_val_workers,
        num_test_workers = args.num_test_workers,
        work_dir=work_dir,
        max_epochs=args.max_epochs,
        load_from=args.load_from,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        resume=args.resume
    )
    runner.train()


def main():
    import config.mambastr as config
    args = parse_args(config)

    from models.mambastr import get_model
    from models.mambastr.runner import get_runner

    model = get_model(args.model_name)
    work_dir = config.train_workdir
    runner = get_runner(
        model=model,
        train_batch_size=config.train_batch_size,
        val_batch_size=config.val_batch_size,
        # test_batch_size=config.test_batch_size,
        train_lmdb_paths=config.train_lmdb_paths,
        val_lmdb_paths=config.val_lmdb_paths,
        # test_lmdb_paths=config.test_lmdb_paths,
        num_train_workers=config.num_train_workers,
        num_val_workers = config.num_val_workers,
        num_test_workers = config.num_test_workers,
        work_dir=work_dir,
        max_epochs=args.max_epochs,
        load_from=args.load_from,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        resume=args.resume
    )

    runner.train()


if __name__ == '__main__':
    main()