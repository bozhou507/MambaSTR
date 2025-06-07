from typing import Literal

# Yang X, Qiao Z, Wei J, et al. Masked and permuted implicit context learning for scene text recognition[J]. IEEE Signal Processing Letters, 2024.
benchmark_val_datasets = [
    'data/test/benchmark/IIIT5k',  # 3000 samples
    'data/test/benchmark/IC13_1015',  # here different with benchmark_test_datasets
    'data/test/benchmark/SVT',  # 647 samples
    'data/test/benchmark/CUTE80',  # 288 samples
    'data/test/benchmark/IC15_2077',  # here different with benchmark_test_datasets
    'data/test/benchmark/SVTP',  # 645 samples
]

# Yang X, Qiao Z, Wei J, et al. Masked and permuted implicit context learning for scene text recognition[J]. IEEE Signal Processing Letters, 2024.
benchmark_test_datasets = [
    'data/test/benchmark/IIIT5k',  # 3000 samples
    'data/test/benchmark/IC13_857',  # here different with benchmark_val_datasets
    'data/test/benchmark/SVT',  # 647 samples
    'data/test/benchmark/CUTE80',  # 288 samples
    'data/test/benchmark/IC15_1811',  # here different with benchmark_val_datasets
    'data/test/benchmark/SVTP',  # 645 samples
]

MJ_ST_datasets = [
    # Total: 8919241 + 6976115 = 15895356
    # MJ: 7224586 + 802731 + 891924 = 8919241 samples
    'data/train/MJ/MJ_train',  # 7224586 samples
    'data/train/MJ/MJ_valid',  # 802731 samples
    'data/train/MJ/MJ_test',  # 891924 samples
    'data/train/ST',  # 6976115 samples
]

# Jiang Q, Wang J, Peng D, et al. Revisiting scene text recognition: A data perspective[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2023: 20543-20554.
Union14M_L_LMDB_datasets = [
    # Total: 482877 + 308025 + 145525 + 218154 + 2076161 = 3230742 samples
    'data/train/Union14M-L-LMDB/train_challenging',  # 482877 samples
    'data/train/Union14M-L-LMDB/train_hard',  # 308025 samples
    'data/train/Union14M-L-LMDB/train_medium',  # 145525 samples
    'data/train/Union14M-L-LMDB/train_normal',  # 218154 samples
    'data/train/Union14M-L-LMDB/train_easy',  # 2076161 samples
]

# Du Y, Chen Z, Xie H, et al. SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition[J]. arXiv preprint arXiv:2411.15858, 2024.
Union14M_L_LMDB_Filtered_datasets = [
    # Total: 481803 + 306771 + 144677 + 217070 + 2073822 = 3224143 samples
    'data/train/Union14M-L-LMDB-Filtered/filter_train_challenging',  # 481803 samples
    'data/train/Union14M-L-LMDB-Filtered/filter_train_hard',  # 306771 samples
    'data/train/Union14M-L-LMDB-Filtered/filter_train_medium',  # 144677 samples
    'data/train/Union14M-L-LMDB-Filtered/filter_train_normal',  # 217070 samples
    'data/train/Union14M-L-LMDB-Filtered/filter_train_easy',  # 2073822 samples
]

# Jiang Q, Wang J, Peng D, et al. Revisiting scene text recognition: A data perspective[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2023: 20543-20554.
Union14M_L_benchmark = [
    # Total: 2426 + 1369 + 900 + 779 + 1585 + 829 + 400000 = 407888 samples
    'data/test/Union14M-benchmark/curve',  # 2426 samples
    'data/test/Union14M-benchmark/multi_oriented',  # 1369 samples
    'data/test/Union14M-benchmark/artistic',  # 900 samples
    'data/test/Union14M-benchmark/contextless',  # 779 samples
    'data/test/Union14M-benchmark/salient',  # 1585 samples
    'data/test/Union14M-benchmark/multi_words',  # 829 samples
    'data/test/Union14M-benchmark/general',  # 400000 samples
]

other_benchmark = [
    'data/test/other_benchmark/wordart',  # 1511 samples
]

USE_TRAIN_DATASETS: Literal['Synth', 'Union14M_L_LMDB', 'Union14M_L_LMDB_Filtered'] = 'Union14M_L_LMDB_Filtered'
if USE_TRAIN_DATASETS == 'Union14M_L_LMDB':
    train_lmdb_paths = Union14M_L_LMDB_datasets
    val_lmdb_paths = benchmark_test_datasets
    test_lmdb_paths = Union14M_L_benchmark
elif USE_TRAIN_DATASETS == 'Union14M_L_LMDB_Filtered':
    train_lmdb_paths = Union14M_L_LMDB_Filtered_datasets
    val_lmdb_paths = benchmark_test_datasets
    test_lmdb_paths = Union14M_L_benchmark
else:
    train_lmdb_paths = MJ_ST_datasets
    val_lmdb_paths = benchmark_val_datasets
    test_lmdb_paths = benchmark_test_datasets