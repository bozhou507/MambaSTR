from mmengine.evaluator import BaseMetric
import numpy as np
from datetime import datetime
from bisect import bisect_right


class WordAccuracy(BaseMetric):

    def __init__(self, save_failure_cases=False):
        super().__init__(prefix="please don't warning")
        self.prefix = None
        self.save_failure_cases = save_failure_cases
        if self.save_failure_cases:
            self.processed_num = 0

    def process(self, data_batch, data_samples):
        pred, gt = data_samples
        result = pred == gt
        self.results.extend(result)
        if self.save_failure_cases:
            dataset_names = self.dataset_meta['dataset_names']
            end_ids = self.dataset_meta['cumulative_sizes']
            if self.processed_num == end_ids[-1]:
                self.processed_num = 0
            if self.processed_num == 0:
                from . import osutil
                self.failure_cases_dir = f'failure_cases.log/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                self.failure_cases_path = f'{self.failure_cases_dir}/_failure_cases.txt'
                if osutil.is_path_exist(self.failure_cases_dir):
                    osutil.remove_dir(self.failure_cases_dir)
                osutil.make_dir(self.failure_cases_dir)
                osutil.make_file(self.failure_cases_path)
                self.save_failure_case_num = 0

            from PIL import Image
            if not isinstance(data_batch[0], list):
                xs = data_batch[0].detach().cpu().numpy()
                xs = (xs * 0.5 + 0.5) * 255
                xs = xs.transpose((0, 2, 3, 1)).astype(np.uint8)
            
            i_offset = self.processed_num
            self.processed_num += len(result)
            for i, r in enumerate(result):
                if not r:
                    dataset_idx = bisect_right(end_ids, i + i_offset)
                    dataset_name = dataset_names[dataset_idx]
                    dataset_offset = end_ids[dataset_idx - 1] if dataset_idx > 0 else 0
                    self.save_failure_case_num += 1
                    with open(self.failure_cases_path, 'a') as f:
                        print(f'{dataset_name}-{i + i_offset - dataset_offset + 1}: ', pred[i], '!=', gt[i], file=f)
                    xs_idx = 0
                    for item in data_batch:
                        if not isinstance(item, list):
                            break
                        imgs = item
                        img_save_path = f"{self.failure_cases_dir}/{self.save_failure_case_num}_{xs_idx}.png"
                        imgs[i].save(img_save_path)
                        xs_idx += 1
                    if xs_idx == 0:
                        x_save_path = f"{self.failure_cases_dir}/{self.save_failure_case_num}-transformed.png"
                        Image.fromarray(xs[i], "RGB").save(x_save_path)

    def compute_metrics(self, results):
        results = np.array(results)
        dataset_names = self.dataset_meta['dataset_names']
        end_ids = self.dataset_meta['cumulative_sizes']
        start_ids = [0] + end_ids[:-1]
        metrics = {}
        for s, t, name in zip(start_ids, end_ids, dataset_names):
            part_res = results[s:t]
            metrics[f"{name}/word_acc"] = part_res.mean() * 100
        word_acc = sum(metrics.values()) / len(metrics)
        metrics["wavg/word_acc"] = results.mean() * 100
        metrics["avg/word_acc"] = word_acc
        return metrics