"""Training callbacks: random reference styles and strict/voting dev F1."""
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

from swift.callbacks import TrainerCallback
from swift.callbacks.mapping import callbacks_map

from bioner_common import STYLES, update_best_symlink, write_checkpoint_summary


class BioNERF1CallbackBase(TrainerCallback):
    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.output_dir = Path(args.output_dir)
        self.model = os.environ.get("BIONER_MODEL")
        if not self.model:
            raise ValueError(f"BIONER_MODEL must be set for {type(self).__name__}.")
        self.eval_gpu = os.environ.get("BIONER_EVAL_GPU") or os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        self.eval_backend = os.environ.get("BIONER_EVAL_BACKEND", "pt")
        self.eval_gpu_mem = os.environ.get("BIONER_EVAL_GPU_MEM", "0.4")
        self.best_f1 = -1.0
        self.rows = []

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        checkpoint = self.output_dir / f"checkpoint-{state.global_step}"
        prefix = f"[{type(self).__name__}]"
        if not checkpoint.is_dir():
            print(f"{prefix} Skip missing checkpoint: {checkpoint}")
            return control

        metrics_path = checkpoint / self.metrics_filename
        try:
            metrics = self._evaluate(checkpoint, metrics_path)
        except subprocess.CalledProcessError as exc:
            print(f"{prefix} Dev evaluation failed; skip checkpoint={checkpoint.name}. "
                  f"returncode={exc.returncode}")
            return control

        precision, recall, f1 = (float(metrics[key]) for key in ("precision", "recall", "f1"))
        self._log_swanlab(state.global_step, precision, recall, f1)
        metadata = self._summary_metadata()
        row = {"checkpoint": str(checkpoint), "metrics_path": str(metrics_path), **metadata, "f1": f1}
        self.rows = [previous for previous in self.rows if previous["checkpoint"] != row["checkpoint"]]
        self.rows.append(row)
        self.rows.sort(key=lambda row: int(Path(row["checkpoint"]).name.split("-")[-1]))
        write_checkpoint_summary(
            self.rows, self.output_dir / self.summary_filename, self.selection_target, **metadata,
        )
        if f1 > self.best_f1:
            self.best_f1 = f1
            update_best_symlink(self.output_dir, checkpoint)
        print(f"{prefix} checkpoint={checkpoint.name} {self.log_prefix}_f1={f1:.6f} "
              f"precision={precision:.6f} recall={recall:.6f} best_f1={self.best_f1:.6f}")
        return control

    def _summary_metadata(self):
        return {}

    def _run_infer(self, checkpoint, dev_dataset, pred_path):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.eval_gpu
        pending_path = pred_path.with_name(f".{pred_path.stem}.tmp.jsonl")
        pending_path.unlink(missing_ok=True)
        cmd = [
            "swift", "infer", "--model", self.model,
            "--adapters", str(checkpoint), "--val_dataset", str(dev_dataset),
            "--result_path", str(pending_path), "--max_new_tokens", "256",
            "--enable_thinking", "false",
        ]
        if self.eval_backend == "vllm":
            cmd.extend([
                "--infer_backend", "vllm", "--vllm_gpu_memory_utilization", self.eval_gpu_mem,
                "--vllm_max_model_len", "2048", "--vllm_tensor_parallel_size", "1",
                "--vllm_max_lora_rank", "16",
            ])
        else:
            cmd.extend(["--infer_backend", "pt", "--max_batch_size", "1"])
        try:
            subprocess.run(cmd, check=True, env=env)
            os.replace(pending_path, pred_path)
        finally:
            pending_path.unlink(missing_ok=True)

    @staticmethod
    def _run_tool(name, *arguments):
        subprocess.run([sys.executable, str(Path(__file__).with_name(name)), *map(str, arguments)], check=True)

    def _run_eval(self, pred_path, metrics_path):
        self._run_tool("eval_swift_bioner_f1.py", "--input", pred_path, "--output", metrics_path)

    def _log_swanlab(self, step, precision, recall, f1):
        try:
            import swanlab
            swanlab.log({
                f"{self.log_prefix}/precision": precision,
                f"{self.log_prefix}/recall": recall,
                f"{self.log_prefix}/f1": f1,
            }, step=step)
        except Exception as exc:
            print(f"[{type(self).__name__}] SwanLab log failed: {exc}")



class BioNERStrictF1Callback(BioNERF1CallbackBase):
    """Evaluate strict dev micro-F1 after each saved checkpoint.

    Set BIONER_MODEL and BIONER_DEV_DATA. Optional BIONER_EVAL_GPU,
    BIONER_EVAL_BACKEND (pt/vllm) and BIONER_EVAL_GPU_MEM configure inference.
    """

    metrics_filename = "dev_metrics.json"
    summary_filename = "dev_checkpoint_metrics.json"
    selection_target = "strict dev micro-F1"
    log_prefix = "strict_dev"

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.dev_dataset = os.environ.get("BIONER_DEV_DATA")
        if not self.dev_dataset:
            raise ValueError("BIONER_DEV_DATA must be set for bioner_strict_f1 callback.")

    def _evaluate(self, checkpoint, metrics_path):
        if not metrics_path.exists():
            pred_path = checkpoint / "dev_predictions.jsonl"
            self._run_infer(checkpoint, self.dev_dataset, pred_path)
            self._run_eval(pred_path, metrics_path)
        with metrics_path.open(encoding="utf-8") as handle:
            return json.load(handle)["metrics"]



class BioNERCodeVoteF1Callback(BioNERF1CallbackBase):
    """Select checkpoints by Python/Java/C++ dev voting F1 after each save."""

    metrics_filename = "dev_code_vote_metrics.json"
    summary_filename = "dev_code_vote_checkpoint_metrics.json"
    selection_target = "code-style dev vote micro-F1"
    log_prefix = "dev_vote"

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        data_dir = os.environ.get("BIONER_CODE_DATA_DIR")
        if not data_dir:
            raise ValueError("BIONER_CODE_DATA_DIR must be set for bioner_code_vote_f1 callback.")
        self.dev_datasets = {
            style: str(Path(data_dir) / f"{style}_dev.jsonl") for style in STYLES
        }
        missing = [path for path in self.dev_datasets.values() if not Path(path).is_file()]
        if missing:
            raise ValueError(f"Missing code-style dev files: {missing}")
        self.vote_threshold = os.environ.get("BIONER_CODE_VOTE_THRESHOLD", "2")
        if self.vote_threshold not in {"1", "2", "3"}:
            raise ValueError("BIONER_CODE_VOTE_THRESHOLD must be one of: 1, 2, 3.")

    def _summary_metadata(self):
        return {"vote_threshold": self.vote_threshold}

    def _evaluate(self, checkpoint, metrics_path):
        if not metrics_path.exists():
            prediction_paths = {}
            for style in STYLES:
                pred_path = checkpoint / f"dev_{style}_predictions.jsonl"
                style_metrics_path = checkpoint / f"dev_{style}_metrics.json"
                prediction_paths[style] = pred_path
                if not pred_path.exists():
                    self._run_infer(checkpoint, self.dev_datasets[style], pred_path)
                if not style_metrics_path.exists():
                    self._run_eval(pred_path, style_metrics_path)
            self._run_tool(
                "eval_swift_bioner_f1.py",
                "--python", prediction_paths["python"],
                "--java", prediction_paths["java"],
                "--cpp", prediction_paths["cpp"],
                "--output", metrics_path,
            )
        with metrics_path.open(encoding="utf-8") as handle:
            return json.load(handle)["thresholds"][self.vote_threshold]



class RandomCodeStyleCallback(TrainerCallback):
    """Select one aligned code-style variant per sample at every real epoch."""

    def __init__(self, args, trainer):
        from datasets import concatenate_datasets
        from swift.dataset import load_dataset

        super().__init__(args, trainer)
        data_dir = os.environ.get("BIONER_CODE_DATA_DIR")
        if not data_dir:
            raise ValueError("BIONER_CODE_DATA_DIR must be set for random_code_style callback.")
        self.seed = int(os.environ.get("BIONER_CODE_STYLE_SEED", "42"))
        self.trainer = trainer

        datasets = []
        lengths = {}
        for style in STYLES:
            path = os.path.join(data_dir, f"{style}_train.jsonl")
            dataset, _ = load_dataset([path], split_dataset_ratio=0, shuffle=False)
            datasets.append(dataset)
            lengths[style] = len(dataset)
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Code-style train files are not aligned in length: {lengths}")

        self.sample_count = next(iter(lengths.values()))
        if len(trainer.train_dataset) != self.sample_count:
            raise ValueError(
                "Trainer dataset length differs from code-style files: "
                f"trainer={len(trainer.train_dataset)}, styles={lengths}"
            )
        if not hasattr(trainer.train_dataset, "dataset"):
            raise TypeError(
                "random_code_style requires ms-swift lazy tokenization; "
                "set --lazy_tokenize true."
            )
        self.all_styles = concatenate_datasets(datasets)

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch or 0)
        rng = random.Random(self.seed + epoch)
        selected_indices = []
        counts = Counter()
        for sample_index in range(self.sample_count):
            style_index = rng.randrange(len(STYLES))
            style = STYLES[style_index]
            selected_indices.append(style_index * self.sample_count + sample_index)
            counts[style] += 1

        # LazyLLMDataset encodes on access. Its DataLoader keeps a reference to
        # this wrapper, so replacing only the aligned HF dataset is sufficient.
        self.trainer.train_dataset.dataset = self.all_styles.select(selected_indices)
        summary = ", ".join(f"{style}={counts[style]}" for style in STYLES)
        print(f"[RandomCodeStyleCallback] epoch={epoch + 1} seed={self.seed + epoch}: {summary}")
        return control



callbacks_map.update({
    "bioner_strict_f1": BioNERStrictF1Callback,
    "bioner_code_vote_f1": BioNERCodeVoteF1Callback,
    "random_code_style": RandomCodeStyleCallback,
})
