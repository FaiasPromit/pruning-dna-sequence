"""
Extract frozen embeddings of one Hyena-16k model for DART-Eval Task 4.

Goes to: DART-Eval/src/dnalm_bench/task_2_5_single/experiments/task_4_chromatin_activity/extract_embeddings/hyena16k_hrg.py
It is DART-Eval's extract_embeddings/hyenadna.py with the model taken from environment
variables (see dart_task4.sbatch), so the same file serves every pretraining variant.

Usage:
    python -m dnalm_bench.task_2_5_single.experiments.task_4_chromatin_activity.extract_embeddings.hyena16k_hrg <cell line> <category>
where <category> is peaks, nonpeaks or idr.
"""
import os
import sys

from ....hyena16k_hrg_embeddings import Hyena16kHRGEmbeddingExtractor
from ....components import SimpleSequence

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    model_name = os.environ["HYENA16K_RUN_NAME"]       # names the output folders
    model_path = os.environ["HYENA16K_MODEL_DIR"]
    model_code_dir = os.environ["HYENA16K_CODE_DIR"]

    genome_fa = os.path.join(
        root_output_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    )

    cell_line = sys.argv[1]
    category = sys.argv[2]          # peaks, nonpeaks or idr
    if category == "idr":
        elements_tsv = os.path.join(
            root_output_dir,
            f"task_4_chromatin_activity/processed_data/cell_line_idr_peaks/{cell_line}.bed",
        )
    else:
        elements_tsv = os.path.join(
            root_output_dir,
            f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_{category}.bed",
        )

    chroms = None
    batch_size = int(os.environ.get("HYENA16K_EMB_BATCH", 256))
    num_workers = 0
    seed = 0
    device = "cuda"

    out_dir = os.path.join(
        root_output_dir, f"task_4_chromatin_activity/embeddings/{model_name}/"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cell_line}_{category}.h5")

    dataset = SimpleSequence(genome_fa, elements_tsv, chroms, seed)
    extractor = Hyena16kHRGEmbeddingExtractor(
        model_path, model_code_dir, batch_size, num_workers, device
    )
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)
