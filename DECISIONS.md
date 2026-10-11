# Decision log

Decisions made while building the project (deadline-driven, no back-and-forth), in order.

## Environment
- Hardware: CPU only (16 logical cores, no CUDA; torch 2.9.1+cpu, Python 3.14). Models kept small so each run is minutes.
- `datasets` library not installed; used `huggingface_hub.hf_hub_download` to fetch raw JSONL files instead (fewer dependencies).
- No system pandoc; used `pypandoc_binary` (bundles pandoc 3.9) + MiKTeX `xelatex` for the PDF.

## Step 1 - Data
- **Dataset: `openbmb/UltraFeedback`** (licence: MIT on the HF card). Verified fields: `source`, `instruction` (prompt), `completions[].model`, `completions[].response`. 63,967 prompts x 4 completions, 17 models. Each prompt was answered by 4 randomly drawn models, so prompts repeat across models in the raw data.
- **Families (by developer)**: GPT = {gpt-3.5-turbo, gpt-4}; LLaMA-2 = {llama-2-7b/13b/70b-chat}; Bard = {bard}; Falcon = {falcon-40b-instruct}; MPT = {mpt-30b-chat}. 5 classes.
- **Dropped models**: vicuna-33b, wizardlm-7b/13b/70b, ultralm-13b/65b, alpaca-7b are LLaMA fine-tunes from other labs: putting them in "LLaMA" or in their own families would be ambiguous (the base model is LLaMA but the fine-tuning data is often distilled from GPT). starchat (code-specialised) and pythia-12b (834 rows, rare) also dropped.
- **Cleaning**: strip whitespace; drop empty prompt/response; drop exact duplicate responses; drop duplicate (prompt, model); keep responses with 10-1000 whitespace words.
- **Balancing**: random downsample to 4,000 per family (Bard, the smallest family, had 8,266 after cleaning). 20,000 rows total. 4k chosen so the BiLSTM trains in a few minutes on CPU.
- **Task label**: `truthful_qa`/`false_qa` -> factual_qa; otherwise regex heuristics on the first 600 chars of the prompt: code keywords -> coding; math keywords/arithmetic -> math; any remaining `flan_*` -> nlp_task; writing keywords -> writing; else chat. Heuristic is noisy; documented as a limitation.
- **Split**: 80/10/10 over unique prompts (seed 448), so a prompt never appears in two splits (assert in code).
- **Observation**: after random downsampling only 1,880 / 17,899 prompts have >1 family, so most prompts appear once. Bard has 0 examples from `flan_v2_niv2`/`flan_v2_p3`/`flan2021` (Bard was not run on those subsets) - a source/label correlation that input-only models can exploit (RQ2).
- User instruction (mid-run): do not push anything. Commits are local only; no remote configured.

## Step 2 - Models (speed fallback)
- First BiLSTM (hidden 128, packed sequences, 4 jobs in parallel) took >10 min per epoch on CPU. Benchmarked: packing was ~3x slower than an unpacked LSTM on this machine. Fallback: **unpacked BiLSTM, hidden 64 per direction, masked mean+max pooling (final hidden state dropped because it would see padding)**, length-bucketed batches to cut padding, max 6 epochs with early stopping (patience 2) on val accuracy. CNN uses the same loop for consistency.
- Tokeniser is case-preserving (regex `\n|\w+|[^\w\s]`, newline kept as `<nl>`) so that the "lowercase + strip punctuation" ablation in RQ4 actually removes information. Vocab: train split only, min freq 2, max 30k.
- RQ2 "input + output": prompt truncated to 64 tokens, then `<sep>`, then the response, total max 256 tokens.
- Added optional CUDA support to train.py (falls back to CPU) and scripts/icds_array.slurm so experiments can be run as a Slurm job array on Penn State ICDS Roar Collab. Each run JSON records the device it used.
- train.py now skips a run whose result JSON already exists (use --force to redo), so 'bash scripts/resume_local.sh' resumes after any interruption. Each run's log is written only when the run starts, so a killed run simply restarts from scratch.

## Steps 3-5 - Runs and infrastructure
- **Incident:** when the first (too slow) grid was killed at ~12:05, only the python processes were stopped, not the `xargs` queue, so the old queue kept launching runs in parallel with the new grid. Some of those late runs overwrote 4 RQ1/RQ2 JSON files (`cnn_both_none_s1/s2`, `rnn_input_none_s1/s2`) at ~13:45; two of them used the old 128-hidden packed BiLSTM. Fixed by restoring the committed versions (produced by the correct grid) and validating that all 60 result files have `epochs=6`, no subsampling, and the final BiLSTM size. The two re-run CNN files had identical test accuracy, a useful reproducibility check. Side effect: RQ1/RQ2 wall-clock times are inflated by the extra concurrent jobs, so the report gives them as approximate.
- **ICDS:** GPU jobs on the `open` account crashed with a CUDA error (traceback not inspected because of time). Fallback: the same CPU code as a Slurm array on the `basic` partition (`--account=open`), 42 RQ3/RQ4 runs in parallel, finished in ~20 min. One RQ3 run (`cnn_output_none_holdout-writing_s2`) is from the earlier GPU attempt (device=cuda) and completed normally; everything else ran on CPU. Duplicate local RQ3/RQ4 runs were stopped once the ICDS results were copied back.
- **Model choice:** the CNN is the better neural model (65.6 vs 59.6 test acc), so RQ3 uses the CNN for all 6 tasks; the BiLSTM is also run for writing and coding. RQ4 ablations were run for both models.
