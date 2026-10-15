# 13 Production-trace cost projection and queueing check

Replays the first 48 MiB HTTP-range slice of AzureLLMInferenceTrace_conv_1week.csv
(Azure Public Dataset, release dataset-llm-2024; cite Stojkovic et al., DynamoLLM, HPCA 2025).

The trace carries no prompt content (privacy), so no semantic hit rate can be
measured. The script (a) projects the audit cost onto production-scale arrival
dynamics using the measured M3 Pro cost medians (no token-count scaling), and
(b) runs a single-server FCFS queueing check driven by the trace arrival times,
under fixed-median service times and a bootstrap resampling of the measured
timing samples in `e8_timings_m3pro.npz` (seed 42).

Raw slice: `data/azure_conv_slice.csv`. Re-fetch with:

    curl -L -r 0-50331647 -o azure_conv_slice.csv https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv

Run: `python replay_azure_trace.py` (expects `azure_conv_slice.csv` alongside
the script — copy it from `../data` — and `e8_timings_m3pro.npz`, which is
included here). Outputs `azure_replay_results.json` and `fig_azure_replay.pdf`.
