# SLIM verl Ascend package

This directory is the canonical Ascend-adapted VERL package. The launcher loads
`SLIM/verl_ascend/__init__.py` under the runtime module name `verl`, so existing
absolute imports such as `from verl.utils import ...` continue to work.

Use the Search-QA launcher with:

```bash
export VERL_BACKEND=ascend
export ASCEND_VERL_ROOT=/path/to/SLIM/verl_ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash scripts/run_search_qa_slim_full.sh
```

Install Ascend-specific Python dependencies from `requirements-npu.txt`.
Additionally, install a mutually compatible CANN, PyTorch, torch-npu, vLLM, and
vllm-ascend stack. Those core packages are intentionally not pinned here because
their versions must match the target Ascend firmware and CANN installation.
