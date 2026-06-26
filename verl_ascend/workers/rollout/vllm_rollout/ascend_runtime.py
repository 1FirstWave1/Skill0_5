import ctypes.util
import os
from pathlib import Path

from verl.utils.device import is_npu_available


def prepare_ascend_vllm_runtime():
    if not is_npu_available:
        return

    try:
        from vllm_ascend.patch import platform, worker  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "vLLM-Ascend is required for NPU rollout. Install vllm-ascend in the training environment."
        ) from exc

    if ctypes.util.find_library("atb"):
        return

    candidates = [
        "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0/lib",
        "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib",
        "/usr/local/Ascend/nnal/atb/latest/lib",
        "/usr/local/Ascend/nnal/atb/lib",
    ]
    existing = [path for path in candidates if (Path(path) / "libatb.so").exists()]
    if existing:
        raise RuntimeError(
            "libatb.so exists but is not visible to the dynamic linker. "
            "Source /usr/local/Ascend/nnal/atb/set_env.sh before launching Ray/training, "
            "or add the ATB lib directory to LD_LIBRARY_PATH. "
            f"Detected candidate dirs: {existing}. Current LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}"
        )

    raise RuntimeError(
        "libatb.so was not found. Install Ascend NNAL/ATB and source "
        "/usr/local/Ascend/nnal/atb/set_env.sh before launching Ray/training. "
        f"Current LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}"
    )


def ascend_vllm_engine_kwargs(config, engine_kwargs):
    if not is_npu_available:
        return {}

    npu_kwargs = {
        "enable_sleep_mode": False,
        "enable_prefix_caching": False,
    }

    if "additional_config" not in engine_kwargs:
        graph_enabled = int(os.environ.get("VLLM_ENABLE_GRAPH_MODE", "0"))
        npu_kwargs["additional_config"] = {
            "torchair_graph_config": {
                "enabled": graph_enabled,
                "use_cached_graph": False,
                "graph_batch_sizes_init": False,
                "graph_batch_sizes": [config.max_num_seqs],
                "enable_multistream_mla": False,
                "enable_multistream_moe": False,
                "enable_view_optimize": False,
                "enable_kv_nz": False,
                "enable_frozen_parameter": False,
            },
            "ascend_scheduler_config": {
                "enabled": True,
            },
            "refresh": True,
        }

    return npu_kwargs
