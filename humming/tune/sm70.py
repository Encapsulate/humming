"""Conservative launch heuristics for Volta (SM70).

Volta has the same 64 KiB default shared-memory launch limit as the SM75
profile used by Humming.  The kernel backend is deliberately kept separate:
these heuristics only select safe tile sizes for the Volta implementation.
"""

from humming.tune.sm75 import Sm75Heuristics


class Sm70Heuristics(Sm75Heuristics):
    """Volta baseline, initially limited to the FP16 activation path."""

    sm_version = 70

    @classmethod
    def get_config(cls, *args, **kwargs):
        config = super().get_config(*args, **kwargs)
        # The current epilogue stores one 32-lane warp tile.  Keep the Volta
        # baseline at one complete warp even for very small token batches;
        # bounds checks in the kernel discard inactive rows.
        block_m, block_n, block_k = config["block_shape"]
        warp_m, warp_n, warp_k = config["warp_shape"]
        if block_m < 32:
            config["block_shape"] = (32, block_n, block_k)
            config["warp_shape"] = (32, warp_n, warp_k)
        return config
