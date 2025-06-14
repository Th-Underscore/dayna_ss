import torch
import gc
import os
import psutil
import mmap
import tempfile
from pathlib import Path
import traceback

from extensions.dayna_ss.utils.helpers import (
    _ERROR,
    _SUCCESS,
    _INPUT,
    _GRAY,
    _HILITE,
    _BOLD,
    _RESET,
    _DEBUG,
)


class VRAMManager:
    def __init__(self, virtual_memory_dir: str | None = None, max_vram_mb: int | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.virtual_memory_dir = Path(virtual_memory_dir or tempfile.gettempdir()) / "dss_virtual_memory"
        self.virtual_memory_dir.mkdir(parents=True, exist_ok=True)

        # Track active virtual memory pages
        self.active_page_files = []
        self.max_vram_mb = max_vram_mb

        # Sequence tracking
        self.current_position = 0  # Current position in sequence
        self.max_position = 0  # Total number of positions in sequence
        self.load_times = []  # Track load times for performance monitoring

        # Cache map
        self.context_cache_map: dict[int, dict] = {}

    def increment_position(self):
        """Move to next position in sequence"""
        self.current_position += 1
        if self.current_position > self.max_position:
            self.max_position = self.current_position
        print(f"{_SUCCESS}Advanced to position {self.current_position} of {self.max_position}{_RESET}")

    def reset_sequence(self):
        """Reset sequence tracking"""
        self.current_position = 0
        print(f"{_SUCCESS}Reset sequence. Max positions: {self.max_position}{_RESET}")

    def _get_kv_cache(self, model) -> dict[str, torch.Tensor]:
        """Extract KV cache from the model's attention layers"""
        cache = {}

        try:
            print(f"{_BOLD}Attempting to access model cache structures{_RESET}")

            print(f"{_SUCCESS}Checking for LlamaCpp model{_RESET}")
            # For LlamaCpp models
            # Check if model is wrapped in LlamacppHF
            if hasattr(model, "model"):
                print(f"{_SUCCESS}Found wrapped model, checking inner model{_RESET}")
                inner_model = model.model
            else:
                print(f"{_SUCCESS}Using direct model{_RESET}")
                inner_model = model

            print(f"{_SUCCESS}Checking for _ctx: {hasattr(inner_model, '_ctx')}{_RESET}")
            print(f"{_SUCCESS}Checking for model: {hasattr(inner_model, 'model')}{_RESET}")
            if hasattr(inner_model, "_ctx"):
                print(
                    f"{_SUCCESS}Found LlamaCpp model{_RESET} {hasattr(inner_model, 'n_tokens')} {hasattr(inner_model, 'input_ids')} {hasattr(inner_model, 'scores')}"
                )
                if hasattr(inner_model, "n_tokens") and hasattr(inner_model, "input_ids") and hasattr(inner_model, "scores"):
                    print(f"{_SUCCESS}Found all required LlamaCpp attributes{_RESET}")
                    # Create a copy of the current cache state
                    cache = {
                        "n_tokens": inner_model.n_tokens,
                        "input_ids": (
                            inner_model.input_ids.copy()
                            if isinstance(inner_model.input_ids, (list, torch.Tensor))
                            else inner_model.input_ids
                        ),
                        "scores": (
                            inner_model.scores.copy()
                            if isinstance(inner_model.scores, (list, torch.Tensor))
                            else inner_model.scores
                        ),
                        "ctx": inner_model._ctx.ctx,
                    }
                    print(f"{_SUCCESS}LlamaCpp cache extracted with {len(cache)} items{_RESET}")
                    return cache
                else:
                    print(f"{_ERROR}LlamaCpp model missing required attributes{_RESET}")
            else:
                print(f"{_SUCCESS}Not a LlamaCpp model, checking other types{_RESET}")

            print(f"{_SUCCESS}Checking for GGUF HF model{_RESET}")
            # For GGUF HF models
            if hasattr(model, "kv_cache"):
                print(f"{_SUCCESS}Found GGUF HF model kv_cache{_RESET}")
                if isinstance(model.kv_cache, dict):
                    print(f"{_SUCCESS}Found dictionary-style kv_cache{_RESET}")
                    # Direct KV cache format
                    cache = {k: v.clone() for k, v in model.kv_cache.items()}
                elif isinstance(model.kv_cache, (list, tuple)):
                    print(f"{_SUCCESS}Found list/tuple-style kv_cache{_RESET}")
                    # past_key_values format
                    for i, layer_cache in enumerate(model.kv_cache):
                        if layer_cache is not None:
                            cache[f"layer_{i}_k"] = layer_cache[0].clone()
                            cache[f"layer_{i}_v"] = layer_cache[1].clone()
                print(f"{_SUCCESS}GGUF HF cache extracted with {len(cache)} items{_RESET}")
                return cache
            else:
                print(f"{_SUCCESS}Not a GGUF HF model, checking other types{_RESET}")

            print(f"{_SUCCESS}Checking for transformers model{_RESET}")
            # For transformers library models
            if hasattr(model, "past_key_values"):
                print(f"{_SUCCESS}Found transformers-style past_key_values cache{_RESET}")
                for i, layer_cache in enumerate(model.past_key_values):
                    if layer_cache is not None:
                        print(f"{_SUCCESS}Processing cache layer {i}{_RESET}")
                        cache[f"layer_{i}_k"] = layer_cache[0]
                        cache[f"layer_{i}_v"] = layer_cache[1]
                print(f"{_SUCCESS}Transformers cache extracted with {len(cache)} items{_RESET}")
                return cache
            else:
                print(f"{_SUCCESS}Not a transformers model, checking other types{_RESET}")

            print(f"{_SUCCESS}Checking for attention cache{_RESET}")
            # For models using attention cache
            if hasattr(model, "attention_cache"):
                print(f"{_SUCCESS}Found attention_cache{_RESET}")
                cache = model.attention_cache
                print(f"{_SUCCESS}Attention cache extracted{_RESET}")
                return cache
            else:
                print(f"{_SUCCESS}No attention cache, checking other types{_RESET}")

            print(f"{_SUCCESS}Checking for custom TGWUI cache{_RESET}")
            # For custom TGWUI caching
            if hasattr(model, "cache"):
                print(f"{_SUCCESS}Found custom TGWUI cache{_RESET}")
                cache = model.cache
                print(f"{_SUCCESS}Custom TGWUI cache extracted{_RESET}")
                return cache
            else:
                print(f"{_SUCCESS}No custom cache found{_RESET}")
        except Exception as e:
            print(f"{_ERROR}Warning: Error accessing model cache: {e}{_RESET}")
            print(f"{_ERROR}{traceback.format_exc()}{_RESET}")

        print(f"{_SUCCESS}No cache found{_RESET}")
        return cache

    def _get_llamacpp_cache_size(self, cache: dict) -> int:
        """Calculate size of LlamaCpp cache in bytes"""
        print(f"{_BOLD}Calculating LlamaCpp cache size{_RESET}")
        total_size = 0

        try:
            # Print cache keys for debugging
            print(f"{_SUCCESS}Cache keys: {list(cache.keys())}{_RESET}")

            # n_tokens: int (8 bytes) - this represents actual tokens used
            n_tokens = cache.get("n_tokens", 0)
            total_size += 8
            print(f"{_SUCCESS}n_tokens: {n_tokens}, size: {total_size}{_RESET}")

            # input_ids: list of ints (4 bytes per int)
            if "input_ids" in cache:
                input_shape = None
                if isinstance(cache["input_ids"], (list, tuple)):
                    size = len(cache["input_ids"]) * 4
                    input_shape = f"list[{len(cache['input_ids'])}]"
                elif hasattr(cache["input_ids"], "nbytes"):
                    # Only count up to n_tokens if available
                    if n_tokens > 0:
                        size = min(n_tokens * 4, cache["input_ids"].nbytes)
                    else:
                        size = cache["input_ids"].nbytes
                    input_shape = f"array{cache['input_ids'].shape}"
                else:
                    size = 0
                    input_shape = "unknown"
                total_size += size
                print(f"{_SUCCESS}input_ids shape: {input_shape}, +size: {size}, total: {total_size}{_RESET}")

            # scores: list of floats (8 bytes per float)
            if "scores" in cache:
                scores_shape = None
                if isinstance(cache["scores"], (list, tuple)):
                    size = len(cache["scores"]) * 8
                    scores_shape = f"list[{len(cache['scores'])}]"
                elif hasattr(cache["scores"], "nbytes"):
                    scores = cache["scores"]
                    scores_shape = f"array{scores.shape}"
                    # Analyze scores structure
                    rows, vocab_size = scores.shape
                    theoretical_size = n_tokens * vocab_size * 8
                    actual_size = scores.nbytes
                    print(f"{_SUCCESS}Scores analysis:")
                    print(f" - Shape: {scores.shape} ({rows} x {vocab_size})")
                    print(f" - Theoretical size for {n_tokens} tokens: {theoretical_size / (1024*1024):.2f} MB")
                    print(f" - Actual allocated size: {actual_size / (1024*1024):.2f} MB")
                    print(f" - Bytes per row: {actual_size / rows:.2f}")
                    print(f" - Bytes per token: {actual_size / n_tokens:.2f}")

                    # For now, use actual size since we don't fully understand the structure
                    size = actual_size
                else:
                    size = 0
                    scores_shape = "unknown"
                total_size += size
                print(f"{_SUCCESS}scores shape: {scores_shape}, size used: {size}, total: {total_size}{_RESET}")

            # ctx: actual context data
            if "ctx" in cache and cache["ctx"] is not None:
                ctx = cache["ctx"]
                ctx_size = 0
                if hasattr(ctx, "nbytes"):
                    ctx_size = ctx.nbytes
                elif hasattr(ctx, "__sizeof__"):
                    ctx_size = ctx.__sizeof__()
                total_size += ctx_size
                print(f"{_SUCCESS}ctx size: {ctx_size}, total: {total_size}{_RESET}")

                # Try to inspect ctx structure
                print(f"{_SUCCESS}ctx analysis:")
                print(f" - Type: {type(ctx)}")
                if hasattr(ctx, "shape"):
                    print(f" - Shape: {ctx.shape}")
                if hasattr(ctx, "dtype"):
                    print(f" - dtype: {ctx.dtype}")
            else:
                # ctx pointer: 8 bytes on 64-bit systems
                total_size += 8
                print(f"{_SUCCESS}ctx pointer: +8 bytes, total: {total_size}{_RESET}")

            # position: int (8 bytes)
            if "position" in cache:
                total_size += 8
                print(f"{_SUCCESS}position: +8 bytes, total: {total_size}{_RESET}")

            print(f"{_SUCCESS}Final LlamaCpp cache size: {total_size / (1024 ** 2):.2f} MB{_RESET}")

        except Exception as e:
            print(f"{_ERROR}Error calculating LlamaCpp cache size: {str(e)}{_RESET}")
            print(f"{_ERROR}{traceback.format_exc()}{_RESET}")

        return total_size

    def _get_cache_size(self, cache_value) -> int:
        """Get size of a cache value in bytes, handling both tensor and non-tensor types"""
        try:
            # Handle LlamaCpp cache dictionary
            if isinstance(cache_value, dict) and "n_tokens" in cache_value and "input_ids" in cache_value:
                print(f"{_SUCCESS}Getting LlamaCpp cache size{_RESET}")
                return self._get_llamacpp_cache_size(cache_value)

            # Handle tensors
            if isinstance(cache_value, torch.Tensor):
                return cache_value.numel() * cache_value.element_size()

            # Handle numpy arrays
            if hasattr(cache_value, "nbytes"):
                return cache_value.nbytes

            # Handle recursive structures
            if isinstance(cache_value, (list, tuple)):
                return sum(self._get_cache_size(item) for item in cache_value)

            if isinstance(cache_value, dict):
                return sum(self._get_cache_size(v) for v in cache_value.values())

            # Handle primitive types
            if isinstance(cache_value, (int, float)):
                return 8  # 64-bit numbers

            # Handle objects with __sizeof__
            if hasattr(cache_value, "__sizeof__"):
                return cache_value.__sizeof__()

            return 0  # Default case

        except Exception as e:
            print(f"{_ERROR}Error calculating cache size: {str(e)}{_RESET}")
            return 0

    def has_active_cache(self) -> bool:
        """Check if there's an active KV cache in the model"""
        try:
            print(f"{_SUCCESS}Checking for active KV cache{_RESET}")
            from modules.shared import model

            if model is not None:
                cache = self._get_kv_cache(model)
                if cache:
                    print(f"{_SUCCESS}Found cache with {len(cache)} items{_RESET}")
                    return any(not self._is_cache_empty(v) for v in cache.values())
                print(f"{_SUCCESS}No cache found{_RESET}")
                return False
        except ImportError:
            print("Warning: Could not access shared model")
        return False

    def _offload_cache_to_virtual_memory(self, cache: dict[str, torch.Tensor]):
        """Offload context cache to virtual memory"""
        page_file = self.virtual_memory_dir / f"cache_state_{len(self.active_page_files)}.bin"

        # Save cache tensors to file
        cache_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in cache.items()}
        torch.save(cache_cpu, page_file)

        # Clear cache tensors
        for k in cache:
            if isinstance(cache[k], torch.Tensor):
                cache[k] = torch.empty(0)

        self.active_page_files.append((page_file, "cache"))

    def restore_cache_from_virtual_memory(self):
        """Restore context cache from virtual memory"""
        if not self.active_page_files:
            return

        try:
            print(f"{_BOLD}Attempting to restore cache from virtual memory{_RESET}")
            from modules.shared import model

            if model is not None:
                print(f"{_SUCCESS}Found model, looking for cache files{_RESET}")
                # Find latest cache state
                cache_files = [(i, f) for i, (f, type_) in enumerate(self.active_page_files) if type_ == "cache"]
                if not cache_files:
                    print(f"{_SUCCESS}No cache files found{_RESET}")
                    return

                idx, page_file = cache_files[-1]
                cache_dict = torch.load(page_file)

                # Restore cache to model
                cache = self._get_kv_cache(model)
                if cache:
                    for k, v in cache_dict.items():
                        if k in cache:
                            if isinstance(v, torch.Tensor):
                                cache[k] = v.to(self.device)
                            else:
                                cache[k] = v

                # Clean up page file
                page_file.unlink()
                self.active_page_files.pop(idx)

        except ImportError:
            print("Warning: Could not access shared model")

    def get_context_cache(self) -> dict | None:
        """Get cache for current position"""
        print(f"{_BOLD}Retrieving cache for position {self.current_position}{_RESET}")

        # Check RAM cache
        if self.current_position in self.context_cache_map:
            cache = self.context_cache_map[self.current_position]
            if cache:
                print(f"{_SUCCESS}Found cache in RAM{_RESET}")
                cache_size_gb = self._get_cache_size(cache) / (1024**3)
                ram_available = psutil.virtual_memory().available / (1024**3)
                print(f"{_SUCCESS}Cache size: {cache_size_gb:.2f} GB, Available RAM: {ram_available:.2f} GB{_RESET}")
                return cache

        print(f"{_ERROR}No cache found for position {self.current_position}{_RESET}")
        return None

    def save_context_cache(self):
        """Save current cache state"""
        print(f"{_BOLD}Saving cache at position {self.current_position}{_RESET}")
        print(f"{_SUCCESS}Memory before save - {self.get_memory_usage()}{_RESET}")

        try:
            from modules.shared import model

            if model is not None:
                cache = None
                if hasattr(model, "model") and hasattr(model.model, "_ctx"):
                    # LlamaCpp model
                    cache = self._save_llamacpp_cache(model)
                else:
                    # Other model types
                    cache = self._get_kv_cache(model)
                    if cache:
                        cache["position"] = self.current_position

                if cache:
                    print(f"{_SUCCESS}Found cache for position {self.current_position}{_RESET}")
                    cache_size_gb = self._get_cache_size(cache) / (1024**3)
                    ram_available = psutil.virtual_memory().available / (1024**3)
                    print(f"{_SUCCESS}Cache size: {cache_size_gb:.2f} GB, Available RAM: {ram_available:.2f} GB{_RESET}")

                    # If RAM is low, offload to virtual memory
                    if ram_available < cache_size_gb * 2:  # Keep 2x safety margin
                        print(f"{_SUCCESS}Low RAM, offloading to virtual memory{_RESET}")
                        self._offload_cache_to_virtual_memory(cache)
                    else:
                        # Save to RAM
                        print(f"{_SUCCESS}Saving to RAM{_RESET}")
                        self.context_cache_map[self.current_position] = cache

                    print(f"{_SUCCESS}Memory after save - {self.get_memory_usage()}{_RESET}")
                    return True
        except Exception as e:
            print(f"{_ERROR}Error saving cache: {str(e)}{_RESET}")
        return False

    def preload_next_context(self):
        """Preload the next context in sequence"""
        next_position = self.current_position + 1
        if next_position > self.max_position:
            print(f"{_SUCCESS}No more positions to preload (at {self.current_position} of {self.max_position}){_RESET}")
            return

        print(f"{_BOLD}Preloading cache for position {next_position}{_RESET}")
        cached_kv = self.get_context_cache()

        if cached_kv is not None:
            try:
                from modules.shared import model

                if model is not None:
                    print(f"{_SUCCESS}Memory before loading - {self.get_memory_usage()}{_RESET}")

                    if hasattr(model, "model") and hasattr(model.model, "_ctx"):
                        # LlamaCpp model
                        self._load_llamacpp_cache(model, cached_kv)
                    else:
                        # Other model types
                        current_cache = self._get_kv_cache(model)
                        if current_cache:
                            for k, v in cached_kv.items():
                                if k in current_cache and k != "position":
                                    if isinstance(v, torch.Tensor):
                                        current_cache[k] = v.to(self.device)
                                    else:
                                        current_cache[k] = v

                    print(f"{_SUCCESS}Memory after loading - {self.get_memory_usage()}{_RESET}")

                    # Calculate average load time
                    if self.load_times:
                        avg_time = sum(self.load_times) / len(self.load_times)
                        print(f"{_SUCCESS}Average load time: {avg_time:.4f}ms{_RESET}")

            except ImportError:
                print(f"{_ERROR}Could not access model for cache preload{_RESET}")

    def clear_context_cache(self):
        """Clear saved context caches"""
        self.context_cache_map.clear()
        # Remove context cache files
        self.active_page_files = [(f, t) for f, t in self.active_page_files if t != "context_cache"]

    def _save_llamacpp_cache(self, model):
        """Save LlamaCpp model cache"""
        print(f"{_BOLD}Saving LlamaCpp cache for position {self.current_position}{_RESET}")

        def deep_copy_array(arr):
            if arr is None:
                return None
            import numpy as np

            if isinstance(arr, np.ndarray):
                return arr.copy()
            if isinstance(arr, (list, torch.Tensor)):
                return arr.copy()
            return arr

        cache = {
            "n_tokens": model.model.n_tokens,
            "input_ids": deep_copy_array(model.model.input_ids),
            "scores": deep_copy_array(model.model.scores),
            "ctx": model.model._ctx.ctx,  # C pointer, can't copy
            "last_n_tokens_data": (
                deep_copy_array(model.model.last_n_tokens_data) if hasattr(model.model, "last_n_tokens_data") else None
            ),
            "prefix_cache": (deep_copy_array(model.model.prefix_cache) if hasattr(model.model, "prefix_cache") else None),
            "position": self.current_position,
        }

        print(f"{_SUCCESS}Cache saved at position {cache['position']}{_RESET}")
        return cache

    def _load_llamacpp_cache(self, model, cache: dict):
        """Load cache into LlamaCpp model"""
        print(f"{_BOLD}Loading LlamaCpp cache for position {cache['position']}{_RESET}")
        try:
            import time

            start_time = time.time()

            model.model.n_tokens = int(cache["n_tokens"])
            model.model.input_ids = cache["input_ids"]
            model.model.scores = cache["scores"]
            model.model._ctx.ctx = cache["ctx"]
            if cache.get("last_n_tokens_data") is not None:
                model.model.last_n_tokens_data = cache["last_n_tokens_data"]
            if cache.get("prefix_cache") is not None:
                model.model.prefix_cache = cache["prefix_cache"]

            load_time = (time.time() - start_time) * 1000  # Convert to ms
            self.load_times.append(load_time)

            print(f"{_SUCCESS}Cache loaded with {model.model.n_tokens} tokens in {load_time:.4f}ms{_RESET}")
            print(f"{_SUCCESS}Cache shapes - input_ids: {cache['input_ids'].shape}, scores: {cache['scores'].shape}{_RESET}")

            # Update position after successful load
            self.current_position = cache["position"]

        except Exception as e:
            print(f"{_ERROR}Error loading cache: {str(e)}{_RESET}")

    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics"""
        stats = {
            "ram_used_gb": psutil.Process().memory_info().rss / (1024**3),
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
            "virtual_memory_pages": len(self.active_page_files),
        }

        if self.device.type == "cuda":
            stats.update(
                {
                    "vram_used_gb": torch.cuda.memory_allocated() / (1024**3),
                    "vram_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                    "vram_max_gb": torch.cuda.get_device_properties(self.device).total_memory / (1024**3),
                }
            )

        return stats

    def __del__(self):
        """Cleanup any remaining page files"""
        for page_file, _ in self.active_page_files:
            try:
                page_file.unlink()
            except:
                pass
