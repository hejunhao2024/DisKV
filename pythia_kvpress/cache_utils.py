def get_cache_from_kwargs(kwargs):
    cache = kwargs.get("layer_past", None)
    if cache is None:
        cache = kwargs.get("past_key_values", None)
    return cache


def get_layer_kv(cache, layer_idx):
    layer = cache.layers[layer_idx]
    return layer.keys, layer.values


def set_layer_kv(cache, layer_idx, keys, values):
    layer = cache.layers[layer_idx]
    layer.keys = keys.contiguous()
    layer.values = values.contiguous()


def get_cache_length(cache, layer_idx=0):
    return cache.get_seq_length(layer_idx)