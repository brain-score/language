from brainscore_core.metrics import Score


def ceiling_normalize(raw_score: Score, ceiling: Score) -> Score:
    # normalize by ceiling, but not above 1
    score = raw_score / ceiling
    score.attrs['raw'] = raw_score
    score.attrs['ceiling'] = ceiling
    if score > 1 or score < 0:
        out_of_range_value = score.item()
        # ideally we would just update the value, but I could not figure out how to update a scalar DataArray
        attrs = score.attrs
        in_range_value = 1 if score > 1 else 0
        score = type(score)(in_range_value, coords=score.coords, dims=score.dims)
        score.attrs = attrs
        score.attrs['original_out_of_range_score'] = out_of_range_value
    return score
