from brainscore_core.metrics import Score


def ceiling_normalize(raw_score: Score, ceiling: Score) -> Score:
    # normalize by ceiling, but not above 1
    score = raw_score / ceiling
    score.attrs['raw'] = raw_score
    score.attrs['ceiling'] = ceiling
    if score > 1:
        overshoot_value = score.item()
        # ideally we would just update the value, but I could not figure out how to update a scalar DataArray
        attrs = score.attrs
        score = type(score)(1, coords=score.coords, dims=score.dims)
        score.attrs = attrs
        score.attrs['overshoot'] = overshoot_value
    return score
