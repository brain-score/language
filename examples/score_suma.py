from brainscore_language import score

model_score = score(model_identifier='suma', benchmark_identifier='Pereira2018.243sentences-linear')
print(model_score)

'''
array(0.98581247)
Attributes:
    raw:                   <xarray.Score ()>\narray(0.34876988)
    ceiling:               <xarray.Score 'data' ()>\narray(0.35378928)
    model_identifier:      suma
    benchmark_identifier:  Pereira2018.243sentences-linear
'''