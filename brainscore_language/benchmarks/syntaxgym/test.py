import numpy as np
from pprint import pprint
import pytest

from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.benchmarks.syntaxgym import SyntaxGymSingleTSE
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject


# Reference region surprisals computed with syntaxgym-core
# for the test suite subordination_src-src.
#
# See notebook in https://colab.research.google.com/drive/1qziyPcu65jffizSPi-ZGHKR0x7BaHFMS#scrollTo=RgtnScy6LLKi .
DISTILGPT2_SUBORDINATION_SRC_REFERENCE = \
[{('sub_no-matrix', 1): 19.84858652913709,
  ('sub_no-matrix', 2): 38.27761140375277,
  ('sub_no-matrix', 3): 29.751378006746812,
  ('sub_no-matrix', 4): 47.65413128319506,
  ('sub_no-matrix', 5): 1.5145187795075914,
  ('no-sub_no-matrix', 1): 21.183614927141107,
  ('no-sub_no-matrix', 2): 37.427751739260216,
  ('no-sub_no-matrix', 3): 30.294518824193762,
  ('no-sub_no-matrix', 4): 48.05019176370221,
  ('no-sub_no-matrix', 5): 1.5036962552540964,
  ('sub_matrix', 1): 19.84858652913709,
  ('sub_matrix', 2): 38.27761140375277,
  ('sub_matrix', 3): 29.751378006746812,
  ('sub_matrix', 4): 47.65412715561145,
  ('sub_matrix', 5): 42.39613042087248,
  ('no-sub_matrix', 1): 21.183614927141107,
  ('no-sub_matrix', 2): 37.427751739260216,
  ('no-sub_matrix', 3): 30.294518824193762,
  ('no-sub_matrix', 4): 48.05020414645307,
  ('no-sub_matrix', 5): 48.51086651309947},
 {('sub_no-matrix', 1): 15.481061657043739,
  ('sub_no-matrix', 2): 20.541090323257418,
  ('sub_no-matrix', 3): 21.032507186474724,
  ('sub_no-matrix', 4): 21.510654556282063,
  ('sub_no-matrix', 5): 1.6935814401584692,
  ('no-sub_no-matrix', 1): 16.892056167795065,
  ('no-sub_no-matrix', 2): 18.94204688636498,
  ('no-sub_no-matrix', 3): 23.362924732356944,
  ('no-sub_no-matrix', 4): 21.192542546547394,
  ('no-sub_no-matrix', 5): 2.0879328430726654,
  ('sub_matrix', 1): 15.481061657043739,
  ('sub_matrix', 2): 20.541090323257418,
  ('sub_matrix', 3): 21.032507186474724,
  ('sub_matrix', 4): 21.510654556282063,
  ('sub_matrix', 5): 30.0925069705944,
  ('no-sub_matrix', 1): 16.892056167795065,
  ('no-sub_matrix', 2): 18.94204688636498,
  ('no-sub_matrix', 3): 23.362924732356944,
  ('no-sub_matrix', 4): 21.192542546547394,
  ('no-sub_matrix', 5): 34.16022583829358},
 {('sub_no-matrix', 1): 17.458578714557415,
  ('sub_no-matrix', 2): 44.434026029563846,
  ('sub_no-matrix', 3): 23.76181615478645,
  ('sub_no-matrix', 4): 52.25378930833386,
  ('sub_no-matrix', 5): 3.098146723472592,
  ('no-sub_no-matrix', 1): 17.20280267654123,
  ('no-sub_no-matrix', 2): 45.41114477251028,
  ('no-sub_no-matrix', 3): 24.130059021957983,
  ('no-sub_no-matrix', 4): 52.415132940431654,
  ('no-sub_no-matrix', 5): 3.374929754361067,
  ('sub_matrix', 1): 17.458578714557415,
  ('sub_matrix', 2): 44.434026029563846,
  ('sub_matrix', 3): 23.76181615478645,
  ('sub_matrix', 4): 52.25378930833386,
  ('sub_matrix', 5): 28.35409464085287,
  ('no-sub_matrix', 1): 17.20280267654123,
  ('no-sub_matrix', 2): 45.41114477251028,
  ('no-sub_matrix', 3): 24.130059021957983,
  ('no-sub_matrix', 4): 52.415132940431654,
  ('no-sub_matrix', 5): 29.672878159595665},
 {('sub_no-matrix', 1): 17.88796779896659,
  ('sub_no-matrix', 2): 54.61975407730992,
  ('sub_no-matrix', 3): 24.482945063783852,
  ('sub_no-matrix', 4): 32.95154483871123,
  ('sub_no-matrix', 5): 3.0266738294617244,
  ('no-sub_no-matrix', 1): 17.74169449061596,
  ('no-sub_no-matrix', 2): 52.93976150234407,
  ('no-sub_no-matrix', 3): 24.55781392721866,
  ('no-sub_no-matrix', 4): 34.741309554165205,
  ('no-sub_no-matrix', 5): 1.7552664575434327,
  ('sub_matrix', 1): 17.88796779896659,
  ('sub_matrix', 2): 54.61975407730992,
  ('sub_matrix', 3): 24.482945063783852,
  ('sub_matrix', 4): 32.95154999819075,
  ('sub_matrix', 5): 26.49159734446178,
  ('no-sub_matrix', 1): 17.74169449061596,
  ('no-sub_matrix', 2): 52.93976150234407,
  ('no-sub_matrix', 3): 24.55781392721866,
  ('no-sub_matrix', 4): 34.74130092278592,
  ('no-sub_matrix', 5): 30.13520682315062},
 {('sub_no-matrix', 1): 28.40811284364502,
  ('sub_no-matrix', 2): 30.774620432862854,
  ('sub_no-matrix', 3): 30.953590222882262,
  ('sub_no-matrix', 4): 30.672586393776314,
  ('sub_no-matrix', 5): 4.813360313439236,
  ('no-sub_no-matrix', 1): 32.95567740982881,
  ('no-sub_no-matrix', 2): 30.49579012049851,
  ('no-sub_no-matrix', 3): 32.08415326278971,
  ('no-sub_no-matrix', 4): 31.28184181376385,
  ('no-sub_no-matrix', 5): 3.1110602127952767,
  ('sub_matrix', 1): 28.40811284364502,
  ('sub_matrix', 2): 30.774620432862854,
  ('sub_matrix', 3): 30.953590222882262,
  ('sub_matrix', 4): 30.672586393776314,
  ('sub_matrix', 5): 38.83770360519069,
  ('no-sub_matrix', 1): 32.95567740982881,
  ('no-sub_matrix', 2): 30.49579012049851,
  ('no-sub_matrix', 3): 32.08415326278971,
  ('no-sub_matrix', 4): 31.28184181376385,
  ('no-sub_matrix', 5): 42.83607193078196},
 {('sub_no-matrix', 1): 20.154073120489265,
  ('sub_no-matrix', 2): 30.172181545242267,
  ('sub_no-matrix', 3): 25.562914763408436,
  ('sub_no-matrix', 4): 29.54087003821218,
  ('sub_no-matrix', 5): 3.5408751262479874,
  ('no-sub_no-matrix', 1): 20.996927071698927,
  ('no-sub_no-matrix', 2): 30.329587290583248,
  ('no-sub_no-matrix', 3): 26.868092304684843,
  ('no-sub_no-matrix', 4): 30.128514806222896,
  ('no-sub_no-matrix', 5): 2.0285295195166984,
  ('sub_matrix', 1): 20.154073120489265,
  ('sub_matrix', 2): 30.172181545242267,
  ('sub_matrix', 3): 25.562914763408436,
  ('sub_matrix', 4): 29.54087003821218,
  ('sub_matrix', 5): 58.487986484654996,
  ('no-sub_matrix', 1): 20.996927071698927,
  ('no-sub_matrix', 2): 30.329587290583248,
  ('no-sub_matrix', 3): 26.868092304684843,
  ('no-sub_matrix', 4): 30.128514806222896,
  ('no-sub_matrix', 5): 62.33690317651638},
 {('sub_no-matrix', 1): 18.542096244286643,
  ('sub_no-matrix', 2): 39.8330400403864,
  ('sub_no-matrix', 3): 26.94842264918045,
  ('sub_no-matrix', 4): 48.0418985452993,
  ('sub_no-matrix', 5): 3.3899132268695364,
  ('no-sub_no-matrix', 1): 19.875457098507894,
  ('no-sub_no-matrix', 2): 41.05056576373733,
  ('no-sub_no-matrix', 3): 27.56687828107589,
  ('no-sub_no-matrix', 4): 47.62817914417553,
  ('no-sub_no-matrix', 5): 2.7966109175760363,
  ('sub_matrix', 1): 18.542096244286643,
  ('sub_matrix', 2): 39.8330400403864,
  ('sub_matrix', 3): 26.94842264918045,
  ('sub_matrix', 4): 48.0418985452993,
  ('sub_matrix', 5): 38.41174987423249,
  ('no-sub_matrix', 1): 19.875457098507894,
  ('no-sub_matrix', 2): 41.05056576373733,
  ('no-sub_matrix', 3): 27.56687828107589,
  ('no-sub_matrix', 4): 47.62817914417553,
  ('no-sub_matrix', 5): 39.380539497397656},
 {('sub_no-matrix', 1): 22.058236137756936,
  ('sub_no-matrix', 2): 59.756337208919504,
  ('sub_no-matrix', 3): 30.511204925495342,
  ('sub_no-matrix', 4): 55.01269470653908,
  ('sub_no-matrix', 5): 1.465839606128826,
  ('no-sub_no-matrix', 1): 23.56970497693413,
  ('no-sub_no-matrix', 2): 59.835000009632346,
  ('no-sub_no-matrix', 3): 31.51849114522963,
  ('no-sub_no-matrix', 4): 54.874384280914725,
  ('no-sub_no-matrix', 5): 1.4070206796204912,
  ('sub_matrix', 1): 22.058236137756936,
  ('sub_matrix', 2): 59.756337208919504,
  ('sub_matrix', 3): 30.511204925495342,
  ('sub_matrix', 4): 55.012681979822915,
  ('sub_matrix', 5): 40.21390851942091,
  ('no-sub_matrix', 1): 23.56970497693413,
  ('no-sub_matrix', 2): 59.835000009632346,
  ('no-sub_matrix', 3): 31.51849114522963,
  ('no-sub_matrix', 4): 54.874387634576415,
  ('no-sub_matrix', 5): 45.11965086161496},
 {('sub_no-matrix', 1): 20.208583365645538,
  ('sub_no-matrix', 2): 32.71071151331525,
  ('sub_no-matrix', 3): 23.929238664163798,
  ('sub_no-matrix', 4): 36.27710164167995,
  ('sub_no-matrix', 5): 2.3424855684932644,
  ('no-sub_no-matrix', 1): 22.040704914258587,
  ('no-sub_no-matrix', 2): 32.724291263427375,
  ('no-sub_no-matrix', 3): 24.517287763386697,
  ('no-sub_no-matrix', 4): 35.637334951585494,
  ('no-sub_no-matrix', 5): 3.1888551890518615,
  ('sub_matrix', 1): 20.208583365645538,
  ('sub_matrix', 2): 32.71071151331525,
  ('sub_matrix', 3): 23.929238664163798,
  ('sub_matrix', 4): 36.27709957788814,
  ('sub_matrix', 5): 53.605858478311646,
  ('no-sub_matrix', 1): 22.040704914258587,
  ('no-sub_matrix', 2): 32.724291263427375,
  ('no-sub_matrix', 3): 24.517287763386697,
  ('no-sub_matrix', 4): 35.637334951585494,
  ('no-sub_matrix', 5): 57.65807396994879},
 {('sub_no-matrix', 1): 19.235979511249543,
  ('sub_no-matrix', 2): 29.229340836421937,
  ('sub_no-matrix', 3): 30.48163268066213,
  ('sub_no-matrix', 4): 33.534677813859204,
  ('sub_no-matrix', 5): 3.453748027390285,
  ('no-sub_no-matrix', 1): 19.29213597433987,
  ('no-sub_no-matrix', 2): 29.560537974164852,
  ('no-sub_no-matrix', 3): 30.8270084980988,
  ('no-sub_no-matrix', 4): 34.09155505761322,
  ('no-sub_no-matrix', 5): 4.115376980079769,
  ('sub_matrix', 1): 19.235979511249543,
  ('sub_matrix', 2): 29.229340836421937,
  ('sub_matrix', 3): 30.48163268066213,
  ('sub_matrix', 4): 33.534677813859204,
  ('sub_matrix', 5): 31.08725943869911,
  ('no-sub_matrix', 1): 19.29213597433987,
  ('no-sub_matrix', 2): 29.560537974164852,
  ('no-sub_matrix', 3): 30.8270084980988,
  ('no-sub_matrix', 4): 34.09155505761322,
  ('no-sub_matrix', 5): 38.23530109190076},
 {('sub_no-matrix', 1): 20.368089707084152,
  ('sub_no-matrix', 2): 49.25627457838232,
  ('sub_no-matrix', 3): 27.831169868066894,
  ('sub_no-matrix', 4): 40.37500888205905,
  ('sub_no-matrix', 5): 2.651787079006879,
  ('no-sub_no-matrix', 1): 22.02301684258265,
  ('no-sub_no-matrix', 2): 48.816563439233654,
  ('no-sub_no-matrix', 3): 29.33092944044274,
  ('no-sub_no-matrix', 4): 43.41645344149012,
  ('no-sub_no-matrix', 5): 3.236936378753577,
  ('sub_matrix', 1): 20.368089707084152,
  ('sub_matrix', 2): 49.25627457838232,
  ('sub_matrix', 3): 27.831169868066894,
  ('sub_matrix', 4): 40.37500888205905,
  ('sub_matrix', 5): 37.53002449026615,
  ('no-sub_matrix', 1): 22.02301684258265,
  ('no-sub_matrix', 2): 48.816563439233654,
  ('no-sub_matrix', 3): 29.33092944044274,
  ('no-sub_matrix', 4): 43.41645344149012,
  ('no-sub_matrix', 5): 41.495497436252926},
 {('sub_no-matrix', 1): 18.693022715239017,
  ('sub_no-matrix', 2): 32.35393178427385,
  ('sub_no-matrix', 3): 17.16735963193817,
  ('sub_no-matrix', 4): 33.140066410798106,
  ('sub_no-matrix', 5): 2.7831408923951595,
  ('no-sub_no-matrix', 1): 21.630230481791305,
  ('no-sub_no-matrix', 2): 31.841969357636522,
  ('no-sub_no-matrix', 3): 18.760749483148793,
  ('no-sub_no-matrix', 4): 34.65003788302557,
  ('no-sub_no-matrix', 5): 3.0621545382653252,
  ('sub_matrix', 1): 18.693022715239017,
  ('sub_matrix', 2): 32.35393178427385,
  ('sub_matrix', 3): 17.16735963193817,
  ('sub_matrix', 4): 33.140066410798106,
  ('sub_matrix', 5): 25.79307925254729,
  ('no-sub_matrix', 1): 21.630230481791305,
  ('no-sub_matrix', 2): 31.841969357636522,
  ('no-sub_matrix', 3): 18.760749483148793,
  ('no-sub_matrix', 4): 34.65003788302557,
  ('no-sub_matrix', 5): 29.401350735786},
 {('sub_no-matrix', 1): 23.349119910699418,
  ('sub_no-matrix', 2): 36.79505525594273,
  ('sub_no-matrix', 3): 27.448890271382062,
  ('sub_no-matrix', 4): 27.75725241265667,
  ('sub_no-matrix', 5): 6.4009658721533915,
  ('no-sub_no-matrix', 1): 25.37373717424875,
  ('no-sub_no-matrix', 2): 36.617579479211365,
  ('no-sub_no-matrix', 3): 27.65311554169803,
  ('no-sub_no-matrix', 4): 28.50976026573242,
  ('no-sub_no-matrix', 5): 3.7340522310725253,
  ('sub_matrix', 1): 23.349119910699418,
  ('sub_matrix', 2): 36.79505525594273,
  ('sub_matrix', 3): 27.448890271382062,
  ('sub_matrix', 4): 27.75725241265667,
  ('sub_matrix', 5): 43.75054754099549,
  ('no-sub_matrix', 1): 25.37373717424875,
  ('no-sub_matrix', 2): 36.617579479211365,
  ('no-sub_matrix', 3): 27.65311554169803,
  ('no-sub_matrix', 4): 28.50976026573242,
  ('no-sub_matrix', 5): 44.61412307986834},
 {('sub_no-matrix', 1): 18.45109196854789,
  ('sub_no-matrix', 2): 24.060699579747908,
  ('sub_no-matrix', 3): 26.564201712332675,
  ('sub_no-matrix', 4): 30.81202197289828,
  ('sub_no-matrix', 5): 2.2963507064513635,
  ('no-sub_no-matrix', 1): 17.20280267654123,
  ('no-sub_no-matrix', 2): 24.094664100250412,
  ('no-sub_no-matrix', 3): 27.832402983673582,
  ('no-sub_no-matrix', 4): 30.335879318860723,
  ('no-sub_no-matrix', 5): 1.7021960511401872,
  ('sub_matrix', 1): 18.45109196854789,
  ('sub_matrix', 2): 24.060699579747908,
  ('sub_matrix', 3): 26.564201712332675,
  ('sub_matrix', 4): 30.81202197289828,
  ('sub_matrix', 5): 40.518402942860256,
  ('no-sub_matrix', 1): 17.20280267654123,
  ('no-sub_matrix', 2): 24.094664100250412,
  ('no-sub_matrix', 3): 27.832402983673582,
  ('no-sub_matrix', 4): 30.335879318860723,
  ('no-sub_matrix', 5): 47.22794172102688},
 {('sub_no-matrix', 1): 21.58307008720231,
  ('sub_no-matrix', 2): 40.25998439141357,
  ('sub_no-matrix', 3): 27.037717417369624,
  ('sub_no-matrix', 4): 53.95255426259367,
  ('sub_no-matrix', 5): 6.407459937049971,
  ('no-sub_no-matrix', 1): 24.13069862543654,
  ('no-sub_no-matrix', 2): 39.806348676937795,
  ('no-sub_no-matrix', 3): 28.21254707178265,
  ('no-sub_no-matrix', 4): 52.36760893152307,
  ('no-sub_no-matrix', 5): 2.310403924801316,
  ('sub_matrix', 1): 21.58307008720231,
  ('sub_matrix', 2): 40.25998439141357,
  ('sub_matrix', 3): 27.037717417369624,
  ('sub_matrix', 4): 53.95256595741393,
  ('sub_matrix', 5): 48.12866878015645,
  ('no-sub_matrix', 1): 24.13069862543654,
  ('no-sub_matrix', 2): 39.806348676937795,
  ('no-sub_matrix', 3): 28.21254707178265,
  ('no-sub_matrix', 4): 52.36760893152307,
  ('no-sub_matrix', 5): 50.13949044999237},
 {('sub_no-matrix', 1): 21.749740537945264,
  ('sub_no-matrix', 2): 52.052811457427154,
  ('sub_no-matrix', 3): 18.66615145793761,
  ('sub_no-matrix', 4): 41.294273607159816,
  ('sub_no-matrix', 5): 2.207527514690852,
  ('no-sub_no-matrix', 1): 20.455624748789702,
  ('no-sub_no-matrix', 2): 51.89108430401795,
  ('no-sub_no-matrix', 3): 19.884462110107073,
  ('no-sub_no-matrix', 4): 42.374576093213825,
  ('no-sub_no-matrix', 5): 2.31318935581469,
  ('sub_matrix', 1): 21.749740537945264,
  ('sub_matrix', 2): 52.052811457427154,
  ('sub_matrix', 3): 18.66615145793761,
  ('sub_matrix', 4): 41.29429630886973,
  ('sub_matrix', 5): 35.2251669629595,
  ('no-sub_matrix', 1): 20.455624748789702,
  ('no-sub_matrix', 2): 51.89108430401795,
  ('no-sub_matrix', 3): 19.884462110107073,
  ('no-sub_matrix', 4): 42.374576093213825,
  ('no-sub_matrix', 5): 38.56545610077077},
 {('sub_no-matrix', 1): 22.700034802871336,
  ('sub_no-matrix', 2): 38.939772814729324,
  ('sub_no-matrix', 3): 25.22128171841117,
  ('sub_no-matrix', 4): 32.84511113945031,
  ('sub_no-matrix', 5): 3.0570659155915965,
  ('no-sub_no-matrix', 1): 24.47903916580011,
  ('no-sub_no-matrix', 2): 40.169493483887635,
  ('no-sub_no-matrix', 3): 26.33333460957155,
  ('no-sub_no-matrix', 4): 32.82603860743493,
  ('no-sub_no-matrix', 5): 3.5915728599999297,
  ('sub_matrix', 1): 22.700034802871336,
  ('sub_matrix', 2): 38.939772814729324,
  ('sub_matrix', 3): 25.22128171841117,
  ('sub_matrix', 4): 32.84511113945031,
  ('sub_matrix', 5): 31.485425760774746,
  ('no-sub_matrix', 1): 24.47903916580011,
  ('no-sub_matrix', 2): 40.169493483887635,
  ('no-sub_matrix', 3): 26.33333460957155,
  ('no-sub_matrix', 4): 32.82603860743493,
  ('no-sub_matrix', 5): 36.499637388885326},
 {('sub_no-matrix', 1): 19.271887424957885,
  ('sub_no-matrix', 2): 40.79664559050661,
  ('sub_no-matrix', 3): 26.810901623923552,
  ('sub_no-matrix', 4): 40.094853357378476,
  ('sub_no-matrix', 5): 6.078382829444246,
  ('no-sub_no-matrix', 1): 17.145941084582365,
  ('no-sub_no-matrix', 2): 42.24564895502921,
  ('no-sub_no-matrix', 3): 27.403583677786717,
  ('no-sub_no-matrix', 4): 39.54429974371404,
  ('no-sub_no-matrix', 5): 2.8781360255471844,
  ('sub_matrix', 1): 19.271887424957885,
  ('sub_matrix', 2): 40.79664559050661,
  ('sub_matrix', 3): 26.810901623923552,
  ('sub_matrix', 4): 40.094853357378476,
  ('sub_matrix', 5): 43.77389057421445,
  ('no-sub_matrix', 1): 17.145941084582365,
  ('no-sub_matrix', 2): 42.24564895502921,
  ('no-sub_matrix', 3): 27.403583677786717,
  ('no-sub_matrix', 4): 39.54429974371404,
  ('no-sub_matrix', 5): 52.64193467429601},
 {('sub_no-matrix', 1): 20.832624478093045,
  ('sub_no-matrix', 2): 36.50940755129819,
  ('sub_no-matrix', 3): 29.083164698269048,
  ('sub_no-matrix', 4): 49.64570143007171,
  ('sub_no-matrix', 5): 1.697204770646847,
  ('no-sub_no-matrix', 1): 25.069685375033682,
  ('no-sub_no-matrix', 2): 35.617647581593275,
  ('no-sub_no-matrix', 3): 30.66911784767371,
  ('no-sub_no-matrix', 4): 49.587663390789615,
  ('no-sub_no-matrix', 5): 1.963616731740342,
  ('sub_matrix', 1): 20.832624478093045,
  ('sub_matrix', 2): 36.50940755129819,
  ('sub_matrix', 3): 29.083164698269048,
  ('sub_matrix', 4): 49.64570143007171,
  ('sub_matrix', 5): 42.74057116863641,
  ('no-sub_matrix', 1): 25.069685375033682,
  ('no-sub_matrix', 2): 35.617647581593275,
  ('no-sub_matrix', 3): 30.66911784767371,
  ('no-sub_matrix', 4): 49.587663390789615,
  ('no-sub_matrix', 5): 46.617001988948125},
 {('sub_no-matrix', 1): 22.448761958585344,
  ('sub_no-matrix', 2): 32.15143459563375,
  ('sub_no-matrix', 3): 22.292358012185638,
  ('sub_no-matrix', 4): 46.77310135407315,
  ('sub_no-matrix', 5): 3.3833823576856723,
  ('no-sub_no-matrix', 1): 23.15318776204613,
  ('no-sub_no-matrix', 2): 31.538401404737364,
  ('no-sub_no-matrix', 3): 23.30091827939752,
  ('no-sub_no-matrix', 4): 45.351021013629854,
  ('no-sub_no-matrix', 5): 2.148098221754605,
  ('sub_matrix', 1): 22.448761958585344,
  ('sub_matrix', 2): 32.15143459563375,
  ('sub_matrix', 3): 22.292358012185638,
  ('sub_matrix', 4): 46.77310135407315,
  ('sub_matrix', 5): 42.947408167488376,
  ('no-sub_matrix', 1): 23.15318776204613,
  ('no-sub_matrix', 2): 31.538401404737364,
  ('no-sub_matrix', 3): 23.30091827939752,
  ('no-sub_matrix', 4): 45.351021013629854,
  ('no-sub_matrix', 5): 45.507769785244534},
 {('sub_no-matrix', 1): 19.73122556818831,
  ('sub_no-matrix', 2): 27.639846048288057,
  ('sub_no-matrix', 3): 28.083093150738456,
  ('sub_no-matrix', 4): 33.82243514369056,
  ('sub_no-matrix', 5): 2.2183565742850795,
  ('no-sub_no-matrix', 1): 20.0670925508501,
  ('no-sub_no-matrix', 2): 28.689049255551836,
  ('no-sub_no-matrix', 3): 28.262896024718646,
  ('no-sub_no-matrix', 4): 35.92639040646556,
  ('no-sub_no-matrix', 5): 1.909283112766509,
  ('sub_matrix', 1): 19.73122556818831,
  ('sub_matrix', 2): 27.639846048288057,
  ('sub_matrix', 3): 28.083093150738456,
  ('sub_matrix', 4): 33.82243514369056,
  ('sub_matrix', 5): 53.05216282760726,
  ('no-sub_matrix', 1): 20.0670925508501,
  ('no-sub_matrix', 2): 28.689049255551836,
  ('no-sub_matrix', 3): 28.262896024718646,
  ('no-sub_matrix', 4): 35.92639040646556,
  ('no-sub_matrix', 5): 58.56106948024324},
 {('sub_no-matrix', 1): 15.421868668269777,
  ('sub_no-matrix', 2): 54.27397435861769,
  ('sub_no-matrix', 3): 19.669065878686965,
  ('sub_no-matrix', 4): 25.491622666159284,
  ('sub_no-matrix', 5): 4.734922122068425,
  ('no-sub_no-matrix', 1): 16.0240858702534,
  ('no-sub_no-matrix', 2): 53.57980663235027,
  ('no-sub_no-matrix', 3): 20.270908244483035,
  ('no-sub_no-matrix', 4): 25.9813102008265,
  ('no-sub_no-matrix', 5): 3.89920510690427,
  ('sub_matrix', 1): 15.421868668269777,
  ('sub_matrix', 2): 54.27397435861769,
  ('sub_matrix', 3): 19.669065878686965,
  ('sub_matrix', 4): 25.491619914436868,
  ('sub_matrix', 5): 34.71751137683846,
  ('no-sub_matrix', 1): 16.0240858702534,
  ('no-sub_matrix', 2): 53.57980663235027,
  ('no-sub_matrix', 3): 20.270908244483035,
  ('no-sub_matrix', 4): 25.9813102008265,
  ('no-sub_matrix', 5): 41.833321597185034},
 {('sub_no-matrix', 1): 17.907130793856325,
  ('sub_no-matrix', 2): 31.11938278818389,
  ('sub_no-matrix', 3): 25.531152491499135,
  ('sub_no-matrix', 4): 46.53901611194908,
  ('sub_no-matrix', 5): 2.2405772487245073,
  ('no-sub_no-matrix', 1): 17.20280267654123,
  ('no-sub_no-matrix', 2): 30.99227574239337,
  ('no-sub_no-matrix', 3): 27.091719453792596,
  ('no-sub_no-matrix', 4): 47.86441244960888,
  ('no-sub_no-matrix', 5): 2.8696366429415443,
  ('sub_matrix', 1): 17.907130793856325,
  ('sub_matrix', 2): 31.11938278818389,
  ('sub_matrix', 3): 25.531152491499135,
  ('sub_matrix', 4): 46.53901611194908,
  ('sub_matrix', 5): 56.498433565285985,
  ('no-sub_matrix', 1): 17.20280267654123,
  ('no-sub_matrix', 2): 30.99227574239337,
  ('no-sub_matrix', 3): 27.091719453792596,
  ('no-sub_matrix', 4): 47.86441244960888,
  ('no-sub_matrix', 5): 60.744979821369654}]


@pytest.fixture
def subordination_src_benchmark():
    return SyntaxGymSingleTSE("subordination_src-src")


@pytest.fixture(scope="session")
def distilgpt2():
  return HuggingfaceSubject(model_id="distilgpt2", region_layer_mapping={})


def test_subordination_match(distilgpt2, subordination_src_benchmark):
    """
    The region-level surprisals computed on the subordination_src-src
    test suite should match those of the reference implementation.
    """

    benchmark = subordination_src_benchmark
    subject = distilgpt2

    region_totals = benchmark.get_region_totals(subject)

    keys = region_totals[0].keys()
    assert set(keys) == set(DISTILGPT2_SUBORDINATION_SRC_REFERENCE[0].keys())

    # Convert to ndarray for easy comparison + easy visualization of mismatches
    result_ndarray = np.concatenate([np.array([region_totals_i[key] for key in keys])
                                     for region_totals_i in region_totals])
    reference_ndarray = np.concatenate([np.array([region_totals_i[key] for key in keys])
                                        for region_totals_i in DISTILGPT2_SUBORDINATION_SRC_REFERENCE])
    pprint(sorted(zip(keys, np.abs(result_ndarray - reference_ndarray)),
                  key=lambda x: -x[1]))
    np.testing.assert_allclose(result_ndarray, reference_ndarray, atol=1e-3)


def test_syntaxgym2020_data():
    # TODO: Load SyntaxGym 2020 dataset and verify e.g. number of suites
    pass