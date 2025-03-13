from brainscore_language import metric_registry
from .metric import RDMCrossValidated

metric_registry['rdm'] = RDMCrossValidated

BIBTEX = """@article{kriegeskorte2008representational,
  title={Representational similarity analysis-connecting the branches of systems neuroscience},
  author={Kriegeskorte, Nikolaus and Mur, Marieke and Bandettini, Peter A},
  journal={Frontiers in systems neuroscience},
  pages={4},
  year={2008},
  publisher={Frontiers}
}"""