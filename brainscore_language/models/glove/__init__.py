class Glove(KeyedVectorModel):
    """
    Pennington et al., 2014
    http://www.aclweb.org/anthology/D14-1162
    """

    identifier = 'glove'

    def __init__(self, weights='glove.840B.300d.txt', random_embeddings=False, **kwargs):
        from gensim.scripts.glove2word2vec import glove2word2vec
        weights_file = os.path.join(_ressources_dir, 'glove', weights)
        word2vec_weightsfile = weights_file + '.word2vec'
        if not os.path.isfile(word2vec_weightsfile):
            glove2word2vec(weights_file, word2vec_weightsfile)
        super(Glove, self).__init__(
            identifier=self.identifier + ('-untrained' if random_embeddings else ''), weights_file=word2vec_weightsfile,
            # std from https://gist.github.com/MatthieuBizien/de26a7a2663f00ca16d8d2558815e9a6#file-fast_glove-py-L16
            random_std=.01, random_embeddings=random_embeddings, **kwargs)
