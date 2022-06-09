import argcomplete, argparse
import brainscore_language as lbs
import xarray as xr
import wandb
import os


def main(args):

    print(args)
    if args.dry_run:
        os.environ["LBS_DRY_RUN"] = "TRUE"
        return

    if args.model_type is None:
        if "bert" in args.model_name_or_path:
            args.model_type = "bert"
        elif "gpt2" in args.model_name_or_path:
            args.model_type = "gpt"
        else:
            raise argparse.ArgumentError(
                f"could not infer model_type based on {args.model_name_or_path}"
            )

    # what should wandb log?
    # - workflow of testing/run (e.g., alpha)
    # - model_name_or_path
    # - dataset_name_or_path
    # - params otherwise used for caching (which are used to uniquely identify a cacheable object)
    # - resultant brainscore
    if args.use_wandb:
        lbs.utils.logging.init_wandb(
            project=f"langbrainscore_{args.project}",
            group=args.benchmark_name_or_path,
        )

    os.environ["LBS_CACHE"] = args.cache_prefix
    args.cache = not args.no_write_cache

    #### step 1: load benchmark/dataset
    # TODO: dataset caching behavior is stochastically failing; let's disable it for now
    if False and not args.recompute:
        try:
            dataset = lbs.dataset.Dataset(
                xr.DataArray(),
                dataset_name=args.benchmark_name_or_path,
                _skip_checks=True,
            )
            dataset.load_cache()
        # cache does not exist
        except FileNotFoundError:
            dataset = lbs.benchmarks.load_benchmark(args.benchmark_name_or_path)
    else:
        dataset = lbs.benchmarks.load_benchmark(
            args.benchmark_name_or_path, load_cache=False
        )

    #### step 2: set up encoders
    ann_enc = lbs.encoder.HuggingFaceEncoder(
        model_id=args.model_name_or_path,
        emb_preproc=args.emb_preproc,
        context_dimension=args.context_dimension,
        bidirectional=args.bidirectional,
        emb_aggregation=args.emb_agg,
    )
    brain_enc = lbs.encoder.BrainEncoder()

    #### step 3: encode
    ann_enc_out = ann_enc.encode(
        dataset, read_cache=not args.recompute, write_cache=args.cache
    )
    brain_enc_out = brain_enc.encode(dataset)
    lbs.utils.logging.log(f"Obtained ann-encoded data of shape: {ann_enc_out.shape}")
    lbs.utils.logging.log(
        f"Obtained brain-encoded data of shape: {brain_enc_out.shape}"
    )

    #### step 4: define mapping
    if args.mapping_class == "identity":
        mapping = lbs.mapping.IdentityMap()
    else:
        # TODO work out whether/how we want to pass additional kwargs for the
        # specific mapping class during instantiation; e.g., alpha for ridge regression
        mapping = lbs.mapping.LearnedMap(
            args.mapping_class, split_coord=args.sample_split_coord
        )

    #### step 5: define metric
    metric_class = lbs.metrics.metric_classes[args.metric_class]
    metric = metric_class()

    #### step 6: compute brainscore
    encoded_reprs_order = [ann_enc_out, brain_enc_out]
    brainscore = lbs.BrainScore(
        *(
            encoded_reprs_order
            if args.mode == "encoding"
            else reversed(encoded_reprs_order)
        ),
        mapping=mapping,
        metric=metric,
        sample_split_coord=args.sample_split_coord,
        neuroid_split_coord=args.neuroid_split_coord,
    )
    if args.compute_brainscore:
        brainscore.score()
        s = brainscore.scores.mean()
        lbs.utils.logging.log(f"brainscore = {s}", cmap="ANNOUNCE", type="INFO")
        if args.use_wandb:
            lbs.utils.logging.log_to_wandb({"brainscore": s.item()}, commit=False)
    if args.compute_ceiling:
        brainscore.ceiling()
        c = brainscore.ceilings.mean()
        lbs.utils.logging.log(f"ceiling = {c}", cmap="ANNOUNCE", type="INFO")
        if args.use_wandb:
            lbs.utils.logging.log_to_wandb({"ceiling": c.item()}, commit=False)
    if args.compute_null_permutation:
        brainscore.null(
            sample_split_coord=args.sample_split_coord,
            neuroid_split_coord=args.neuroid_split_coord,
        )
        n = brainscore.nulls.mean()
        lbs.utils.logging.log(f"null = {n}", cmap="ANNOUNCE", type="INFO")
        if args.use_wandb:
            lbs.utils.logging.log_to_wandb({"null": n.item()}, commit=False)

    if args.use_wandb:
        lbs.utils.logging.log_to_wandb(
            dict(
                metric=args.metric_class,
                mapping=args.mapping_class,
                model_id=args.model_name_or_path,
                model_type=args.model_type,
                benchmark=args.benchmark_name_or_path,
            ),
            dataset,
            ann_enc,
            brain_enc,
            ann_enc_out,  # brain_enc_out,
            mapping,
            metric,
            brainscore,
            commit=True,
        )

    lbs.utils.logging.log(f"Finished.")

    if args.cache:
        dataset.to_cache()
        brainscore.to_cache()  # also caches mapping, metric automatically


if __name__ == "__main__":

    parser = score_parser = argparse.ArgumentParser("LangBrainScore (alpha)")

    # subparsers = parser.add_subparsers(help="commands", required=True)
    # score_parser = subparsers.add_parser("score")

    ################################################################
    # basic info
    ################################################################

    score_parser.add_argument(
        "-wandb",
        "--use_wandb",
        action="store_true",
        help="Use `wandb` (weights and biases) to log the current experiment?",
    )
    score_parser.add_argument(
        "--project",
        type=str,
        default="alpha-fuzzy-potato",
        help="A short identifier of the project this run is intended for, e.g., `alpha` (testing), ...",
    )
    score_parser.add_argument(
        "--mode",
        type=str,
        default="encoding",
        choices=("encoding", "decoding"),
        help="Mode for computing the langbrainscore: **encoding** or **decoding**. An **encoding** model maps "
        "ANN model representations *onto* human behavioral or neuroimaging data. A **decoding** model maps "
        "*from* human data onto ANN representations",
    )
    score_parser.add_argument(
        "--recompute",
        action="store_true",
        help="Skip loading from cache if it exists and recompute any values",
    )
    score_parser.add_argument(
        "--no_write_cache",
        action="store_true",
        help="""If flag is enabled, representations and results will not 
                be cached for future use. By default, everything is cached.""",
    )
    score_parser.add_argument(
        "--cache_prefix",
        default="~/.cache",
        help="""Default directory to use for caching. (A langbrainscore subdirectory 
                is created within the prefix directory if it doesn't already exist)""",
    )
    score_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run through the pipeline without performing too much computation",
    )

    score_parser_benchmark = score_parser.add_argument_group("Dataset options")
    score_parser_encoder = score_parser.add_argument_group("ANN Language Model options")
    score_parser_mapping = score_parser.add_argument_group("Mapping options")
    score_parser_metric = score_parser.add_argument_group(
        "Metric (Similarity evaluation) options"
    )
    score_parser_brainscore = score_parser.add_argument_group(
        "Language Brain Score computation options"
    )

    #### benchmark arguments
    score_parser_benchmark.add_argument(
        "-b",
        "--benchmark_name_or_path",
        type=str,
        required=True,
        # default="pereira2018_mean_froi_nat_stories",
        help=f"""Identifier of a pre-packaged benchmark in `langbrainscore`, i.e.  (one of:
                 {','.join(lbs.benchmarks.supported_benchmarks.keys())}), or a path to a folder containing a
                 benchmark and a loading script (`NotImplemented`), or a dataset ID from HuggingFace Datasets
                 repository (must be LanguageBrainScore-formatted; `NotImplemented`).
                 Default: F. Pereira et al. (2018) Natural Stories dataset, averaged over 
                 Language Network (Fedorenko et al., 2010) functional Regions of Interest (ROIs)""",
    )

    #### encoder arguments
    score_parser_encoder.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        # default="bert-base-uncased",
        required=True,
        help="HuggingFace model name or path to a pretrained model that can be loaded using `AutoModel.from_pretrained`",
    )
    score_parser_encoder.add_argument(
        "-mt",
        "--model_type",
        type=str,
        choices=("bert", "gpt"),
        required=False,
        help="""Model type (must be one of gpt and bert; support for more types will be added based
                on requirement).  If not provided, will be inferred automatically if possible, else, an
                error will be raised.""",
    )
    score_parser_encoder.add_argument(
        "--bidirectional",
        action="store_true",
        help="Whether to encode using bidirectional context (`model_type` must be `bert` and `context_group` must be used)",
    )
    score_parser_encoder.add_argument(
        "--context_dimension",
        type=str,
        default=None,
        help="""The dimension to use for constructing the context of a stimulus to use for encoding using an ANN LM.
                if None, each sampleid (stimuli) will be treated as a single context group.
                if a string is specified, the string must refer to the name of a dimension in the xarray-like dataset
                object (langbrainscore.dataset.Dataset) that provides groupings of sampleids (stimuli) that should be
                used as context when generating encoder representations [default: None]""",
    )
    score_parser_encoder.add_argument(
        "-emb_preproc",
        # "--preprocess_representations",
        dest="emb_preproc",
        default=(),
        help="Preprocessing steps to apply to encoder representations obtained from the ANN LM before using them for BrainScore.",
        choices=lbs.utils.preprocessing.preprocessor_classes.keys(),
        nargs="*",
    )
    score_parser_encoder.add_argument(
        "-emb_agg",
        # "--aggregate_representations",
        dest="emb_agg",
        help="How should encoder representations obtained from the ANN LM be aggregated (`all` = no aggregation).",
        choices=("all", "last", "first", "mean", "median", "sum"),
        default="mean",
    )

    #### mapping arguments
    score_parser_mapping.add_argument(
        "-mapping",
        "--mapping_class",
        help=f"""What mapping class should be used to construct a mapping between ANN encoder
                 representations and Brain encoder representations. Must be either `identity`
                 (no transformation; used in conjunction with RSA, CKA, etc.) or the name of a
                 learned mapping class, i.e., one of:
                 {lbs.mapping.mapping.mapping_classes_params.keys()}""",
        choices=("identity",)
        + tuple(lbs.mapping.mapping.mapping_classes_params.keys()),
        type=str,
        required=True,
    )

    #### metric arguments
    score_parser_metric.add_argument(
        "-metric",
        "--metric_class",
        help="""What mapping class should be used to construct a mapping between ANN encoder
                representations and Brain encoder representations""",
        choices=lbs.metrics.metric_classes.keys(),
        type=str,
        required=True,
    )

    #### brainscore arguments
    score_parser_brainscore.add_argument(
        "--sample_split_coord",
        help="""Coordinate (in dataset/benchmark) along the sample/stimulus dimension to use for
                grouping data during cross-validation to prevent bleeding out of information and
                inflation of BrainScore. E.g. passage,paragraph, etc.""",
        type=str,
        required=False,
    )
    score_parser_brainscore.add_argument(
        "--neuroid_split_coord",
        help="""Coordinate (in dataset/benchmark) along the neuroid (human behavioral/neural)
                dimension to use for grouping data during cross-validation to prevent bleeding out
                of information and inflation of BrainScore. E.g. subjectid""",
        type=str,
        required=False,
    )
    score_parser_brainscore.add_argument(
        "-brainscore",
        "--compute_brainscore",
        action="store_true",
        help="Should we compute the brainscore?",
    )
    score_parser_brainscore.add_argument(
        "-ceiling",
        "--compute_ceiling",
        action="store_true",
        help="Should we compute the ceiling?",
    )
    score_parser_brainscore.add_argument(
        "-null",
        "--compute_null_permutation",
        action="store_true",
        help="Should we compute the null permutation score with 100 iterations?",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    main(args)
