import logging
import pathlib

import click
from click.exceptions import UsageError

from .parse import parse_corpus, parse_corpora
from .bleu import bleu_score
from .log import setup_base_logging, FORMATS


logger = logging.getLogger(__name__)


def calculate_verbosity(verbose: int) -> int:
    if verbose == 0:
        return logging.ERROR
    elif verbose == 1:
        return logging.WARNING
    elif verbose == 2:
        return logging.INFO
    else:
        return logging.DEBUG


@click.command
@click.option(
    "-r",
    "--reference",
    "reference_files",
    help="A file containing REFERENCE corpus. May be specified multiple times.",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    multiple=True,
)
@click.option(
    "-c",
    "--candidate",
    "candidate_files",
    help="A file containing CANDIDATE corpora. May be specified multiple times.",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    multiple=True,
)
@click.option(
    "-v",
    "--verbose",
    help="Verbosity level. May be repeated 1-3 times to increase verbosity.",
    count=True,
)
def cli(
    reference_files: tuple[pathlib.Path],
    candidate_files: tuple[pathlib.Path],
    verbose: int,
) -> None:
    """
    Calculate BLEU metric for each candidate in CANDIDATE file using references
    found in REFERENCE files. One file may specify several references or
    candidates by delimiting reference texts with an empty line (\\n\\n).

    Supported FILE formats: doc, docx, txt. Other file extensions will be
    treated as UTF-8 text.
    """

    log_level = calculate_verbosity(verbose)

    if log_level == logging.DEBUG:
        setup_base_logging(level=log_level, format=FORMATS["debug"])
    else:
        setup_base_logging(level=log_level)

    if not reference_files:
        raise UsageError("No reference files provided. Exiting.")

    if not candidate_files:
        raise UsageError("No reference files provided. Exiting.")

    logger.debug(f"{reference_files = }, {candidate_files = }, {verbose = }")

    # data/
    #   refs1.txt
    #     - ref_1
    #     - ref_2
    #   refs2.txt
    #     - ref_3
    #   cands_1.txt
    #     - cand1
    #   cands_2.txt
    #     - cand2
    #     - cand3
    #
    # bleu -r refs_1.txt -r refs_2.txt -c cands_1.txt -c cands_2.txt
    #   => [[ref_1, ref_2], ref_3], [[cand1], [cand2, cand3]]
    #   => [ref_1, ref_2, ref_3], [cand1, cand2, cand3]

    references = [s for l in map(parse_corpus, reference_files) for s in l]
    candidates = [s for l in map(parse_corpora, candidate_files) for s in l]

    for candidate in candidates:
        score = bleu_score(
            references,
            candidate,
            n=4,
            smoothing_function=lambda fr: fr.numerator + 0.1 / fr.denominator,
        )

        print(score)
