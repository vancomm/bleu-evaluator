import logging
import pathlib

import click

from .parse import parse
from .bleu import bleu_score
from .log import setup_base_logging, FORMATS


logger = logging.getLogger(__name__)


@click.command
@click.argument(
    "reference_file",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "candidate_files",
    metavar="[CANDIDATE_FILE]...",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    nargs=-1,
)
@click.option(
    "-v",
    "--verbose",
    help="Verbosity level. May be repeated 1-3 times to increase verbosity.",
    count=True,
)
def cli(
    reference_file: pathlib.Path, candidate_files: tuple[pathlib.Path], verbose: int
):
    """Calculate BLEU metric for each CANDIDATE_FILE using REFERENCE_FILE."""

    if verbose == 0:
        log_level = logging.ERROR
    elif verbose == 1:
        log_level = logging.WARNING
    elif verbose == 2:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    if log_level == logging.DEBUG:
        setup_base_logging(level=log_level, format=FORMATS["debug"])
    else:
        setup_base_logging(level=log_level)

    if not candidate_files:
        logger.error("No candidate files provided. Exiting.")
        exit(1)

    logger.debug(f"{reference_file = }, {candidate_files = }, {verbose = }")

    references = parse(reference_file)
    candidates = list(map(parse, candidate_files))

    for candidate in candidates:
        score = bleu_score(
            references,
            candidate,
            n=4,
            smoothing_function=lambda fr: fr.numerator + 1e-10 / fr.denominator,
        )

        print(score)
