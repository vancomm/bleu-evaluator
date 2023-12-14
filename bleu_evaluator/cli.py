import time
import logging
import pathlib

import click

from .parse import get_parser
from .bleu import BLEU
from .log import setup_base_logging, LOG_FORMATS


logger = logging.getLogger(__name__)


def calculate_log_level(verbosity: int) -> int:
    if verbosity == 0:
        return logging.ERROR
    elif verbosity == 1:
        return logging.WARNING
    elif verbosity == 2:
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
    "-h",
    "--hypothesis",
    "hypothesis_files",
    help="A file containing HYPOTHESIS corpora. May be specified multiple times.",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    multiple=True,
)
@click.option(
    "-i",
    "--interactive",
    help="Force interactive prompting of references and hypotheses.",
    is_flag=True,
    default=False,
)
@click.option(
    "-v",
    "--verbose",
    "verbosity",
    help="Verbosity level. May be repeated 1-3 times to increase verbosity.",
    count=True,
)
def cli(
    reference_files: tuple[pathlib.Path],
    hypothesis_files: tuple[pathlib.Path],
    verbosity: int,
    interactive: bool,
) -> None:
    """
    Calculate BLEU score for each hypothesis in HYPOTHESIS file(s) using references
    found in REFERENCE file(s). One file may specify several references or
    hypotheses by delimiting items with empty lines.

    Supported file formats: txt, doc, docx, pdf. Files with other extensions
    will be treated as UTF-8 text.

    You can also supply references and hypotheses in interactive mode by invoking the
    script without -r or -h options. You will be prompted for missing data
    interactively. Alternatively, supply -i option to force interactive prompt
    regardless of other options.
    """
    log_level = calculate_log_level(verbosity)
    log_format = "debug" if log_level == logging.DEBUG else "default"
    setup_base_logging(level=log_level, format=LOG_FORMATS[log_format])

    start = time.perf_counter()
    logger.info("CLI invoked")

    references = [get_parser(file).read_all() for file in reference_files]
    hypotheses = [get_parser(file).read_all() for file in hypothesis_files]

    if not references or interactive:
        ask_again = True
        while ask_again:
            value = click.prompt(
                "Please enter a reference translation (newlines not allowed)",
                prompt_suffix=":\n> ",
            )
            references.extend(value)
            ask_again = click.confirm("Do you want to add another reference?")

    if not hypotheses or interactive:
        ask_again = True
        while ask_again:
            value = click.prompt(
                "Please enter a hypothesis (newlines not allowed)",
                prompt_suffix=":\n> ",
            )
            hypotheses.append(value)
            ask_again = click.confirm("Do you want to add another hypothesis?")

    logging.info("BLEU calculation initiated")

    bleu = BLEU(references)

    for hypothesis in hypotheses:
        score = bleu.corpus_score(hypothesis)
        click.echo(score.format(verbose=True))

    logging.info("BLEU calculation complete")

    end = time.perf_counter()
    logger.info(f"CLI workflow complete, time elapsed = {end - start:.3f}ms")
