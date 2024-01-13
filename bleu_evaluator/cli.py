import time
import logging
import pathlib

import click

from .read import get_reader
from .bleu import BLEU
from .log import LOG_FORMATS, setup_file_logging


logger = logging.getLogger(__name__)


@click.command()
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
    "-l",
    "--log-file",
    help="A file to write logs to.",
    type=click.Path(exists=False, path_type=pathlib.Path),
)
@click.option(
    "-v",
    "--verbose",
    "verbose",
    help="Increase logging verbosity.",
)
@click.version_option(message="%(version)s")
def cli(
    reference_files: tuple[pathlib.Path],
    hypothesis_files: tuple[pathlib.Path],
    verbose: bool,
    log_file: pathlib.Path | None,
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

    if log_file:
        log_level = logging.DEBUG if verbose else logging.INFO
        log_format = "debug" if verbose else "default"
        setup_file_logging(
            logfile=log_file, level=log_level, format=LOG_FORMATS[log_format]
        )

    start = time.perf_counter()
    logger.info("CLI invoked")

    references = [get_reader(file).read_all() for file in reference_files]
    hypotheses = [get_reader(file).read_all() for file in hypothesis_files]

    if not references or interactive:
        ask_again = True
        while ask_again:
            value = click.prompt(
                "Please enter a reference translation (newlines not allowed)",
                prompt_suffix=":\n> ",
            )
            references.append(value)
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

    logger.debug(f"{references = }")
    logger.debug(f"{hypotheses = }")

    logging.info("BLEU calculation initiated")

    bleu = BLEU(references)

    for hypothesis in hypotheses:
        try:
            score = bleu.corpus_score(hypothesis)
            click.echo(score.format(verbose=True))
        except click.ClickException as e:
            logger.error(e)
            raise e
        except Exception as e:
            logger.error(e)
            raise click.ClickException(str(e))

    logging.info("BLEU calculation complete")

    end = time.perf_counter()
    logger.info(f"CLI workflow complete, time elapsed = {end - start:.3f}ms")
