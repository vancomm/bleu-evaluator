import time
import logging
from pathlib import Path

import click

from .read import get_reader
from .scan import scan_directory
from .bleu import BLEU
from .log import LOG_FORMATS, setup_stream_logging, setup_file_logging


logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-r",
    "--reference",
    "reference_files",
    help="A file containing REFERENCE corpus. May be specified multiple times.",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
)
@click.option(
    "-h",
    "--hypothesis",
    "hypothesis_files",
    help="A file containing HYPOTHESIS corpora. May be specified multiple times.",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
)
@click.option(
    "-d",
    "--dir",
    "directories",
    help="A directory containing REFERENCE and/or HYPOTHESIS corpora. May be specified multiple times.",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
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
    type=click.Path(exists=False, path_type=Path),
)
@click.option(
    "-v",
    "--verbose",
    help="Increase logging verbosity.",
    is_flag=True,
    default=False,
)
@click.version_option(message="%(version)s")
def cli(
    reference_files: tuple[Path, ...],
    hypothesis_files: tuple[Path, ...],
    directories: tuple[Path, ...],
    verbose: bool,
    log_file: Path | None,
    interactive: bool,
) -> None:
    """
    Calculate BLEU score for each hypothesis in HYPOTHESIS file(s) using
    references found in REFERENCE file(s). One file may specify several
    references or hypotheses by delimiting items with empty lines.

    Supported file formats: txt, doc, docx, pdf. Files with other extensions
    will be treated as UTF-8 text.

    You can also supply references and hypotheses in interactive mode by
    invoking the script without -r or -h options. You will be prompted for
    missing data interactively. Alternatively, supply -i option to force
    interactive prompt regardless of other options.

    Files with references and hypotheses can be detected automatically by
    specifying a DIRECTORY (or several) via -d option. Detection rules:

      - all files in DIRECTORY that start with "ref_" are considered to be
        REFERENCE files;

      - all files in DIRECTORY that start with "hyp_" are considered to be
        HYPOTHESIS files.
    """
    start = time.perf_counter()

    # setup_stream_logging(level=log_level, format=log_format)
    if log_file:
        log_level = logging.DEBUG if verbose else logging.INFO
        log_format_key = "debug" if verbose else "default"
        log_format = LOG_FORMATS[log_format_key]
        setup_file_logging(logfile=log_file, level=log_level, format=log_format)

    logger.info("CLI invoked")

    if directories:
        if len(directories) == 1:
            _reference_files, _hypothesis_files = scan_directory(directories[0])
        else:
            _files: zip[tuple[Path, ...]] = zip(*map(scan_directory, set(directories)))
            _reference_files, _hypothesis_files = _files

        if _reference_files:
            click.echo(
                "Detected reference files:\n%s\n%s"
                % (
                    "\n".join(f"- {file}" for file in _reference_files),
                    "These will be appended to the list of references.",
                )
            )
        if _hypothesis_files:
            click.echo(
                "Detected hypothesis files:\n%s\n%s"
                % (
                    "\n".join(f"- {file}" for file in _hypothesis_files),
                    "These will be appended to the list of hypotheses.",
                )
            )

        reference_files = (*reference_files, *_reference_files)
        hypothesis_files = (*hypothesis_files, *_hypothesis_files)

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

    calc_start = time.perf_counter()
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
    logger.info(
        f"Program workflow complete; "
        f"time reading input = {calc_start - start:.3f}ms, "
        f"time of calculation = {end - calc_start:.3f}ms, "
        f"total time elapsed = {end - start:.3f}ms"
    )
