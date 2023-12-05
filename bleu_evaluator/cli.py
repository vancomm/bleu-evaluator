import logging
import pathlib

import click

from .read import reader_factory


logger = logging.getLogger(__name__)


@click.command
@click.argument("reference", required=False)
@click.argument("candidates", metavar="[CANDIDATE]...", required=False, nargs=-1)
@click.option(
    "-r",
    "--reference-file",
    help="File containing the reference string. Overrides REFERENCE argument",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.option(
    "-c",
    "--candidates-file",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="File containing the candidate string. Overrides CANDIDATE arguments",
)
def cli(
    reference: str | None,
    candidates: tuple[str],
    reference_file: pathlib.Path | None,
    candidates_file: pathlib.Path | None,
):
    """Calculate BLEU metric using provided REFERENCE and CANDIDATE strings."""
    logger.debug(
        f"{reference = }, {reference_file = }, {candidates = }, {candidates_file = }"
    )

    if not reference:
        if not reference_file:
            print("Error: reference required!")
            exit(1)

        reader = reader_factory(reference_file)
        reference = reader(reference_file)

        if not reference:
            print(f"Error: reference string not found in {reference_file}!")
            exit(1)

    if not candidates:
        if not candidates_file:
            print("Error: at least one candidate required!")
            exit(1)

        reader = reader_factory(candidates_file)
        candidates_text = reader(candidates_file)

    print("Hello world!")
