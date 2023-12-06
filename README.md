# BLEU Evaluator

A CLI utility that calculates [the BLEU metric](https://en.wikipedia.org/wiki/BLEU) of a translated text.

## Installation

```sh
pip install git+ssh://git@github.com/vancomm/bleu-evaluator.git
```

If you are using Ubuntu, chances are you will have to use [`pipx`](https://pipx.pypa.io/stable/) instead of `pip`, otherwise the command is unchanged:

```sh
pipx install git+ssh://git@github.com/vancomm/bleu-evaluator.git
```

## Uninstallation

```sh
pip uninstall bleu-evaluator
```

or, if you used `pipx` to install:

```sh
pipx uninstall bleu-evaluator
```

## Usage

```
Usage: bleu [OPTIONS]

  Calculate BLEU metric for each candidate in CANDIDATE file using references
  found in REFERENCE files. One file may specify several references or
  candidates by delimiting reference texts with an empty line (\n\n).

  Supported FILE formats: doc, docx, txt. Other file extensions will be
  treated as UTF-8 text.

Options:
  -r, --reference FILE  A file containing REFERENCE corpus. May be specified
                        multiple times.
  -c, --candidate FILE  A file containing CANDIDATE corpora. May be specified
                        multiple times.
  -v, --verbose         Verbosity level. May be repeated 1-3 times to increase
                        verbosity.
  --help                Show this message and exit.
```
