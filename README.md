# BLEU Evaluator

A CLI utility that calculates [the BLEU score](https://aclanthology.org/P02-1040.pdf) of a translated text.

## Installation

Requires Python 3.10+.

You can clone this Git repository and run `pip install .` or install it from GitHub directly:

```sh
pip install git+https://github.com/vancomm/bleu-evaluator.git
```

If you are using Ubuntu 23.04 or newer, chances are you will have to use [`pipx`](https://pipx.pypa.io/stable/) instead of `pip`, otherwise the command is the same:

```sh
pipx install git+https://github.com/vancomm/bleu-evaluator.git
```

**Note:** this script requires a subset (~50MB) of [NLTK data](https://www.nltk.org/data.html) to be present on your system. If automatic data discovery performed by NLTK package fails, script downloads required packages to the default directory (current user's home directory). To learn how to override this behaviour please consult NLTK documentation.

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

Options:
  -r, --reference FILE   A file containing REFERENCE corpus. May be specified
                         multiple times.
  -h, --hypothesis FILE  A file containing HYPOTHESIS corpora. May be
                         specified multiple times.
  -d, --dir DIRECTORY    A directory containing REFERENCE and/or HYPOTHESIS
                         corpora. May be specified multiple times.
  -i, --interactive      Force interactive prompting of references and
                         hypotheses.
  -l, --log-file PATH    A file to write logs to.
  -v, --verbose          Enable logging to console.
  -D, --debug            Include debugging information in logs.
  --version              Show the version and exit.
  --help                 Show this message and exit.
```
