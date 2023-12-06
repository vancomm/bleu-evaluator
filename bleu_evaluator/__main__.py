def main():
    from collections.abc import Sequence

    from .parse import tokenize
    from .bleu import bleu_score

    example_1 = (
        ("the the the the the the the",),
        ("The cat is on the mat", "There is a cat on the mat"),
    )

    example_2 = (
        (
            "It is a guide to action which ensures that the military always obeys the commands of the party.",
            "It is to insure the troops forever hearing the activity guidebook that party direct.",
        ),
        (
            "It is a guide to action that ensures that the military will forever heed Party commands.",
            "It is the guiding principle which guarantees the military forces always being under the command of the Party.",
            "It is the practical guide for the army always to heed the directions of the party.",
        ),
    )

    examples: Sequence[tuple[Sequence[str], Sequence[str]]] = (example_1, example_2)

    for candidates, references in examples:
        references = list(map(tokenize, references))
        for candidate in map(tokenize, candidates):
            bleu = bleu_score(
                references,
                [candidate],
                n=4,
                smoothing_function=lambda fr: fr.numerator + 0.0000001 / fr.denominator,
            )
            print(f"{bleu = }")


if __name__ == "__main__":
    import logging
    from .log import setup_base_logging, FORMATS

    setup_base_logging(level=logging.DEBUG, format=FORMATS["debug"])
    main()
