from dataclasses import dataclass


@dataclass
class BLEUScore:
    score: float
    precisions: list[float]
    correct: list[int]
    total: list[int]
    bp: float
    hyp_len: int
    ref_len: int

    def __post_init__(self) -> None:
        precisions_str = "/".join(f"{p:.1f}" for p in self.precisions)
        ratio = self.ref_len / self.hyp_len
        self.verbose = (
            f"{precisions_str} (BP = {self.bp:.3f}, "
            f"ratio = {ratio:.3f}, hyp_len = {self.hyp_len}, ref_len = {self.ref_len})"
        )

    def format(self, *, width: int = 2, verbose: bool = False) -> str:
        text = f"BLEU = {self.score:.{width}f}"
        if verbose:
            text = "%s %s" % (text, self.verbose)
        return text
