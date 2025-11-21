import polars as pl


@pl.api.register_expr_namespace("bio_ext")
class BioExtensionNamespace:
    """
    Polars expression namespace for biological sequence conversions and analysis.
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    # -------------------------------------------------------------------------
    # Nucleic Acid Conversions
    # -------------------------------------------------------------------------

    def dna_to_rna(self) -> pl.Expr:
        """Convert DNA sequences (A, T, G, C) to RNA sequences (A, U, G, C)."""
        return self._expr.str.replace_all("T", "U")

    def rna_to_dna(self) -> pl.Expr:
        """Convert RNA sequences (A, U, G, C) to DNA sequences (A, T, G, C)."""
        return self._expr.str.replace_all("U", "T")

    def dna_complement(self) -> pl.Expr:
        """Return the DNA complement (A↔T, C↔G)."""
        return (
            self._expr
            .str.replace_all("A", "t")
            .str.replace_all("T", "a")
            .str.replace_all("G", "c")
            .str.replace_all("C", "g")
            .str.to_uppercase()
        )

    def dna_reverse_complement(self) -> pl.Expr:
        """Return the reverse complement of DNA sequence."""
        return self._expr.bio_ext.dna_complement().str.reverse()

    def dna_transcribe(self) -> pl.Expr:
        """Transcribe DNA to RNA (same as dna_to_rna)."""
        return self._expr.bio_ext.dna_to_rna()

    # -------------------------------------------------------------------------
    # Sequence Metrics
    # -------------------------------------------------------------------------

    def sequence_length(self) -> pl.Expr:
        """Return the sequence length."""
        return self._expr.str.len_chars()

    def gc_content(self) -> pl.Expr:
        """Return GC content as a percentage."""
        g = self._expr.str.count_matches("G")
        c = self._expr.str.count_matches("C")
        total = self._expr.str.len_chars()
        return ((g + c) / total * 100).round(2)

    def at_content(self) -> pl.Expr:
        """Return AT content as a percentage."""
        a = self._expr.str.count_matches("A")
        t = self._expr.str.count_matches("T")
        total = self._expr.str.len_chars()
        return ((a + t) / total * 100).round(2)

    def gc_skew(self) -> pl.Expr:
        """Compute GC skew = (G - C) / (G + C)."""
        g = self._expr.str.count_matches("G")
        c = self._expr.str.count_matches("C")
        return ((g - c) / (g + c)).fill_nan(0).round(3)

    def count_nucleotides(self) -> pl.Expr:
        """Return a struct of counts for A, T, G, and C."""
        return pl.struct(
            [
                self._expr.str.count_matches("A").alias("A"),
                self._expr.str.count_matches("T").alias("T"),
                self._expr.str.count_matches("G").alias("G"),
                self._expr.str.count_matches("C").alias("C"),
            ]
        )

    def count_codons(self) -> pl.Expr:
        """Return the number of codons (sequence length / 3)."""
        return (self._expr.str.len_chars() / 3).floor()

    # -------------------------------------------------------------------------
    # Validation & Motifs
    # -------------------------------------------------------------------------

    def is_valid_dna(self) -> pl.Expr:
        """Return True if the sequence only contains valid DNA bases (A, T, G, C, N)."""
        return self._expr.str.contains(r"^[ATGCN]+$")

    def is_valid_rna(self) -> pl.Expr:
        """Return True if the sequence only contains valid RNA bases (A, U, G, C, N)."""
        return self._expr.str.contains(r"^[AUGCN]+$")

    def contains_motif(self, motif: str) -> pl.Expr:
        """Return True if sequence contains the given motif."""
        return self._expr.str.contains(motif)

    def count_motif(self, motif: str) -> pl.Expr:
        """Count occurrences of a motif in the sequence."""
        return self._expr.str.count_matches(motif)

    # -------------------------------------------------------------------------
    # Distance & Similarity
    # -------------------------------------------------------------------------

    def hamming_distance(self, other: pl.Expr) -> pl.Expr:
        """
        Return Hamming distance between two equal-length sequences.

        Parameters
        ----------
        other : pl.Expr
            Another expression containing sequences of equal length.
        """
        return (
            self._expr
            .str.zip_with(other, separator="")
            .str.split("")
            .arr.eval(
                pl.element().alias("seq").arr.first().ne(pl.element().alias("seq").arr.last())
            )
            .arr.sum()
        )

    # -------------------------------------------------------------------------
    # Sequence Manipulation
    # -------------------------------------------------------------------------

    def reverse_sequence(self) -> pl.Expr:
        """Reverse a sequence string."""
        return self._expr.str.reverse()

    def repeat_sequence(self, n: int) -> pl.Expr:
        """Repeat the sequence n times."""
        return self._expr.repeat_by(n).str.concat()

    def mutate_sequence(self, position: int, new_base: str) -> pl.Expr:
        """Mutate a sequence by replacing one base at a given position (0-indexed)."""
        return (
            self._expr
            .str.slice(0, position)
            + pl.lit(new_base)
            + self._expr.str.slice(position + 1)
        )

    def insert_sequence(self, position: int, subseq: str) -> pl.Expr:
        """Insert a subsequence at the given position."""
        return (
            self._expr.str.slice(0, position)
            + pl.lit(subseq)
            + self._expr.str.slice(position)
        )

    def delete_sequence(self, start: int, end: int) -> pl.Expr:
        """Delete a segment of the sequence from start to end."""
        return (
            self._expr.str.slice(0, start)
            + self._expr.str.slice(end)
        )
