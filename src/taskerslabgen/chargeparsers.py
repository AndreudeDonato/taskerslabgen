def parse_hirshfeld_fhi_aims(output_path):
    """
    Parse Hirshfeld charges from an FHI-aims output file.

    Parameters
    ----------
    output_path : str or Path
        Path to the FHI-aims ``.out`` file.

    Returns
    -------
    list of float
        Hirshfeld charges in atom order.
    """
    charges = []
    in_block = False
    with open(output_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "Performing Hirshfeld analysis of fragment charges and moments." in line:
                in_block = True
                continue
            if not in_block:
                continue
            if "Hirshfeld charge" in line:
                # Example: "|   Hirshfeld charge        :      0.35343123"
                try:
                    charges.append(float(line.split(":")[1].strip()))
                except (IndexError, ValueError):
                    continue
    return charges
