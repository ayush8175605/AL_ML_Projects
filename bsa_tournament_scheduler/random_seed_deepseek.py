import random
from AL_ML_Projects.bsa_tournament_scheduler.players_info import players_info

def generate_draw_with_missing_seeds(seeds, unseeded, rng_seed=None):
    # Validate input
    total_players = len(seeds) + len(unseeded)
    if total_players != 32:
        raise ValueError(f"Total players must be 32, got {total_players}")

    # If all 16 seeds are present, use the original logic
    if len(seeds) == 16:
        return generate_draw_with_brackets(seeds, unseeded)

    if rng_seed is not None:
        random.seed(rng_seed)
    # For cases with missing seeds
    s = len(seeds)
    brackets = [[] for _ in range(4)]  # Initialize 4 empty brackets

    # Distribute seeds into brackets
    num_groups = s // 4
    remainder = s % 4

    # Process full groups of 4
    for i in range(num_groups):
        group = seeds[i * 4: (i + 1) * 4]
        random.shuffle(group)
        for j in range(4):
            brackets[j].append(group[j])

    # Process remainder seeds
    if remainder > 0:
        group_rem = seeds[-remainder:]
        # Select random brackets for the remaining seeds
        selected_brackets = random.sample(range(4), remainder)
        for idx, bracket_idx in enumerate(selected_brackets):
            brackets[bracket_idx].append(group_rem[idx])

    # Calculate how many unseeded players we need to add to reach 4 per bracket
    unseeded_needed = 16 - s
    if unseeded_needed > len(unseeded):
        raise ValueError("Not enough unseeded players to fill the brackets")

    # Add unseeded players to brackets
    random.shuffle(unseeded)
    unseeded_idx = 0
    for bracket in brackets:
        while len(bracket) < 4 and unseeded_idx < unseeded_needed:
            bracket.append(unseeded[unseeded_idx])
            unseeded_idx += 1
    # Randomize the order of players within each bracket
    for bracket in brackets:
        random.shuffle(bracket)
    # Flatten the brackets to create Group P
    group_p = []
    for bracket in brackets:
        group_p.extend(bracket)

    # The remaining unseeded players form Group Q
    group_q = unseeded[unseeded_needed:]
    random.shuffle(group_q)

    # Create Round of 32 matches
    r32_matches = list(zip(group_p, group_q))

    # Generate the draw dictionary
    draw = {
        'R32': r32_matches,
        'R16': [],
        'QF': [],
        'SF': [],
        'F': []
    }

    # Generate subsequent rounds
    for i in range(0, 16, 2):
        match1 = f'W(R32 M{i + 1})'
        match2 = f'W(R32 M{i + 2})'
        draw['R16'].append((match1, match2))

    for i in range(0, 8, 2):
        match1 = f'W(R16 M{i + 1})'
        match2 = f'W(R16 M{i + 2})'
        draw['QF'].append((match1, match2))

    for i in range(0, 4, 2):
        match1 = f'W(QF M{i + 1})'
        match2 = f'W(QF M{i + 2})'
        draw['SF'].append((match1, match2))

    draw['F'] = [(f'W(SF M1)', f'W(SF M2)')]

    return draw

def generate_draw_with_brackets(seeds, unseeded, rng_seed=None):
    if len(seeds) != 16:
        raise ValueError("There must be exactly 16 seeded players.")
    if len(unseeded) != 16:
        raise ValueError("There must be exactly 16 unseeded players.")

    if rng_seed is not None:
        random.seed(rng_seed)

    # Slice the seeds list into groups based on ranking
    highest_seeds = seeds[0:4]  # Seeds 1-4 (S1, S2, S3, S4)
    high_seeds = seeds[4:8]  # Seeds 5-8 (S5, S6, S7, S8)
    low_seeds = seeds[8:12]  # Seeds 9-12 (S9, S10, S11, S12)
    lowest_seeds = seeds[12:16]  # Seeds 13-16 (S13, S14, S15, S16)

    # Shuffle each group to randomize assignment to brackets
    random.shuffle(highest_seeds)
    random.shuffle(low_seeds)
    random.shuffle(high_seeds)
    random.shuffle(lowest_seeds)

    # Create the ordered seed list for the brackets: each bracket has one from each group in order: Highest, Low, High, Lowest
    ordered_seeds = []
    for i in range(4):
        ordered_seeds.append(highest_seeds[i])
        ordered_seeds.append(low_seeds[i])
        ordered_seeds.append(high_seeds[i])
        ordered_seeds.append(lowest_seeds[i])

    # Shuffle the unseeded players
    random.shuffle(unseeded)

    # Create Round of 32 matches by pairing ordered seeds with unseeded players
    r32_matches = list(zip(ordered_seeds, unseeded))

    # Generate the draw dictionary
    draw = {
        'R32': r32_matches,
        'R16': [],
        'QF': [],
        'SF': [],
        'F': []
    }

    # Generate R16 matches: winners of consecutive pairs of R32 matches
    for i in range(0, 16, 2):
        match1 = f'W(R32 M{i + 1})'
        match2 = f'W(R32 M{i + 2})'
        draw['R16'].append((match1, match2))

    # Generate QF matches: winners of consecutive pairs of R16 matches
    for i in range(0, 8, 2):
        match1 = f'W(R16 M{i + 1})'
        match2 = f'W(R16 M{i + 2})'
        draw['QF'].append((match1, match2))

    # Generate SF matches: winners of consecutive pairs of QF matches
    for i in range(0, 4, 2):
        match1 = f'W(QF M{i + 1})'
        match2 = f'W(QF M{i + 2})'
        draw['SF'].append((match1, match2))

    # Generate F match: winners of the two SF matches
    draw['F'] = [(f'W(SF M1)', f'W(SF M2)')]

    return draw

def print_stylized_bracket(
    draw,
    col_pad: int = 2,
    header_pad_rows: int = 2,
    charset: str = "ascii",              # "ascii" (robust) or "unicode" (pretty)
    annotate_brackets: bool = True,      # show A/B/C/D blocks for R32
    bracket_labels = ("A", "B", "C", "D")
) -> None:
    """
    Render a stylized bracket with safe padding and optional ASCII mode,
    and annotate the 4 seeded R32 brackets (A–D), each spanning 4 matches.
    """
    rounds_order = ["R32", "R16", "QF", "SF", "F"]

    # Character sets to avoid glyph width issues
    if charset == "unicode":
        CH = dict(h="─", v="│", tee="┤", ctop="┐", cbot="┘", sep="═")
    else:  # ascii
        CH = dict(h="-", v="|", tee="+", ctop="+", cbot="+", sep="=")

    # Build labels
    r32_flat = []
    for (a, b) in draw["R32"]:
        r32_flat.extend([a, b])
    round_labels = [r32_flat]

    def flatten_round(rnd: str):
        labels = []
        for (a, b) in draw[rnd]:
            labels.extend([a, b])
        return labels

    round_labels += [
        flatten_round("R16"),  # 16
        flatten_round("QF"),   # 8
        flatten_round("SF"),   # 4
        flatten_round("F"),    # 2
    ]

    # Row placement with top padding (prevents header/first-row overlap)
    y_positions = []
    y0 = [header_pad_rows + i * 2 for i in range(32)]  # <- offset by header_pad_rows
    y_positions.append(y0)

    def mids_from(prev_positions):
        mids = []
        for i in range(0, len(prev_positions), 2):
            mids.append((prev_positions[i] + prev_positions[i + 1]) // 2)
        return mids

    y1 = mids_from(y0)
    y2 = mids_from(y1)
    y3 = mids_from(y2)
    y4 = mids_from(y3)
    y_positions += [y1, y2, y3, y4]

    # Canvas
    all_labels = [lbl for col in round_labels for lbl in col]
    max_label_len = max(len(s) for s in all_labels) if all_labels else 10
    col_width = max(12, min(28, max_label_len + 2))
    connector_span = 6
    cols = len(rounds_order) * col_width + (len(rounds_order) - 1) * connector_span
    rows = max(y0) + 1  # include padding
    grid = [[" "] * cols for _ in range(rows + 2)]

    # Helpers
    def put_text(y: int, x: int, text: str):
        for i, ch in enumerate(text[: col_width - col_pad]):
            if 0 <= y < len(grid) and 0 <= x + i < cols:
                grid[y][x + i] = ch

    def draw_hline(y: int, x1: int, x2: int, char: str):
        if x1 > x2:
            x1, x2 = x2, x1
        for x in range(x1, x2 + 1):
            if 0 <= y < len(grid) and 0 <= x < cols:
                grid[y][x] = char

    def draw_vline(x: int, y1: int, y2: int, char: str):
        if y1 > y2:
            y1, y2 = y2, y1
        for y in range(y1, y2 + 1):
            if 0 <= y < len(grid) and 0 <= x < cols:
                grid[y][x] = char

    # Place text
    col_starts = []
    for r in range(len(rounds_order)):
        x = r * (col_width + connector_span)
        col_starts.append(x)
        for idx, label in enumerate(round_labels[r]):
            put_text(y_positions[r][idx], x, label)

    # Headers on their own row(s)
    for r, name in enumerate(rounds_order):
        put_text(0, col_starts[r], f"[ {name} ]")

    # Optional: annotate the 4 R32 brackets (A–D) as horizontal separators
    if annotate_brackets:
        # Each bracket is 4 matches = 8 participants in R32
        # Start participant index for bracket g: p_start = g * 8
        for g, lab in enumerate(bracket_labels):
            p_start = g * 8
            if p_start >= len(y_positions[0]):
                break
            y_label = y_positions[0][p_start] - 1
            if y_label < 1:
                y_label = 1  # keep above the first participants, below header
            # draw a full-width separator
            draw_hline(y_label, 0, cols - 1, CH["sep"])
            # write the label at the left
            label_text = f"[ Bracket {lab} ]"
            for i, ch in enumerate(label_text):
                if i < cols:
                    grid[y_label][i] = ch

    # Connectors
    for r in range(len(rounds_order) - 1):
        x_text = col_starts[r]
        x_next_text = col_starts[r + 1]
        x_arm_end = x_text + col_width
        x_spine = x_arm_end + 1
        x_to_next = x_next_text - 1

        num_matches = len(y_positions[r]) // 2
        for m in range(num_matches):
            top_idx = 2 * m
            bot_idx = 2 * m + 1
            y_top = y_positions[r][top_idx]
            y_bot = y_positions[r][bot_idx]
            y_mid = (y_top + y_bot) // 2

            draw_hline(y_top, x_arm_end, x_spine - 1, CH["h"])
            draw_hline(y_bot, x_arm_end, x_spine - 1, CH["h"])
            draw_vline(x_spine, y_top, y_bot, CH["v"])
            grid[y_top][x_spine] = CH["ctop"]
            grid[y_bot][x_spine] = CH["cbot"]
            grid[y_mid][x_spine] = CH["tee"]
            draw_hline(y_mid, x_spine + 1, x_to_next, CH["h"])

    print("\n".join("".join(row).rstrip() for row in grid))

# Example usage:
# MD Seeded players (with seed numbers)
seeded_players_md = players_info["seeded_players_md"]
unseeded_players_md = players_info["unseeded_players_md"]
unseeded_players_extra_md = players_info["unseeded_players_extra_md"]
if len(seeded_players_md) <16:
    random.shuffle(unseeded_players_extra_md)
    n = 16 - len(seeded_players_md)
    unseeded_players_md.extend(unseeded_players_extra_md[:n])
demo_draw_md = generate_draw_with_missing_seeds(seeded_players_md, unseeded_players_md, rng_seed=145)

# WD Seeded Players
seeded_players_wd = players_info["seeded_players_wd"]
unseeded_players_wd = players_info["unseeded_players_wd"]
unseeded_players_extra_wd = players_info["unseeded_players_extra_wd"]
if len(seeded_players_wd) <16:
    random.shuffle(unseeded_players_extra_wd)
    n = 16 - len(seeded_players_wd)
    unseeded_players_md.extend(unseeded_players_extra_wd[:n])
demo_draw_wd = generate_draw_with_missing_seeds(seeded_players_wd, unseeded_players_wd, rng_seed=135)

# XD seeded Players
seeded_players_xd = players_info["seeded_players_xd"]
unseeded_players_xd = players_info["unseeded_players_xd"]
unseeded_players_extra_xd = players_info["unseeded_players_extra_xd"]
if len(seeded_players_xd) <16:
    random.shuffle(unseeded_players_extra_xd)
    n = 16 - len(seeded_players_xd)
    unseeded_players_md.extend(unseeded_players_extra_xd[:n])
demo_draw_xd = generate_draw_with_missing_seeds(seeded_players_wd, unseeded_players_wd, rng_seed = 155)

print("-------------------------Men's Doubles Schedule--------------------------------")
print_stylized_bracket(demo_draw_md, charset="unicode")
print("-------------------------Women's Doubles Schedule--------------------------------")
print_stylized_bracket(demo_draw_wd, charset="unicode")
print("-------------------------Mixed Doubles Schedule--------------------------------")
print_stylized_bracket(demo_draw_xd, charset="unicode")