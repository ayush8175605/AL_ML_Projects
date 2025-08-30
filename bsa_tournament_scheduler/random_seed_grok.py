import random
import os
import sys

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from AL_ML_Projects.bsa_tournament_scheduler.players_info import players_info
import streamlit as st


def generate_tournament_schedule(seeds, unseeded, random_seed=None):
    """
    Generates a randomized tournament draw for a 32-player single-elimination tournament.
    - Seeds are grouped into Highest (1-4), High (5-8), Low (9-12), Lowest (13-16) and shuffled within groups.
    - The draw is divided into 4 brackets, each with 4 matches.
    - Each bracket gets one seed from each group, placed in order: Highest, Low, High, Lowest.
    - Unseeded players are shuffled and paired against the ordered seeds.
    - Returns a dictionary with keys 'R32', 'R16', 'QF', 'SF', 'F' and values as lists of (top_label, bottom_label) tuples.
    - For R32: (seed, unseeded); for later rounds: ('Winner M{id1}', 'Winner M{id2}').
    """
    if len(seeds) != 16 or len(unseeded) != 16:
        raise ValueError("Both seeds and unseeded lists must have exactly 16 entries.")

    # Group seeds assuming they are ordered S1 (top) to S16
    highest = seeds[0:4]   # S1-S4
    high = seeds[4:8]      # S5-S8
    low = seeds[8:12]      # S9-S12
    lowest = seeds[12:16]  # S13-S16
    if random_seed is not None:
        random.seed(random_seed)
    # Shuffle within each group to randomize assignment to brackets
    random.shuffle(highest)
    random.shuffle(high)
    random.shuffle(low)
    random.shuffle(lowest)

    # Build ordered seeds for the draw: 4 brackets, each with Highest, Low, High, Lowest
    ordered_seeds = []
    for i in range(4):
        ordered_seeds.extend([highest[i], low[i], high[i], lowest[i]])

    # Shuffle unseeded for random pairings
    random.shuffle(unseeded)

    # Create Round of 32 pairs: ordered seeds vs shuffled unseeded (seed as top, unseeded as bottom)
    r32_pairs = [(seed, unseeded_player) for seed, unseeded_player in zip(ordered_seeds, unseeded)]

    # Assign match IDs starting from 1
    match_id = 1
    r32_matches = []
    r32_match_ids = []
    for top, bottom in r32_pairs:
        r32_matches.append((top, bottom))
        r32_match_ids.append(match_id)
        match_id += 1

    schedule = {'R32': r32_matches}

    # Round of 16: Pair winners of adjacent R32 matches
    r16_matches = []
    r16_match_ids = []
    for i in range(0, 16, 2):
        w1 = r32_match_ids[i]
        w2 = r32_match_ids[i + 1]
        r16_matches.append((f"Winner M{w1}", f"Winner M{w2}"))
        r16_match_ids.append(match_id)
        match_id += 1
    schedule['R16'] = r16_matches

    # Quarterfinals: Pair winners of adjacent R16 matches
    qf_matches = []
    qf_match_ids = []
    for i in range(0, 8, 2):
        w1 = r16_match_ids[i]
        w2 = r16_match_ids[i + 1]
        qf_matches.append((f"Winner M{w1}", f"Winner M{w2}"))
        qf_match_ids.append(match_id)
        match_id += 1
    schedule['QF'] = qf_matches

    # Semifinals: Pair winners of adjacent QF matches
    sf_matches = []
    sf_match_ids = []
    for i in range(0, 4, 2):
        w1 = qf_match_ids[i]
        w2 = qf_match_ids[i + 1]
        sf_matches.append((f"Winner M{w1}", f"Winner M{w2}"))
        sf_match_ids.append(match_id)
        match_id += 1
    schedule['SF'] = sf_matches

    # Final: Pair winners of SF matches
    f_matches = []
    w1 = sf_match_ids[0]
    w2 = sf_match_ids[1]
    f_matches.append((f"Winner M{w1}", f"Winner M{w2}"))
    schedule['F'] = f_matches

    return schedule

def generate_tournament_schedule_new(seeds, unseeded, random_seed=None):
    """
        Generates a randomized tournament draw for a 32-player single-elimination tournament.
        - If len(seeds) == 16, uses the original logic with Highest-Low-High-Lowest order per bracket.
        - If len(seeds) < 16, distributes participating seeds (sorted ascending by rank) into tiers and brackets,
          then fills remaining positions with promoted unseeded players.
        - Unseeded length must be 32 - len(seeds).
        - Returns a dictionary with keys 'R32', 'R16', 'QF', 'SF', 'F' and values as lists of (top_label, bottom_label) tuples.
        - Optional random_seed for reproducibility.
        """
    if random_seed is not None:
        random.seed(random_seed)

    n = len(seeds)
    if n + len(unseeded) != 32:
        raise ValueError("Total of seeds and unseeded must be 32.")
    if n > 16:
        raise ValueError("Seeds cannot exceed 16.")

    # Define 4 brackets, each with 4 positions (lists to hold entrants)
    brackets = [[] for _ in range(4)]

    if n == 16:
        # Original logic: Group into tiers of 4, shuffle within, assign in Highest-Low-High-Lowest order per bracket
        highest = seeds[0:4]
        high = seeds[4:8]
        low = seeds[8:12]
        lowest = seeds[12:16]
        random.shuffle(highest)
        random.shuffle(high)
        random.shuffle(low)
        random.shuffle(lowest)
        for i in range(4):
            brackets[i].extend([highest[i], low[i], high[i], lowest[i]])
    else:
        # New logic for n < 16
        # Step 1: Group seeds into tiers
        tiers = []
        full_tiers = n // 4
        remainder = n % 4
        start = 0
        for _ in range(full_tiers):
            tier = seeds[start:start + 4]
            random.shuffle(tier)  # Shuffle within tier
            tiers.append(tier)
            start += 4
        if remainder > 0:
            rem_tier = seeds[start:start + remainder]
            random.shuffle(rem_tier)
            tiers.append(rem_tier)

        # Step 2: Distribute to brackets
        for tier in tiers:
            if len(tier) == 4:
                # Full tier: Assign one to each bracket
                for bracket_idx, seed in enumerate(tier):
                    # Choose a random position in the bracket (which has empty slots)
                    pos = random.randint(0, len(brackets[bracket_idx]))
                    brackets[bracket_idx].insert(pos, seed)
            else:
                # Remainder: Select len(tier) unique random brackets
                selected_brackets = random.sample(range(4), len(tier))
                for i, seed in enumerate(tier):
                    bracket_idx = selected_brackets[i]
                    pos = random.randint(0, len(brackets[bracket_idx]))
                    brackets[bracket_idx].insert(pos, seed)

        # Step 3: Fill remaining positions with promoted unseeded
        promoted = random.sample(unseeded, 16 - n)
        unseeded = [u for u in unseeded if u not in promoted]  # Remove promoted from unseeded

        # Assign promoted to empty positions across brackets
        promoted_idx = 0
        for bracket in brackets:
            while len(bracket) < 4:
                # Insert at random position to mix
                pos = random.randint(0, len(bracket))
                bracket.insert(pos, promoted[promoted_idx])
                promoted_idx += 1
            random.shuffle(bracket)
    # Now, flatten the brackets into ordered "seeds" (tops for R32)
    ordered_tops = []
    for bracket in brackets:
        # Optionally shuffle within bracket for random order (but per logic, can keep as is; adding light shuffle)
        # random.shuffle(bracket)
        ordered_tops.extend(bracket)

    # Shuffle remaining unseeded for bottoms
    random.shuffle(unseeded)

    # Create Round of 32 pairs: ordered tops vs shuffled unseeded
    r32_pairs = [(top, bottom) for top, bottom in zip(ordered_tops, unseeded)]

    # Assign match IDs starting from 1
    match_id = 1
    r32_matches = []
    r32_match_ids = []
    for top, bottom in r32_pairs:
        r32_matches.append((top, bottom))
        r32_match_ids.append(match_id)
        match_id += 1

    schedule = {'R32': r32_matches}

    # Round of 16: Pair winners of adjacent R32 matches
    r16_matches = []
    r16_match_ids = []
    for i in range(0, 16, 2):
        w1 = r32_match_ids[i]
        w2 = r32_match_ids[i + 1]
        r16_matches.append((f"Winner M{w1}", f"Winner M{w2}"))
        r16_match_ids.append(match_id)
        match_id += 1
    schedule['R16'] = r16_matches

    # Quarterfinals: Pair winners of adjacent R16 matches
    qf_matches = []
    qf_match_ids = []
    for i in range(0, 8, 2):
        w1 = r16_match_ids[i]
        w2 = r16_match_ids[i + 1]
        qf_matches.append((f"Winner M{w1}", f"Winner M{w2}"))
        qf_match_ids.append(match_id)
        match_id += 1
    schedule['QF'] = qf_matches

    # Semifinals: Pair winners of adjacent QF matches
    sf_matches = []
    sf_match_ids = []
    for i in range(0, 4, 2):
        w1 = qf_match_ids[i]
        w2 = qf_match_ids[i + 1]
        sf_matches.append((f"Winner M{w1}", f"Winner M{w2}"))
        sf_match_ids.append(match_id)
        match_id += 1
    schedule['SF'] = sf_matches

    # Final: Pair winners of SF matches
    f_matches = []
    w1 = sf_match_ids[0]
    w2 = sf_match_ids[1]
    f_matches.append((f"Winner M{w1}", f"Winner M{w2}"))
    schedule['F'] = f_matches

    return schedule

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

def get_stylized_bracket(
    draw,
    col_pad: int = 2,
    header_pad_rows: int = 2,
    charset: str = "unicode",
    annotate_brackets: bool = True,
    bracket_labels=("A", "B", "C", "D")
) -> str:
    """
    Modified to return the bracket as a string for Streamlit display.
    """
    # ... (entire function body as provided, but replace all print() with building a string)
    # For brevity, assume we capture output:
    from io import StringIO
    import sys
    old_stdout = sys.stdout
    output = StringIO()
    sys.stdout = output
    print_stylized_bracket(draw, col_pad, header_pad_rows, charset, annotate_brackets, bracket_labels)  # Call original
    sys.stdout = old_stdout
    return output.getvalue()

# Streamlit App
st.title("Bangalore Smash Academy Tournament Simulator")
st.header("Where Professionals Meet Badminton Passion")
st.markdown("""
Welcome to BSA's virtual draw generator. Select a discipline, tweak settings, and simulate a weekly tournament.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Simulation Settings")
    discipline = st.selectbox("Choose Discipline", ["MD (Men's Doubles)", "WD (Women's Doubles)", "XD (Mixed Doubles)"])
    random_seed = st.number_input("Random Seed (for reproducibility)", value=42, step=1)
    charset = st.selectbox("Bracket Style", ["unicode (Pretty)", "ascii (Simple)"])
    regenerate = st.button("Regenerate/Simulate Draw")

# Map discipline to keys
disc_key = discipline[:2]
seeded_key = f"seeded_players_{disc_key.lower()}"
unseeded_key = f"unseeded_players_{disc_key.lower()}"
extra_key = f"unseeded_players_extra_{disc_key.lower()}"

# Prepare players
seeded = players_info[seeded_key]
unseeded = players_info[unseeded_key].copy()  # Copy to avoid mutation
extra = players_info[extra_key]


if regenerate:
    if len(seeded) < 16:
        random.shuffle(extra)
        n = 16 - len(seeded)
        unseeded.extend(extra[:n])

    st.session_state.draw = generate_tournament_schedule_new(seeded, unseeded, random_seed=random_seed)
    st.session_state.simulated = True

else:
    st.session_state.simulated = st.session_state.get('simulated', False)

if st.session_state.get('simulated', False):
    # Display the bracket
    st.subheader(f"{discipline} Tournament Draw")
    bracket_str = get_stylized_bracket(
        st.session_state.draw,
        charset=charset.split()[0]
    )
    st.code(bracket_str, language="text")  # Monospace for alignment

    # Additional functionalities
    st.header("Extra Features")
    if st.button("Export to Text"):
        st.download_button("Download Bracket", bracket_str, file_name=f"{disc_key}_bracket.txt")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Show Player Roster"):
            seeded_df = pd.DataFrame({
                "Rank": list(range(1, len(seeded) + 1)),
                "Seeded Pair": seeded
            })
            unseeded_df = pd.DataFrame({
                "Unseeded Pair": unseeded
            })

            st.subheader("Seeded Pairs")
            st.dataframe(seeded_df, use_container_width=True)  # Or st.table for non-interactive

            st.subheader("Unseeded Pairs")
            st.dataframe(unseeded_df, use_container_width=True)
    with col2:
        if st.button("Simulate Next Quarter"):
            st.write("Coming soon: Reset rankings and run multiple sims!")

# st.markdown("Balance your sims like Avika juggles WD with Anjali and XD romance—refresh for new paths!")
# # --- Demo with both modes ---
# # MD Seeded players (with seed numbers)
# seeded_players_md = players_info["seeded_players_md"]
# unseeded_players_md = players_info["unseeded_players_md"]
# unseeded_players_extra_md = players_info["unseeded_players_extra_md"]
# if len(seeded_players_md) <16:
#     random.shuffle(unseeded_players_extra_md)
#     n = 16 - len(seeded_players_md)
#     unseeded_players_md.extend(unseeded_players_extra_md[:n])
# demo_draw_md = generate_tournament_schedule_new(seeded_players_md, unseeded_players_md, random_seed=145)
#
# # WD Seeded Players
# seeded_players_wd = players_info["seeded_players_wd"]
# unseeded_players_wd = players_info["unseeded_players_wd"]
# unseeded_players_extra_wd = players_info["unseeded_players_extra_wd"]
# if len(seeded_players_wd) <16:
#     random.shuffle(unseeded_players_extra_wd)
#     n = 16 - len(seeded_players_wd)
#     unseeded_players_md.extend(unseeded_players_extra_wd[:n])
# demo_draw_wd = generate_tournament_schedule_new(seeded_players_wd, unseeded_players_wd, random_seed=135)
#
# # XD seeded Players
# seeded_players_xd = players_info["seeded_players_xd"]
# unseeded_players_xd = players_info["unseeded_players_xd"]
# unseeded_players_extra_xd = players_info["unseeded_players_extra_xd"]
# if len(seeded_players_xd) <16:
#     random.shuffle(unseeded_players_extra_xd)
#     n = 16 - len(seeded_players_xd)
#     unseeded_players_md.extend(unseeded_players_extra_xd[:n])
# demo_draw_xd = generate_tournament_schedule_new(seeded_players_wd, unseeded_players_wd, random_seed=155)
#
# print("-------------------------Men's Doubles Schedule--------------------------------")
# print_stylized_bracket(demo_draw_md, charset="unicode")
# print("-------------------------Women's Doubles Schedule--------------------------------")
# print_stylized_bracket(demo_draw_wd, charset="unicode")
# print("-------------------------Mixed Doubles Schedule--------------------------------")
# print_stylized_bracket(demo_draw_xd, charset="unicode")

