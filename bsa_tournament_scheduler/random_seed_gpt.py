import random
import os
import sys
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from AL_ML_Projects.bsa_tournament_scheduler.players_info import players_info
import streamlit as st

@st.cache_data
def fig_to_bytes(fig, fmt="png", scale=2, width=None, height=None):
    """
    Convert a Plotly figure to image bytes with kaleido.
    fmt: "png" | "svg" | "pdf"
    scale >1 = higher DPI; width/height override figure size (pixels).
    """
    return pio.to_image(fig, format=fmt, scale=scale, width=width, height=height)

def generate_draw_seeded(
    seeds,        # 16 names, ordered by rank: seeds[0] is rank 1 ... seeds[15] is rank 16
    unseeded,     # 16 names
    rng_seed=None,
    shuffle_unseeded=True
):
    """
    Build a 32-draw with seeded placement by brackets:
      - 4 brackets (each = 4 matches in a row: M1-4, M5-8, M9-12, M13-16)
      - In every bracket, place seeds in order: Highest(1-4), Low(9-12), High(5-8), Lowest(13-16)
      - Each seed meets an unseeded player in R32
      - Next rounds: winners of consecutive matches meet (standard progression)

    Returns: dict with keys "R32","R16","QF","SF","F"
             where R32 is a list of 16 (seed, unseeded) pairs;
             later rounds are placeholders like "W(R32 M3)".
    """
    if len(seeds) != 16 or len(unseeded) != 16:
        raise ValueError("Pass exactly 16 seeded names (ranked) and 16 unseeded names.")

    rng = random.Random(rng_seed)

    # Tier indices (0-based for list indexing)
    highest_idx = list(range(0, 4))    # ranks 1–4
    high_idx    = list(range(4, 8))    # ranks 5–8
    low_idx     = list(range(8, 12))   # ranks 9–12
    lowest_idx  = list(range(12, 16))  # ranks 13–16

    # Shuffle inside each tier so every bracket gets a random seed from each tier
    rng.shuffle(highest_idx)
    rng.shuffle(high_idx)
    rng.shuffle(low_idx)
    rng.shuffle(lowest_idx)

    # Build the 16-seed sheet in bracket blocks (4 blocks × 4 matches)
    # Order inside each bracket: Highest → Low → High → Lowest
    seed_order = []
    for i in range(4):  # 4 brackets
        seed_order.extend([
            seeds[highest_idx[i]],  # first match in this bracket
            seeds[low_idx[i]],      # second
            seeds[high_idx[i]],     # third
            seeds[lowest_idx[i]],   # fourth
        ])

    # Shuffle unseeded and pair 1:1 with the above seed order
    u = unseeded[:]
    if shuffle_unseeded:
        rng.shuffle(u)

    r32 = [(seed_order[i], u[i]) for i in range(16)]

    # Helpers to wire later rounds
    def w(round_name, match_no):  # winner token
        return f"W({round_name} M{match_no})"

    def advance(prev_round_name, prev_matches):
        nxt = []
        for i in range(0, len(prev_matches), 2):
            nxt.append((w(prev_round_name, i+1), w(prev_round_name, i+2)))
        return nxt

    r16 = advance("R32", r32)   # 8 matches
    qf  = advance("R16", r16)   # 4 matches
    sf  = advance("QF", qf)     # 2 matches
    f   = advance("SF", sf)     # 1 match

    return {"R32": r32, "R16": r16, "QF": qf, "SF": sf, "F": f}

def generate_draw_flexible(
    seeds,                # list of participating seeds, ascending by rank (length K, 0..16)
    unseeded,             # list of unseeded players, length must be 32 - K
    rng_seed=None,
    shuffle_unseeded=True,
    randomize_full_batches=False,  # only used when some seeds rest (K < 16)
):
    """
    Returns a dict: {"R32": [(A,B)*16], "R16": [...8], "QF": [...4], "SF": [...2], "F": [...1]}
    - If K == 16 (all seeds participate):
        Place seeds in 4 brackets; within each: Highest(1–4) → Low(9–12) → High(5–8) → Lowest(13–16)
        Then pair vs shuffled unseeded.
    - If K < 16 (some seeds rest):
        Batch seeds in groups of 4 (top-4, next-4, ...); each full batch spreads 1 to each bracket A–D.
        If remainder r in {1,2,3}, place them in r distinct random brackets.
        Fill remaining seed-side slots with (16 - K) extra unseeded.
        The other 16 unseeded are opponents (shuffled).
    """

    K = len(seeds)
    if not (0 <= K <= 16):
        raise ValueError("Number of participating seeds must be between 0 and 16.")
    if len(unseeded) != 32 - K:
        raise ValueError(f"unseeded must have length 32 - K (= {32 - K}), got {len(unseeded)}.")

    rng = random.Random(rng_seed)

    # --- Helper: wire later rounds ---
    def w(round_name, match_no):
        return f"W({round_name} M{match_no})"

    def advance(prev_round_name, prev_matches):
        nxt = []
        for i in range(0, len(prev_matches), 2):
            nxt.append((w(prev_round_name, i + 1), w(prev_round_name, i + 2)))
        return nxt

    # --- Case 1: all 16 seeds participate (original seeding logic) ---
    if K == 16:
        # Tiers: indices are 0-based on the already rank-ordered seed list
        highest = seeds[0:4]     # 1–4
        high    = seeds[4:8]     # 5–8
        low     = seeds[8:12]    # 9–12
        lowest  = seeds[12:16]   # 13–16

        # Shuffle within tiers so which exact seed from a tier lands in which bracket is randomized
        highest = highest[:]; rng.shuffle(highest)
        high    = high[:];    rng.shuffle(high)
        low     = low[:];     rng.shuffle(low)
        lowest  = lowest[:];  rng.shuffle(lowest)

        # Build the 16 seed-side slots: 4 brackets × 4 matches each
        seed_side_order = []
        for i in range(4):  # brackets A,B,C,D
            seed_side_order.extend([highest[i], low[i], high[i], lowest[i]])  # fixed in-bracket order

        # Opponents: shuffle all 16 unseeded
        opponents = unseeded[:]
        if shuffle_unseeded:
            rng.shuffle(opponents)

        r32 = [(seed_side_order[i], opponents[i]) for i in range(16)]

    # --- Case 2: some seeds rest (K < 16) ---
    else:
        # Prepare four bracket buckets (A,B,C,D) that each hold up to 4 seeds on the seed-side
        bracket_seeds = [[] for _ in range(4)]  # each element becomes a list of seeds in that bracket

        # Split seeds into batches of 4
        batches = [seeds[i:i+4] for i in range(0, K, 4)]
        full_batches = batches[:-1] if (K % 4) != 0 else batches
        remainder = batches[-1] if (K % 4) != 0 else []

        # Distribute each full batch of 4 across the 4 brackets
        for batch in full_batches:
            if len(batch) != 4:
                continue
            # determine mapping to brackets
            mapping = [0,1,2,3]
            if randomize_full_batches:
                rng.shuffle(mapping)
            # assign in order: batch[j] -> bracket mapping[j]
            for j, seed_name in enumerate(batch):
                bracket_seeds[mapping[j]].append(seed_name)

        # Distribute remainder seeds (size r in {1,2,3}) to r distinct random brackets
        if remainder:
            r = len(remainder)
            chosen_brackets = rng.sample(range(4), r)  # distinct
            for seed_name, b_idx in zip(remainder, chosen_brackets):
                bracket_seeds[b_idx].append(seed_name)

        # Now fill each bracket up to 4 seed-side slots with extra unseeded as needed
        seed_side_order = []
        # We'll pick (16 - K) extra unseeded to act as seed-side replacements
        extra_needed = 16 - K
        extra_pool = unseeded[:]  # will sample from this for seed-side replacements
        if extra_needed > 0:
            extra_seed_side = rng.sample(extra_pool, extra_needed)
            # remove those from extra_pool
            extra_set = set(extra_seed_side)
            extra_pool = [x for x in extra_pool if x not in extra_set]
        else:
            extra_seed_side = []

        # iterator over the chosen extra unseeded for seed-side slots
        extra_iter = iter(extra_seed_side)

        for b in range(4):
            # Place bracket's participating seeds in top-to-bottom order (order inside bracket doesn't matter)
            slots = bracket_seeds[b][:4]
            # Fill remaining seed-side slots in this bracket with extra unseeded
            while len(slots) < 4:
                try:
                    slots.append(next(extra_iter))
                except StopIteration:
                    # Should not happen if counts are correct, but guard anyway
                    raise RuntimeError("Ran out of extra unseeded while filling seed-side slots.")
            rng.shuffle(slots)
            seed_side_order.extend(slots)

        # Now opponents are the remaining 16 unseeded from extra_pool
        opponents = extra_pool  # should now be exactly 16 left
        if len(opponents) != 16:
            raise RuntimeError(f"Expected 16 opponent-side unseeded, got {len(opponents)}.")
        if shuffle_unseeded:
            rng.shuffle(opponents)

        r32 = [(seed_side_order[i], opponents[i]) for i in range(16)]

    # Build later rounds
    r16 = advance("R32", r32)
    qf  = advance("R16", r16)
    sf  = advance("QF",  qf)
    f   = advance("SF",  sf)

    return {"R32": r32, "R16": r16, "QF": qf, "SF": sf, "F": f}

# import plotly.graph_objects as go

def render_bracket_plotly(
    draw,
    seeds=None,
    title: str | None = None,
    show_bracket_labels: bool = True,
    # box geometry (independent of spacing)
    box_w: float = 1.6,
    box_h: float = 0.8,
    size_scale: float = 1.0,          # ONLY scales box size
    # spacing (gaps) — independent of box size
    dx: float = 6,                   # horiz gap between round columns
    vgap: float = 2.5,                 # vertical gap between consecutive R32 matches
    # group_gap: float = 3.6,            # extra gap inserted between R32 brackets A/B/C/D
    gap_scale: float = 1.2,           # ONLY scales gaps (dx, vgap, group_gap) → use this to “open up” the chart
    # typography
    font_size: float = 8.7,
    # fullscreen comfort: scale figure height more than usual so expanded view stays airy
    height_gain: float = 1.35,         # ↑ if fullscreen still feels tight
):
    """
    Plotly bracket renderer with:
      - size_scale   -> grows boxes only
      - gap_scale    -> increases gaps only (more space)
      - group_gap    -> extra vertical padding between R32 brackets A–D
      - height_gain  -> multiplies computed figure height so fullscreen isn't cramped
    """

    # apply scales
    box_w *= size_scale
    box_h *= size_scale
    dx *= gap_scale
    vgap *= gap_scale
    # group_gap *= gap_scale

    seed_set = set(seeds or [])
    rounds = ["R32", "R16", "QF", "SF", "F"]
    match_counts = {"R32": 16, "R16": 8, "QF": 4, "SF": 2, "F": 1}

    # x positions per round
    x_pos = {r: i * dx for i, r in enumerate(rounds)}

    # y positions
    y_pos = {}

    # R32 with extra group gaps between brackets (after M4, M8, M12)
    y_vals = []
    y = 0.0
    for i in range(match_counts["R32"]):
        y_vals.append(y)
        y += vgap
    y_pos["R32"] = y_vals

    # subsequent rounds are midpoints of children
    for ri in range(1, len(rounds)):
        prev, cur = rounds[ri - 1], rounds[ri]
        y_cur = []
        for i in range(match_counts[cur]):
            y_mid = (y_pos[prev][2 * i] + y_pos[prev][2 * i + 1]) / 2.0
            y_cur.append(y_mid)
        y_pos[cur] = y_cur

    def tag(name: str) -> str:
        return f"★ {name}" if name in seed_set else name

    def add_match_box(fig, rnd: str, i: int, a: str, b: str):
        xc = x_pos[rnd]
        yc = y_pos[rnd][i]
        x0, x1 = xc - box_w / 2, xc + box_w / 2
        y0, y1 = yc - box_h / 2, yc + box_h / 2

        fig.add_shape(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="#9AA4B2", width=1.4),
            fillcolor="#F6F8FB"
        )
        fig.add_annotation(
            x=xc, y=yc,
            text=f"{tag(a)}<br><span style='font-size:90%'>vs</span><br>{tag(b)}",
            showarrow=False,
            font=dict(size=font_size),
            align="center"
        )

    def add_connector(fig, rnd: str, i: int):
        r_index = rounds.index(rnd)
        if r_index >= len(rounds) - 1:
            return
        nxt = rounds[r_index + 1]

        x_right = x_pos[rnd] + box_w / 2
        y_mid = y_pos[rnd][i]
        x_left_next = x_pos[nxt] - box_w / 2
        y_next = y_pos[nxt][i // 2]
        x_elbow = (x_right + x_left_next) / 2.0

        fig.add_shape(type="line", x0=x_right, y0=y_mid, x1=x_elbow, y1=y_mid,
                      line=dict(color="#C1C7D0", width=1.2))
        fig.add_shape(type="line", x0=x_elbow, y0=y_mid, x1=x_elbow, y1=y_next,
                      line=dict(color="#C1C7D0", width=1.2))
        fig.add_shape(type="line", x0=x_elbow, y0=y_next, x1=x_left_next, y1=y_next,
                      line=dict(color="#C1C7D0", width=1.2))

    fig = go.Figure()


    for rnd in rounds:
        for i, (a, b) in enumerate(draw[rnd]):
            add_match_box(fig, rnd, i, a, b)
            add_connector(fig, rnd, i)

    # # Bracket labels (A–D) above R32
    # if show_bracket_labels:
    #     labels = ["A", "B", "C", "D"]
    #     for g, lab in enumerate(labels):
    #         idx0 = g * 4
    #         if idx0 < len(y_pos["R32"]):
    #             y_lab = y_pos["R32"][idx0] + box_h / 2 + 0.5
    #             fig.add_annotation(
    #                 x=x_pos["R32"] - box_w / 2,
    #                 y=y_lab,
    #                 text=f"<b>Bracket {lab}</b>",
    #                 showarrow=False,
    #                 xanchor="left",
    #                 font=dict(size=font_size + 1, color="#475569")
    #             )

    # Emphasize Final
    if draw["F"]:
        xcf, ycf = x_pos["F"], y_pos["F"][0]
        fig.add_shape(
            type="rect",
            x0=xcf - box_w / 2, y0=ycf - box_h / 2, x1=xcf + box_w / 2, y1=ycf + box_h / 2,
            line=dict(color="#2A72D4", width=2.6),
            fillcolor="rgba(0,0,0,0)"
        )

    # canvas extents (make it tall enough so fullscreen stays airy)
    top_pad = 1.2
    bottom_pad = 1.0
    max_y = y_pos["R32"][-1] + box_h / 2 + top_pad
    min_y = -bottom_pad
    max_x = x_pos["F"] + box_w / 2 + 1.0
    # Add round headers (R32, R16, QF, SF, F)
    header_y = max_y - 0.1  # a bit above the top boxes
    for rnd in rounds:
        fig.add_annotation(
            x=x_pos[rnd],
            y=header_y,
            text=f"<b>{rnd}</b>",
            showarrow=False,
            font=dict(size=font_size + 3, color="#1E293B"),
            align="center",
            xanchor="center"
        )
    fig.update_xaxes(visible=False, range=[-1.0, max_x])
    fig.update_yaxes(visible=False, range=[min_y, max_y])

    # Height scales with vertical data units; height_gain bumps it further for fullscreen
    units_tall = (max_y - min_y)
    px_per_unit = 60  # base vertical pixel density
    print(int(units_tall * px_per_unit * height_gain))
    fig.update_layout(
        height=int(units_tall * px_per_unit * height_gain),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        title=title or None,
    )
    return fig


# Streamlit App
st.title("Bangalore Smash Academy Tournament Simulator")
st.header("Where Professionals Meet Badminton Passion")
st.markdown("""
Welcome to BSA's virtual draw generator. Select a discipline, tweak settings, and simulate a weekly tournament.
""")

with st.sidebar:
    st.header("Draw Controls")

    discipline = st.selectbox(
        "Discipline",
        ["MD (Men's Doubles)", "WD (Women's Doubles)", "XD (Mixed Doubles)"],
        key="discipline_sel",
    )

    random_seed = st.number_input(
        "Random Seed",
        value=42, step=1,
        key="rng_seed_input",
    )

    # # spacing controls for the Plotly bracket
    # gap_scale  = st.slider(
    #     "Gap scale (spacing)", 1.0, 2.0, 1.35, 0.05,
    #     key="gap_scale_main",
    # )
    # size_scale = st.slider(
    #     "Box size", 0.9, 1.5, 1.20, 0.05,
    #     key="size_scale_main",
    # )
    # # if you also expose height gain, give it a key too:
    # height_gain = st.slider("Fullscreen height gain", 1.0, 2.0, 1.35, 0.05, key="height_gain_main")
    start_btn = st.button("Find/Regenerate Draw(using latest top 16 unseeded entries)",key='strt_btn')
    st.markdown("---")
    st.caption("Choose resting seeds and choose unseeded entries from the 26-player pool (16 qualified + 10 extra) to update/simulate draw for the new unseeded/seeded pair changes.")


# Map discipline to players_info keys
disc_key = discipline[:2]          # "MD", "WD", "XD"
seeded_key   = f"seeded_players_{disc_key.lower()}"
unseeded_key = f"unseeded_players_{disc_key.lower()}"
extra_key    = f"unseeded_players_extra_{disc_key.lower()}"

seeded_all   = players_info[seeded_key]                 # 16 seeds in rank order
unseeded_16  = players_info[unseeded_key]               # qualified 16
unseeded_10  = players_info[extra_key]                  # extra 10
unseeded_26  = unseeded_16 + unseeded_10                # pool to choose from (no dups expected)

# after you compute disc_key, seeded_all, unseeded_26, etc.
with st.sidebar:
    resting_seeds = st.multiselect(
        "Seeds resting",
        options=seeded_all,
        default=[],
        key="resting_seeds_ms",
    )

    K = 16 - len(resting_seeds)
    needed_unseeded = 32 - K
    default_selection = players_info[unseeded_key][:min(16, needed_unseeded)]

    selected_unseeded = st.multiselect(
        f"Unseeded entries (choose exactly {needed_unseeded})",
        options=unseeded_26,
        default=default_selection,
        key="unseeded_26_ms",
    )

    generate_btn = st.button("Update Draw", use_container_width=True, key="generate_btn_main")

# def make_draw_csv(draw, seeds=None):
#     seed_set = set(seeds or [])
#     rows = []
#     for rnd in ["R32","R16","QF","SF","F"]:
#         for i, (a, b) in enumerate(draw[rnd], start=1):
#             rows.append({
#                 "Round": rnd,
#                 "Match": f"M{i}",
#                 "Side A": a,
#                 "Side B": b,
#                 "Seed A?": a in seed_set,
#                 "Seed B?": b in seed_set,
#             })
#     return pd.DataFrame(rows).to_csv(index=False)


# Initial default draw (first load): assume all 16 seeds + the qualified 16
if start_btn:
    st.session_state.draw = generate_draw_flexible(
        seeds_all := seeded_all,                 # all 16 participating
        unseeded   := unseeded_16,               # qualified 16
        rng_seed=random_seed
    )
    st.session_state.simulated = True
    st.session_state.seeds = seeded_all
    st.session_state.unseeded = unseeded_16
    st.session_state.extra = unseeded_10

# On click: validate counts, then regenerate
if generate_btn:
    if len(selected_unseeded) != needed_unseeded:
        st.error(f"Please select exactly {needed_unseeded} unseeded entries "
                 f"(currently {len(selected_unseeded)}).")
    else:
        participating_seeds = [s for s in seeded_all if s not in resting_seeds]
        st.session_state.draw = generate_draw_flexible(
            participating_seeds,
            selected_unseeded,
            rng_seed=random_seed
        )
        st.session_state.simulated = True
        st.success("Draw updated.")
        st.session_state.seeds = participating_seeds
        st.session_state.unseeded = selected_unseeded
        st.session_state.extra = list(set(unseeded_26) - set(selected_unseeded))

if st.session_state.get('simulated', False):
    # Display the bracket
    st.subheader(f"{discipline} Tournament Draw")
    # with st.sidebar:
    #     gap_scale = st.slider("Gap scale (spacing)", 1.0, 2.0, 1.35, 0.05)
    #     size_scale = st.slider("Box size", 0.9, 1.5, 1.20, 0.05)
    #     height_gain = st.slider("Fullscreen height gain", 1.0, 2.0, 1.35, 0.05)

    fig = render_bracket_plotly(
        st.session_state.draw,
        seeds=[s for s in seeded_all if s not in resting_seeds],
        title=f"{discipline} – Brackets",
        # gap_scale=gap_scale,  # spreads rounds & rows apart
        # size_scale=size_scale,  # box-only scaling
        # height_gain=height_gain,  # keeps fullscreen airy
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Additional functionalities
    st.header("Extra Features")
    roster_button = st.button("Show Player Roster", use_container_width=True)
    if roster_button:
        seeded_df = pd.DataFrame({
            "Seeded Pairs": st.session_state.seeds,
        })
        unseeded_df = pd.DataFrame({
            "Latest Top Qualified Unseeded Pairs ": st.session_state.unseeded,
        })
        extra_unseeded_df = pd.DataFrame({
            "Unseeded Remaining Pairs": st.session_state.extra,
        })

        st.subheader("Seeded Pairs")
        st.dataframe(seeded_df, use_container_width=True)  # Or st.table for non-interactive

        st.subheader("Unseeded Pairs Qualified")
        st.dataframe(unseeded_df, use_container_width=True)

        st.subheader("Unseeded Remaining Pairs")
        st.dataframe(extra_unseeded_df, use_container_width=True)

    # Your download button(s) – CSV version
    # --- Download buttons for the image ---
    # Choose a consistent export size (pixels). You can tune these.
    export_width = 1800
    export_height = fig.layout.height or 1200

    png_bytes = fig_to_bytes(fig, fmt="png", scale=2, width=export_width, height=export_height)
    # svg_bytes = fig_to_bytes(fig, fmt="svg", width=export_width, height=export_height)
    # csv_data = make_draw_csv(st.session_state.draw, seeds=seeded)
    st.download_button(
        "⬇️ Download Bracket (PNG)",
        data=png_bytes,
        file_name=f"{disc_key}_bracket.png",
        mime="image/png",
        use_container_width=True,
    )
    #
    # roster_button = st.button("Show Player Roster", use_container_width=True)
    # if roster_button :
    #     seeded_df = pd.DataFrame({
    #         "Seeded Pair": seeded
    #     })
    #     unseeded_df = pd.DataFrame({
    #         "Unseeded Pairs Qualified(Prev Tournament)": unseeded
    #     })
    #     extra_unseeded_df = pd.DataFrame({
    #         "Unseeded Remaining Pairs": extra
    #     })
    #
    #     st.subheader("Seeded Pairs")
    #     st.dataframe(seeded_df, use_container_width=True)  # Or st.table for non-interactive
    #
    #     st.subheader("Unseeded Pairs Qualified(Prev Tournament)")
    #     st.dataframe(unseeded_df, use_container_width=True)
    #
    #     st.subheader("Unseeded Remaining Pairs")
    #     st.dataframe(extra_unseeded_df, use_container_width=True)

# # --- Demo with both modes ---
# # MD Seeded players (with seed numbers)
# seeded_players_md = players_info["seeded_players_md"]
# unseeded_players_md = players_info["unseeded_players_md"]
# unseeded_players_extra_md = players_info["unseeded_players_extra_md"]
# if len(seeded_players_md) <16:
#     random.shuffle(unseeded_players_extra_md)
#     n = 16 - len(seeded_players_md)
#     unseeded_players_md.extend(unseeded_players_extra_md[:n])
# demo_draw_md = generate_draw_flexible(seeded_players_md, unseeded_players_md, rng_seed=145)
#
# # WD Seeded Players
# seeded_players_wd = players_info["seeded_players_wd"]
# unseeded_players_wd = players_info["unseeded_players_wd"]
# unseeded_players_extra_wd = players_info["unseeded_players_extra_wd"]
# if len(seeded_players_wd) <16:
#     random.shuffle(unseeded_players_extra_wd)
#     n = 16 - len(seeded_players_wd)
#     unseeded_players_md.extend(unseeded_players_extra_wd[:n])
# demo_draw_wd = generate_draw_flexible(seeded_players_wd, unseeded_players_wd, rng_seed=135)
#
# # XD seeded Players
# seeded_players_xd = players_info["seeded_players_xd"]
# unseeded_players_xd = players_info["unseeded_players_xd"]
# unseeded_players_extra_xd = players_info["unseeded_players_extra_xd"]
# if len(seeded_players_xd) <16:
#     random.shuffle(unseeded_players_extra_xd)
#     n = 16 - len(seeded_players_xd)
#     unseeded_players_md.extend(unseeded_players_extra_xd[:n])
# demo_draw_xd = generate_draw_flexible(seeded_players_wd, unseeded_players_wd, rng_seed=155)
#
# print("-------------------------Men's Doubles Schedule--------------------------------")
# print_stylized_bracket(demo_draw_md, charset="unicode")
# print("-------------------------Women's Doubles Schedule--------------------------------")
# print_stylized_bracket(demo_draw_wd, charset="unicode")
# print("-------------------------Mixed Doubles Schedule--------------------------------")
# print_stylized_bracket(demo_draw_xd, charset="unicode")

# def print_stylized_bracket(
#     draw,
#     col_pad: int = 2,
#     header_pad_rows: int = 2,
#     charset: str = "ascii",              # "ascii" (robust) or "unicode" (pretty)
#     annotate_brackets: bool = True,      # show A/B/C/D blocks for R32
#     bracket_labels = ("A", "B", "C", "D")
# ) -> None:
#     """
#     Render a stylized bracket with safe padding and optional ASCII mode,
#     and annotate the 4 seeded R32 brackets (A–D), each spanning 4 matches.
#     """
#     rounds_order = ["R32", "R16", "QF", "SF", "F"]
#
#     # Character sets to avoid glyph width issues
#     if charset == "unicode":
#         CH = dict(h="─", v="│", tee="┤", ctop="┐", cbot="┘", sep="═")
#     else:  # ascii
#         CH = dict(h="-", v="|", tee="+", ctop="+", cbot="+", sep="=")
#
#     # Build labels
#     r32_flat = []
#     for (a, b) in draw["R32"]:
#         r32_flat.extend([a, b])
#     round_labels = [r32_flat]
#
#     def flatten_round(rnd: str):
#         labels = []
#         for (a, b) in draw[rnd]:
#             labels.extend([a, b])
#         return labels
#
#     round_labels += [
#         flatten_round("R16"),  # 16
#         flatten_round("QF"),   # 8
#         flatten_round("SF"),   # 4
#         flatten_round("F"),    # 2
#     ]
#
#     # Row placement with top padding (prevents header/first-row overlap)
#     y_positions = []
#     y0 = [header_pad_rows + i * 2 for i in range(32)]  # <- offset by header_pad_rows
#     y_positions.append(y0)
#
#     def mids_from(prev_positions):
#         mids = []
#         for i in range(0, len(prev_positions), 2):
#             mids.append((prev_positions[i] + prev_positions[i + 1]) // 2)
#         return mids
#
#     y1 = mids_from(y0)
#     y2 = mids_from(y1)
#     y3 = mids_from(y2)
#     y4 = mids_from(y3)
#     y_positions += [y1, y2, y3, y4]
#
#     # Canvas
#     all_labels = [lbl for col in round_labels for lbl in col]
#     max_label_len = max(len(s) for s in all_labels) if all_labels else 10
#     col_width = max(12, min(28, max_label_len + 2))
#     connector_span = 6
#     cols = len(rounds_order) * col_width + (len(rounds_order) - 1) * connector_span
#     rows = max(y0) + 1  # include padding
#     grid = [[" "] * cols for _ in range(rows + 2)]
#
#     # Helpers
#     def put_text(y: int, x: int, text: str):
#         for i, ch in enumerate(text[: col_width - col_pad]):
#             if 0 <= y < len(grid) and 0 <= x + i < cols:
#                 grid[y][x + i] = ch
#
#     def draw_hline(y: int, x1: int, x2: int, char: str):
#         if x1 > x2:
#             x1, x2 = x2, x1
#         for x in range(x1, x2 + 1):
#             if 0 <= y < len(grid) and 0 <= x < cols:
#                 grid[y][x] = char
#
#     def draw_vline(x: int, y1: int, y2: int, char: str):
#         if y1 > y2:
#             y1, y2 = y2, y1
#         for y in range(y1, y2 + 1):
#             if 0 <= y < len(grid) and 0 <= x < cols:
#                 grid[y][x] = char
#
#     # Place text
#     col_starts = []
#     for r in range(len(rounds_order)):
#         x = r * (col_width + connector_span)
#         col_starts.append(x)
#         for idx, label in enumerate(round_labels[r]):
#             put_text(y_positions[r][idx], x, label)
#
#     # Headers on their own row(s)
#     for r, name in enumerate(rounds_order):
#         put_text(0, col_starts[r], f"[ {name} ]")
#
#     # Optional: annotate the 4 R32 brackets (A–D) as horizontal separators
#     if annotate_brackets:
#         # Each bracket is 4 matches = 8 participants in R32
#         # Start participant index for bracket g: p_start = g * 8
#         for g, lab in enumerate(bracket_labels):
#             p_start = g * 8
#             if p_start >= len(y_positions[0]):
#                 break
#             y_label = y_positions[0][p_start] - 1
#             if y_label < 1:
#                 y_label = 1  # keep above the first participants, below header
#             # draw a full-width separator
#             draw_hline(y_label, 0, cols - 1, CH["sep"])
#             # write the label at the left
#             label_text = f"[ Bracket {lab} ]"
#             for i, ch in enumerate(label_text):
#                 if i < cols:
#                     grid[y_label][i] = ch
#
#     # Connectors
#     for r in range(len(rounds_order) - 1):
#         x_text = col_starts[r]
#         x_next_text = col_starts[r + 1]
#         x_arm_end = x_text + col_width
#         x_spine = x_arm_end + 1
#         x_to_next = x_next_text - 1
#
#         num_matches = len(y_positions[r]) // 2
#         for m in range(num_matches):
#             top_idx = 2 * m
#             bot_idx = 2 * m + 1
#             y_top = y_positions[r][top_idx]
#             y_bot = y_positions[r][bot_idx]
#             y_mid = (y_top + y_bot) // 2
#
#             draw_hline(y_top, x_arm_end, x_spine - 1, CH["h"])
#             draw_hline(y_bot, x_arm_end, x_spine - 1, CH["h"])
#             draw_vline(x_spine, y_top, y_bot, CH["v"])
#             grid[y_top][x_spine] = CH["ctop"]
#             grid[y_bot][x_spine] = CH["cbot"]
#             grid[y_mid][x_spine] = CH["tee"]
#             draw_hline(y_mid, x_spine + 1, x_to_next, CH["h"])
#
#     print("\n".join("".join(row).rstrip() for row in grid))

# def get_stylized_bracket(
#     draw,
#     col_pad: int = 2,
#     header_pad_rows: int = 2,
#     charset: str = "unicode",
#     annotate_brackets: bool = True,
#     bracket_labels=("A", "B", "C", "D")
# ) -> str:
#     """
#     Modified to return the bracket as a string for Streamlit display.
#     """
#     # ... (entire function body as provided, but replace all print() with building a string)
#     # For brevity, assume we capture output:
#     from io import StringIO
#     import sys
#     old_stdout = sys.stdout
#     output = StringIO()
#     sys.stdout = output
#     print_stylized_bracket(draw, col_pad, header_pad_rows, charset, annotate_brackets, bracket_labels)  # Call original
#     sys.stdout = old_stdout
#     return output.getvalue()