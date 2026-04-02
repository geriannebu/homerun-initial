"""
quiz.py
=======
Streamlit quiz component for the HDB Flat Recommender.

Returns two things the rest of the app needs:
    scoring_weights  — rank-sum weights from the final confirmed ranking
                       (used for amenity_score calculation in recommender.py)
    final_ranking    — ordered list of amenity keys, most → least important

Flow
----
    Step 1  Amenity multi-select checkboxes
    Step 2  Up to 4 filtered quiz questions (radio buttons)
            → normalised_quiz_weights  (used ONLY to suggest ranking order)
    Step 3  Tie-break sliders  (only shown if two amenities scored within TIE_THRESHOLD)
    Step 4  User confirms / adjusts the ranking via selectboxes
    Step 5  rank_sum_weights(final_ranking)  → scoring_weights shown + returned

Usage
-----
    from backend.quiz import render_quiz, reset_quiz

    scoring_weights, final_ranking = render_quiz()

    if not scoring_weights:
        st.stop()   # quiz not done yet — wait for user

    # scoring_weights → e.g. {"train": 0.5, "hawker": 0.333, "bus": 0.167}
    # final_ranking   → e.g. ["train", "hawker", "bus"]
"""

from __future__ import annotations
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────

AMENITY_LABELS: dict[str, str] = {
    "train":          "MRT / Train",
    "bus":            "Bus Stop",
    "hawker":         "Hawker Centre",
    "mall":           "Shopping Mall",
    "supermarket":    "Supermarket",
    "polyclinic":     "Polyclinic",
    "primary_school": "Primary School",
}

QUESTION_BANK = [
    {
        "id": "q1",
        "text": "What makes your daily commute feel easiest?",
        "options": [
            {"id": "q1_a", "label": "A fast MRT connection", "amenity": "train"},
            {"id": "q1_b", "label": "A bus stop very close to home", "amenity": "bus"},
            {"id": "q1_c", "label": "I don’t mind either, as long as essentials are nearby", "amenity": "mall"},
        ],
    },
    {
        "id": "q2",
        "text": "On most days, how do you usually handle meals?",
        "options": [
            {"id": "q2_a", "label": "I like affordable cooked food nearby", "amenity": "hawker"},
            {"id": "q2_b", "label": "I usually buy food while running errands at a mall", "amenity": "mall"},
            {"id": "q2_c", "label": "I prefer buying groceries and preparing food at home", "amenity": "supermarket"},
        ],
    },
    {
        "id": "q3",
        "text": "On a busy weekday, which nearby option would help you the most?",
        "options": [
            {"id": "q3_a", "label": "MRT access", "amenity": "train"},
            {"id": "q3_b", "label": "A one-stop place for errands and essentials", "amenity": "mall"},
            {"id": "q3_c", "label": "A nearby clinic or polyclinic", "amenity": "polyclinic"},
        ],
    },
    {
        "id": "q4",
        "text": "Which of these matters more for your household right now?",
        "options": [
            {"id": "q4_a", "label": "Good school access", "amenity": "primary_school"},
            {"id": "q4_b", "label": "Healthcare nearby", "amenity": "polyclinic"},
            {"id": "q4_c", "label": "Good public transport connectivity", "amenity": "train"},
        ],
    },
    {
        "id": "q5",
        "text": "What sounds most like your usual weekend?",
        "options": [
            {"id": "q5_a", "label": "Eating around the neighbourhood and staying close to home", "amenity": "hawker"},
            {"id": "q5_b", "label": "Shopping, errands, cafés, or mall time", "amenity": "mall"},
            {"id": "q5_c", "label": "Family-oriented routines where nearby schools and amenities matter", "amenity": "primary_school"},
        ],
    },
    {
        "id": "q6",
        "text": "If you had to prioritise one, which would you choose?",
        "options": [
            {"id": "q6_a", "label": "Being near MRT over having more food options", "amenity": "train"},
            {"id": "q6_b", "label": "Having food options nearby over faster transport", "amenity": "hawker"},
            {"id": "q6_c", "label": "Having everyday essentials in one place", "amenity": "mall"},
        ],
    },
]

NO_PREF_LABEL   = "No preference"
TIE_THRESHOLD   = 0.001   # quiz weights within this gap → treated as tied
QUIZ_SCORE_BASE = 0.25    # baseline score added to every selected amenity


# ── Pure logic helpers (no Streamlit) ─────────────────────────────────────────

def _build_active_questions(selected: list[str]) -> list[dict]:
    """
    Filter QUESTION_BANK to questions with ≥2 options whose amenity was selected.
    Append 'No preference' option to every kept question. Return at most 4.
    """
    sel, active = set(selected), []
    for q in QUESTION_BANK:
        valid = [o for o in q["options"] if o["amenity"] in sel]
        if len(valid) >= 2:
            active.append({
                **q,
                "options": valid + [{"label": NO_PREF_LABEL, "amenity": None}],
            })
    return active[:4]


def _compute_normalised_weights(
    selected: list[str],
    answers: dict[str, str | None],
) -> dict[str, float]:
    """
    Quiz scoring:
        baseline +0.25 per selected amenity
        +1.0 per answer that maps to that amenity
        'No preference' adds 0
    Normalise so scores sum to 1.
    Result used ONLY to suggest ranking order — not for final scoring.
    """
    scores = {a: QUIZ_SCORE_BASE for a in selected}
    for amenity in answers.values():
        if amenity and amenity in scores:
            scores[amenity] += 1.0
    total = sum(scores.values())
    if total == 0:
        n = len(selected)
        return {a: round(1 / n, 4) for a in selected}
    return {a: round(v / total, 4) for a, v in scores.items()}


def rank_sum_weights(ranking: list[str]) -> dict[str, float]:
    """
    Convert an ordered amenity list → Rank-Sum weights that sum to 1.
    These are the weights used for computing amenity_score during scoring.

    Formula: weight(rank i) = (n - i + 1) / sum(1..n)

    Example — ['train', 'hawker', 'bus']  (n=3, denom=6):
        train  rank 1  →  3/6 = 0.5000
        hawker rank 2  →  2/6 = 0.3333
        bus    rank 3  →  1/6 = 0.1667
    """
    n     = len(ranking)
    denom = n * (n + 1) / 2
    return {a: round((n - i) / denom, 6) for i, a in enumerate(ranking)}


def _find_ties(
    ranking: list[str],
    weights: dict[str, float],
) -> list[tuple[str, str]]:
    """Return adjacent pairs in ranking whose quiz weights are within TIE_THRESHOLD."""
    ties = []
    for i in range(len(ranking) - 1):
        a1, a2 = ranking[i], ranking[i + 1]
        if abs(weights[a1] - weights[a2]) <= TIE_THRESHOLD:
            ties.append((a1, a2))
    return ties


# ── Session state initialiser ──────────────────────────────────────────────────

def _init_state(ss) -> None:
    defaults = {
        "quiz_step":               "select",
        "quiz_selected":           [],
        "quiz_answers":            {},
        "quiz_normalised_weights": {},
        "quiz_ranking":            [],
        "quiz_ties":               [],
        "quiz_tiebreak":           {},
        "quiz_final_ranking":      [],
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v


# ── Main Streamlit component ───────────────────────────────────────────────────

def render_quiz() -> tuple[dict[str, float], list[str]]:
    """
    Render the full multi-step quiz UI in Streamlit.

    Returns
    -------
    (scoring_weights, final_ranking)
        scoring_weights : rank-sum weights by amenity key, sums to 1.0
                          pass to run_recommender(amenity_weights=scoring_weights)
        final_ranking   : amenity keys ordered most → least important
                          pass to run_recommender(amenity_ranking=final_ranking)

    Returns ({}, []) while still in progress.
    """
    ss = st.session_state
    _init_state(ss)

    # ── Step 1: Amenity selection ──────────────────────────────────────────────
    if ss.quiz_step == "select":
        st.subheader("Step 1 — What amenities matter to you?")
        st.caption("Select everything you care about. We'll personalise the quiz to match.")

        chosen = []
        cols = st.columns(2)
        for i, (key, label) in enumerate(AMENITY_LABELS.items()):
            with cols[i % 2]:
                if st.checkbox(label, key=f"_qcb_{key}"):
                    chosen.append(key)

        st.button(
            "Next →",
            disabled=len(chosen) < 1,
            on_click=lambda: (
                ss.update({"quiz_selected": chosen, "quiz_step": "quiz"})
            ),
        )
        return {}, []

    # ── Step 2: Quiz questions ─────────────────────────────────────────────────
    if ss.quiz_step == "quiz":
        questions = _build_active_questions(ss.quiz_selected)

        if not questions:
            weights = _compute_normalised_weights(ss.quiz_selected, {})
            ranking = sorted(weights, key=lambda a: weights[a], reverse=True)
            ss.quiz_normalised_weights = weights
            ss.quiz_ranking            = ranking
            ss.quiz_ties               = []
            ss.quiz_tiebreak           = {}
            ss.quiz_step               = "adjust"
            st.rerun()
            return {}, []

        st.subheader("Step 2 — Quick quiz")
        st.caption("Your answers help us suggest an amenity ranking.")

        answers: dict[str, str | None] = {}
        for q in questions:
            st.markdown(f"**{q['text']}**")
            option_labels = [o["label"] for o in q["options"]]
            option_keys   = [o["amenity"] for o in q["options"]]
            prev_key = ss.quiz_answers.get(q["id"])
            prev_idx = option_keys.index(prev_key) if prev_key in option_keys else 0
            choice   = st.radio(
                label=q["id"],
                options=option_labels,
                index=prev_idx,
                horizontal=True,
                label_visibility="collapsed",
                key=f"_qr_{q['id']}",
            )
            answers[q["id"]] = option_keys[option_labels.index(choice)]
            st.divider()

        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("← Back", key="_qback2"):
                ss.quiz_step = "select"
                st.rerun()
        with c2:
            if st.button("See my ranking →", key="_qnext2"):
                ss.quiz_answers            = answers
                weights                    = _compute_normalised_weights(ss.quiz_selected, answers)
                ranking                    = sorted(weights, key=lambda a: weights[a], reverse=True)
                ss.quiz_normalised_weights = weights
                ss.quiz_ranking            = ranking
                ties                       = _find_ties(ranking, weights)
                ss.quiz_ties               = ties
                ss.quiz_tiebreak           = {f"{a1}__{a2}": 0 for a1, a2 in ties}
                ss.quiz_step               = "tiebreak" if ties else "adjust"
                st.rerun()
        return {}, []

    # ── Step 3: Tie-break sliders ──────────────────────────────────────────────
    if ss.quiz_step == "tiebreak":
        ties    = ss.quiz_ties
        ranking = list(ss.quiz_ranking)

        st.subheader("Step 3 — Break the tie")
        st.caption(
            "Some amenities scored equally. Drag the slider to set your preference. "
            "This only affects ranking order, not the weights used for scoring."
        )

        for a1, a2 in ties:
            label1, label2 = AMENITY_LABELS[a1], AMENITY_LABELS[a2]
            key = f"{a1}__{a2}"
            st.markdown(f"**{label1}** vs **{label2}**")
            val = st.slider(
                label=key,
                min_value=-5, max_value=5,
                value=ss.quiz_tiebreak.get(key, 0),
                label_visibility="collapsed",
                key=f"_qtb_{key}",
                help=f"-5 = strongly prefer {label1} | 0 = no preference | +5 = strongly prefer {label2}",
            )
            cl, _, cr = st.columns([2, 1, 2])
            with cl: st.caption(f"◄ {label1}")
            with cr: st.caption(f"{label2} ►")
            ss.quiz_tiebreak[key] = val
            st.divider()

        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("← Back", key="_qback3"):
                ss.quiz_step = "quiz"
                st.rerun()
        with c2:
            if st.button("Apply →", key="_qnext3"):
                for a1, a2 in ties:
                    val = ss.quiz_tiebreak.get(f"{a1}__{a2}", 0)
                    if val > 0:
                        i1, i2 = ranking.index(a1), ranking.index(a2)
                        ranking[i1], ranking[i2] = ranking[i2], ranking[i1]
                ss.quiz_ranking = ranking
                ss.quiz_step    = "adjust"
                st.rerun()
        return {}, []

    # ── Step 4: Confirm / adjust ranking ──────────────────────────────────────
    if ss.quiz_step == "adjust":
        ranking = list(ss.quiz_ranking)
        nweights = ss.quiz_normalised_weights

        st.subheader("Step 4 — Confirm your ranking")
        st.caption(
            "This is the suggested order from your quiz. "
            "Use the dropdowns to reorder, then confirm."
        )

        # Show quiz-derived order and weights for transparency
        for i, a in enumerate(ranking):
            w   = nweights.get(a, 0)
            bar = "█" * int(w * 25)
            st.markdown(f"**{i+1}. {AMENITY_LABELS[a]}** — quiz score `{w:.3f}` {bar}")

        st.info(
            "💡 The quiz score only sets the suggested order. "
            "Scoring uses **rank-sum weights** calculated from whichever order you confirm below.",
            icon="ℹ️",
        )

        st.markdown("**Adjust order (optional):**")
        label_to_key  = {AMENITY_LABELS[a]: a for a in ranking}
        new_order_labels: list[str] = []
        for i in range(len(ranking)):
            remaining = [
                AMENITY_LABELS[a] for a in ranking
                if AMENITY_LABELS[a] not in new_order_labels
            ]
            default = AMENITY_LABELS[ranking[i]] if AMENITY_LABELS[ranking[i]] in remaining else remaining[0]
            chosen  = st.selectbox(
                f"Position {i + 1}",
                options=remaining,
                index=remaining.index(default),
                key=f"_qord_{i}",
            )
            new_order_labels.append(chosen)

        # Build final_ranking from selectbox choices
        seen: set[str] = set()
        final_ranking: list[str] = []
        for label in new_order_labels:
            key = label_to_key.get(label)
            if key and key not in seen:
                final_ranking.append(key)
                seen.add(key)
        for a in ranking:
            if a not in seen:
                final_ranking.append(a)

        # Preview rank-sum weights for the current order
        preview_w = rank_sum_weights(final_ranking)
        n, denom  = len(final_ranking), int(len(final_ranking) * (len(final_ranking) + 1) / 2)
        st.markdown("**Rank-sum weights that will be used for scoring:**")
        for i, a in enumerate(final_ranking):
            pts = n - i
            w   = preview_w[a]
            bar = "█" * int(w * 30)
            st.markdown(f"{i+1}. {AMENITY_LABELS[a]} — {pts}/{denom} pts → `{w:.4f}` {bar}")

        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("← Back", key="_qback4"):
                ss.quiz_step = "tiebreak" if ss.quiz_ties else "quiz"
                st.rerun()
        with c2:
            if st.button("Confirm ranking →", key="_qnext4"):
                ss.quiz_final_ranking = final_ranking
                ss.quiz_step          = "done"
                st.rerun()
        return {}, []

    # ── Step 5: Done ──────────────────────────────────────────────────────────
    if ss.quiz_step == "done":
        final_ranking   = ss.quiz_final_ranking
        scoring_weights = rank_sum_weights(final_ranking)
        n               = len(final_ranking)
        denom           = int(n * (n + 1) / 2)

        st.subheader("Your amenity ranking")
        st.caption("These rank-sum weights will be used to score listings.")
        for i, a in enumerate(final_ranking):
            pts = n - i
            w   = scoring_weights[a]
            bar = "█" * int(w * 30)
            st.markdown(
                f"**{i+1}. {AMENITY_LABELS[a]}** — "
                f"{pts}/{denom} pts → weight `{w:.4f}` {bar}"
            )

        if st.button("← Redo quiz"):
            reset_quiz()
            st.rerun()

        return scoring_weights, final_ranking

    return {}, []


def reset_quiz() -> None:
    """Call to wipe all quiz state and restart from Step 1."""
    for key in [
        "quiz_step", "quiz_selected", "quiz_answers",
        "quiz_normalised_weights", "quiz_ranking",
        "quiz_ties", "quiz_tiebreak", "quiz_final_ranking",
    ]:
        st.session_state.pop(key, None)
