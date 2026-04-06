import copy
import numpy as np
import pandas as pd
import streamlit as st

from backend.utils.formatters import fmt_sgd, valuation_tag_html
from backend.utils.scoring import compute_listing_scores
from frontend.components.listing_detail import show_listing_detail


def _safe_str(x):
    return "" if pd.isna(x) else str(x)


# ---------------------------------------------------------------------------
# Feature DF (historical transactions 2017–2026)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_feature_df_cached():
    """Load and preprocess historical HDB transactions from feature_df.csv."""
    try:
        df = pd.read_csv(
            "backend_predictor_listings/price_predictor/csv_outputs/feature_df.csv",
            low_memory=False,
        )
    except Exception:
        return pd.DataFrame()

    # Nominal transacted price = real_price × (rpi / 100)
    if "real_price" in df.columns and "rpi" in df.columns:
        real = pd.to_numeric(df["real_price"], errors="coerce")
        rpi = pd.to_numeric(df["rpi"], errors="coerce")
        df["transacted_price"] = (real * rpi / 100).round(0)

    # Use block + street_name for matching (cleaner than full_address which has postal)
    df["display_address"] = (
        df["block"].fillna("").astype(str).str.strip()
        + " "
        + df["street_name"].fillna("").str.strip()
    ).str.upper().str.strip()

    # Transaction date from month_index (0 = Jan 2017)
    if "month_index" in df.columns:
        mi = pd.to_numeric(df["month_index"], errors="coerce").fillna(0).astype(int)
        df["txn_year"] = 2017 + mi // 12
        df["txn_month"] = mi % 12 + 1
        df["txn_date"] = (
            df["txn_year"].astype(str)
            + "-"
            + df["txn_month"].apply(lambda m: f"{int(m):02d}")
        )

    return df


def _search_feature_df_addresses(query: str, feature_df: pd.DataFrame, max_results: int = 20) -> list:
    """Token-prefix fuzzy search on display_address. Returns list of unique matched addresses."""
    if not query or not query.strip():
        return []

    q = query.strip().upper()
    q_tokens = q.split()
    unique_addrs = feature_df["display_address"].dropna().unique()

    scored = []
    for addr in unique_addrs:
        a = str(addr)
        # Full substring match → highest priority
        if q in a:
            scored.append((addr, 2.0))
            continue
        # Token-prefix overlap: "AVE" matches "AVENUE", "10" matches "10", etc.
        a_tokens = set(a.split())
        matched = sum(
            1 for t in q_tokens
            if any(t == at or at.startswith(t) or t.startswith(at) for at in a_tokens)
        )
        score = matched / len(q_tokens) if q_tokens else 0
        if score >= 0.5:
            scored.append((addr, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return [addr for addr, _ in scored[:max_results]]


def _compute_feature_df_median(
    feature_df: pd.DataFrame,
    town: str,
    flat_type: str,
    floor_area: float,
    remaining_lease: int = None,
    months_back: int = 24,
) -> tuple:
    """Compute median nominal transacted price from feature_df for a given flat profile.
    Returns (median_price, transaction_count).
    """
    if feature_df is None or feature_df.empty or "transacted_price" not in feature_df.columns:
        return None, 0

    f = feature_df.copy()

    # Restrict to recent transactions only
    if "month_index" in f.columns:
        # Apr 2026 = month_index 111
        current_mi = (2026 - 2017) * 12 + 3
        cutoff = current_mi - months_back
        mi_vals = pd.to_numeric(f["month_index"], errors="coerce")
        f = f[mi_vals >= cutoff]

    if town and "town" in f.columns:
        f = f[f["town"].str.upper().fillna("") == town.upper()]

    if flat_type and "flat_type" in f.columns:
        ft = flat_type.upper().replace("_", " ")
        f = f[f["flat_type"].str.upper().fillna("") == ft]

    if floor_area and "floor_area_sqm" in f.columns:
        area_vals = pd.to_numeric(f["floor_area_sqm"], errors="coerce")
        f = f[(area_vals >= floor_area - 20) & (area_vals <= floor_area + 20)]

    if remaining_lease and "remaining_lease" in f.columns:
        lease_vals = pd.to_numeric(f["remaining_lease"], errors="coerce")
        f = f[lease_vals >= remaining_lease]

    prices = pd.to_numeric(f["transacted_price"], errors="coerce").dropna()
    if prices.empty:
        return None, 0

    return float(prices.median()), len(prices)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _get_active_session_obj():
    active_session_id = st.session_state.get("active_session_id")
    for s in st.session_state.get("search_sessions", []):
        if s.get("session_id") == active_session_id:
            return s
    return None


def _save_extra_row(row_dict: dict):
    session = _get_active_session_obj()
    if session is None:
        st.warning("No active session found.")
        return False

    session.setdefault("extra_saved_rows", [])

    listing_id = str(row_dict.get("listing_id", "")).strip()
    address = str(row_dict.get("address", "")).strip().lower()

    liked_ids = [str(x).strip() for x in session.get("liked_ids", [])]
    if listing_id and listing_id in liked_ids:
        st.info("This flat is already saved.")
        return False

    for existing in session["extra_saved_rows"]:
        ex_id = str(existing.get("listing_id", "")).strip()
        ex_addr = str(existing.get("address", "")).strip().lower()

        if listing_id and ex_id == listing_id:
            st.info("This flat is already saved.")
            return False

        if address and ex_addr == address:
            st.info("This flat is already saved.")
            return False

    row_dict["session_id"] = session.get("session_id", "na")
    session["extra_saved_rows"].append(copy.deepcopy(row_dict))
    return True


def _is_row_already_saved(row_dict: dict) -> bool:
    session = _get_active_session_obj()
    if session is None:
        return False

    listing_id = str(row_dict.get("listing_id", "")).strip()
    address = str(row_dict.get("address", "")).strip().lower()

    liked_ids = [str(x).strip() for x in session.get("liked_ids", [])]
    if listing_id and listing_id in liked_ids:
        return True

    for existing in session.get("extra_saved_rows", []):
        ex_id = str(existing.get("listing_id", "")).strip()
        ex_addr = str(existing.get("address", "")).strip().lower()

        if listing_id and ex_id == listing_id:
            return True
        if address and ex_addr == address:
            return True

    return False


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_flat_snapshot(row: pd.Series):
    address = row.get("address", row.get("listing_id", "Selected flat"))
    predicted_price = row.get("predicted_price", np.nan)
    predicted_low = row.get("predicted_price_lower", np.nan)
    predicted_high = row.get("predicted_price_upper", np.nan)
    valuation_label = row.get("valuation_label", "")
    asking_vs_predicted_pct = row.get("asking_vs_predicted_pct", np.nan)

    if pd.isna(asking_vs_predicted_pct):
        asking_vs_predicted_pct = row.get("valuation_pct", np.nan)

    flat_type = _safe_str(row.get("flat_type", "—"))
    area = row.get("floor_area_sqm", np.nan)
    storey = row.get("storey_range", np.nan)

    meta_parts = [flat_type]
    if pd.notna(area):
        meta_parts.append(f"{float(area):.1f} sqm")
    if pd.notna(storey) and str(storey).strip() and str(storey).lower() != "nan":
        meta_parts.append(f"Storey {storey}")

    meta_text = " · ".join(meta_parts)

    tag_html = valuation_tag_html(valuation_label) if valuation_label else ""
    diff_text = f"{asking_vs_predicted_pct:+.1f}% vs model" if pd.notna(asking_vs_predicted_pct) else ""

    diff_html = ""
    if diff_text:
        diff_html = f"<span style='font-size:0.76rem;color:#9ca3af;'>{diff_text}</span>"

    range_html = ""
    if pd.notna(predicted_low) and pd.notna(predicted_high):
        range_html = (
            f"<div style='font-size:0.78rem;color:#6b7280;margin-top:4px;'>"
            f"95% range: {fmt_sgd(predicted_low)} to {fmt_sgd(predicted_high)}"
            f"</div>"
        )

    card_html = (
        f"<div style='border:1px solid #e4e7ed;background:rgba(255,255,255,0.96);"
        f"border-radius:16px;padding:16px;margin-bottom:10px;box-shadow:0 2px 10px rgba(0,0,0,0.04);'>"
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:16px;'>"
                f"<div style='min-width:0;flex:1;'>"
                    f"<div style='font-size:1rem;font-weight:800;color:#0f172a;line-height:1.35;'>"
                        f"{address}"
                    f"</div>"
                    f"<div style='font-size:0.82rem;color:#6b7280;margin-top:4px;'>"
                        f"{meta_text}"
                    f"</div>"
                f"</div>"
                f"<div style='text-align:right;flex-shrink:0;'>"
                    f"<div style='font-size:1.1rem;font-weight:800;color:#0f172a;'>"
                        f"{fmt_sgd(predicted_price) if pd.notna(predicted_price) else '—'}"
                    f"</div>"
                    f"<div style='font-size:0.82rem;color:#6b7280;margin-top:4px;'>"
                        f"Predicted price"
                    f"</div>"
                    f"{range_html}"
                f"</div>"
            f"</div>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-top:10px;flex-wrap:wrap;'>"
                f"{tag_html}"
                f"{diff_html}"
            f"</div>"
        f"</div>"
    )

    st.markdown(card_html, unsafe_allow_html=True)


def _enrich_explore_row(row_dict: dict, inputs):
    one_row_df = pd.DataFrame([row_dict])

    try:
        scored_df = compute_listing_scores(
            listings_df=one_row_df,
            budget=getattr(inputs, "budget", None) if inputs is not None else None,
            amenity_weights=getattr(inputs, "amenity_weights", {}) if inputs is not None else {},
            ranking_profile=getattr(inputs, "ranking_profile", "balanced") if inputs is not None else "balanced",
        )
        return scored_df.iloc[0].to_dict()
    except Exception:
        return row_dict


def _estimate_hypothetical_amenities(result: dict, listings_df: pd.DataFrame) -> dict:
    if listings_df is None or listings_df.empty:
        return {}

    df = listings_df.copy()

    if "town" not in df.columns or "flat_type" not in df.columns:
        return {}

    df["town"] = df["town"].fillna("").str.upper()
    df["flat_type"] = df["flat_type"].fillna("").str.upper()

    target_town = str(result["town"]).upper()
    target_flat_type = str(result["flat_type"]).upper()
    target_area = float(result["floor_area_sqm"])
    target_lease = float(result["remaining_lease"])
    target_storey = float(result["storey"])

    base = df[(df["town"] == target_town) & (df["flat_type"] == target_flat_type)].copy()
    if base.empty:
        base = df[df["town"] == target_town].copy()

    if base.empty:
        return {}

    if "floor_area_sqm" in base.columns:
        base["floor_area_sqm_num"] = pd.to_numeric(base["floor_area_sqm"], errors="coerce")
    else:
        base["floor_area_sqm_num"] = np.nan

    if "remaining_lease_years" in base.columns:
        base["lease_num"] = pd.to_numeric(base["remaining_lease_years"], errors="coerce")
    elif "remaining_lease" in base.columns:
        base["lease_num"] = pd.to_numeric(base["remaining_lease"], errors="coerce")
    else:
        base["lease_num"] = np.nan

    if "storey_midpoint" in base.columns:
        base["storey_num"] = pd.to_numeric(base["storey_midpoint"], errors="coerce")
    else:
        base["storey_num"] = np.nan

    matched = pd.DataFrame()

    bands = [
        (10, 3, 3),
        (15, 5, 5),
        (20, 8, 8),
        (30, 12, 12),
        (40, 20, 15),
        (60, 30, 20),
    ]

    for area_band, lease_band, storey_band in bands:
        candidate = base.copy()

        if candidate["floor_area_sqm_num"].notna().any():
            candidate = candidate[
                candidate["floor_area_sqm_num"].between(target_area - area_band, target_area + area_band)
            ]

        if candidate["lease_num"].notna().any():
            candidate = candidate[
                candidate["lease_num"].between(target_lease - lease_band, target_lease + lease_band)
            ]

        if candidate["storey_num"].notna().any():
            candidate = candidate[
                candidate["storey_num"].between(target_storey - storey_band, target_storey + storey_band)
            ]

        if len(candidate) >= 5:
            matched = candidate
            break

        if len(candidate) > len(matched):
            matched = candidate

    if len(matched) < 5:
        matched = base.copy()

    amenity_cols = [
        "train_1_dist_m",
        "bus_1_dist_m",
        "school_1_dist_m",
        "hawker_1_dist_m",
        "mall_1_dist_m",
        "polyclinic_1_dist_m",
        "supermarket_1_dist_m",
        "walk_train_min1",
        "walk_bus_min1",
        "walk_primary_school_min1",
        "walk_hawker_min1",
        "walk_mall_min1",
        "walk_polyclinic_min1",
        "walk_supermarket_min1",
    ]

    usable_cols = [c for c in amenity_cols if c in matched.columns]
    if not usable_cols:
        return {"similar_flats_used_for_amenities": len(matched)}

    numeric = matched[usable_cols].apply(pd.to_numeric, errors="coerce")

    medians = numeric.median(numeric_only=True).to_dict()
    medians = {k: v for k, v in medians.items() if pd.notna(v)}
    medians["similar_flats_used_for_amenities"] = len(matched)

    return medians


def _build_hypothetical_result_row(result: dict) -> dict:
    row = {
        "listing_id": (
            f"HYP-{result['town']}-{result['flat_type']}-"
            f"{int(result['floor_area_sqm'])}sqm-{int(result['remaining_lease'])}y-{int(result['storey'])}"
        ),
        "address": (
            f"Hypothetical flat · {result['town']} · {result['flat_type']} · "
            f"{int(result['floor_area_sqm'])} sqm · Storey {int(result['storey'])}"
        ),
        "town": result["town"],
        "flat_type": result["flat_type"],
        "floor_area_sqm": result["floor_area_sqm"],
        "remaining_lease_years": result["remaining_lease"],
        "remaining_lease": result["remaining_lease"],
        "storey_range": str(result["storey"]),
        "storey_midpoint": result["storey"],
        "predicted_price": result["predicted_price"],
        "predicted_price_lower": result.get("confidence_low"),
        "predicted_price_upper": result.get("confidence_high"),
        "comparison_source": "Explore",
        "is_hypothetical": True,
        "similar_flats_used_for_amenities": result.get("similar_flats_used_for_amenities"),
    }

    amenity_cols = [
        "train_1_dist_m",
        "bus_1_dist_m",
        "school_1_dist_m",
        "hawker_1_dist_m",
        "mall_1_dist_m",
        "polyclinic_1_dist_m",
        "supermarket_1_dist_m",
        "walk_train_min1",
        "walk_bus_min1",
        "walk_primary_school_min1",
        "walk_hawker_min1",
        "walk_mall_min1",
        "walk_polyclinic_min1",
        "walk_supermarket_min1",
    ]

    for col in amenity_cols:
        if col in result:
            row[col] = result[col]

    return row


# ---------------------------------------------------------------------------
# Editorial UI helpers
# ---------------------------------------------------------------------------

def _tab_intro(eyebrow: str, headline: str, body: str):
    """Magazine-style section intro — no borders, pure type hierarchy."""
    st.markdown(
        f"""
        <div style="padding:2px 0 22px;border-bottom:1px solid #f0f0f0;margin-bottom:28px;">
            <div style="font-size:0.67rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.14em;color:#FF4458;margin-bottom:9px;">{eyebrow}</div>
            <div style="font-size:1.12rem;font-weight:800;color:#1a1a2e;
                        line-height:1.3;margin-bottom:9px;">{headline}</div>
            <div style="font-size:0.87rem;color:#555577;line-height:1.68;
                        max-width:540px;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_rule(label: str):
    """Thin rule + uppercase label — editorial section break."""
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;margin:26px 0 14px;">
            <div style="font-size:0.63rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.14em;color:#c0c0cc;white-space:nowrap;">{label}</div>
            <div style="flex:1;height:1px;background:#f0f0f0;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _no_results_state(query: str):
    """Centered empty state with redirect nudge to Tab 2."""
    st.markdown(
        f"""
        <div style="text-align:center;padding:44px 20px 36px;">
            <div style="font-size:1.6rem;color:#e0e0e0;margin-bottom:14px;
                        letter-spacing:0.2em;">— ○ —</div>
            <div style="font-size:1rem;font-weight:800;color:#1a1a2e;margin-bottom:10px;">
                Nothing found for &ldquo;{query}&rdquo;
            </div>
            <div style="font-size:0.85rem;color:#555577;line-height:1.7;
                        max-width:360px;margin:0 auto 20px;">
                This block may not have been transacted since 2017 — or try spelling
                out the street name in full
                (e.g. <em>Ang Mo Kio Ave 10</em>, not <em>AMK Ave 10</em>).
            </div>
            <div style="display:inline-block;font-size:0.83rem;font-weight:700;
                        color:#FF4458;border-bottom:1.5px solid #FF4458;padding-bottom:1px;">
                Try the Explore a flat profile tab to estimate by town &amp; type →
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _address_result_header(address: str, town: str, flat_type: str, txn_count: int):
    """Bold editorial property header — feels like a report dateline."""
    txn_label = f"{txn_count} transaction{'s' if txn_count != 1 else ''} on record"
    st.markdown(
        f"""
        <div style="padding:18px 0 16px;border-bottom:2px solid #1a1a2e;margin-bottom:22px;">
            <div style="font-size:0.63rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.14em;color:#c0c0cc;margin-bottom:9px;">
                HDB Resale &nbsp;·&nbsp; Transaction Record
            </div>
            <div style="font-size:1.45rem;font-weight:800;color:#1a1a2e;
                        line-height:1.25;margin-bottom:10px;">{address}</div>
            <div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px;
                        font-size:0.82rem;color:#555577;">
                <span>{flat_type}</span>
                <span style="color:#d0d0d8;">·</span>
                <span>{town}</span>
                <span style="color:#d0d0d8;">·</span>
                <span style="color:#FF4458;font-weight:700;">{txn_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _price_estimate_card(predicted_price, confidence_low, confidence_high,
                         floor_area, flat_type, storey, lease):
    """Full-width light card — the hero result for Tab 1."""
    ci_text = ""
    if confidence_low and confidence_high:
        ci_text = f"{fmt_sgd(confidence_low)} &nbsp;—&nbsp; {fmt_sgd(confidence_high)}"
    detail = (
        f"{flat_type} &nbsp;·&nbsp; {float(floor_area):.0f} sqm"
        f" &nbsp;·&nbsp; Storey {storey}"
        f" &nbsp;·&nbsp; {int(lease)} yrs remaining lease"
    )
    year = pd.Timestamp.today().year
    st.markdown(
        f"""
        <div style="
            background:#f8fafc;
            border:1.5px solid #e2e8f0;
            border-top:3px solid #6366f1;
            border-radius:16px;
            padding:26px 28px;
            margin:18px 0 14px;
        ">
            <div style="font-size:0.63rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.14em;color:#6366f1;margin-bottom:14px;">
                Model estimate &nbsp;·&nbsp; {year}
            </div>
            <div style="font-size:2.4rem;font-weight:800;color:#0f172a;
                        line-height:1;letter-spacing:-0.01em;margin-bottom:6px;">
                {fmt_sgd(predicted_price) if pd.notna(predicted_price) else "—"}
            </div>
            <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:18px;">
                {"95% confidence range: " + ci_text if ci_text else "&nbsp;"}
            </div>
            <div style="border-top:1px solid #e2e8f0;padding-top:14px;">
                <div style="font-size:0.79rem;color:#475569;margin-bottom:8px;">{detail}</div>
                <div style="font-size:0.73rem;color:#94a3b8;line-height:1.6;">
                    <strong style="color:#64748b;">How this works:</strong>
                    We use this flat's actual floor area, storey, and remaining lease from its last transaction.
                    Three ML models (CatBoost, XGBoost, LightGBM) — trained on 228,000 HDB resale transactions
                    from 2017–2026 — produce a fair-value estimate, which is then scaled to today's
                    price level using the HDB Resale Price Index (currently at {pd.Timestamp.today().year}).
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _two_price_cards(model_price, conf_low, conf_high, median_price, median_count):
    """Light side-by-side comparison — model estimate vs market median."""
    ci_text = (
        f"{fmt_sgd(conf_low)} &nbsp;—&nbsp; {fmt_sgd(conf_high)}"
        if conf_low and conf_high else ""
    )
    median_display = fmt_sgd(median_price) if median_price else "—"
    count_text = f"Based on {median_count:,} transactions" if median_count else "Insufficient data"
    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0 14px;">
            <div style="background:#f8fafc;border:1.5px solid #e2e8f0;
                        border-top:3px solid #6366f1;border-radius:16px;padding:22px 24px;">
                <div style="font-size:0.63rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.14em;color:#6366f1;margin-bottom:12px;">
                    Model estimate
                </div>
                <div style="font-size:1.85rem;font-weight:800;color:#0f172a;
                            line-height:1;letter-spacing:-0.01em;margin-bottom:6px;">
                    {fmt_sgd(model_price)}
                </div>
                <div style="font-size:0.74rem;color:#94a3b8;margin-bottom:14px;">
                    {"95% range: " + ci_text if ci_text else "&nbsp;"}
                </div>
                <div style="border-top:1px solid #e2e8f0;padding-top:12px;
                            font-size:0.73rem;color:#64748b;line-height:1.6;">
                    <strong>What this is:</strong> Our ML model's fair-value estimate based on
                    your chosen profile. Location features (MRT proximity, schools, etc.) are
                    filled in from the typical values for that town and flat type.
                </div>
            </div>
            <div style="background:#f0fdf4;border:1.5px solid #bbf7d0;
                        border-top:3px solid #0d9488;border-radius:16px;padding:22px 24px;">
                <div style="font-size:0.63rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.14em;color:#0d9488;margin-bottom:12px;">
                    Median transacted &nbsp;·&nbsp; last 24 months
                </div>
                <div style="font-size:1.85rem;font-weight:800;color:#0f172a;
                            line-height:1;letter-spacing:-0.01em;margin-bottom:6px;">
                    {median_display}
                </div>
                <div style="font-size:0.74rem;color:#0d9488;margin-bottom:14px;">
                    {count_text}
                </div>
                <div style="border-top:1px solid #bbf7d0;padding-top:12px;
                            font-size:0.73rem;color:#374151;line-height:1.6;">
                    <strong>What this is:</strong> The median price buyers actually paid for
                    similar flats (same town, flat type, ±20 sqm, similar lease) over
                    the past 2 years — real market data, not a model.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 1: Look up a flat
# ---------------------------------------------------------------------------

def _render_flat_lookup(inputs, feature_df: pd.DataFrame):
    _tab_intro(
        eyebrow="Look up a flat",
        headline="Search by block and street",
        body=(
            "See every resale on record since 2017 for any HDB block — "
            "then get a model estimate of what it's worth today."
        ),
    )

    if feature_df is None or feature_df.empty:
        st.info("Historical transaction data is not available.")
        return

    search_query = st.text_input(
        "Search by block and street",
        placeholder="e.g. 406 Ang Mo Kio Ave 10",
        key="explore_address_search",
    )

    if not search_query.strip():
        return

    matched_addrs = _search_feature_df_addresses(search_query, feature_df)

    if not matched_addrs:
        _no_results_state(search_query.strip())
        return

    selected_addr = st.selectbox(
        "Matching addresses",
        options=matched_addrs,
        key="explore_feature_addr_dropdown",
        label_visibility="collapsed",
    )

    addr_df = feature_df[feature_df["display_address"] == selected_addr].copy()

    flat_types = sorted(addr_df["flat_type"].dropna().str.upper().unique().tolist())

    if len(flat_types) > 1:
        selected_flat_type = st.selectbox(
            "Flat type at this block",
            options=flat_types,
            key="explore_feature_flat_type",
        )
        addr_df = addr_df[addr_df["flat_type"].str.upper() == selected_flat_type]
    elif flat_types:
        selected_flat_type = flat_types[0]
    else:
        selected_flat_type = None

    if addr_df.empty:
        st.warning("No transactions found for this selection.")
        return

    if "txn_date" in addr_df.columns:
        addr_df = addr_df.sort_values("txn_date", ascending=False).reset_index(drop=True)

    town_label = str(addr_df.iloc[0].get("town", "")).title()
    _address_result_header(
        address=selected_addr,
        town=town_label,
        flat_type=str(selected_flat_type or ""),
        txn_count=len(addr_df),
    )

    _section_rule("Price History")

    def _fmt_lease(val):
        """61.333 → '61 yrs 4 mths'"""
        try:
            v = float(val)
            yrs = int(v)
            mths = round((v - yrs) * 12)
            if mths == 0:
                return f"{yrs} yrs"
            return f"{yrs} yrs {mths} mths"
        except Exception:
            return "—"

    def _fmt_storey_range(mid):
        """11.0 → '10 – 12'  (reconstructs HDB 3-floor band from midpoint)"""
        try:
            m = int(float(mid))
            return f"{m - 1:02d} – {m + 1:02d}"
        except Exception:
            return "—"

    disp_cols = {}
    if "txn_date" in addr_df.columns:
        disp_cols["txn_date"] = "Month"
    if "floor_area_sqm" in addr_df.columns:
        disp_cols["floor_area_sqm"] = "Floor area (sqm)"
    if "storey_midpoint" in addr_df.columns:
        disp_cols["storey_midpoint"] = "Storey range"
    if "remaining_lease" in addr_df.columns:
        disp_cols["remaining_lease"] = "Remaining lease"
    if "transacted_price" in addr_df.columns:
        disp_cols["transacted_price"] = "Transacted price"

    show_df = addr_df[list(disp_cols.keys())].rename(columns=disp_cols).head(30)

    if "Transacted price" in show_df.columns:
        show_df["Transacted price"] = show_df["Transacted price"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) and float(x) > 0 else "—"
        )
    if "Floor area (sqm)" in show_df.columns:
        show_df["Floor area (sqm)"] = show_df["Floor area (sqm)"].apply(
            lambda x: f"{float(x):.0f}" if pd.notna(x) else "—"
        )
    if "Remaining lease" in show_df.columns:
        show_df["Remaining lease"] = show_df["Remaining lease"].apply(_fmt_lease)
    if "Storey range" in show_df.columns:
        show_df["Storey range"] = show_df["Storey range"].apply(_fmt_storey_range)

    st.dataframe(show_df, use_container_width=True, hide_index=True)

    _section_rule("Current valuation")

    first = addr_df.iloc[0]
    town_for_pred = str(first.get("town", "")).upper()
    flat_type_for_pred = str(selected_flat_type or first.get("flat_type", "")).upper()

    # --- Floor area: dropdown of unique values from past transactions ---
    raw_areas = sorted(pd.to_numeric(addr_df["floor_area_sqm"], errors="coerce").dropna().unique())
    area_options = [f"{int(a)} sqm" for a in raw_areas] if raw_areas else ["90 sqm"]
    area_counts = addr_df["floor_area_sqm"].dropna().apply(lambda x: int(float(x))).value_counts()
    most_common_area = int(area_counts.index[0]) if not area_counts.empty else 90
    default_area_idx = next(
        (i for i, s in enumerate(area_options) if s == f"{most_common_area} sqm"), 0
    )
    selected_area_str = st.selectbox(
        "Floor area",
        options=area_options,
        index=default_area_idx,
        key="explore_lookup_area",
        help="Choose from floor areas recorded in past transactions at this block.",
    )
    selected_area = float(selected_area_str.replace(" sqm", ""))

    # --- Remaining lease: auto-calculated from lease_commence_date ---
    today = pd.Timestamp.today()
    lease_commence = None
    if "lease_commence_date" in addr_df.columns:
        lc_vals = pd.to_numeric(addr_df["lease_commence_date"], errors="coerce").dropna()
        if not lc_vals.empty:
            lease_commence = int(lc_vals.iloc[0])
    if lease_commence:
        expiry_year = lease_commence + 99
        remaining_today = max(0.0, expiry_year - today.year - (today.month - 1) / 12)
    else:
        latest_rl = float(first.get("remaining_lease", 70))
        latest_mi = int(first.get("month_index", 0))
        current_mi = (today.year - 2017) * 12 + (today.month - 1)
        remaining_today = max(0.0, latest_rl - (current_mi - latest_mi) / 12)
        expiry_year = today.year + remaining_today

    sale_mode = st.radio(
        "Estimate for",
        options=["Today", "A future sale date"],
        horizontal=True,
        key="explore_lookup_sale_mode",
    )
    if sale_mode == "A future sale date":
        max_yr = max(today.year + 1, int(expiry_year) - 1)
        sale_year = st.slider(
            "Estimated sale year",
            min_value=today.year,
            max_value=max_yr,
            value=min(today.year + 5, max_yr),
            key="explore_lookup_sale_year",
        )
        remaining_for_pred = max(1, int(expiry_year - sale_year))
        st.caption(
            f"At {sale_year}: ~{remaining_for_pred} yrs remaining lease "
            f"(lease expires {int(expiry_year)})"
        )
    else:
        remaining_for_pred = max(1, int(round(remaining_today)))
        lc_str = str(lease_commence) if lease_commence else "—"
        st.caption(
            f"Remaining lease today: ~{remaining_for_pred} yrs "
            f"(commenced {lc_str}, expires {int(expiry_year)})"
        )
        sale_year = today.year

    # --- Storey: number input, default = median storey at this address ---
    if "storey_midpoint" in addr_df.columns:
        med_storey = pd.to_numeric(addr_df["storey_midpoint"], errors="coerce").median()
        default_storey = max(1, int(round(med_storey))) if pd.notna(med_storey) else 5
    else:
        default_storey = 5
    selected_storey = st.number_input(
        "Storey",
        min_value=1,
        max_value=50,
        value=default_storey,
        step=1,
        key="explore_lookup_storey",
        help="Higher floors are priced higher. Defaults to the median storey for this block.",
    )

    result_key = (
        f"explore_lookup_{selected_addr}_{flat_type_for_pred}_"
        f"{int(selected_area)}_{selected_storey}_{remaining_for_pred}"
    )

    if st.button("Predict current value", type="primary", key="explore_lookup_predict_btn"):
        try:
            from backend_predictor_listings.price_predictor.notebooks.predict_hypothetical import (
                predict_hypothetical,
            )
        except Exception as e:
            st.error(f"Predictor unavailable: {e}")
            return

        with st.spinner("Running model — this may take a few seconds on first load…"):
            try:
                result = predict_hypothetical(
                    floor_area_sqm=selected_area,
                    town=town_for_pred,
                    flat_type=flat_type_for_pred,
                    remaining_lease_years=remaining_for_pred,
                    storey=selected_storey,
                )
                result["_lookup_addr"] = selected_addr
                result["_lookup_ft"] = selected_flat_type
                st.session_state[result_key] = result
            except Exception as e:
                st.error(f"Could not generate prediction: {e}")
                return

    result = st.session_state.get(result_key)
    if result:
        _price_estimate_card(
            predicted_price=result["predicted_price"],
            confidence_low=result.get("confidence_low"),
            confidence_high=result.get("confidence_high"),
            floor_area=selected_area,
            flat_type=flat_type_for_pred,
            storey=selected_storey,
            lease=remaining_for_pred,
        )

        hyp_row = _build_hypothetical_result_row(result)
        hyp_row["address"] = selected_addr
        hyp_row["town"] = town_for_pred
        hyp_row["flat_type"] = flat_type_for_pred

        if not _is_row_already_saved(hyp_row):
            if st.button("Save flat", key="explore_lookup_save_btn", use_container_width=True):
                hyp_row["comparison_source"] = "Explore"
                if _save_extra_row(hyp_row):
                    st.success("Flat saved to Saved tab.")
                    st.rerun()
        else:
            st.button(
                "Saved ♥",
                key="explore_lookup_saved_disabled",
                disabled=True,
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Tab 2: Explore a flat profile (predict + median side-by-side)
# ---------------------------------------------------------------------------

def _render_explore_flat_profile(inputs=None, listings_df: pd.DataFrame = None, feature_df: pd.DataFrame = None):
    _tab_intro(
        eyebrow="Explore a flat profile",
        headline="No specific block in mind?",
        body=(
            "Pick a town, flat type, size, and lease. We'll show our <strong>model's fair-value estimate</strong> "
            "alongside the <strong>median price buyers actually paid</strong> for similar flats in the past 2 years."
        ),
    )

    town_options = [
        "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH",
        "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG",
        "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
        "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL",
        "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES",
        "TOA PAYOH", "WOODLANDS", "YISHUN"
    ]
    flat_type_options = ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI_GENERATION"]

    inputs_town = getattr(inputs, "town", None) if inputs is not None else None
    inputs_flat_type = getattr(inputs, "flat_type", None) if inputs is not None else None
    inputs_floor_area = getattr(inputs, "floor_area_sqm", None) if inputs is not None else None
    inputs_lease = getattr(inputs, "remaining_lease_years", None) if inputs is not None else None

    default_town = str(inputs_town).upper() if inputs_town else "ANG MO KIO"
    if default_town not in town_options:
        default_town = "ANG MO KIO"
    default_flat_type = inputs_flat_type if inputs_flat_type in flat_type_options else "4 ROOM"
    default_area = float(inputs_floor_area) if inputs_floor_area else 90.0
    default_lease = int(inputs_lease) if inputs_lease else 70

    c1, c2 = st.columns(2)

    with c1:
        hyp_town = st.selectbox(
            "Town",
            options=town_options,
            index=town_options.index(default_town),
            key="explore_profile_town",
        )
        hyp_floor_area = st.slider(
            "Floor area (sqm)",
            min_value=20.0,
            max_value=300.0,
            value=float(default_area),
            step=1.0,
            key="explore_profile_area",
        )
        hyp_storey = st.number_input(
            "Storey",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            key="explore_profile_storey",
        )

    with c2:
        hyp_flat_type = st.selectbox(
            "Flat type",
            options=flat_type_options,
            index=flat_type_options.index(default_flat_type),
            key="explore_profile_flat_type",
        )
        hyp_remaining_lease = st.slider(
            "Remaining lease (years)",
            min_value=1,
            max_value=99,
            value=int(default_lease),
            step=1,
            key="explore_profile_lease",
        )

    estimate_clicked = st.button("Estimate price", type="primary", key="explore_profile_submit")

    if estimate_clicked:
        try:
            from backend_predictor_listings.price_predictor.notebooks.predict_hypothetical import (
                predict_hypothetical,
            )
        except Exception as e:
            st.error(f"Hypothetical predictor is unavailable: {e}")
            return

        with st.spinner("Running model — this may take a few seconds on first load…"):
            try:
                result = predict_hypothetical(
                    floor_area_sqm=hyp_floor_area,
                    town=hyp_town,
                    flat_type=hyp_flat_type,
                    remaining_lease_years=hyp_remaining_lease,
                    storey=hyp_storey,
                )
                amenity_estimates = _estimate_hypothetical_amenities(result, listings_df)
                result.update(amenity_estimates)

                # Compute median from feature_df for this profile
                fdf = feature_df if feature_df is not None else pd.DataFrame()
                median_price, median_count = _compute_feature_df_median(
                    fdf,
                    town=hyp_town,
                    flat_type=hyp_flat_type,
                    floor_area=hyp_floor_area,
                    remaining_lease=hyp_remaining_lease,
                )
                result["_profile_median"] = median_price
                result["_profile_median_count"] = median_count

                st.session_state["explore_profile_result"] = result
            except Exception as e:
                st.error(f"Could not generate estimate: {e}")
                return

    result = st.session_state.get("explore_profile_result")
    if not result:
        return

    _section_rule("Valuation")

    _two_price_cards(
        model_price=result["predicted_price"],
        conf_low=result.get("confidence_low"),
        conf_high=result.get("confidence_high"),
        median_price=result.get("_profile_median"),
        median_count=result.get("_profile_median_count", 0),
    )

    used_n = result.get("similar_flats_used_for_amenities")
    if used_n:
        st.caption(f"Amenity estimates based on {int(used_n)} similar listings.")

    hyp_row = _build_hypothetical_result_row(result)

    if _is_row_already_saved(hyp_row):
        st.button(
            "Saved ♥",
            key="explore_profile_saved_disabled",
            use_container_width=True,
            disabled=True,
        )
    else:
        if st.button(
            "Save hypothetical flat",
            key="explore_profile_save",
            use_container_width=True,
        ):
            if _save_extra_row(hyp_row):
                st.success("Hypothetical flat saved to Saved tab.")
                st.rerun()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_explore_page(inputs=None, listings_df: pd.DataFrame = None):
    st.markdown(
        """
        <div style="padding:0 0 28px;">
            <div style="font-size:0.67rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.14em;color:#FF4458;margin-bottom:10px;">
                Research tools
            </div>
            <div style="font-size:1.9rem;font-weight:800;color:#1a1a2e;
                        line-height:1.15;margin-bottom:9px;">
                Explore
            </div>
            <div style="font-size:0.9rem;color:#555577;line-height:1.65;max-width:500px;">
                Look up the complete transaction history for any HDB block,
                or model-price any flat profile to understand what the
                market should say it's worth.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if listings_df is None:
        listings_df = pd.DataFrame()

    feature_df = _load_feature_df_cached()

    tab1, tab2 = st.tabs([
        "🔎 Look up a flat",
        "🏠 Explore a flat profile",
    ])

    with tab1:
        _render_flat_lookup(inputs, feature_df)

    with tab2:
        _render_explore_flat_profile(inputs, listings_df, feature_df)
