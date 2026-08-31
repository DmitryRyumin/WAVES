"""
File: about.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: About-page content for the WAVES Gradio application.

License: MIT License
"""

from html import escape
from typing import Final

from waves.config import get_config_str
from waves.localization import get_localized_text

ABOUT_FEATURE_KEYS: Final[tuple[tuple[str, str], ...]] = (
    (
        "About_FEATURE_ENHANCE_TITLE",
        "About_FEATURE_ENHANCE_TEXT",
    ),
    (
        "About_FEATURE_COMPARE_TITLE",
        "About_FEATURE_COMPARE_TEXT",
    ),
    (
        "About_FEATURE_ROUTING_TITLE",
        "About_FEATURE_ROUTING_TEXT",
    ),
)

ABOUT_METRIC_KEYS: Final[tuple[tuple[str, str], ...]] = (
    (
        "About_EXPERTS_VALUE",
        "About_EXPERTS_LABEL",
    ),
    (
        "About_ROUTING_VALUE",
        "About_ROUTING_LABEL",
    ),
    (
        "About_ACTIVE_PARAMS_VALUE",
        "About_ACTIVE_PARAMS_LABEL",
    ),
    (
        "About_PESQ_VALUE",
        "About_PESQ_LABEL",
    ),
)


def _get_required_about_value(
    config_key: str,
) -> str:
    """Return a required scalar About configuration value."""

    value = get_config_str(
        config_key,
        "",
    ).strip()

    if value:
        return value

    msg = f"Missing required About configuration value: {config_key}."

    raise ValueError(msg)


def _create_feature_cards_html(
    language_index: int,
) -> str:
    """Create the three compact WAVES capability cards."""

    cards: list[str] = []

    for (
        index,
        (
            title_key,
            text_key,
        ),
    ) in enumerate(
        ABOUT_FEATURE_KEYS,
        start=1,
    ):
        title = escape(
            get_localized_text(
                title_key,
                language_index,
            )
        )

        text = escape(
            get_localized_text(
                text_key,
                language_index,
            )
        )

        cards.append(f"""
            <article
                class="about-feature-card"
                style="--about-card-index: {index};"
            >
                <span
                    class="about-feature-index"
                    aria-hidden="true"
                >
                    {index:02d}
                </span>

                <h3 class="about-feature-title">
                    {title}
                </h3>

                <p class="about-feature-text">
                    {text}
                </p>
            </article>
            """)

    return "".join(cards)


def _create_metric_cards_html(
    language_index: int,
) -> str:
    """Create compact model snapshot metrics."""

    metrics: list[str] = []

    for (
        value_key,
        label_key,
    ) in ABOUT_METRIC_KEYS:
        value = escape(_get_required_about_value(value_key))

        label = escape(
            get_localized_text(
                label_key,
                language_index,
            )
        )

        metrics.append(f"""
            <div class="about-metric">
                <strong class="about-metric-value">
                    {value}
                </strong>

                <span class="about-metric-label">
                    {label}
                </span>
            </div>
            """)

    return "".join(metrics)


def create_about_content_html(
    language_index: int,
) -> str:
    """Create the complete localized WAVES About content."""

    badge = escape(
        get_localized_text(
            "About_BADGE",
            language_index,
        )
    )

    full_name = escape(
        get_localized_text(
            "About_FULL_NAME",
            language_index,
        )
    )

    description = escape(
        get_localized_text(
            "About_HERO_DESCRIPTION",
            language_index,
        )
    )

    open_source_label = escape(
        get_localized_text(
            "About_OPEN_SOURCE",
            language_index,
        )
    )

    manuscript_status = escape(
        get_localized_text(
            "About_MANUSCRIPT_STATUS",
            language_index,
        )
    )

    flow_input = escape(
        get_localized_text(
            "About_FLOW_INPUT",
            language_index,
        )
    )

    flow_model = escape(
        get_localized_text(
            "About_FLOW_MODEL",
            language_index,
        )
    )

    flow_output = escape(
        get_localized_text(
            "About_FLOW_OUTPUT",
            language_index,
        )
    )

    snapshot_title = escape(
        get_localized_text(
            "About_SNAPSHOT_TITLE",
            language_index,
        )
    )

    snapshot_note = escape(
        get_localized_text(
            "About_SNAPSHOT_NOTE",
            language_index,
        )
    )

    benchmark_note = escape(
        get_localized_text(
            "About_BENCHMARK_NOTE",
            language_index,
        )
    )

    code_label = escape(
        get_localized_text(
            "About_CODE_LABEL",
            language_index,
        )
    )

    code_url = escape(
        _get_required_about_value("About_CODE_URL"),
        quote=True,
    )

    return f"""
    <section class="about-shell">
        <div class="about-hero">
            <div class="about-hero-copy">
                <div class="about-badge-row">
                    <span class="about-badge">
                        {badge}
                    </span>

                    <span class="about-status-chip">
                        <span
                            class="about-status-dot"
                            aria-hidden="true"
                        ></span>

                        {manuscript_status}
                    </span>
                </div>

                <h2 class="about-full-name">
                    {full_name}
                </h2>

                <p class="about-hero-description">
                    {description}
                </p>

                <div class="about-hero-actions">
                    <a
                        class="about-code-link"
                        href="{code_url}"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        <span>
                            {code_label}
                        </span>
                    </a>

                    <span class="about-open-source">
                        {open_source_label}
                    </span>
                </div>
            </div>

            <div
                class="about-flow"
                aria-label="
                    {flow_input}
                    →
                    {flow_model}
                    →
                    {flow_output}
                "
            >
                <span class="about-flow-node">
                    {flow_input}
                </span>

                <span
                    class="about-flow-arrow"
                    aria-hidden="true"
                >
                    →
                </span>

                <span class="about-flow-node is-model">
                    {flow_model}
                </span>

                <span
                    class="about-flow-arrow"
                    aria-hidden="true"
                >
                    →
                </span>

                <span class="about-flow-node">
                    {flow_output}
                </span>
            </div>
        </div>

        <div class="about-feature-grid">
            {_create_feature_cards_html(language_index)}
        </div>

        <section class="about-snapshot">
            <div class="about-snapshot-header">
                <div>
                    <span class="about-snapshot-eyebrow">
                        {snapshot_title}
                    </span>

                    <p class="about-snapshot-note">
                        {snapshot_note}
                    </p>
                </div>

                <span class="about-benchmark-note">
                    {benchmark_note}
                </span>
            </div>

            <div class="about-metric-grid">
                {_create_metric_cards_html(language_index)}
            </div>
        </section>
    </section>
    """
