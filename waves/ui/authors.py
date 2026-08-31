"""
File: authors.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Author profile cards for the WAVES Gradio application.

License: MIT License
"""

from dataclasses import dataclass
from html import escape
from typing import Final

import gradio as gr

from waves.config import (
    get_config_str_list,
    get_config_str_mapping,
)

AUTHOR_LINK_ORDER: Final[
    tuple[
        str,
        ...,
    ]
] = (
    "LINK_GITHUB",
    "LINK_WEBSITE",
    "LINK_WEB_OF_SCIENCE",
    "LINK_SCOPUS",
    "LINK_GOOGLE_SCHOLAR",
    "LINK_ORCID",
    "LINK_HUGGING_FACE",
    "LINK_TELEGRAM",
    "LINK_EMAIL",
)

AUTHOR_LINK_LABELS: Final[
    dict[
        str,
        str,
    ]
] = {
    "LINK_GITHUB": "GitHub",
    "LINK_WEBSITE": "Website",
    "LINK_WEB_OF_SCIENCE": "Web of Science",
    "LINK_SCOPUS": "Scopus",
    "LINK_GOOGLE_SCHOLAR": "Google Scholar",
    "LINK_ORCID": "ORCID",
    "LINK_HUGGING_FACE": "Hugging Face",
    "LINK_TELEGRAM": "Telegram",
    "LINK_EMAIL": "Email",
}


@dataclass(
    frozen=True,
    slots=True,
)
class AuthorLink:
    """One external author profile link."""

    label: str
    url: str


@dataclass(
    frozen=True,
    slots=True,
)
class AuthorProfile:
    """One WAVES author profile."""

    name: str
    initials: str
    links: tuple[
        AuthorLink,
        ...,
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class AuthorsTabComponents:
    """Components created inside the Authors tab."""

    content: gr.HTML


def _create_author_link_label(
    config_key: str,
    url: str,
) -> str:
    """Create the visible label for one author profile link."""

    label = AUTHOR_LINK_LABELS.get(
        config_key,
        config_key,
    )

    if config_key != "LINK_ORCID":
        return label

    orcid_identifier = (
        url.rstrip("/")
        .rsplit(
            "/",
            maxsplit=1,
        )[-1]
        .strip()
    )

    if not orcid_identifier:
        return label

    return f"{label} · {orcid_identifier}"


def _get_author_profile(
    author_key: str,
) -> AuthorProfile:
    """Load one author profile from config.toml."""

    section_name = f"Author_{author_key}"

    values = get_config_str_mapping(section_name)

    name = values.get(
        "NAME",
        "",
    ).strip()

    initials = values.get(
        "INITIALS",
        "",
    ).strip()

    if not name:
        msg = f"Author configuration section '{section_name}' must define NAME."

        raise ValueError(msg)

    if not initials:
        msg = f"Author configuration section '{section_name}' must define INITIALS."

        raise ValueError(msg)

    links: list[AuthorLink] = []

    for config_key in AUTHOR_LINK_ORDER:
        url = values.get(
            config_key,
            "",
        ).strip()

        if not url:
            continue

        links.append(
            AuthorLink(
                label=(
                    _create_author_link_label(
                        config_key,
                        url,
                    )
                ),
                url=url,
            )
        )

    return AuthorProfile(
        name=name,
        initials=initials,
        links=tuple(links),
    )


def get_author_profiles() -> tuple[
    AuthorProfile,
    ...,
]:
    """Return WAVES authors in configured display order."""

    author_keys = get_config_str_list(
        "Authors_ORDER",
        [],
    )

    if not author_keys:
        msg = "Authors_ORDER must contain at least one configured author."

        raise ValueError(msg)

    return tuple(_get_author_profile(author_key) for author_key in author_keys)


def _create_author_link_html(
    link: AuthorLink,
) -> str:
    """Render one author profile link."""

    url = escape(
        link.url,
        quote=True,
    )

    label = escape(link.label)

    is_external_browser_link = not link.url.startswith(
        (
            "mailto:",
            "tel:",
        )
    )

    external_attributes = ' target="_blank" rel="noopener noreferrer"' if is_external_browser_link else ""

    return (
        '<a class="author-profile-link" '
        f'href="{url}"'
        f"{external_attributes}"
        f' aria-label="{label}">'
        '<span class="author-profile-link-label">'
        f"{label}"
        "</span>"
        "</a>"
    )


def _create_author_card_html(
    profile: AuthorProfile,
    index: int,
) -> str:
    """Render one WAVES author card."""

    name = escape(profile.name)

    initials = escape(profile.initials)

    links_html = "".join(_create_author_link_html(link) for link in profile.links)

    links_block = (f'<div class="author-profile-links">{links_html}</div>') if links_html else ""

    return f"""
    <article
        class="author-profile-card"
        style="--author-card-index: {index};"
    >
        <div class="author-profile-header">
            <div
                class="author-profile-monogram"
                aria-hidden="true"
            >
                {initials}
            </div>

            <div class="author-profile-identity">
                <h3 class="author-profile-name">
                    {name}
                </h3>
            </div>
        </div>

        {links_block}
    </article>
    """


def create_authors_content_html() -> str:
    """Create the complete WAVES author-card grid."""

    profiles = get_author_profiles()

    cards = "".join(
        _create_author_card_html(
            profile,
            index,
        )
        for (
            index,
            profile,
        ) in enumerate(profiles)
    )

    return f"""
    <section class="authors-shell">
        <div class="authors-grid">
            {cards}
        </div>
    </section>
    """


def create_authors_tab(
    language_index: int = 0,
) -> AuthorsTabComponents:
    """Create the WAVES Authors tab."""

    del language_index

    content = gr.HTML(
        value=create_authors_content_html(),
        elem_id="authors-content",
        elem_classes="authors-content",
    )

    return AuthorsTabComponents(
        content=content,
    )
