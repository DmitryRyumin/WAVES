"""
File: requirements.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Runtime dependency information and version checks for the WAVES Requirements tab.

License: MIT License
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from importlib import metadata
import json
import platform
from threading import Lock
import time
import tomllib
from typing import Final, cast
from urllib.parse import quote
from urllib.request import Request, urlopen

import gradio as gr
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from waves.config import (
    PROJECT_ROOT,
    get_config_str,
)
from waves.localization import (
    get_language_index,
    get_localized_text,
)
from waves.logger import get_logger

LOGGER = get_logger(__name__)

PYPROJECT_PATH: Final = PROJECT_ROOT / "pyproject.toml"

PYPROJECT_GITHUB_URL: Final = "https://github.com/DmitryRyumin/WAVES/blob/main/pyproject.toml"

PYPI_HOME_URL: Final = "https://pypi.org/"

PYPI_JSON_URL_TEMPLATE: Final = "https://pypi.org/pypi/{package_name}/json"

PYPI_TIMEOUT_SECONDS: Final = 4.0
PYPI_CACHE_TTL_SECONDS: Final = 1800.0
PYPI_MAX_WORKERS: Final = 4

PACKAGE_DISPLAY_NAMES: Final[
    dict[
        str,
        str,
    ]
] = {
    "gradio": "Gradio",
    "librosa": "librosa",
    "numpy": "NumPy",
    "packaging": "packaging",
    "plotly": "Plotly",
    "psutil": "psutil",
    "pyyaml": "PyYAML",
    "safetensors": "safetensors",
    "torch": "PyTorch",
    "torchcodec": "TorchCodec",
}

PACKAGE_PURPOSE_KEYS: Final[
    dict[
        str,
        str,
    ]
] = {
    "gradio": ("Requirements_PURPOSE_GRADIO"),
    "librosa": ("Requirements_PURPOSE_LIBROSA"),
    "numpy": ("Requirements_PURPOSE_NUMPY"),
    "packaging": ("Requirements_PURPOSE_PACKAGING"),
    "plotly": ("Requirements_PURPOSE_PLOTLY"),
    "psutil": ("Requirements_PURPOSE_PSUTIL"),
    "pyyaml": ("Requirements_PURPOSE_PYYAML"),
    "safetensors": ("Requirements_PURPOSE_SAFETENSORS"),
    "torch": ("Requirements_PURPOSE_TORCH"),
    "torchcodec": ("Requirements_PURPOSE_TORCHCODEC"),
}

_STATUS_LOCALIZATION_KEYS: Final[
    dict[
        str,
        str,
    ]
] = {
    "current": ("Requirements_STATUS_CURRENT"),
    "update": ("Requirements_STATUS_UPDATE_AVAILABLE"),
    "constrained": ("Requirements_STATUS_CONSTRAINED"),
    "missing": ("Requirements_STATUS_MISSING"),
    "incompatible": ("Requirements_STATUS_INCOMPATIBLE"),
    "unavailable": ("Requirements_STATUS_UNAVAILABLE"),
}

_STATUS_CSS_CLASSES: Final[
    dict[
        str,
        str,
    ]
] = {
    "current": "is-current",
    "update": "is-update",
    "constrained": "is-constrained",
    "missing": "is-error",
    "incompatible": "is-error",
    "unavailable": "is-unavailable",
}


@dataclass(
    frozen=True,
    slots=True,
)
class RequirementsTabComponents:
    """Components created inside the Requirements tab."""

    title: gr.Markdown
    description: gr.Markdown
    content: gr.HTML


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimeDependency:
    """One direct runtime dependency declared by WAVES."""

    requirement: Requirement
    raw_requirement: str
    installed_version: str | None

    @property
    def canonical_name(
        self,
    ) -> str:
        """Return the normalized package name used for cache lookup."""

        return canonicalize_name(self.requirement.name)


@dataclass(
    frozen=True,
    slots=True,
)
class LatestVersionCacheEntry:
    """Cached latest-version result for one package."""

    version: str | None
    checked_at: float


_LATEST_VERSION_CACHE: dict[
    str,
    LatestVersionCacheEntry,
] = {}

_LATEST_VERSION_CACHE_LOCK = Lock()

_LAST_PYPI_CHECK_AT: datetime | None = None

_LAST_PYPI_CHECK_AT_LOCK = Lock()


def _mark_pypi_check_completed() -> None:
    """Record the completion time of the latest PyPI version check."""

    global _LAST_PYPI_CHECK_AT

    with _LAST_PYPI_CHECK_AT_LOCK:
        _LAST_PYPI_CHECK_AT = datetime.now(UTC)


def _get_last_pypi_check_at() -> datetime | None:
    """Return the latest completed PyPI check timestamp."""

    with _LAST_PYPI_CHECK_AT_LOCK:
        return _LAST_PYPI_CHECK_AT


def _read_project_metadata() -> tuple[
    str,
    list[str],
]:
    """Read Python and runtime requirements from pyproject.toml."""

    with PYPROJECT_PATH.open("rb") as file:
        data = tomllib.load(file)

    project_value = data.get("project")

    if not isinstance(
        project_value,
        dict,
    ):
        msg = "pyproject.toml must contain a [project] table."

        raise TypeError(msg)

    project = cast(
        dict[
            str,
            object,
        ],
        project_value,
    )

    requires_python_value = project.get(
        "requires-python",
        "",
    )

    requires_python = (
        requires_python_value
        if isinstance(
            requires_python_value,
            str,
        )
        else ""
    )

    dependencies_value = project.get(
        "dependencies",
        [],
    )

    if not isinstance(
        dependencies_value,
        list,
    ) or not all(
        isinstance(
            dependency,
            str,
        )
        for dependency in dependencies_value
    ):
        msg = "pyproject.toml project.dependencies must be a list of strings."

        raise TypeError(msg)

    return (
        requires_python,
        cast(
            list[str],
            dependencies_value,
        ),
    )


def _get_installed_version(
    package_name: str,
) -> str | None:
    """Return the installed distribution version."""

    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def get_runtime_dependencies() -> tuple[
    str,
    tuple[
        RuntimeDependency,
        ...,
    ],
]:
    """Resolve direct WAVES dependencies from the active environment."""

    (
        requires_python,
        raw_dependencies,
    ) = _read_project_metadata()

    dependencies: list[RuntimeDependency] = []

    for raw_requirement in raw_dependencies:
        try:
            requirement = Requirement(raw_requirement)
        except InvalidRequirement as error:
            msg = f"Invalid runtime requirement in pyproject.toml: {raw_requirement!r}."

            raise ValueError(msg) from error

        dependencies.append(
            RuntimeDependency(
                requirement=(requirement),
                raw_requirement=(raw_requirement),
                installed_version=(_get_installed_version(requirement.name)),
            )
        )

    return (
        requires_python,
        tuple(dependencies),
    )


def _get_cached_latest_version(
    package_name: str,
) -> tuple[
    bool,
    str | None,
]:
    """Return a fresh cached PyPI result when available."""

    canonical_name = canonicalize_name(package_name)

    now = time.monotonic()

    with _LATEST_VERSION_CACHE_LOCK:
        entry = _LATEST_VERSION_CACHE.get(canonical_name)

    if entry is None or (now - entry.checked_at) > PYPI_CACHE_TTL_SECONDS:
        return (
            False,
            None,
        )

    return (
        True,
        entry.version,
    )


def _set_cached_latest_version(
    package_name: str,
    version: str | None,
) -> None:
    """Cache one PyPI latest-version lookup."""

    canonical_name = canonicalize_name(package_name)

    with _LATEST_VERSION_CACHE_LOCK:
        _LATEST_VERSION_CACHE[canonical_name] = LatestVersionCacheEntry(
            version=version,
            checked_at=(time.monotonic()),
        )


def _fetch_latest_pypi_version(
    package_name: str,
) -> str | None:
    """Fetch the latest published version from PyPI."""

    package_url_name = quote(
        package_name,
        safe="",
    )

    app_version = get_config_str(
        "App_VERSION",
        "1.0.0",
    )

    request = Request(
        PYPI_JSON_URL_TEMPLATE.format(package_name=(package_url_name)),
        headers={
            "Accept": ("application/json"),
            "User-Agent": (f"WAVES/{app_version}"),
        },
    )

    try:
        with urlopen(
            request,
            timeout=(PYPI_TIMEOUT_SECONDS),
        ) as response:
            payload = cast(
                dict[
                    str,
                    object,
                ],
                json.loads(response.read().decode("utf-8")),
            )

    except (
        OSError,
        TimeoutError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        LOGGER.warning(
            "Could not retrieve the latest PyPI version for '%s': %s",
            package_name,
            error,
        )

        return None

    info_value = payload.get("info")

    if not isinstance(
        info_value,
        dict,
    ):
        return None

    info = cast(
        dict[
            str,
            object,
        ],
        info_value,
    )

    version_value = info.get("version")

    if (
        not isinstance(
            version_value,
            str,
        )
        or not version_value.strip()
    ):
        return None

    return version_value.strip()


def _get_package_display_name(
    dependency: RuntimeDependency,
) -> str:
    """Create a display name including configured extras."""

    display_name = PACKAGE_DISPLAY_NAMES.get(
        dependency.canonical_name,
        dependency.requirement.name,
    )

    if not (dependency.requirement.extras):
        return display_name

    extras = ", ".join(sorted(dependency.requirement.extras))

    return f"{display_name} [{extras}]"


def _get_dependency_purpose(
    dependency: RuntimeDependency,
    language_index: int,
) -> str:
    """Return the localized WAVES-specific dependency purpose."""

    localization_key = PACKAGE_PURPOSE_KEYS.get(dependency.canonical_name)

    if localization_key is None:
        return get_localized_text(
            "Requirements_PURPOSE_GENERIC",
            language_index,
        )

    return get_localized_text(
        localization_key,
        language_index,
    )


def _resolve_dependency_status(
    dependency: RuntimeDependency,
    latest_version: str | None,
    *,
    pending: bool,
) -> str | None:
    """Resolve the dependency status displayed in the table."""

    installed_version = dependency.installed_version

    if installed_version is None:
        return "missing"

    try:
        installed = Version(installed_version)
    except InvalidVersion:
        return "incompatible"

    specifier = dependency.requirement.specifier

    if specifier and not specifier.contains(
        installed,
        prereleases=True,
    ):
        return "incompatible"

    if pending:
        return None

    if latest_version is None:
        return "unavailable"

    try:
        latest = Version(latest_version)
    except InvalidVersion:
        return "unavailable"

    if installed >= latest:
        return "current"

    if not specifier or specifier.contains(
        latest,
        prereleases=True,
    ):
        return "update"

    return "constrained"


def _create_loader_html(
    language_index: int,
) -> str:
    """Create the neutral orbit loader used during PyPI lookup."""

    aria_label = escape(
        get_localized_text(
            "Requirements_CHECKING_LATEST",
            language_index,
        ),
        quote=True,
    )

    return f'<span class="requirements-version-loader" role="status" aria-label="{aria_label}"></span>'


def _create_version_badge_html(
    version: str,
) -> str:
    """Create a compact neutral version badge."""

    return f'<span class="requirements-version-badge">{escape(version)}</span>'


def _create_status_html(
    status: str | None,
    language_index: int,
) -> str:
    """Create a localized dependency status badge."""

    if status is None:
        return '<span class="requirements-status-pending" aria-hidden="true"></span>'

    localization_key = _STATUS_LOCALIZATION_KEYS[status]

    css_class = _STATUS_CSS_CLASSES[status]

    label = get_localized_text(
        localization_key,
        language_index,
    )

    return (
        "<span "
        'class="requirements-status-badge '
        f'{css_class}">'
        "<span "
        'class="requirements-status-dot" '
        'aria-hidden="true">'
        "</span>"
        f"{escape(label)}"
        "</span>"
    )


def _create_package_cell_html(
    dependency: RuntimeDependency,
) -> str:
    """Create the linked package-name cell."""

    package_name = _get_package_display_name(dependency)

    project_url = f"https://pypi.org/project/{quote(dependency.requirement.name, safe='')}/"

    return (
        "<a "
        'class="requirements-package-link" '
        f'href="{escape(project_url, quote=True)}" '
        'target="_blank" '
        'rel="noopener noreferrer">'
        f"{escape(package_name)}"
        "</a>"
    )


def _create_pyproject_link_html() -> str:
    """Create the GitHub pyproject.toml source link."""

    return (
        "<a "
        'class="requirements-source-link" '
        f'href="{escape(PYPROJECT_GITHUB_URL, quote=True)}" '
        'target="_blank" '
        'rel="noopener noreferrer">'
        "<span>pyproject.toml</span>"
        "<span "
        'class="requirements-source-link-arrow" '
        'aria-hidden="true">'
        "↗"
        "</span>"
        "</a>"
    )


def _create_pypi_link_html() -> str:
    """Create the primary PyPI source link."""

    return (
        "<a "
        'class="requirements-summary-primary-link" '
        f'href="{escape(PYPI_HOME_URL, quote=True)}" '
        'target="_blank" '
        'rel="noopener noreferrer">'
        "<span>PyPI</span>"
        "<span "
        'class="requirements-summary-primary-link-arrow" '
        'aria-hidden="true">'
        "↗"
        "</span>"
        "</a>"
    )


def _create_runtime_summary_html(
    requires_python: str,
    dependency_count: int,
    language_index: int,
) -> str:
    """Create compact runtime summary cards."""

    python_label = get_localized_text(
        "Requirements_SUMMARY_PYTHON",
        language_index,
    )

    dependencies_label = get_localized_text(
        "Requirements_SUMMARY_DEPENDENCIES",
        language_index,
    )

    source_label = get_localized_text(
        "Requirements_SUMMARY_SOURCE",
        language_index,
    )

    source_meta = get_localized_text(
        "Requirements_SUMMARY_SOURCE_META",
        language_index,
    )

    required_label = get_localized_text(
        "Requirements_REQUIRED",
        language_index,
    )

    installed_python = platform.python_version()

    pyproject_link = _create_pyproject_link_html()

    pypi_link = _create_pypi_link_html()

    return f"""
    <div class="requirements-summary-grid">
        <div class="requirements-summary-card">
            <span class="requirements-summary-label">
                {escape(python_label)}
            </span>

            <strong class="requirements-summary-value">
                {escape(installed_python)}
            </strong>

            <span class="requirements-summary-meta">
                {escape(required_label)}
                {escape(requires_python or "—")}
            </span>
        </div>

        <div class="requirements-summary-card">
            <span class="requirements-summary-label">
                {escape(dependencies_label)}
            </span>

            <strong class="requirements-summary-value">
                {dependency_count}
            </strong>

            <span class="requirements-summary-meta">
                {pyproject_link}
            </span>
        </div>

        <div class="requirements-summary-card">
            <span class="requirements-summary-label">
                {escape(source_label)}
            </span>

            <strong
                class="
                    requirements-summary-value
                    requirements-summary-source
                "
            >
                {pypi_link}
            </strong>

            <span class="requirements-summary-meta">
                {pyproject_link}

                <span class="requirements-source-separator">
                    +
                </span>

                <span>
                    {escape(source_meta)}
                </span>
            </span>
        </div>
    </div>
    """


def _create_last_checked_html(
    language_index: int,
) -> str:
    """Create the latest PyPI check timestamp."""

    checked_at = _get_last_pypi_check_at()

    if checked_at is None:
        return ""

    label = get_localized_text(
        "Requirements_LAST_CHECKED",
        language_index,
    )

    checked_time = checked_at.strftime("%H:%M")

    return (
        "<span "
        'class="requirements-last-checked">'
        f"{escape(label)}"
        '<span aria-hidden="true">'
        " · "
        "</span>"
        f"<strong>"
        f"{checked_time} UTC"
        f"</strong>"
        "</span>"
    )


def create_requirements_content_html(
    language_index: int,
    *,
    latest_versions: (
        dict[
            str,
            str | None,
        ]
        | None
    ) = None,
    pending_names: set[str] | None = None,
    resolved_name: str | None = None,
) -> str:
    """Render the complete Requirements table."""

    (
        requires_python,
        dependencies,
    ) = get_runtime_dependencies()

    resolved_latest_versions = latest_versions or {}

    resolved_pending_names = pending_names or set()

    package_label = get_localized_text(
        "Requirements_COLUMN_PACKAGE",
        language_index,
    )

    purpose_label = get_localized_text(
        "Requirements_COLUMN_PURPOSE",
        language_index,
    )

    requirement_label = get_localized_text(
        "Requirements_COLUMN_REQUIREMENT",
        language_index,
    )

    installed_label = get_localized_text(
        "Requirements_COLUMN_INSTALLED",
        language_index,
    )

    latest_label = get_localized_text(
        "Requirements_COLUMN_LATEST",
        language_index,
    )

    status_label = get_localized_text(
        "Requirements_COLUMN_STATUS",
        language_index,
    )

    unavailable_label = get_localized_text(
        "Requirements_VALUE_UNAVAILABLE",
        language_index,
    )

    rows: list[str] = []

    for dependency in dependencies:
        canonical_name = dependency.canonical_name

        pending = canonical_name in resolved_pending_names

        latest_version = resolved_latest_versions.get(canonical_name)

        status = _resolve_dependency_status(
            dependency,
            latest_version,
            pending=pending,
        )

        installed_html = (
            _create_version_badge_html(dependency.installed_version)
            if (dependency.installed_version is not None)
            else (f'<span class="requirements-value-unavailable">{escape(unavailable_label)}</span>')
        )

        if pending:
            latest_html = _create_loader_html(language_index)

        elif latest_version is not None:
            latest_html = _create_version_badge_html(latest_version)

        else:
            latest_html = f'<span class="requirements-value-unavailable">{escape(unavailable_label)}</span>'

        row_class = " is-resolved" if canonical_name == resolved_name else ""

        rows.append(
            """
            <tr class="requirements-dependency-row{row_class}">
                <td class="requirements-package-cell">
                    {package}
                </td>

                <td class="requirements-purpose-cell">
                    {purpose}
                </td>

                <td>
                    <code class="requirements-specifier">
                        {requirement}
                    </code>
                </td>

                <td class="requirements-version-cell">
                    {installed}
                </td>

                <td
                    class="
                        requirements-version-cell
                        requirements-latest-cell
                    "
                >
                    {latest}
                </td>

                <td class="requirements-status-cell">
                    {status}
                </td>
            </tr>
            """.format(
                row_class=(row_class),
                package=(_create_package_cell_html(dependency)),
                purpose=escape(
                    _get_dependency_purpose(
                        dependency,
                        language_index,
                    )
                ),
                requirement=escape(dependency.raw_requirement),
                installed=(installed_html),
                latest=(latest_html),
                status=(
                    _create_status_html(
                        status,
                        language_index,
                    )
                ),
            )
        )

    footer = get_localized_text(
        "Requirements_FOOTER",
        language_index,
    )

    last_checked_html = _create_last_checked_html(language_index) if not resolved_pending_names else ""

    return f"""
    <section class="requirements-shell">
        {
        _create_runtime_summary_html(
            requires_python,
            len(dependencies),
            language_index,
        )
    }

        <div class="requirements-table-wrapper">
            <table class="requirements-table">
                <thead>
                    <tr>
                        <th>
                            {escape(package_label)}
                        </th>
                        <th>
                            {escape(purpose_label)}
                        </th>
                        <th>
                            {escape(requirement_label)}
                        </th>
                        <th>
                            {escape(installed_label)}
                        </th>
                        <th>
                            {escape(latest_label)}
                        </th>
                        <th>
                            {escape(status_label)}
                        </th>
                    </tr>
                </thead>

                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>

        <div class="requirements-footer-row">
            <p class="requirements-footer">
                {escape(footer)}
            </p>

            {last_checked_html}
        </div>
    </section>
    """


def create_initial_requirements_content_html(
    language_index: int,
) -> str:
    """Create cached Requirements content without network access."""

    (
        _,
        dependencies,
    ) = get_runtime_dependencies()

    latest_versions: dict[
        str,
        str | None,
    ] = {}

    pending_names: set[str] = set()

    for dependency in dependencies:
        (
            cache_hit,
            latest_version,
        ) = _get_cached_latest_version(dependency.requirement.name)

        if cache_hit:
            latest_versions[dependency.canonical_name] = latest_version
        else:
            pending_names.add(dependency.canonical_name)

    return create_requirements_content_html(
        language_index,
        latest_versions=(latest_versions),
        pending_names=(pending_names),
    )


def stream_requirements_content_html(
    language: str,
) -> Iterator[str]:
    """Stream PyPI version results into the Requirements table."""

    language_index = get_language_index(language)

    (
        _,
        dependencies,
    ) = get_runtime_dependencies()

    latest_versions: dict[
        str,
        str | None,
    ] = {}

    pending_names: set[str] = set()

    dependencies_to_check: dict[
        str,
        RuntimeDependency,
    ] = {}

    for dependency in dependencies:
        (
            cache_hit,
            latest_version,
        ) = _get_cached_latest_version(dependency.requirement.name)

        if cache_hit:
            latest_versions[dependency.canonical_name] = latest_version

            continue

        pending_names.add(dependency.canonical_name)

        dependencies_to_check[dependency.canonical_name] = dependency

    yield (
        create_requirements_content_html(
            language_index,
            latest_versions=(latest_versions),
            pending_names=(pending_names),
        )
    )

    if not dependencies_to_check:
        return

    max_workers = min(
        PYPI_MAX_WORKERS,
        len(dependencies_to_check),
    )

    with ThreadPoolExecutor(max_workers=(max_workers)) as executor:
        futures = {
            executor.submit(
                _fetch_latest_pypi_version,
                dependency.requirement.name,
            ): dependency
            for dependency in dependencies_to_check.values()
        }

        for future in as_completed(futures):
            dependency = futures[future]

            try:
                latest_version = future.result()
            except Exception:
                LOGGER.exception(
                    "Unexpected error while checking the latest version of '%s'.",
                    dependency.requirement.name,
                )

                latest_version = None

            _set_cached_latest_version(
                dependency.requirement.name,
                latest_version,
            )

            latest_versions[dependency.canonical_name] = latest_version

            pending_names.discard(dependency.canonical_name)

            _mark_pypi_check_completed()

            yield (
                create_requirements_content_html(
                    language_index,
                    latest_versions=(latest_versions),
                    pending_names=(pending_names),
                    resolved_name=(dependency.canonical_name),
                )
            )


def create_requirements_tab(
    language_index: int = 0,
) -> RequirementsTabComponents:
    """Create the WAVES Requirements tab."""

    title = gr.Markdown(
        (
            f"### "
            f"{
                get_localized_text(
                    'Texts_REQUIREMENTS_TITLE',
                    language_index,
                )
            }"
        ),
        elem_classes=("requirements-title"),
    )

    description = gr.Markdown(
        get_localized_text(
            "Requirements_DESCRIPTION",
            language_index,
        ),
        elem_classes=("requirements-description"),
    )

    content = gr.HTML(
        value=(create_initial_requirements_content_html(language_index)),
        elem_id="requirements-content",
        elem_classes=("requirements-content"),
    )

    return RequirementsTabComponents(
        title=title,
        description=description,
        content=content,
    )
