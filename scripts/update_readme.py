from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import json
from pathlib import Path
import re
import sys
import tomllib
from typing import Any
from urllib.parse import quote

type Config = dict[str, Any]
type DependencyMap = dict[str, str]


ROOT = Path(__file__).resolve().parents[1]

PYPROJECT_PATH = ROOT / "pyproject.toml"
PACKAGE_JSON_PATH = ROOT / "package.json"
PYTHON_VERSION_PATH = ROOT / ".python-version"
NODE_VERSION_PATH = ROOT / ".nvmrc"
DOCKERFILE_PATH = ROOT / "Dockerfile"
README_PATH = ROOT / "README.md"

README_BLOCK_START = "<!-- BEGIN PROJECT BADGES -->"
README_BLOCK_END = "<!-- END PROJECT BADGES -->"

PRETTIER_IGNORE_START = "<!-- prettier-ignore-start -->"
PRETTIER_IGNORE_END = "<!-- prettier-ignore-end -->"

GITHUB_OWNER = "DmitryRyumin"
HUGGING_FACE_OWNER = GITHUB_OWNER

PYTHON_DEPENDENCY_PATTERN = re.compile(
    r"^([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*(.*)$",
)

MINIMUM_VERSION_PATTERN = re.compile(
    r">=\s*([0-9]+(?:\.[0-9A-Za-z]+)*)",
)

EXACT_VERSION_PATTERN = re.compile(
    r"==\s*([0-9]+(?:\.[0-9A-Za-z]+)*)",
)

COMPATIBLE_VERSION_PATTERN = re.compile(
    r"~=\s*([0-9]+(?:\.[0-9A-Za-z]+)*)",
)

NPM_PACKAGE_MANAGER_PATTERN = re.compile(r"npm@(.+)")


@dataclass(
    frozen=True,
    slots=True,
)
class ProjectMetadata:
    """Canonical project naming and repository metadata."""

    package_name: str
    display_name: str
    github_repository: str
    hugging_face_space: str
    hugging_face_app_url: str

    @classmethod
    def from_pyproject(
        cls,
        pyproject: Config,
    ) -> ProjectMetadata:
        """Create project metadata from pyproject.toml."""

        package_name = _get_project_name(pyproject)

        display_name = package_name.upper()

        github_repository = f"{GITHUB_OWNER}/{display_name}"

        hugging_face_space = f"{HUGGING_FACE_OWNER}/{display_name}"

        app_slug = f"{HUGGING_FACE_OWNER}-{package_name}".lower().replace("_", "-")

        return cls(
            package_name=package_name,
            display_name=display_name,
            github_repository=github_repository,
            hugging_face_space=hugging_face_space,
            hugging_face_app_url=f"https://{app_slug}.hf.space",
        )


@dataclass(
    frozen=True,
    slots=True,
)
class Badge:
    """Description of a static Shields.io badge."""

    label: str
    message: str
    color: str
    alt: str
    logo: str | None = None
    logo_color: str | None = None

    def render(self) -> str:
        """Render the badge as an HTML image."""

        return _image(
            _badge_url(
                self.label,
                self.message,
                self.color,
                logo=self.logo,
                logo_color=self.logo_color,
            ),
            self.alt,
        )


def _load_toml(
    path: Path,
) -> Config:
    """Load a TOML document."""

    with path.open("rb") as file:
        return tomllib.load(file)


def _load_json(
    path: Path,
) -> Config:
    """Load a JSON object."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        msg = f"Expected a JSON object in {path}."
        raise TypeError(msg)

    return data


def _read_text_value(
    path: Path,
) -> str:
    """Read a required one-line text value."""

    value = path.read_text(encoding="utf-8").strip()

    if not value:
        msg = f"Expected a non-empty value in {path}."
        raise ValueError(msg)

    return value


def _get_project_table(
    pyproject: Config,
) -> Config:
    """Return the required [project] table."""

    project = pyproject.get("project")

    if not isinstance(project, dict):
        msg = "Missing [project] table in pyproject.toml."
        raise ValueError(msg)

    return project


def _get_required_string(
    mapping: Config,
    key: str,
    *,
    source: str,
) -> str:
    """Return a required non-empty string from a configuration mapping."""

    value = mapping.get(key)

    if not isinstance(value, str) or not value.strip():
        msg = f"Missing {key!r} in {source}."
        raise ValueError(msg)

    return value.strip()


def _get_project_name(
    pyproject: Config,
) -> str:
    """Return the canonical project package name."""

    return _get_required_string(
        _get_project_table(pyproject),
        "name",
        source="pyproject.toml [project]",
    )


def _get_project_version(
    pyproject: Config,
) -> str:
    """Return the project version."""

    return _get_required_string(
        _get_project_table(pyproject),
        "version",
        source="pyproject.toml [project]",
    )


def _normalize_package_name(
    name: str,
) -> str:
    """Normalize a Python package name according to PEP-style matching."""

    return re.sub(
        r"[-_.]+",
        "-",
        name.strip().lower(),
    )


def _parse_python_dependency(
    dependency: str,
) -> tuple[str, str]:
    """Split a Python dependency into normalized name and version specifier."""

    match = PYTHON_DEPENDENCY_PATTERN.match(dependency.strip())

    if match is None:
        return "", ""

    name = _normalize_package_name(match.group(1))

    specification = match.group(2).strip()

    if ";" in specification:
        specification = specification.partition(";")[0].strip()

    return name, specification


def _add_python_dependencies(
    destination: DependencyMap,
    values: object,
) -> None:
    """Add string dependency specifications from a sequence."""

    if not isinstance(values, list):
        return

    for dependency in values:
        if not isinstance(dependency, str):
            continue

        name, specification = _parse_python_dependency(dependency)

        if name:
            destination[name] = specification


def _collect_python_dependencies(
    pyproject: Config,
) -> DependencyMap:
    """Collect runtime, optional, and development Python dependencies."""

    dependencies: DependencyMap = {}

    project = pyproject.get("project")

    if isinstance(project, dict):
        _add_python_dependencies(
            dependencies,
            project.get("dependencies"),
        )

        optional_dependencies = project.get("optional-dependencies")

        if isinstance(optional_dependencies, dict):
            for group in optional_dependencies.values():
                _add_python_dependencies(
                    dependencies,
                    group,
                )

    dependency_groups = pyproject.get("dependency-groups")

    if isinstance(dependency_groups, dict):
        for group in dependency_groups.values():
            _add_python_dependencies(
                dependencies,
                group,
            )

    return dependencies


def _minimum_version(
    specification: str,
) -> tuple[str, bool] | None:
    """Return a displayable version and whether it represents a minimum."""

    for pattern, minimum in (
        (MINIMUM_VERSION_PATTERN, True),
        (EXACT_VERSION_PATTERN, False),
        (COMPATIBLE_VERSION_PATTERN, True),
    ):
        match = pattern.search(specification)

        if match is not None:
            return match.group(1), minimum

    return None


def _dependency_badge_version(
    specification: str,
) -> str | None:
    """Convert a dependency specification into a compact badge version."""

    result = _minimum_version(specification)

    if result is None:
        return None

    version, minimum = result

    return f"{version}+" if minimum else version


def _strip_version_prefix(
    value: str,
) -> str:
    """Remove a conventional leading v from a version."""

    return value.removeprefix("v").strip()


def _get_uv_version(
    pyproject: Config,
) -> str | None:
    """Return the configured uv version requirement."""

    tool = pyproject.get("tool")

    if not isinstance(tool, dict):
        return None

    uv = tool.get("uv")

    if not isinstance(uv, dict):
        return None

    value = uv.get("required-version")

    if not isinstance(value, str):
        value = uv.get("required_version")

    if not isinstance(value, str):
        return None

    return _dependency_badge_version(value.strip())


def _get_engine_version(
    package_json: Config,
    engine_name: str,
) -> str | None:
    """Return a displayable package.json engine version."""

    engines = package_json.get("engines")

    if not isinstance(engines, dict):
        return None

    specification = engines.get(engine_name)

    if not isinstance(specification, str):
        return None

    return _dependency_badge_version(specification.strip())


def _get_npm_version(
    package_json: Config,
) -> str | None:
    """Return the canonical npm version."""

    package_manager = package_json.get("packageManager")

    if isinstance(package_manager, str):
        match = NPM_PACKAGE_MANAGER_PATTERN.fullmatch(
            package_manager.strip(),
        )

        if match is not None:
            return match.group(1).strip()

    return _get_engine_version(
        package_json,
        "npm",
    )


def _get_node_version(
    package_json: Config,
) -> str:
    """Return the canonical Node.js version."""

    if NODE_VERSION_PATH.is_file():
        return _strip_version_prefix(
            _read_text_value(NODE_VERSION_PATH),
        )

    version = _get_engine_version(
        package_json,
        "node",
    )

    if version is not None:
        return version

    msg = "Node.js version was not found in .nvmrc or package.json."
    raise ValueError(msg)


def _get_javascript_dependency_version(
    package_json: Config,
    package_name: str,
) -> str | None:
    """Return a JavaScript dependency version from package.json."""

    for section_name in (
        "dependencies",
        "devDependencies",
        "optionalDependencies",
    ):
        section = package_json.get(section_name)

        if not isinstance(section, dict):
            continue

        value = section.get(package_name)

        if not isinstance(value, str):
            continue

        return value.strip().lstrip("^~")

    return None


def _encode_badge_part(
    value: str,
) -> str:
    """Encode text for the Shields.io static badge path."""

    return quote(
        value,
        safe="",
    ).replace("-", "--")


def _badge_url(
    label: str,
    message: str,
    color: str,
    *,
    logo: str | None = None,
    logo_color: str | None = None,
) -> str:
    """Build a static Shields.io badge URL."""

    url = (
        "https://img.shields.io/badge/"
        f"{_encode_badge_part(label)}-"
        f"{_encode_badge_part(message)}-"
        f"{color}"
        "?style=flat-square"
    )

    if logo is not None:
        url += f"&logo={quote(logo, safe='')}"

    if logo_color is not None:
        url += f"&logoColor={quote(logo_color, safe='')}"

    return url


def _image(
    source: str,
    alt: str,
) -> str:
    """Build an HTML image element."""

    return f'<img src="{html.escape(source, quote=True)}" alt="{html.escape(alt, quote=True)}">'


def _link(
    href: str,
    content: str,
) -> str:
    """Build an HTML anchor element."""

    return f'<a href="{html.escape(href, quote=True)}">{content}</a>'


def _linked_badge(
    href: str,
    badge: Badge,
) -> str:
    """Render a badge wrapped in a link."""

    return _link(
        href,
        badge.render(),
    )


def _python_dependency_badge(
    dependencies: DependencyMap,
    package_name: str,
    *,
    label: str,
    color: str,
    logo: str | None = None,
    logo_color: str | None = None,
) -> str | None:
    """Build a configured Python dependency badge."""

    specification = dependencies.get(
        _normalize_package_name(package_name),
    )

    if specification is None:
        return None

    version = _dependency_badge_version(specification)

    if version is None:
        return None

    return Badge(
        label=label,
        message=version,
        color=color,
        alt=label,
        logo=logo,
        logo_color=logo_color,
    ).render()


def _javascript_dependency_badge(
    package_json: Config,
    package_name: str,
    *,
    label: str,
    color: str,
    logo: str | None = None,
    logo_color: str | None = None,
) -> str | None:
    """Build a configured JavaScript dependency badge."""

    version = _get_javascript_dependency_version(
        package_json,
        package_name,
    )

    if version is None:
        return None

    return Badge(
        label=label,
        message=version,
        color=color,
        alt=label,
        logo=logo,
        logo_color=logo_color,
    ).render()


def _existing_badges(
    badges: tuple[str | None, ...],
) -> list[str]:
    """Filter optional badge values."""

    return [badge for badge in badges if badge is not None]


def _build_project_badges(
    pyproject: Config,
    metadata: ProjectMetadata,
) -> list[str]:
    """Build project metadata badges."""

    python_version = _strip_version_prefix(
        _read_text_value(PYTHON_VERSION_PATH),
    )

    return [
        Badge(
            label="version",
            message=_get_project_version(pyproject),
            color="2F81F7",
            alt="Version",
        ).render(),
        _link(
            "LICENSE",
            _image(
                (f"https://img.shields.io/github/license/{metadata.github_repository}?style=flat-square"),
                "License",
            ),
        ),
        Badge(
            label="Python",
            message=python_version,
            color="3776AB",
            alt="Python",
            logo="python",
            logo_color="white",
        ).render(),
    ]


def _build_runtime_badges(
    pyproject: Config,
    package_json: Config,
    dependencies: DependencyMap,
) -> list[str]:
    """Build runtime technology badges."""

    badges = _existing_badges(
        (
            _python_dependency_badge(
                dependencies,
                "torch",
                label="PyTorch",
                color="EE4C2C",
                logo="pytorch",
                logo_color="white",
            ),
            _python_dependency_badge(
                dependencies,
                "torchcodec",
                label="TorchCodec",
                color="EE4C2C",
            ),
            _python_dependency_badge(
                dependencies,
                "gradio",
                label="Gradio",
                color="F97316",
            ),
        )
    )

    if DOCKERFILE_PATH.is_file():
        badges.append(
            Badge(
                label="Docker",
                message="enabled",
                color="2496ED",
                alt="Docker",
                logo="docker",
                logo_color="white",
            ).render()
        )

    uv_version = _get_uv_version(pyproject)

    if uv_version is not None:
        badges.append(
            Badge(
                label="uv",
                message=uv_version,
                color="261230",
                alt="uv",
            ).render()
        )

    badges.append(
        Badge(
            label="Node.js",
            message=_get_node_version(package_json),
            color="5FA04E",
            alt="Node.js",
            logo="nodedotjs",
            logo_color="white",
        ).render()
    )

    npm_version = _get_npm_version(package_json)

    if npm_version is not None:
        badges.append(
            Badge(
                label="npm",
                message=npm_version,
                color="CB3837",
                alt="npm",
                logo="npm",
                logo_color="white",
            ).render()
        )

    return badges


def _build_code_quality_badges(
    package_json: Config,
    dependencies: DependencyMap,
) -> list[str]:
    """Build code-quality tool badges."""

    return _existing_badges(
        (
            _python_dependency_badge(
                dependencies,
                "ruff",
                label="Ruff",
                color="D7FF64",
            ),
            _python_dependency_badge(
                dependencies,
                "mypy",
                label="mypy",
                color="2A6DB2",
            ),
            _python_dependency_badge(
                dependencies,
                "deptry",
                label="deptry",
                color="6B7280",
            ),
            _python_dependency_badge(
                dependencies,
                "pre-commit",
                label="pre-commit",
                color="FAB040",
            ),
            _javascript_dependency_badge(
                package_json,
                "stylelint",
                label="Stylelint",
                color="263238",
            ),
            _javascript_dependency_badge(
                package_json,
                "prettier",
                label="Prettier",
                color="F7B93E",
                logo="prettier",
                logo_color="black",
            ),
        )
    )


def _github_badge_url(
    metric: str,
    metadata: ProjectMetadata,
) -> str:
    """Build a dynamic GitHub Shields.io badge URL."""

    return f"https://img.shields.io/github/{metric}/{metadata.github_repository}?style=flat-square"


def _github_url(
    metadata: ProjectMetadata,
    suffix: str = "",
) -> str:
    """Build a GitHub repository URL."""

    return f"https://github.com/{metadata.github_repository}{suffix}"


def _build_repository_badges(
    metadata: ProjectMetadata,
) -> list[str]:
    """Build live GitHub repository badges."""

    return [
        _image(
            _github_badge_url(
                "repo-size",
                metadata,
            ),
            "Repository size",
        ),
        _image(
            _github_badge_url(
                "last-commit",
                metadata,
            ),
            "Last commit",
        ),
        _link(
            _github_url(
                metadata,
                "/graphs/contributors",
            ),
            _image(
                _github_badge_url(
                    "contributors",
                    metadata,
                ),
                "Contributors",
            ),
        ),
    ]


def _build_community_badges(
    metadata: ProjectMetadata,
) -> list[str]:
    """Build live GitHub community badges."""

    return [
        _link(
            _github_url(
                metadata,
                "/stargazers",
            ),
            _image(
                _github_badge_url(
                    "stars",
                    metadata,
                ),
                "Stars",
            ),
        ),
        _link(
            _github_url(
                metadata,
                "/forks",
            ),
            _image(
                _github_badge_url(
                    "forks",
                    metadata,
                ),
                "Forks",
            ),
        ),
        _link(
            _github_url(
                metadata,
                "/issues",
            ),
            _image(
                _github_badge_url(
                    "issues",
                    metadata,
                ),
                "Issues",
            ),
        ),
    ]


def _build_application_badges(
    metadata: ProjectMetadata,
) -> list[str]:
    """Build application badges."""

    return [
        _linked_badge(
            metadata.hugging_face_app_url,
            Badge(
                label="🤗 Space",
                message=metadata.display_name,
                color="FFD21E",
                alt=(f"{metadata.display_name} on Hugging Face"),
            ),
        )
    ]


def _table_row(
    title: str,
    badges: list[str],
) -> list[str]:
    """Build one README metadata table row."""

    return [
        "  <tr>",
        f"    <td><strong>{html.escape(title)}</strong></td>",
        f"    <td>{' '.join(badges)}</td>",
        "  </tr>",
    ]


def _build_badge_block() -> str:
    """Build the generated README metadata table."""

    pyproject = _load_toml(PYPROJECT_PATH)

    package_json = _load_json(PACKAGE_JSON_PATH)

    metadata = ProjectMetadata.from_pyproject(pyproject)

    dependencies = _collect_python_dependencies(pyproject)

    rows = [
        PRETTIER_IGNORE_START,
        '<table align="center">',
    ]

    sections = (
        (
            "Project",
            _build_project_badges(
                pyproject,
                metadata,
            ),
        ),
        (
            "Runtime",
            _build_runtime_badges(
                pyproject,
                package_json,
                dependencies,
            ),
        ),
        (
            "Code Quality",
            _build_code_quality_badges(
                package_json,
                dependencies,
            ),
        ),
        (
            "Repository",
            _build_repository_badges(metadata),
        ),
        (
            "Community",
            _build_community_badges(metadata),
        ),
        (
            "Application",
            _build_application_badges(metadata),
        ),
    )

    for title, badges in sections:
        if badges:
            rows.extend(
                _table_row(
                    title,
                    badges,
                )
            )

    rows.extend(
        (
            "</table>",
            PRETTIER_IGNORE_END,
        )
    )

    return "\n".join(rows)


def _render_readme(
    source: str,
    generated_block: str,
) -> str:
    """Replace only the generated README metadata region."""

    start_index = source.find(README_BLOCK_START)

    if start_index < 0:
        msg = f"README marker not found: {README_BLOCK_START}"
        raise ValueError(msg)

    content_start = start_index + len(README_BLOCK_START)

    end_index = source.find(
        README_BLOCK_END,
        content_start,
    )

    if end_index < 0:
        msg = f"README marker not found: {README_BLOCK_END}"
        raise ValueError(msg)

    return source[:content_start] + "\n\n" + generated_block + "\n\n" + source[end_index:]


def _update_readme(
    *,
    check: bool,
) -> bool:
    """Update README or verify that generated metadata is current."""

    source = README_PATH.read_text(
        encoding="utf-8",
    )

    rendered = _render_readme(
        source,
        _build_badge_block(),
    )

    if rendered == source:
        print("README metadata is up to date.")
        return True

    if check:
        print(
            "README metadata is out of date. Run `just fix` to regenerate it.",
            file=sys.stderr,
        )
        return False

    README_PATH.write_text(
        rendered,
        encoding="utf-8",
    )

    print("README metadata updated.")

    return True


def _parse_arguments(
    project_name: str,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(f"Generate the technical metadata table in the {project_name} README."),
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help=("Check README metadata without modifying README.md."),
    )

    return parser.parse_args()


def main() -> int:
    """Run the README metadata generator."""

    try:
        pyproject = _load_toml(
            PYPROJECT_PATH,
        )

        metadata = ProjectMetadata.from_pyproject(
            pyproject,
        )

        arguments = _parse_arguments(
            metadata.display_name,
        )

        success = _update_readme(
            check=arguments.check,
        )

    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(
            f"README generation failed: {error}",
            file=sys.stderr,
        )
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
