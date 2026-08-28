from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import sys
import tomllib
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]

PYPROJECT_PATH = ROOT / "pyproject.toml"
PACKAGE_JSON_PATH = ROOT / "package.json"
PYTHON_VERSION_PATH = ROOT / ".python-version"
NODE_VERSION_PATH = ROOT / ".nvmrc"
DOCKERFILE_PATH = ROOT / "Dockerfile"
LICENSE_PATH = ROOT / "LICENSE"
README_PATH = ROOT / "README.md"

README_BLOCK_START = "<!-- BEGIN PROJECT BADGES -->"
README_BLOCK_END = "<!-- END PROJECT BADGES -->"

PRETTIER_IGNORE_START = "<!-- prettier-ignore-start -->"
PRETTIER_IGNORE_END = "<!-- prettier-ignore-end -->"

GITHUB_REPOSITORY = "DmitryRyumin/WAVES"
HUGGING_FACE_SPACE = "DmitryRyumin/WAVES"


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML document."""

    with path.open("rb") as file:
        data = tomllib.load(file)

    return data


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(
        data,
        dict,
    ):
        msg = f"Expected a JSON object in {path}."
        raise TypeError(msg)

    return data


def _read_text_value(path: Path) -> str:
    """Read a required one-line text value."""

    value = path.read_text(encoding="utf-8").strip()

    if not value:
        msg = f"Expected a non-empty value in {path}."
        raise ValueError(msg)

    return value


def _normalize_package_name(name: str) -> str:
    """Normalize a Python package name for matching."""

    return re.sub(
        r"[-_.]+",
        "-",
        name.strip().lower(),
    )


def _parse_python_dependency(dependency: str) -> tuple[str, str]:
    """Split a Python dependency into normalized name and version specifier."""

    dependency = dependency.strip()

    match = re.match(
        r"^([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*(.*)$",
        dependency,
    )

    if match is None:
        return (
            "",
            "",
        )

    name = _normalize_package_name(match.group(1))

    specification = match.group(2).strip()

    if ";" in specification:
        specification = specification.split(
            ";",
            maxsplit=1,
        )[0].strip()

    return (
        name,
        specification,
    )


def _collect_python_dependencies(
    pyproject: dict[str, Any],
) -> dict[str, str]:
    """Collect project and development Python dependencies."""

    dependencies: dict[str, str] = {}

    project = pyproject.get("project", {})

    if isinstance(
        project,
        dict,
    ):
        project_dependencies = project.get("dependencies", [])

        if isinstance(
            project_dependencies,
            list,
        ):
            for dependency in project_dependencies:
                if not isinstance(
                    dependency,
                    str,
                ):
                    continue

                name, specification = _parse_python_dependency(dependency)

                if name:
                    dependencies[name] = specification

        optional_dependencies = project.get("optional-dependencies", {})

        if isinstance(
            optional_dependencies,
            dict,
        ):
            for group_dependencies in optional_dependencies.values():
                if not isinstance(
                    group_dependencies,
                    list,
                ):
                    continue

                for dependency in group_dependencies:
                    if not isinstance(
                        dependency,
                        str,
                    ):
                        continue

                    name, specification = _parse_python_dependency(dependency)

                    if name:
                        dependencies[name] = specification

    dependency_groups = pyproject.get("dependency-groups", {})

    if isinstance(
        dependency_groups,
        dict,
    ):
        for group_dependencies in dependency_groups.values():
            if not isinstance(
                group_dependencies,
                list,
            ):
                continue

            for dependency in group_dependencies:
                if not isinstance(
                    dependency,
                    str,
                ):
                    continue

                name, specification = _parse_python_dependency(dependency)

                if name:
                    dependencies[name] = specification

    return dependencies


def _minimum_version(specification: str) -> str | None:
    """Extract the minimum or exact version from a Python version specifier."""

    minimum_match = re.search(
        r">=\s*([0-9]+(?:\.[0-9A-Za-z]+)*)",
        specification,
    )

    if minimum_match is not None:
        return minimum_match.group(1)

    exact_match = re.search(
        r"==\s*([0-9]+(?:\.[0-9A-Za-z]+)*)",
        specification,
    )

    if exact_match is not None:
        return exact_match.group(1)

    compatible_match = re.search(
        r"~=\s*([0-9]+(?:\.[0-9A-Za-z]+)*)",
        specification,
    )

    if compatible_match is not None:
        return compatible_match.group(1)

    return None


def _dependency_badge_version(
    specification: str,
) -> str | None:
    """Convert a dependency specification into a compact badge version."""

    version = _minimum_version(specification)

    if version is None:
        return None

    if ">=" in specification or "~=" in specification:
        return f"{version}+"

    return version


def _get_project_version(
    pyproject: dict[str, Any],
) -> str:
    """Return the project version."""

    project = pyproject.get("project")

    if not isinstance(
        project,
        dict,
    ):
        msg = "Missing [project] table in pyproject.toml."
        raise ValueError(msg)

    version = project.get("version")

    if (
        not isinstance(
            version,
            str,
        )
        or not version.strip()
    ):
        msg = "Missing project.version in pyproject.toml."
        raise ValueError(msg)

    return version.strip()


def _get_uv_version(
    pyproject: dict[str, Any],
) -> str | None:
    """Return a configured minimum uv version, if available."""

    tool = pyproject.get("tool")

    if not isinstance(
        tool,
        dict,
    ):
        return None

    uv = tool.get("uv")

    if not isinstance(
        uv,
        dict,
    ):
        return None

    value = uv.get("required-version")

    if not isinstance(
        value,
        str,
    ):
        value = uv.get("required_version")

    if not isinstance(
        value,
        str,
    ):
        return None

    return _dependency_badge_version(value.strip())


def _strip_version_prefix(value: str) -> str:
    """Remove a leading v from a tool version."""

    return value.removeprefix("v").strip()


def _get_npm_version(
    package_json: dict[str, Any],
) -> str | None:
    """Return the npm version declared in package.json."""

    package_manager = package_json.get("packageManager")

    if isinstance(
        package_manager,
        str,
    ):
        match = re.fullmatch(
            r"npm@(.+)",
            package_manager.strip(),
        )

        if match is not None:
            return match.group(1).strip()

    engines = package_json.get("engines")

    if not isinstance(
        engines,
        dict,
    ):
        return None

    specification = engines.get("npm")

    if not isinstance(
        specification,
        str,
    ):
        return None

    return _dependency_badge_version(specification)


def _get_node_version(
    package_json: dict[str, Any],
) -> str:
    """Return the canonical Node.js version."""

    if NODE_VERSION_PATH.is_file():
        return _strip_version_prefix(
            _read_text_value(NODE_VERSION_PATH),
        )

    engines = package_json.get("engines")

    if isinstance(
        engines,
        dict,
    ):
        specification = engines.get("node")

        if isinstance(
            specification,
            str,
        ):
            version = _dependency_badge_version(specification)

            if version is not None:
                return version

    msg = "Node.js version was not found in .nvmrc or package.json."
    raise ValueError(msg)


def _get_javascript_dependency_version(
    package_json: dict[str, Any],
    package_name: str,
) -> str | None:
    """Return a JavaScript package version from package.json."""

    for section_name in (
        "dependencies",
        "devDependencies",
        "optionalDependencies",
    ):
        section = package_json.get(section_name)

        if not isinstance(
            section,
            dict,
        ):
            continue

        value = section.get(package_name)

        if not isinstance(
            value,
            str,
        ):
            continue

        cleaned = value.strip()

        cleaned = re.sub(
            r"^[~^]",
            "",
            cleaned,
        )

        return cleaned

    return None


def _badge_url(
    label: str,
    message: str,
    color: str,
    *,
    logo: str | None = None,
    logo_color: str | None = None,
) -> str:
    """Build a static Shields.io badge URL."""

    encoded_label = quote(
        label,
        safe="",
    )

    encoded_message = quote(
        message,
        safe="",
    )

    url = f"https://img.shields.io/badge/{encoded_label}-{encoded_message}-{color}?style=flat-square"

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


def _static_badge(
    label: str,
    message: str,
    color: str,
    alt: str,
    *,
    logo: str | None = None,
    logo_color: str | None = None,
) -> str:
    """Build a complete static badge image."""

    return _image(
        _badge_url(
            label,
            message,
            color,
            logo=logo,
            logo_color=logo_color,
        ),
        alt,
    )


def _python_dependency_badge(
    dependencies: dict[str, str],
    package_name: str,
    *,
    label: str,
    color: str,
    logo: str | None = None,
    logo_color: str | None = None,
) -> str | None:
    """Build a badge for a configured Python dependency."""

    specification = dependencies.get(
        _normalize_package_name(package_name),
    )

    if specification is None:
        return None

    version = _dependency_badge_version(specification)

    if version is None:
        return None

    return _static_badge(
        label,
        version,
        color,
        label,
        logo=logo,
        logo_color=logo_color,
    )


def _javascript_dependency_badge(
    package_json: dict[str, Any],
    package_name: str,
    *,
    label: str,
    color: str,
    logo: str | None = None,
    logo_color: str | None = None,
) -> str | None:
    """Build a badge for a configured JavaScript dependency."""

    version = _get_javascript_dependency_version(
        package_json,
        package_name,
    )

    if version is None:
        return None

    return _static_badge(
        label,
        version,
        color,
        label,
        logo=logo,
        logo_color=logo_color,
    )


def _build_project_badges(
    pyproject: dict[str, Any],
) -> list[str]:
    """Build project metadata badges."""

    project_version = _get_project_version(pyproject)

    python_version = _strip_version_prefix(
        _read_text_value(PYTHON_VERSION_PATH),
    )

    return [
        _static_badge(
            "version",
            project_version,
            "2F81F7",
            "Version",
        ),
        _link(
            "LICENSE",
            _image(
                (f"https://img.shields.io/github/license/{GITHUB_REPOSITORY}?style=flat-square"),
                "License",
            ),
        ),
        _static_badge(
            "Python",
            python_version,
            "3776AB",
            "Python",
            logo="python",
            logo_color="white",
        ),
    ]


def _build_runtime_badges(
    pyproject: dict[str, Any],
    package_json: dict[str, Any],
    dependencies: dict[str, str],
) -> list[str]:
    """Build runtime technology badges."""

    badges: list[str] = []

    for badge in (
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
    ):
        if badge is not None:
            badges.append(badge)

    if DOCKERFILE_PATH.is_file():
        badges.append(
            _static_badge(
                "Docker",
                "enabled",
                "2496ED",
                "Docker",
                logo="docker",
                logo_color="white",
            )
        )

    uv_version = _get_uv_version(pyproject)

    if uv_version is not None:
        badges.append(
            _static_badge(
                "uv",
                uv_version,
                "261230",
                "uv",
            )
        )

    node_version = _get_node_version(package_json)

    badges.append(
        _static_badge(
            "Node.js",
            node_version,
            "5FA04E",
            "Node.js",
            logo="nodedotjs",
            logo_color="white",
        )
    )

    npm_version = _get_npm_version(package_json)

    if npm_version is not None:
        badges.append(
            _static_badge(
                "npm",
                npm_version,
                "CB3837",
                "npm",
                logo="npm",
                logo_color="white",
            )
        )

    return badges


def _build_code_quality_badges(
    package_json: dict[str, Any],
    dependencies: dict[str, str],
) -> list[str]:
    """Build code-quality tool badges."""

    badges: list[str] = []

    for badge in (
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
            logo="precommit",
            logo_color="black",
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
    ):
        if badge is not None:
            badges.append(badge)

    return badges


def _build_repository_badges() -> list[str]:
    """Build live GitHub repository badges."""

    return [
        _image(
            (f"https://img.shields.io/github/repo-size/{GITHUB_REPOSITORY}?style=flat-square"),
            "Repository size",
        ),
        _image(
            (f"https://img.shields.io/github/last-commit/{GITHUB_REPOSITORY}?style=flat-square"),
            "Last commit",
        ),
        _link(
            f"https://github.com/{GITHUB_REPOSITORY}/graphs/contributors",
            _image(
                (f"https://img.shields.io/github/contributors/{GITHUB_REPOSITORY}?style=flat-square"),
                "Contributors",
            ),
        ),
    ]


def _build_community_badges() -> list[str]:
    """Build live GitHub community badges."""

    return [
        _link(
            f"https://github.com/{GITHUB_REPOSITORY}/stargazers",
            _image(
                (f"https://img.shields.io/github/stars/{GITHUB_REPOSITORY}?style=flat-square"),
                "Stars",
            ),
        ),
        _link(
            f"https://github.com/{GITHUB_REPOSITORY}/forks",
            _image(
                (f"https://img.shields.io/github/forks/{GITHUB_REPOSITORY}?style=flat-square"),
                "Forks",
            ),
        ),
        _link(
            f"https://github.com/{GITHUB_REPOSITORY}/issues",
            _image(
                (f"https://img.shields.io/github/issues/{GITHUB_REPOSITORY}?style=flat-square"),
                "Issues",
            ),
        ),
    ]


def _build_application_badges() -> list[str]:
    """Build application badges."""

    return [
        _link(
            f"https://huggingface.co/spaces/{HUGGING_FACE_SPACE}",
            _static_badge(
                "🤗 Space",
                "WAVES",
                "FFD21E",
                "WAVES on Hugging Face",
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

    dependencies = _collect_python_dependencies(pyproject)

    rows: list[str] = [
        PRETTIER_IGNORE_START,
        '<table align="center">',
    ]

    rows.extend(
        _table_row(
            "Project",
            _build_project_badges(pyproject),
        )
    )

    rows.extend(
        _table_row(
            "Runtime",
            _build_runtime_badges(
                pyproject,
                package_json,
                dependencies,
            ),
        )
    )

    code_quality_badges = _build_code_quality_badges(
        package_json,
        dependencies,
    )

    if code_quality_badges:
        rows.extend(
            _table_row(
                "Code Quality",
                code_quality_badges,
            )
        )

    rows.extend(
        _table_row(
            "Repository",
            _build_repository_badges(),
        )
    )

    rows.extend(
        _table_row(
            "Community",
            _build_community_badges(),
        )
    )

    rows.extend(
        _table_row(
            "Application",
            _build_application_badges(),
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

    end_index = source.find(
        README_BLOCK_END,
        start_index + len(README_BLOCK_START),
    )

    if end_index < 0:
        msg = f"README marker not found: {README_BLOCK_END}"
        raise ValueError(msg)

    prefix_end = start_index + len(README_BLOCK_START)

    prefix = source[:prefix_end]

    suffix = source[end_index:]

    return f"{prefix}\n\n{generated_block}\n\n{suffix}"


def _update_readme(
    *,
    check: bool,
) -> bool:
    """Update the README or verify that its generated metadata is current."""

    source = README_PATH.read_text(encoding="utf-8")

    generated_block = _build_badge_block()

    rendered = _render_readme(
        source,
        generated_block,
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


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate the technical metadata table in the WAVES README.",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check README metadata without modifying README.md.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the README metadata generator."""

    arguments = _parse_arguments()

    try:
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
