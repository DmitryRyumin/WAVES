"""
File: app.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: WAVES Gradio application entry point.

License: MIT License
"""

import os
from typing import cast

import gradio as gr

from waves.config import (
    get_config_int,
    get_config_str,
    is_hugging_face_space,
)
from waves.events.event_handlers import setup_app_event_handlers
from waves.logger import setup_logging
from waves.port import ensure_port_available
from waves.ui.language_selector import create_language_selector
from waves.ui.tabs import create_app_tabs


def create_gradio_app() -> gr.Blocks:
    """Create the WAVES Gradio application."""

    static_images_path = get_config_str(
        "StaticPaths_IMAGES",
        "static/images",
    )

    app_title = get_config_str(
        "App_TITLE",
        "WAVES",
    )

    gr.set_static_paths(
        paths=[
            static_images_path,
        ]
    )

    with (
        gr.Blocks(
            title=app_title,
            fill_width=True,
        ) as gradio_app,
        gr.Column(
            elem_classes="app-shell",
        ),
    ):
        language_selector = create_language_selector()

        app_tabs = create_app_tabs()

        setup_app_event_handlers(
            gradio_app=gradio_app,
            language_selector=language_selector,
            app_tabs=app_tabs,
        )

    return cast(
        gr.Blocks,
        gradio_app,
    )


def main() -> None:
    """Run the WAVES Gradio application."""

    configured_server_name = get_config_str(
        "Server_NAME",
        "127.0.0.1",
    )

    configured_server_port = get_config_int(
        "Server_PORT",
        7860,
    )

    default_server_name = "0.0.0.0" if is_hugging_face_space() else configured_server_name

    server_name = os.getenv(
        "GRADIO_SERVER_NAME",
        default_server_name,
    )

    server_port = int(
        os.getenv(
            "PORT",
            os.getenv(
                "GRADIO_SERVER_PORT",
                str(configured_server_port),
            ),
        )
    )

    app_css_path = get_config_str(
        "App_CSS_PATH",
        "app.css",
    )

    ensure_port_available(
        host=server_name,
        port=server_port,
    )

    create_gradio_app().queue(
        api_open=False,
    ).launch(
        theme="default",
        css_paths=app_css_path,
        share=False,
        server_name=server_name,
        server_port=server_port,
        footer_links=[],
    )


if __name__ == "__main__":
    setup_logging()
    main()
