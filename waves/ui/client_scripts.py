"""
File: client_scripts.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Client-side JavaScript helpers for WAVES Gradio events.

License: MIT License
"""

EXAMPLES_UI_JS = """
(language) => {
    const pagesLabel =
        language === "Русский"
            ? "Страницы:"
            : "Pages:";

    const normalizeExampleName = (value) => {
        if (!value) {
            return "";
        }

        return String(value)
            .normalize("NFKC")
            .toLocaleLowerCase()
            .replace(/[^\\p{L}\\p{N}]+/gu, "");
    };

    const setupExamples = (attempt = 0) => {
        const examples = document.querySelector(
            "#application-examples"
        );

        if (!examples) {
            if (attempt < 40) {
                window.setTimeout(
                    () => setupExamples(attempt + 1),
                    50
                );
            }

            return;
        }

        const syncPaginationLabel = () => {
            const paginations =
                examples.querySelectorAll(
                    "div.paginate"
                );

            paginations.forEach(
                (pagination) => {
                    for (
                        const node
                        of pagination.childNodes
                    ) {
                        if (
                            node.nodeType ===
                                Node.TEXT_NODE &&
                            node.textContent &&
                            node.textContent
                                .trim()
                                .length > 0
                        ) {
                            const expected =
                                `${pagesLabel} `;

                            if (
                                node.textContent !==
                                expected
                            ) {
                                node.textContent =
                                    expected;
                            }

                            break;
                        }
                    }
                }
            );
        };

        const syncSelectedExample = () => {
            const selectedKey =
                window.__wavesSelectedExampleKey ??
                "";

            const buttons =
                examples.querySelectorAll(
                    "div.gallery > button.gallery-item"
                );

            buttons.forEach((button) => {
                const buttonKey =
                    normalizeExampleName(
                        button.textContent ?? ""
                    );

                const isSelected =
                    selectedKey.length > 0 &&
                    buttonKey === selectedKey;

                button.classList.toggle(
                    "waves-selected-example",
                    isSelected
                );

                button.setAttribute(
                    "aria-disabled",
                    String(isSelected)
                );
            });
        };

        const selectExample = (button) => {
            window.__wavesSelectedExampleKey =
                normalizeExampleName(
                    button.textContent ?? ""
                );

            syncSelectedExample();
        };

        if (
            window.__wavesExamplesRoot &&
            window.__wavesExamplesRoot !==
                examples &&
            window.__wavesExamplesClickHandler
        ) {
            window.__wavesExamplesRoot
                .removeEventListener(
                    "click",
                    window.__wavesExamplesClickHandler,
                    true
                );
        }

        if (
            window.__wavesExamplesRoot !==
            examples
        ) {
            const clickHandler = (event) => {
                if (
                    !(
                        event.target
                        instanceof Element
                    )
                ) {
                    return;
                }

                const button =
                    event.target.closest(
                        "button.gallery-item"
                    );

                if (
                    !button ||
                    !examples.contains(button)
                ) {
                    return;
                }

                if (
                    button.classList.contains(
                        "waves-selected-example"
                    )
                ) {
                    event.preventDefault();
                    event.stopPropagation();
                    event.stopImmediatePropagation();

                    return;
                }

                selectExample(button);
            };

            examples.addEventListener(
                "click",
                clickHandler,
                true
            );

            window.__wavesExamplesRoot =
                examples;

            window.__wavesExamplesClickHandler =
                clickHandler;
        }

        const syncExamplesUi = () => {
            syncPaginationLabel();
            syncSelectedExample();
        };

        syncExamplesUi();

        requestAnimationFrame(
            syncExamplesUi
        );

        window.setTimeout(
            syncExamplesUi,
            50
        );

        window.setTimeout(
            syncExamplesUi,
            250
        );

        if (
            window.__wavesExamplesObserver
        ) {
            window.__wavesExamplesObserver
                .disconnect();
        }

        const observer =
            new MutationObserver(
                syncExamplesUi
            );

        observer.observe(
            examples,
            {
                childList: true,
                subtree: true,
            }
        );

        window.__wavesExamplesObserver =
            observer;
    };

    setupExamples();

    return [];
}
"""


CLEAR_EXAMPLE_SELECTION_JS = """
(...args) => {
    window.__wavesSelectedExampleKey = "";

    document
        .querySelectorAll(
            "#application-examples " +
            "button.gallery-item"
        )
        .forEach((button) => {
            button.classList.remove(
                "waves-selected-example"
            );

            button.setAttribute(
                "aria-disabled",
                "false"
            );
        });

    return args;
}
"""


MODAL_CLOSE_ANIMATION_JS = """
(...args) => {
    const button = document.activeElement;

    const modal =
        button?.closest(
            "#processing-modal, #audio-info-modal, " +
            "#visualization-info-modal"
        );

    if (!modal) {
        return args;
    }

    modal.classList.add(
        "waves-modal-closing"
    );

    return new Promise(
        (resolve) => {
            window.setTimeout(
                () => resolve(args),
                180
            );
        }
    );
}
"""
