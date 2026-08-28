export default {
    extends: ["stylelint-config-standard", "stylelint-config-recess-order"],
    ignoreFiles: ["app.css", "node_modules/**"],
    rules: {
        "no-descending-specificity": null,
    },
};
