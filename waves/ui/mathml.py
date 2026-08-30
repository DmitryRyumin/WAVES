"""
File: mathml.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Lightweight MathML builder for WAVES UI components.

License: MIT License
"""


class MathML:
    """Build trusted MathML fragments without external dependencies."""

    @staticmethod
    def math(
        body: str,
        *,
        display: bool = False,
    ) -> str:
        """Wrap content in a MathML root element."""

        mode = "block" if display else "inline"

        return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="{mode}">{body}</math>'

    @staticmethod
    def identifier(
        value: str,
        *,
        normal: bool = False,
    ) -> str:
        """Create a MathML identifier."""

        variant = ' mathvariant="normal"' if normal else ""

        return f"<mi{variant}>{value}</mi>"

    @staticmethod
    def number(
        value: int | str,
    ) -> str:
        """Create a MathML number."""

        return f"<mn>{value}</mn>"

    @staticmethod
    def operator(
        value: str,
    ) -> str:
        """Create a MathML operator."""

        return f"<mo>{value}</mo>"

    @staticmethod
    def text(
        value: str,
    ) -> str:
        """Create MathML text."""

        return f"<mtext>{value}</mtext>"

    @staticmethod
    def row(
        *parts: str,
    ) -> str:
        """Create a MathML row."""

        return f"<mrow>{''.join(parts)}</mrow>"

    @classmethod
    def comma_row(
        cls,
        *parts: str,
    ) -> str:
        """Create a comma-separated MathML row."""

        result: list[str] = []

        for index, part in enumerate(parts):
            if index:
                result.append(cls.operator(","))

            result.append(part)

        return cls.row(*result)

    @staticmethod
    def subscript(
        base: str,
        index: str,
    ) -> str:
        """Create a MathML subscript."""

        return f"<msub>{base}{index}</msub>"

    @classmethod
    def indexed(
        cls,
        base: str,
        *indices: str,
        text_index: bool = False,
    ) -> str:
        """Create an indexed mathematical identifier."""

        if text_index:
            if len(indices) != 1:
                msg = "Text subscripts require exactly one index."

                raise ValueError(msg)

            index = cls.text(indices[0])
        else:
            index = cls.comma_row(*(cls.identifier(item) for item in indices))

        return cls.subscript(
            cls.identifier(base),
            index,
        )

    @classmethod
    def delta(
        cls,
        *indices: str,
        text_index: bool = False,
    ) -> str:
        """Create delta with an optional subscript."""

        base = cls.identifier("&#x0394;")

        if not indices:
            return base

        if text_index:
            if len(indices) != 1:
                msg = "Text subscripts require exactly one index."

                raise ValueError(msg)

            index = cls.text(indices[0])
        else:
            index = cls.comma_row(*(cls.identifier(item) for item in indices))

        return cls.subscript(
            base,
            index,
        )

    @staticmethod
    def fraction(
        numerator: str,
        denominator: str,
    ) -> str:
        """Create a MathML fraction."""

        return f"<mfrac>{numerator}{denominator}</mfrac>"

    @classmethod
    def summation(
        cls,
        index: str,
    ) -> str:
        """Create a summation with a lower index."""

        return f"<munder>{cls.operator('&#x2211;')}{index}</munder>"

    @classmethod
    def indexed_sum(
        cls,
        symbol: str,
    ) -> str:
        """Create a summation indexed by one variable."""

        return cls.summation(cls.identifier(symbol))

    @classmethod
    def call(
        cls,
        function: str,
        *arguments: str,
    ) -> str:
        """Create a mathematical function call."""

        return cls.row(
            function,
            cls.operator("("),
            cls.comma_row(*arguments),
            cls.operator(")"),
        )

    @classmethod
    def parentheses(
        cls,
        body: str,
    ) -> str:
        """Wrap an expression in parentheses."""

        return cls.row(
            cls.operator("("),
            body,
            cls.operator(")"),
        )

    @classmethod
    def equation(
        cls,
        lhs: str,
        rhs: str,
    ) -> str:
        """Create an equality expression."""

        return cls.row(
            lhs,
            cls.operator("="),
            rhs,
        )

    @classmethod
    def inline(
        cls,
        body: str,
    ) -> str:
        """Create inline MathML."""

        return cls.math(body)

    @classmethod
    def display(
        cls,
        body: str,
    ) -> str:
        """Create display-style MathML."""

        return cls.math(
            body,
            display=True,
        )
