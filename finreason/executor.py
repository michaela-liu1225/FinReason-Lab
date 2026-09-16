"""Safe execution for the small arithmetic language used by FinQA.

The executor intentionally does not translate programs to Python and never uses
``eval``/``exec``.  Programs are parsed into a tiny, flat DSL and every operator
is dispatched through an explicit allow-list.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import (
    Decimal,
    InvalidOperation,
    Overflow,
    Underflow,
    localcontext,
)
from typing import Final

NumericValue = Decimal
ProgramValue = Decimal | bool


class ProgramError(ValueError):
    """Base class for a rejected or failed FinQA program."""

    code = "program_error"

    def __init__(
        self,
        message: str,
        *,
        step_index: int | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(message)
        self.step_index = step_index
        self.token = token


class ProgramSyntaxError(ProgramError):
    code = "invalid_syntax"


class UnsupportedOperationError(ProgramError):
    code = "unsupported_operation"


class InvalidReferenceError(ProgramError):
    code = "invalid_reference"


class InvalidArgumentError(ProgramError):
    code = "invalid_argument"


class DivisionByZeroError(ProgramError):
    code = "division_by_zero"


class NumericLimitError(ProgramError):
    code = "numeric_limit"


class TableLookupError(InvalidArgumentError):
    code = "table_lookup_error"


@dataclass(frozen=True)
class ProgramStep:
    """One parsed operation in the flattened FinQA DSL."""

    operation: str
    arguments: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class ExecutedStep:
    """One executed operation and its resolved values."""

    index: int
    operation: str
    arguments: tuple[str, ...]
    resolved_arguments: tuple[ProgramValue, ...]
    value: ProgramValue


@dataclass(frozen=True)
class ExecutionResult:
    """Successful deterministic execution of a complete program."""

    program: str
    steps: tuple[ExecutedStep, ...]
    value: ProgramValue

    @property
    def answer(self) -> str:
        if isinstance(self.value, bool):
            return "yes" if self.value else "no"
        return format_decimal(self.value)


_CALL_RE: Final = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$", re.DOTALL)
_REF_RE: Final = re.compile(r"^#([0-9]+)$")
_DECIMAL_RE: Final = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_CONST_RE: Final = re.compile(
    r"^const_(?P<minus>m)?(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$",
    re.IGNORECASE,
)
SUPPORTED_CONSTANTS: Final = frozenset(
    {
        "const_m1",
        "const_0",
        "const_1",
        "const_2",
        "const_3",
        "const_4",
        "const_5",
        "const_6",
        "const_7",
        "const_8",
        "const_9",
        "const_10",
        "const_100",
        "const_1000",
        "const_10000",
        "const_100000",
        "const_1000000",
        "const_10000000",
        "const_1000000000",
    }
)
_DISPLAY_ANNOTATION_RE: Final = re.compile(
    r"^(?P<primary>[$£€]?\s*[+-]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)\s*%?)"
    r"\s*\(\s*[$£€]?\s*[+-]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)\s*%?\s*\)$"
)

_BINARY_OPERATIONS: Final = frozenset({"add", "subtract", "multiply", "divide", "exp", "greater"})
_TABLE_OPERATIONS: Final = frozenset({"table_sum", "table_average", "table_max", "table_min"})
SUPPORTED_OPERATIONS: Final = _BINARY_OPERATIONS | _TABLE_OPERATIONS

MAX_PROGRAM_CHARS: Final = 8_192
MAX_STEPS: Final = 64
MAX_ARGUMENTS_PER_STEP: Final = 128
MAX_NUMERIC_TOKEN_CHARS: Final = 128
MAX_ABS_EXPONENT: Final = 100
MAX_ABS_ADJUSTED_EXPONENT: Final = 1_000

TableValues = Mapping[str, Sequence[str | int | Decimal]]


def _split_at_top_level(text: str) -> tuple[str, ...]:
    """Split comma-separated text without confusing argument commas for steps."""

    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ProgramSyntaxError("unexpected closing parenthesis")

        if character == "," and depth == 0:
            part = "".join(current).strip()
            if not part:
                raise ProgramSyntaxError("empty program step")
            parts.append(part)
            current = []
        else:
            current.append(character)

    if depth != 0:
        raise ProgramSyntaxError("unbalanced parentheses")

    final = "".join(current).strip()
    if final:
        parts.append(final)
    elif parts:
        raise ProgramSyntaxError("empty program step")
    return tuple(parts)


def _split_arguments(body: str, *, step_index: int) -> tuple[str, ...]:
    if not body.strip():
        return ()
    try:
        arguments = _split_at_top_level(body)
    except ProgramSyntaxError as exc:
        raise ProgramSyntaxError(str(exc), step_index=step_index, token=body) from exc
    if len(arguments) > MAX_ARGUMENTS_PER_STEP:
        raise ProgramSyntaxError(
            f"step has more than {MAX_ARGUMENTS_PER_STEP} arguments",
            step_index=step_index,
        )
    return arguments


def parse_program(program: str) -> tuple[ProgramStep, ...]:
    """Parse and validate the structure and operator allow-list of a program."""

    if not isinstance(program, str):
        raise ProgramSyntaxError("program must be a string")
    text = program.strip()
    if not text:
        raise ProgramSyntaxError("program is empty")
    if len(text) > MAX_PROGRAM_CHARS:
        raise ProgramSyntaxError(f"program exceeds {MAX_PROGRAM_CHARS} characters")

    raw_steps = _split_at_top_level(text)
    if len(raw_steps) > MAX_STEPS:
        raise ProgramSyntaxError(f"program exceeds {MAX_STEPS} steps")

    parsed: list[ProgramStep] = []
    for step_index, raw_step in enumerate(raw_steps):
        match = _CALL_RE.fullmatch(raw_step)
        if match is None:
            raise ProgramSyntaxError(
                "each step must be an operation call",
                step_index=step_index,
                token=raw_step,
            )
        operation = match.group(1).lower()
        if operation not in SUPPORTED_OPERATIONS:
            raise UnsupportedOperationError(
                f"unsupported operation: {operation}",
                step_index=step_index,
                token=operation,
            )
        arguments = _split_arguments(match.group(2), step_index=step_index)
        if operation in _BINARY_OPERATIONS and len(arguments) != 2:
            raise InvalidArgumentError(
                f"{operation} requires exactly two arguments",
                step_index=step_index,
            )
        if operation in _TABLE_OPERATIONS and not arguments:
            raise InvalidArgumentError(
                f"{operation} requires at least one argument",
                step_index=step_index,
            )
        parsed.append(ProgramStep(operation, arguments, raw_step))
    return tuple(parsed)


def _validate_decimal(value: Decimal, *, token: str) -> Decimal:
    if not value.is_finite():
        raise NumericLimitError("non-finite numbers are not allowed", token=token)
    if value and abs(value.adjusted()) > MAX_ABS_ADJUSTED_EXPONENT:
        raise NumericLimitError("numeric magnitude exceeds the executor limit", token=token)
    return value


def parse_decimal(token: str) -> Decimal:
    """Parse a FinQA numeric token exactly, including percentages and constants.

    Examples include ``1,200``, ``(4.5)``, ``25%``, ``const_100`` and
    ``const_m1``.  Percentages use their mathematical value (``25%`` is
    ``Decimal('0.25')``).
    """

    if not isinstance(token, str):
        raise InvalidArgumentError("numeric argument must be a string")
    raw = token.strip()
    if not raw or len(raw) > MAX_NUMERIC_TOKEN_CHARS:
        raise InvalidArgumentError("invalid numeric argument", token=raw)

    constant = _CONST_RE.fullmatch(raw)
    if constant:
        if raw.casefold() not in SUPPORTED_CONSTANTS:
            raise InvalidArgumentError("constant is not allow-listed", token=raw)
        number_text = constant.group("number")
        if constant.group("minus"):
            number_text = "-" + number_text
        try:
            return _validate_decimal(Decimal(number_text), token=raw)
        except InvalidOperation as exc:
            raise InvalidArgumentError("invalid constant", token=raw) from exc

    # Some FinQA tables retain an accessibility/accounting duplicate such as
    # ``-13 (13)`` or ``11.4% (11.4%)``.  The signed leading value is canonical;
    # the parenthetical display is not a second operand.
    annotation = _DISPLAY_ANNOTATION_RE.fullmatch(raw)
    if annotation:
        raw = annotation.group("primary").strip()

    negative_parentheses = raw.startswith("(") and raw.endswith(")")
    if negative_parentheses:
        raw = raw[1:-1].strip()

    # Currency marks and thousands separators are presentation, not operations.
    raw = raw.replace("$", "").replace("£", "").replace("€", "")
    raw = raw.replace(",", "").strip()
    percentage = raw.endswith("%")
    if percentage:
        raw = raw[:-1].strip()
    if not _DECIMAL_RE.fullmatch(raw):
        raise InvalidArgumentError("argument is not a numeric literal", token=token)
    if negative_parentheses:
        raw = "-" + raw.lstrip("+")
    try:
        value = Decimal(raw)
        if percentage:
            value /= Decimal(100)
    except (InvalidOperation, Overflow) as exc:
        raise InvalidArgumentError("invalid numeric literal", token=token) from exc
    return _validate_decimal(value, token=token)


def format_decimal(value: Decimal) -> str:
    """Render a Decimal deterministically without binary-float conversion."""

    value = _validate_decimal(value, token=str(value))
    if value.is_zero():
        return "0"
    normalized = value.normalize()
    if -12 <= normalized.adjusted() <= 20:
        return format(normalized, "f")
    return format(normalized, "E").replace("E+", "e").replace("E", "e")


def _ensure_numeric(value: ProgramValue, *, step_index: int, token: str) -> Decimal:
    # bool is a subclass of int, so this explicit check matters.
    if isinstance(value, bool):
        raise InvalidArgumentError(
            "a yes/no result cannot be used as a numeric argument",
            step_index=step_index,
            token=token,
        )
    return value


def _resolve_argument(
    token: str,
    prior_values: list[ProgramValue],
    *,
    step_index: int,
) -> ProgramValue:
    reference = _REF_RE.fullmatch(token.strip())
    if reference:
        reference_index = int(reference.group(1))
        if reference_index >= len(prior_values):
            raise InvalidReferenceError(
                f"reference #{reference_index} does not name a prior step",
                step_index=step_index,
                token=token,
            )
        return prior_values[reference_index]
    try:
        return parse_decimal(token)
    except ProgramError as exc:
        exc.step_index = step_index
        raise


def _normalise_table_label(label: str) -> str:
    return " ".join(label.casefold().split())


def _materialise_table_values(
    table_values: TableValues | None,
    *,
    required_labels: frozenset[str],
) -> dict[str, tuple[Decimal, ...]]:
    materialised: dict[str, tuple[Decimal, ...]] = {}
    if table_values is None or not required_labels:
        return materialised
    for raw_label, raw_values in table_values.items():
        label = _normalise_table_label(str(raw_label))
        # A caller may pass a complete report table.  Only rows explicitly
        # referenced by the program belong to this execution boundary.
        if label not in required_labels:
            continue
        if label in materialised:
            raise TableLookupError(f"duplicate table row label: {raw_label}")
        if isinstance(raw_values, str):
            raise TableLookupError(f"table row {raw_label!r} must contain a sequence of values")
        values: list[Decimal] = []
        for raw_value in raw_values:
            if isinstance(raw_value, bool):
                raise TableLookupError(f"table row {raw_label!r} contains a boolean value")
            if isinstance(raw_value, Decimal):
                values.append(_validate_decimal(raw_value, token=str(raw_value)))
            elif isinstance(raw_value, int | str):
                values.append(parse_decimal(str(raw_value)))
            else:
                raise TableLookupError(f"table row {raw_label!r} contains a non-exact value")
        if not values:
            raise TableLookupError(f"table row {raw_label!r} has no numeric values")
        materialised[label] = tuple(values)
    return materialised


def _is_symbolic_table_step(step: ProgramStep) -> bool:
    return (
        step.operation in _TABLE_OPERATIONS
        and len(step.arguments) == 2
        and step.arguments[1].strip().casefold() == "none"
    )


def _resolve_symbolic_table_step(
    step: ProgramStep,
    table_values: Mapping[str, tuple[Decimal, ...]],
    *,
    step_index: int,
) -> tuple[ProgramValue, ...]:
    label = _normalise_table_label(step.arguments[0])
    values = table_values.get(label)
    if values is None:
        raise TableLookupError(
            f"no grounded values were provided for table row {step.arguments[0]!r}",
            step_index=step_index,
            token=step.arguments[0],
        )
    return values


def _checked(value: Decimal, *, step_index: int) -> Decimal:
    try:
        return _validate_decimal(value, token=format(value, "E"))
    except NumericLimitError as exc:
        exc.step_index = step_index
        raise


def _execute_operation(
    operation: str,
    values: tuple[ProgramValue, ...],
    *,
    step_index: int,
    tokens: tuple[str, ...],
) -> ProgramValue:
    numeric = tuple(
        _ensure_numeric(value, step_index=step_index, token=tokens[index])
        for index, value in enumerate(values)
    )
    try:
        with localcontext() as context:
            context.prec = 50
            if operation == "add":
                result = numeric[0] + numeric[1]
            elif operation == "subtract":
                result = numeric[0] - numeric[1]
            elif operation == "multiply":
                result = numeric[0] * numeric[1]
            elif operation == "divide":
                if numeric[1].is_zero():
                    raise DivisionByZeroError(
                        "division by zero", step_index=step_index, token=tokens[1]
                    )
                result = numeric[0] / numeric[1]
            elif operation == "exp":
                exponent = numeric[1]
                if exponent != exponent.to_integral_value():
                    raise InvalidArgumentError(
                        "exp requires an integer exponent",
                        step_index=step_index,
                        token=tokens[1],
                    )
                integer_exponent = int(exponent)
                if abs(integer_exponent) > MAX_ABS_EXPONENT:
                    raise NumericLimitError(
                        f"absolute exponent exceeds {MAX_ABS_EXPONENT}",
                        step_index=step_index,
                        token=tokens[1],
                    )
                if numeric[0].is_zero() and integer_exponent < 0:
                    raise DivisionByZeroError(
                        "zero cannot be raised to a negative exponent",
                        step_index=step_index,
                        token=tokens[0],
                    )
                result = numeric[0] ** integer_exponent
            elif operation == "greater":
                # FinQA defines this as a strict yes/no question, not max(a, b).
                return numeric[0] > numeric[1]
            elif operation == "table_sum":
                result = sum(numeric, Decimal(0))
            elif operation == "table_average":
                result = sum(numeric, Decimal(0)) / Decimal(len(numeric))
            elif operation == "table_max":
                result = max(numeric)
            elif operation == "table_min":
                result = min(numeric)
            else:  # Defensive: parsing already enforces the allow-list.
                raise UnsupportedOperationError(
                    f"unsupported operation: {operation}", step_index=step_index
                )
    except ProgramError:
        raise
    except (InvalidOperation, Overflow, Underflow) as exc:
        raise NumericLimitError(
            "numeric operation exceeded safe Decimal limits", step_index=step_index
        ) from exc
    return _checked(result, step_index=step_index)


class FinQAExecutor:
    """Stateless allow-listed FinQA program executor."""

    def execute(
        self,
        program: str,
        *,
        table_values: TableValues | None = None,
    ) -> ExecutionResult:
        parsed = parse_program(program)
        required_table_labels = frozenset(
            _normalise_table_label(step.arguments[0])
            for step in parsed
            if _is_symbolic_table_step(step)
        )
        materialised_table_values = _materialise_table_values(
            table_values,
            required_labels=required_table_labels,
        )
        values: list[ProgramValue] = []
        executed: list[ExecutedStep] = []
        for step_index, step in enumerate(parsed):
            if _is_symbolic_table_step(step):
                resolved = _resolve_symbolic_table_step(
                    step,
                    materialised_table_values,
                    step_index=step_index,
                )
                execution_tokens = tuple(
                    f"{step.arguments[0]}[{index}]" for index in range(len(resolved))
                )
            else:
                resolved = tuple(
                    _resolve_argument(argument, values, step_index=step_index)
                    for argument in step.arguments
                )
                execution_tokens = step.arguments
            value = _execute_operation(
                step.operation,
                resolved,
                step_index=step_index,
                tokens=execution_tokens,
            )
            values.append(value)
            executed.append(
                ExecutedStep(
                    index=step_index,
                    operation=step.operation,
                    arguments=step.arguments,
                    resolved_arguments=resolved,
                    value=value,
                )
            )
        return ExecutionResult(program=program, steps=tuple(executed), value=values[-1])


def execute_program(
    program: str,
    *,
    table_values: TableValues | None = None,
) -> ExecutionResult:
    """Convenience wrapper around :class:`FinQAExecutor`."""

    return FinQAExecutor().execute(program, table_values=table_values)


def literal_arguments(steps: Iterable[ProgramStep]) -> tuple[str, ...]:
    """Return direct numeric leaves, excluding references and named constants."""

    literals: list[str] = []
    for step in steps:
        if _is_symbolic_table_step(step):
            continue
        for argument in step.arguments:
            stripped = argument.strip()
            if _REF_RE.fullmatch(stripped):
                continue
            if _CONST_RE.fullmatch(stripped):
                # Reject arbitrary model-invented constants even though valid
                # FinQA transformation constants do not require evidence.
                parse_decimal(stripped)
                continue
            # Validate here so callers cannot accidentally treat arbitrary text as
            # a grounded number.
            parse_decimal(stripped)
            literals.append(stripped)
    return tuple(literals)


__all__ = [
    "SUPPORTED_CONSTANTS",
    "SUPPORTED_OPERATIONS",
    "DivisionByZeroError",
    "ExecutedStep",
    "ExecutionResult",
    "FinQAExecutor",
    "InvalidArgumentError",
    "InvalidReferenceError",
    "NumericLimitError",
    "ProgramError",
    "ProgramStep",
    "ProgramSyntaxError",
    "TableLookupError",
    "TableValues",
    "UnsupportedOperationError",
    "execute_program",
    "format_decimal",
    "literal_arguments",
    "parse_decimal",
    "parse_program",
]
