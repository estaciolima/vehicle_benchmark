"""Generic parser for Carros na Web technical sheet PDFs.

The PDFs from this source usually share a predictable structure:
- page 1 and 2 contain vehicle overview and technical specifications
- later pages contain equipment lists grouped by sections such as Segurança,
  Conforto and Infotenimento

This parser uses lightweight heuristics so it can adapt to similar PDFs with
different line counts and slightly different labels.
"""

from __future__ import annotations

from collections import OrderedDict
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from openai_llm_helper import openai_chat_json
from vehicle_pdf_utils import build_dataframe, load_pdf


SECTION_HEADERS = {
    "MOTOR",
    "TRANSMISSÃO",
    "SUSPENSÃO",
    "FREIOS",
    "DIREÇÃO",
    "PNEUS",
    "DIMENSÕES",
    "DESEMPENHO",
    "CONSUMO",
    "AUTONOMIA",
    "SEGURANÇA",
    "CONFORTO",
    "INFOTENIMENTO",
    "EQUIPAMENTOS",
}

NOISE_PREFIXES = (
    "Página Principal",
    "Carros na Web",
    "https://",
    "http://",
    "Legenda:",
    "Fotos",
    "Post",
    "Ficha Técnica",
    "Busca detalhada",
    "Co",
    "As informações no website",
    "As informações contidas no website",
    "Algumas informações podem não estar atualizadas",
    "Material ilustrativo sem valor",
    "Sobre as informações dos veículos",
    "providenciar uma informação precisa",
    "informações fornecidas",
    "1 Valor aproximado",
    "2 Preço médio aproximado",
    "3 Antes de adquirir o óleo do motor",
    "Fale Conosco",
    "Equipamento de série",
    "Equipamento opcional",
    "Mapa do site",
    "Sobre o site",
    "Privacidade",
    "Termos de uso",
    "Mobile",
)

SUMMARY_LABEL_PREFIXES = (
    "Ano",
    "Preço",
    "Propulsão",
    "Combustível",
    "IPVA",
    "Seguro",
    "Revisões",
    "Procedência",
    "Garantia",
    "Configuração",
    "Porte",
    "Lugares",
    "Portas",
    "Plataforma",
    "Nota do leitor",
    "Índice CNW",
    "Latin NCAP",
    "Ranking CNW",
    "Proteção adulto",
    "Proteção a pedestres",
    "Proteção infantil",
    "Assistência segurança",
)

SUMMARY_PENDING_PREFIXES = (
    "Nota do leitor",
    "Índice CNW",
    "Latin NCAP",
    "Ranking CNW",
    "Proteção adulto",
    "Proteção a pedestres",
    "Proteção infantil",
    "Assistência segurança",
)

DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4},\s*\d{1,2}:\d{2}$")
PAGE_RE = re.compile(r"^\d+/\d+$")
RANGE_VALUE_RE = re.compile(r"^\d+(?:[.,]\d+)?(?:\s+[A-Za-z%/().-]+)?$")


@dataclass
class ParsedRow:
    label: str
    value: str
    section: str | None = None
    subsection: str | None = None
def _iter_block_lines(pdf_path: str | Path):
    """Yield cleaned lines from page blocks in reading order."""
    doc = load_pdf(pdf_path)
    for page_number in range(1, doc.page_count + 1):
        page = doc.load_page(page_number - 1)
        blocks = page.get_text("blocks")
        for block in sorted(blocks, key=lambda b: (round(b[1], 1), b[0])):
            text = block[4].replace("\r", "\n").replace("\u00a0", " ")
            if not text:
                continue
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if line:
                    yield page_number, block[0], block[1], line


def _split_rendered_line(line: str) -> list[str]:
    """Split a rendered line into logical chunks.

    Carros na Web often places two equipment items on the same visual line.
    Splitting on large gaps helps recover the left and right columns separately
    without affecting compact labels such as "Zonas de ar-condicionado: 2".
    """

    parts = [part.strip() for part in re.split(r"\s{2,}", line) if part.strip()]
    return parts or [line]


def _normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _is_noise(line: str) -> bool:
    if not line:
        return True
    if line == "|":
        return True
    if not any(ch.isalpha() for ch in line) and not any(ch.isdigit() for ch in line):
        return True
    if re.match(r"^\d{2}/\d{2}/\d{4}", line):
        return True
    if DATE_RE.match(line) or PAGE_RE.match(line):
        return True
    return any(line.startswith(prefix) for prefix in NOISE_PREFIXES)


def _is_equipment_section(section: str | None) -> bool:
    return section in {"SEGURANÇA", "CONFORTO", "INFOTENIMENTO", "EQUIPAMENTOS"}


def _is_section_header(line: str) -> bool:
    if line in SECTION_HEADERS:
        return True
    tokens = line.split()
    if 1 <= len(tokens) <= 3 and all(tok.isupper() for tok in tokens):
        return True
    return False


def _is_subsection_title(line: str) -> bool:
    if ":" in line:
        return False
    if any(char.isdigit() for char in line):
        return False
    tokens = line.split()
    if not (1 <= len(tokens) <= 5):
        return False
    if _is_section_header(line):
        return False
    # Titles generally have a capitalized first token and then lowercase words.
    if tokens[0][:1].isupper() and all(
        tok[:1].islower() or tok.lower() in {"e", "de", "do", "da", "dos", "das", "a", "o"}
        for tok in tokens[1:]
    ):
        return True
    return False


def _is_pair_line(line: str) -> bool:
    if ":" in line:
        return True
    tokens = line.split()
    if len(tokens) < 2:
        return False
    if any(char.isdigit() for char in line):
        return True
    if tokens[1][:1].isupper():
        return True
    return False


def _is_summary_row(line: str) -> bool:
    return any(line.startswith(prefix) for prefix in SUMMARY_LABEL_PREFIXES)


def _extract_value_from_line(line: str) -> str | None:
    line = _normalize_line(line)
    if not line:
        return None
    if ":" in line:
        return line.split(":", 1)[1].strip() or None
    tokens = line.split()
    if not tokens:
        return None
    if re.match(r"^\d", tokens[0]):
        return tokens[0]
    if tokens[0].lower() in {"r$", "r"} and len(tokens) > 1:
        return f"{tokens[0]} {tokens[1]}".strip()
    if len(tokens) == 1:
        return tokens[0]
    if tokens[0].lower() in {"presente", "sim", "não", "nao"}:
        return tokens[0]
    # Prefer short numeric-like prefixes such as "6,1 Avalie".
    if re.match(r"^\d+(?:[.,]\d+)?$", tokens[0]) and len(tokens) > 1:
        return tokens[0]
    return None


def _strip_footnote_suffix(value: str) -> str:
    value = value.strip()
    currency_match = re.match(r"^(R\$\s*)(\d[\d.]*)(\d)$", value)
    if currency_match:
        last_group = value.split(".")[-1]
        if len(last_group) == 4:
            return value[:-1]
    return value


def _looks_like_value_start(token: str) -> bool:
    if not token:
        return False
    token_lower = token.lower()
    if token[0].isdigit() or token_lower.startswith("r$"):
        return True
    if any(ch.isdigit() for ch in token):
        return True
    if token_lower in {
        "presente",
        "dianteiro",
        "dianteira",
        "traseiro",
        "traseira",
        "transversal",
        "natural",
        "turbocompressor",
        "elétrica",
        "eletrica",
        "hidráulicos",
        "hidraulicos",
        "mcpherson",
        "multibraço",
        "multibraco",
        "disco",
        "flex",
        "importado",
        "nacional",
    }:
        return True
    return False


def _split_pair_line(line: str) -> tuple[str, str] | None:
    line = _normalize_line(line)
    if not line:
        return None
    if ":" in line:
        left, right = line.split(":", 1)
        return left.strip(), right.strip()

    tokens = line.split()
    if len(tokens) < 2:
        return None

    # Find the first token that clearly looks like the start of the value.
    split_index = None
    for idx in range(1, len(tokens)):
        if _looks_like_value_start(tokens[idx]):
            split_index = idx
            break
        if idx >= 2 and tokens[idx][:1].isupper() and tokens[idx - 1][:1].islower():
            split_index = idx
            break

    if split_index is None:
        # Fallback: for short lines assume the first word is the label.
        if len(tokens) <= 4:
            return tokens[0], " ".join(tokens[1:])
        return None

    label = " ".join(tokens[:split_index]).strip()
    value = " ".join(tokens[split_index:]).strip()
    if label and value:
        return label, value
    return None


def _section_prefix(section: str | None, subsection: str | None, label: str) -> str:
    if subsection and section == "MOTOR":
        return f"{subsection} - {label}"
    return label


def _should_keep_as_presence(line: str, current_section: str | None) -> bool:
    if current_section == "EQUIPAMENTOS":
        return True
    if current_section in {"SEGURANÇA", "CONFORTO", "INFOTENIMENTO"}:
        return True
    return False


def _append_pending_label(
    rows: list[ParsedRow],
    pending_label: str | None,
    current_section: str | None,
    current_subsection: str | None,
) -> None:
    if pending_label is None:
        return
    rows.append(
        ParsedRow(
            label=_section_prefix(current_section, current_subsection, pending_label),
            value="Presente",
            section=current_section,
            subsection=current_subsection,
        )
    )


def _parse_equipment_buffer(section: str, lines: list[str]) -> list[ParsedRow]:
    """Parse equipment-section lines deterministically as a local fallback."""

    rows: list[ParsedRow] = []
    for line in lines:
        normalized = _normalize_line(line)
        if not normalized or _is_noise(normalized):
            continue
        if ":" in normalized:
            left, right = normalized.split(":", 1)
            label = left.strip()
            value = right.strip() or "Presente"
        else:
            label = normalized
            value = "Presente"
        rows.append(ParsedRow(label=label, value=value, section=section))
    return rows


def _llm_parse_equipment_buffer(section: str, lines: list[str]) -> list[ParsedRow]:
    """Ask the LLM to normalize one equipment section into structured rows."""

    if not lines:
        return []

    prompt = (
        "Você extrai atributos veiculares de uma ficha técnica do Carros na Web.\n"
        "A seção atual é: {section}.\n"
        "Use somente as linhas fornecidas abaixo. Não invente dados.\n"
        "Se uma linha representar apenas um item ou equipamento sem valor explícito, use value='Presente'.\n"
        "Se uma linha contiver dois itens na mesma linha, separe-os em registros distintos.\n"
        "Ignore rodapé, aviso legal, título de página e texto de navegação.\n"
        "Responda somente em JSON no formato:\n"
        "{{\"rows\": [{{\"label\": \"...\", \"value\": \"...\"}}]}}\n\n"
        "Linhas:\n"
        "{lines}\n"
    ).format(section=section, lines="\n".join(f"- {line}" for line in lines))

    response = openai_chat_json(
        messages=[
            {
                "role": "system",
                "content": "Você devolve apenas JSON válido e preserva os atributos presentes no texto.",
            },
            {"role": "user", "content": prompt},
        ]
    )
    if not response:
        return []

    rows_data = response.get("rows") if isinstance(response, dict) else response
    if not isinstance(rows_data, list):
        return []

    rows: list[ParsedRow] = []
    for item in rows_data:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        value = str(item.get("value", "")).strip() or "Presente"
        if not label:
            continue
        rows.append(ParsedRow(label=label, value=value, section=section))
    return rows


def _llm_parse_context_rows(context: str | None, lines: list[str]) -> list[ParsedRow]:
    """Use the LLM to normalize the rows of one logical document context."""

    cleaned_lines = [line for line in (_normalize_line(line) for line in lines) if line and not _is_noise(line)]
    if not cleaned_lines:
        return []

    context_label = "SUMMARY" if context is None else context
    prompt = (
        "You are extracting attributes from a Carros na Web vehicle spec sheet.\n"
        f"Current context: {context_label}.\n"
        "Only use the lines below. Do not invent or infer values beyond the text.\n"
        "Keep labels exactly as written in the source whenever possible.\n"
        "If a line is just a feature name, set value to Presente.\n"
        "If a rendered line contains two items, split them into separate rows.\n"
        "If a line is a short attribute-value pair, preserve the value exactly.\n"
        "Do not truncate labels and do not merge neighboring items.\n"
        "Ignore any footer, disclaimer, navigation, or page-number text.\n"
        "Return JSON only in this schema:\n"
        "{\"rows\": [{\"label\": \"...\", \"value\": \"...\"}]}\n\n"
        "Lines:\n"
        + "\n".join(f"- {line}" for line in cleaned_lines)
    )

    response = openai_chat_json(
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON with a rows array.",
            },
            {"role": "user", "content": prompt},
        ]
    )
    if not response:
        return []

    rows_data = response.get("rows") if isinstance(response, dict) else response
    if not isinstance(rows_data, list):
        return []

    rows: list[ParsedRow] = []
    for item in rows_data:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        value = _strip_footnote_suffix(str(item.get("value", "")).strip()) or "Presente"
        if not label:
            continue
        rows.append(ParsedRow(label=label, value=value, section=context))
    return rows


def _merge_rows(rows: list[ParsedRow]) -> list[ParsedRow]:
    """Drop exact duplicates while keeping the first occurrence order."""

    seen: set[tuple[str, str, str | None, str | None]] = set()
    merged: list[ParsedRow] = []
    for row in rows:
        key = (row.label.strip(), row.value.strip(), row.section, row.subsection)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def _llm_is_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _looks_low_confidence(lines: list[str]) -> bool:
    """Heuristic signal that a context may benefit from LLM repair."""

    if len(lines) >= 8:
        return True
    if any(":" in line for line in lines):
        return True
    if sum(len(line.split()) for line in lines) >= 24:
        return True
    return False


def _replace_rows_for_context(
    rows: list[ParsedRow],
    context: str | None,
    replacement_rows: list[ParsedRow],
) -> list[ParsedRow]:
    """Replace heuristic rows for one context with LLM-normalized rows."""

    if not replacement_rows:
        return rows

    new_rows: list[ParsedRow] = []
    inserted = False
    context_found = False

    for row in rows:
        row_context = row.section if row.section is not None else None
        if row_context == context:
            context_found = True
            if not inserted:
                new_rows.extend(replacement_rows)
                inserted = True
            continue
        new_rows.append(row)

    if not context_found:
        return rows + replacement_rows

    return new_rows


def _parse_generic_rows(pdf_path: str | Path, use_llm_fallback: bool = True) -> list[ParsedRow]:
    """Parse a PDF using geometry-aware heuristics.

    The deterministic parser handles the easy parts. When enabled, the LLM
    fallback normalizes the equipment sections that tend to have two columns,
    merged labels, or truncated words.
    """

    rows: list[ParsedRow] = []
    context_buffers: "OrderedDict[str | None, list[str]]" = OrderedDict()
    current_section: str | None = None
    current_subsection: str | None = None
    pending_label: str | None = None
    started_body = False
    llm_enabled = bool(use_llm_fallback and _llm_is_configured())

    for _page_number, _x0, _y0, line in _iter_block_lines(pdf_path):
        for chunk in _split_rendered_line(line):
            normalized_chunk = _normalize_line(chunk)
            if _is_noise(normalized_chunk):
                continue

            if not started_body:
                if _is_summary_row(normalized_chunk) or _is_section_header(normalized_chunk):
                    started_body = True
                else:
                    continue

            if _is_section_header(normalized_chunk):
                _append_pending_label(rows, pending_label, current_section, current_subsection)
                pending_label = None
                current_section = normalized_chunk if normalized_chunk in SECTION_HEADERS else normalized_chunk.upper()
                current_subsection = None
                continue

            if llm_enabled:
                context_buffers.setdefault(current_section, []).append(normalized_chunk)

            if _is_subsection_title(normalized_chunk) and not _is_equipment_section(current_section):
                _append_pending_label(rows, pending_label, current_section, current_subsection)
                pending_label = None
                current_subsection = normalized_chunk
                continue

            if pending_label is not None:
                value = _extract_value_from_line(normalized_chunk)
                if value is not None:
                    rows.append(
                        ParsedRow(
                            label=_section_prefix(current_section, current_subsection, pending_label),
                            value=_strip_footnote_suffix(value),
                            section=current_section,
                            subsection=current_subsection,
                        )
                    )
                    pending_label = None
                    continue

                _append_pending_label(rows, pending_label, current_section, current_subsection)
                pending_label = None

            if _should_keep_as_presence(normalized_chunk, current_section) or _is_equipment_section(current_section):
                if ":" in normalized_chunk:
                    left, right = normalized_chunk.split(":", 1)
                    rows.append(
                        ParsedRow(
                            label=_section_prefix(current_section, current_subsection, left.strip()),
                            value=_strip_footnote_suffix(right.strip()) or "Presente",
                            section=current_section,
                            subsection=current_subsection,
                        )
                    )
                else:
                    rows.append(
                        ParsedRow(
                            label=_section_prefix(current_section, current_subsection, normalized_chunk),
                            value="Presente",
                            section=current_section,
                            subsection=current_subsection,
                        )
                    )
                continue

            pair = _split_pair_line(normalized_chunk)
            if pair is not None:
                label, value = pair
                rows.append(
                    ParsedRow(
                        label=_section_prefix(current_section, current_subsection, label),
                        value=_strip_footnote_suffix(value),
                        section=current_section,
                        subsection=current_subsection,
                    )
                )
                continue

            if current_section is None and not _is_summary_row(normalized_chunk):
                continue

            pending_label = normalized_chunk

    _append_pending_label(rows, pending_label, current_section, current_subsection)

    if llm_enabled:
        heuristic_counts: dict[str | None, int] = {}
        for row in rows:
            row_context = row.section if row.section is not None else None
            heuristic_counts[row_context] = heuristic_counts.get(row_context, 0) + 1

        for context, context_lines in context_buffers.items():
            if not _looks_low_confidence(context_lines):
                continue
            parsed = _llm_parse_context_rows(context, context_lines)
            if not parsed:
                if context is None:
                    continue
                parsed = _parse_equipment_buffer(context, context_lines) if _is_equipment_section(context) else []
            if len(parsed) >= heuristic_counts.get(context, 0):
                rows = _replace_rows_for_context(rows, context, parsed)

    return _merge_rows(rows)


def parse_pdf(
    pdf_path: str | Path,
    force_generic: bool = False,
    use_llm_fallback: bool | None = None,
) -> pd.DataFrame:
    """Parse a Carros na Web PDF into a two-column dataframe.

    ``force_generic=True`` keeps the parser in heuristic-only mode. When the
    optional LLM fallback is enabled and an API key is configured, the parser
    asks the model to normalize only the hardest sections.
    """

    if use_llm_fallback is None:
        use_llm_fallback = not force_generic
    rows = _parse_generic_rows(pdf_path, use_llm_fallback=use_llm_fallback)
    return build_dataframe([(row.label, row.value) for row in rows])


def parse_pdf_with_sections(
    pdf_path: str | Path,
    force_generic: bool = False,
    use_llm_fallback: bool | None = None,
) -> pd.DataFrame:
    """Parse a PDF and keep the section context in an auxiliary column."""

    if use_llm_fallback is None:
        use_llm_fallback = not force_generic
    rows = _parse_generic_rows(pdf_path, use_llm_fallback=use_llm_fallback)
    df = pd.DataFrame(
        {
            "section": [row.section for row in rows],
            "subsection": [row.subsection for row in rows],
            "Atributo veicular": [row.label for row in rows],
            "Valor": [row.value for row in rows],
        }
    )
    return df
