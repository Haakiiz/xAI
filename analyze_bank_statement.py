from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib import error, request

DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4-1-fast-reasoning"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/133.0.0.0 Safari/537.36"
)


@dataclass
class Transaction:
    date: str
    description: str
    amount: float


def load_env(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ if missing."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def _pick_column(headers: list[str], candidates: list[str]) -> str:
    normalized = {_normalize_header(h): h for h in headers}
    for candidate in candidates:
        key = _normalize_header(candidate)
        if key in normalized:
            return normalized[key]
    raise ValueError(
        "Fant ikke forventet kolonne. Tilgjengelige kolonner: " + ", ".join(headers)
    )


def _normalize_header(value: str) -> str:
    cleaned = value.strip().lower().replace("\ufeff", "")
    replacements = {
        "Ã¸": "o",
        "ã¸": "o",
        "Ã¥": "a",
        "ã¥": "a",
        "Ã¦": "ae",
        "ã¦": "ae",
        "ø": "o",
        "å": "a",
        "æ": "ae",
    }
    for src, dst in replacements.items():
        cleaned = cleaned.replace(src, dst)
    return re.sub(r"[^a-z0-9]+", "", cleaned)


def _parse_amount(value: str) -> float:
    cleaned = value.strip().replace(" ", "")
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    return float(cleaned)


def read_transactions(csv_path: Path) -> list[Transaction]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t")
            reader = csv.DictReader(handle, dialect=dialect)
        except csv.Error:
            reader = csv.DictReader(handle, delimiter=";")

        if not reader.fieldnames:
            raise ValueError("CSV-filen mangler header-rad.")

        headers = list(reader.fieldnames)
        date_col = _pick_column(
            headers, ["bokforing", "bokforingsdato", "dato", "date", "booking date"]
        )
        description_candidates = [
            "tittel",
            "navn",
            "tekst",
            "description",
            "beskrivelse",
            "detaljer",
            "betalingstype",
        ]
        normalized_description_candidates = {
            _normalize_header(candidate) for candidate in description_candidates
        }
        description_cols = [
            header
            for header in headers
            if _normalize_header(header) in normalized_description_candidates
        ]
        if not description_cols:
            raise ValueError(
                "Fant ikke beskrivelseskolonne. Tilgjengelige kolonner: "
                + ", ".join(headers)
            )
        amount_col = _pick_column(headers, ["belop", "amount", "sum", "debit", "credit"])

        rows: list[Transaction] = []
        for row in reader:
            amount = _parse_amount(row[amount_col])
            description = next(
                (str(row.get(col, "")).strip() for col in description_cols if str(row.get(col, "")).strip()),
                "",
            )
            rows.append(
                Transaction(
                    date=row[date_col].strip(),
                    description=description,
                    amount=amount,
                )
            )
        return rows


def summarize_transactions(transactions: Iterable[Transaction]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = defaultdict(
        lambda: {"count": 0, "total": 0.0, "amounts": []}
    )

    for tx in transactions:
        key = tx.description.lower()
        bucket = grouped[key]
        bucket["description"] = tx.description
        bucket["count"] += 1
        bucket["total"] += tx.amount
        bucket["amounts"].append(tx.amount)

    summary = []
    for bucket in grouped.values():
        amounts = bucket["amounts"]
        summary.append(
            {
                "description": bucket["description"],
                "count": bucket["count"],
                "total_amount": round(float(bucket["total"]), 2),
                "average_amount": round(sum(amounts) / len(amounts), 2),
                "min_amount": round(min(amounts), 2),
                "max_amount": round(max(amounts), 2),
            }
        )

    summary.sort(key=lambda item: (item["count"], abs(item["total_amount"])), reverse=True)
    return summary


def ask_grok(
    summary: list[dict[str, object]],
    api_key: str,
    model: str,
    base_url: str,
    user_agent: str,
) -> str:
    system_prompt = (
        "Du er en norsk okonomiassistent. Finn sannsynlige abonnementer og lopende kostnader "
        "som kan vaere unodvendige. Prioriter tydelighet og konkrete handlinger."
    )
    user_prompt = (
        "Her er en oppsummering av transaksjoner gruppert per beskrivelse. "
        "Beskrivelse representerer fortrinnsvis feltet 'Tittel' i CSV (hvem jeg har betalt til / merchant). "
        "Ikke tolk dette som avsenderkonto. "
        "Lag en rapport med disse seksjonene:\n"
        "1) Sannsynlige abonnementer (med manedlig estimat)\n"
        "2) Andre lopende kostnader a vurdere\n"
        "3) Mulige 'uonskede' kostnader\n"
        "4) Konkrete neste steg\n\n"
        "Bruk norsk sprak. Dersom du er usikker, merk det tydelig.\n\n"
        f"Data:\n{json.dumps(summary, ensure_ascii=False, indent=2)}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    req = request.Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
        data=json.dumps(payload).encode("utf-8"),
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"xAI API-feil {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Nettverksfeil mot xAI API: {exc}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Mangler svar fra Grok (ingen choices).")

    message = choices[0].get("message", {})
    content = message.get("content")
    if not content:
        raise RuntimeError("Mangler tekstinnhold i svar fra Grok.")
    return str(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyser bankutskrift i CSV med Grok for a finne abonnement og uonskede kostnader."
    )
    parser.add_argument("csv_file", type=Path, help="Sti til bankutskrift i CSV-format")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Sti til .env")
    parser.add_argument("--base-url", default=os.getenv("XAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("XAI_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--user-agent",
        default=os.getenv("XAI_USER_AGENT", DEFAULT_USER_AGENT),
        help="User-Agent brukt mot xAI API",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Valgfritt: lagre oppsummerte transaksjoner til JSON",
    )
    args = parser.parse_args()

    load_env(args.env_file)
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise SystemExit("XAI_API_KEY mangler. Legg den i .env eller miljo variabler.")

    transactions = read_transactions(args.csv_file)
    if not transactions:
        raise SystemExit("Ingen transaksjoner funnet i CSV.")

    summary = summarize_transactions(transactions)

    if args.json_out:
        args.json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report = ask_grok(
        summary=summary,
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        user_agent=args.user_agent,
    )

    print("=" * 80)
    print("Grok-analyse av bankutskrift")
    print(f"Kjort: {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 80)
    print(report)


if __name__ == "__main__":
    main()

