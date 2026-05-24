"""
tifa.py — generer bilder via xAI Grok Imagine API.
Konfigurasjon leses fra tifa.yaml (se den filen for alle valg).

Bruk:
    python tifa.py
    python tifa.py --config annen_fil.yaml
    python tifa.py --prompt "custom prompt her"   (overstyrer yaml)
"""

import argparse
import base64
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_client() -> OpenAI:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        # Prøv å hente fra config.yaml (prosjektets eksisterende konfig)
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                proj_cfg = yaml.safe_load(f)
            api_key = proj_cfg.get("xai", {}).get("api_key")
        except FileNotFoundError:
            pass
    if not api_key:
        sys.exit("Feil: XAI_API_KEY er ikke satt. Legg den i miljøvariabel eller config.yaml.")
    return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")


def next_filename(output_dir: Path, prefix: str, index: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{prefix}_{timestamp}_{index:03d}.png"


def generate(cfg: dict, prompt_override: str | None = None) -> None:
    prompt = prompt_override or cfg["prompt"]
    model = cfg.get("model", "grok-imagine-image-quality")
    n = int(cfg.get("n", 1))
    aspect_ratio = cfg.get("aspect_ratio", "1:1")
    resolution = cfg.get("resolution", "1k")
    response_format = cfg.get("response_format", "b64_json")
    output_dir = Path(cfg.get("output_dir", "./generated"))
    prefix = cfg.get("filename_prefix", "image")

    output_dir.mkdir(parents=True, exist_ok=True)
    client = build_client()

    print(f"Modell     : {model}")
    print(f"Prompt     : {prompt}")
    print(f"Antall     : {n}")
    print(f"Ratio      : {aspect_ratio}  |  Oppløsning: {resolution}")
    print(f"Lagres i   : {output_dir.resolve()}")
    print()

    extra = {
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    }

    response = client.images.generate(
        model=model,
        prompt=prompt,
        n=n,
        response_format=response_format,
        extra_body=extra,
    )

    for i, img_data in enumerate(response.data, start=1):
        out_path = next_filename(output_dir, prefix, i)

        if response_format == "b64_json":
            raw = base64.b64decode(img_data.b64_json)
            out_path.write_bytes(raw)
        else:
            # response_format == "url" — last ned fra URL
            import urllib.request
            urllib.request.urlretrieve(img_data.url, out_path)

        print(f"  [{i}/{n}] Lagret: {out_path}")

    print("\nFerdig!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generer bilder med xAI Grok Imagine")
    parser.add_argument("--config", default="tifa.yaml", help="Sti til yaml-konfig (standard: tifa.yaml)")
    parser.add_argument("--prompt", default=None, help="Overstyr prompt fra yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate(cfg, prompt_override=args.prompt)


if __name__ == "__main__":
    main()
