#!/usr/bin/env python3
"""
Quality comparison of quantized Gemma 3 27B models on Dutch text.
Tests summarization of Dutch parliamentary proceedings.
"""

import argparse
import json
import time
from datetime import datetime

import requests

# Dutch parliamentary text sample (Tweede Kamer debate excerpt)
DUTCH_PARLIAMENT_TEXT = """
Voorzitter. De minister-president heeft zojuist een verklaring afgelegd over de klimaatmaatregelen
die het kabinet voornemens is te nemen in de komende jaren. Ik wil hier namens mijn fractie
reageren op deze plannen.

Ten eerste, de voorgestelde CO2-heffing voor de industrie. Wij zijn van mening dat deze heffing
weliswaar noodzakelijk is voor de transitie naar een duurzame economie, maar wij maken ons
ernstige zorgen over de concurrentiepositie van Nederlandse bedrijven ten opzichte van hun
buitenlandse concurrenten. De minister heeft aangegeven dat er compensatieregelingen komen,
maar de details hiervan zijn nog onduidelijk. Kan de minister toezeggen dat er voor 1 maart
een uitgewerkt plan ligt voor deze compensatiemaatregelen?

Ten tweede, de subsidieregeling voor elektrische voertuigen. Het kabinet stelt voor om de
aanschafsubsidie te verhogen van 4.000 naar 5.500 euro. Hoewel wij deze maatregel toejuichen,
vragen wij ons af of dit voldoende is om de doelstelling van 1,9 miljoen elektrische auto's
in 2030 te halen. Uit berekeningen van het Planbureau voor de Leefomgeving blijkt dat het
huidige tempo van elektrificatie onvoldoende is. Welke aanvullende maatregelen overweegt
het kabinet?

Ten derde, de woningisolatie. Het kabinet trekt 1,2 miljard euro uit voor de isolatie van
bestaande woningen. Dit is een aanzienlijk bedrag, maar gezien de opgave van 7 miljoen te
isoleren woningen, lijkt dit onvoldoende. Bovendien zijn er grote zorgen over de capaciteit
in de bouwsector. Er is een tekort van naar schatting 40.000 vakmensen in de isolatiebranche.
Hoe denkt de minister dit capaciteitsprobleem op te lossen?

Tot slot wil ik aandacht vragen voor de sociale aspecten van de energietransitie. De stijgende
energiekosten raken met name huishoudens met lagere inkomens. Het kabinet heeft weliswaar
een energietoeslag aangekondigd, maar deze is tijdelijk van aard. Wij pleiten voor structurele
maatregelen om energie-armoede te bestrijden. Denk hierbij aan gerichte subsidies voor
woningisolatie in achterstandswijken en een sociaal energietarief voor minima.

Voorzitter, ik rond af. De klimaatopgave is immens en vraagt om daadkrachtig beleid. Wij
steunen de richting die het kabinet inslaat, maar hebben nog veel vragen over de uitwerking.
Ik zie uit naar de antwoorden van de minister.
"""

SUMMARIZATION_PROMPT = """Je bent een ervaren parlementair journalist die gespecialiseerd is in het samenvatten van Kamerdebatten.

Maak een gestructureerde samenvatting van onderstaande parlementaire bijdrage. De samenvatting moet:

1. **Hoofdpunten** (maximaal 4 bullets): De kernpunten van het betoog
2. **Vragen aan de minister** (genummerde lijst): Concrete vragen die gesteld worden
3. **Standpunt fractie** (1-2 zinnen): Het algemene standpunt van de spreker
4. **Toon** (1 woord): De toon van het betoog (bijv. kritisch, constructief, bezorgd)

Gebruik formele, zakelijke taal passend bij parlementaire verslaggeving.

---

PARLEMENTAIRE BIJDRAGE:

{text}

---

SAMENVATTING:"""


def query_model(url: str, model: str, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> dict:
    """Query a vLLM model and return response with timing."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    start = time.perf_counter()
    response = requests.post(
        f"{url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )
    elapsed = time.perf_counter() - start

    response.raise_for_status()
    data = response.json()

    return {
        "content": data["choices"][0]["message"]["content"],
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "elapsed": elapsed,
        "tokens_per_sec": data["usage"]["completion_tokens"] / elapsed,
    }


def run_comparison(url: str, models: list[str], temperature: float = 0.3):
    """Run the same prompt through multiple models and compare."""
    prompt = SUMMARIZATION_PROMPT.format(text=DUTCH_PARLIAMENT_TEXT)

    results = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"Testing: {model}")
        print(f"{'='*60}")

        try:
            result = query_model(url, model, prompt, temperature)
            results[model] = result

            print(f"\nTokens: {result['completion_tokens']} in {result['elapsed']:.2f}s ({result['tokens_per_sec']:.1f} tok/s)")
            print(f"\n--- OUTPUT ---\n")
            print(result["content"])
            print(f"\n--- END ---\n")

        except Exception as e:
            print(f"ERROR: {e}")
            results[model] = {"error": str(e)}

    return results


def save_results(results: dict, filename: str):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "test": "Dutch parliamentary summarization",
        "temperature": 0.3,
        "input_text_preview": DUTCH_PARLIAMENT_TEXT[:200] + "...",
        "results": results,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Compare quantized model quality on Dutch text")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--model", required=True, help="Model name to test")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature (default: 0.3)")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("Dutch Parliamentary Text Summarization Test")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Input text: {len(DUTCH_PARLIAMENT_TEXT)} characters")

    prompt = SUMMARIZATION_PROMPT.format(text=DUTCH_PARLIAMENT_TEXT)

    try:
        result = query_model(args.url, args.model, prompt, args.temperature)

        print(f"\n{'='*60}")
        print(f"RESULT")
        print(f"{'='*60}")
        print(f"Tokens: {result['completion_tokens']} in {result['elapsed']:.2f}s ({result['tokens_per_sec']:.1f} tok/s)")
        print(f"\n{result['content']}")

        if args.output:
            save_results({args.model: result}, args.output)

        return result

    except Exception as e:
        print(f"ERROR: {e}")
        return None


if __name__ == "__main__":
    main()
