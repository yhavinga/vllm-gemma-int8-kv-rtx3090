# Dutch Language Quality Comparison

**Date:** 2026-03-15
**Test:** Parliamentary text summarization (Tweede Kamer debate)
**Temperature:** 0.3

## Test Setup

- **Input:** Dutch parliamentary speech (~2300 chars) about climate policy
- **Task:** Structured summarization with specific format requirements
- **Models tested:**
  - RedHatAI/gemma-3-27b-it-quantized.w4a16 (compressed-tensors W4A16)
  - ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g (GPTQ 4-bit 128g)

---

## Results

### RedHatAI W4A16 Output

**Tokens:** 408 in 6.96s (58.6 tok/s)

```markdown
## Samenvatting Parlementaire Bijdrage Klimaatmaatregelen

**1. Hoofdpunten:**

*   De fractie steunt de algemene richting van het klimaatbeleid, maar maakt zich zorgen
    over de uitvoerbaarheid en effectiviteit van de voorgestelde maatregelen.
*   Er zijn vragen over de concurrentiepositie van Nederlandse bedrijven door de CO2-heffing,
    ondanks aangekondigde compensatieregelingen.
*   De fractie acht de voorgestelde subsidieregeling voor elektrische voertuigen mogelijk
    onvoldoende om de gestelde doelen te bereiken.
*   Er is bezorgdheid over de capaciteit in de bouwsector voor woningisolatie en de sociale
    impact van de energietransitie, met name energie-armoede.

**2. Vragen aan de minister:**

1.  Kan de minister toezeggen dat er voor 1 maart een uitgewerkt plan ligt voor de
    compensatiemaatregelen rondom de CO2-heffing voor de industrie?
2.  Welke aanvullende maatregelen overweegt het kabinet om het doel van 1,9 miljoen
    elektrische auto's in 2030 te halen, gezien het huidige tempo van elektrificatie?
3.  Hoe denkt de minister het capaciteitsprobleem in de isolatiebranche, met een tekort
    van naar schatting 40.000 vakmensen, op te lossen?

**3. Standpunt fractie:**

De fractie steunt de ambitie van het kabinet op het gebied van klimaat, maar benadrukt
de noodzaak van een zorgvuldige uitwerking en het adresseren van potentiële knelpunten
op het gebied van concurrentiepositie, capaciteit en sociale rechtvaardigheid.

**4. Toon:**

Constructief-kritisch.
```

---

### ISTA-DASLab GPTQ Output

**Tokens:** 417 in 7.06s (59.1 tok/s)

```markdown
## Samenvatting Parlementaire Bijdrage Klimaatmaatregelen

**1. Hoofdpunten:**

*   De CO2-heffing voor de industrie wordt als noodzakelijk gezien, maar er zijn zorgen
    over de concurrentiepositie van Nederlandse bedrijven.
*   De verhoging van de aanschafsubsidie voor elektrische voertuigen wordt toegejuichd,
    maar mogelijk onvoldoende om de doelstelling van 1,9 miljoen elektrische auto's
    in 2030 te halen.
*   De investering in woningisolatie wordt als onvoldoende beschouwd gezien de omvang
    van de opgave en het tekort aan vakmensen.
*   Er is behoefte aan structurele maatregelen om energie-armoede te bestrijden, naast
    de tijdelijke energietoeslag.

**2. Vragen aan de minister:**

1.  Kan de minister toezeggen dat er voor 1 maart een uitgewerkt plan ligt voor de
    compensatiemaatregelen rondom de CO2-heffing voor de industrie?
2.  Welke aanvullende maatregelen overweegt het kabinet om het tempo van elektrificatie
    te versnellen en de doelstelling van 1,9 miljoen elektrische auto's in 2030 te halen?
3.  Hoe denkt de minister het capaciteitsprobleem in de isolatiebranche, met een tekort
    van naar schatting 40.000 vakmensen, op te lossen?

**3. Standpunt fractie:**

De fractie steunt de algemene richting van het kabinetsbeleid op het gebied van
klimaatmaatregelen, maar is bezorgd over de uitwerking en de impact op de
concurrentiepositie, de haalbaarheid van de doelstellingen en de sociale aspecten
van de energietransitie.

**4. Toon:**

Constructief.
```

---

## Quality Analysis

### Similarities (Both models)

| Aspect | Assessment |
|--------|------------|
| **Dutch grammar** | Correct, formal |
| **Vocabulary** | Appropriate parliamentary register |
| **Structure** | Follows requested format exactly |
| **Content accuracy** | All key points captured |
| **Questions extracted** | Same 3 questions identified |

### Differences

| Aspect | RedHatAI W4A16 | GPTQ 4b-128g |
|--------|----------------|--------------|
| **Tone classification** | "Constructief-kritisch" | "Constructief" |
| **Detail level** | Slightly more interpretive | More factual/direct |
| **Token count** | 408 | 417 |
| **Style** | More analytical framing | More straightforward summary |

### Specific Observations

1. **Tone assessment:** RedHatAI adds "-kritisch" which is arguably more accurate given the questioning nature of the speech.

2. **Hoofdpunt 1:**
   - RedHatAI: "maakt zich zorgen over de uitvoerbaarheid en effectiviteit"
   - GPTQ: "er zijn zorgen over de concurrentiepositie"
   - *RedHatAI provides broader interpretation*

3. **Standpunt fractie:**
   - RedHatAI: mentions "zorgvuldige uitwerking" and "sociale rechtvaardigheid"
   - GPTQ: mentions "haalbaarheid van de doelstellingen"
   - *Both capture the essence, slightly different emphasis*

---

## Conclusion

**Both models produce high-quality Dutch output with no significant quality difference.**

| Metric | RedHatAI W4A16 | GPTQ 4b-128g |
|--------|----------------|--------------|
| Dutch quality | Excellent | Excellent |
| Accuracy | High | High |
| Format adherence | 100% | 100% |
| Speed | 58.6 tok/s | 59.1 tok/s |
| **Recommendation** | **Preferred** | Good alternative |

**RedHatAI is slightly preferred** because:
1. More nuanced tone classification ("constructief-kritisch")
2. Broader interpretive framing in summary
3. Official Red Hat support and maintenance

However, **both are production-ready** for Dutch language tasks.

---

## Files

- `quality-redhat.json` - Full RedHatAI test output
- `quality-gptq.json` - Full GPTQ test output
- `quality_compare.py` - Test script
