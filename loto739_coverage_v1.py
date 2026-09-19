#!/usr/bin/env python3
"""Loto 7/39: ciklusi cifara, tacni prelazi i NEXT sedmorka."""

import argparse
import csv
import math
from collections import Counter
from functools import lru_cache

ALL = (1 << 10) - 1
TOTAL = math.comb(39, 7)
DEFAULT_CSV = "/Users/4c/Desktop/GHQ/data/loto7_4688_k75.csv"
# DEFAULT_CSV = "/Users/4c/Desktop/GHQ/data/loto7_4688_k75_loto_2966.csv"
# DEFAULT_CSV = "/Users/4c/Desktop/GHQ/data/loto7_4688_k75_loto_plus_1722.csv"


def mask(numbers):
    result = 0
    for number in numbers:
        for digit in f"{number:02d}":
            result |= 1 << int(digit)
    return result


def digits(value):
    return "".join(
        str(d) for d in range(10) if value & (1 << d)
    ) or "-"


def digit_arg(value):
    if any(c not in "0123456789" for c in value):
        raise argparse.ArgumentTypeError(
            "Unesi cifre bez razmaka, npr. 478."
        )
    return sum(1 << int(c) for c in set(value))


def load_csv(path, newest):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for line, row in enumerate(csv.reader(handle), 1):
            if not row or not any(cell.strip() for cell in row):
                continue
            try:
                numbers = tuple(int(cell.strip()) for cell in row)
            except ValueError:
                raise ValueError(
                    f"Red {line}: ocekujem sedam celih brojeva."
                )
            if len(numbers) != 7 or len(set(numbers)) != 7:
                raise ValueError(
                    f"Red {line}: potrebno je sedam razlicitih brojeva."
                )
            if any(n < 1 or n > 39 for n in numbers):
                raise ValueError(
                    f"Red {line}: broj izvan raspona 1-39."
                )
            rows.append((line, numbers))
    if not rows:
        raise ValueError("CSV je prazan.")
    return rows[::-1] if newest == "first" else rows


def build_counts():
    """Tacno brojanje sedmorki po uniji cifara, bez simulacije."""
    layers = [Counter() for _ in range(8)]
    layers[0][0] = 1

    for number in range(1, 40):
        number_mask = mask((number,))
        for selected in range(6, -1, -1):
            for previous, count in tuple(layers[selected].items()):
                layers[selected + 1][previous | number_mask] += count

    assert sum(layers[7].values()) == TOTAL
    return dict(layers[7])


class Model:
    def __init__(self):
        self.counts = build_counts()

    @lru_cache(maxsize=None)
    def transitions(self, missing):
        counts = Counter()
        for observed, count in self.counts.items():
            counts[missing & ~observed] += count
        return tuple(sorted(counts.items()))

    @lru_cache(maxsize=None)
    def expected(self, missing):
        if not missing:
            return 0.0
        transitions = self.transitions(missing)
        stay = dict(transitions).get(missing, 0)
        progress = sum(
            count * self.expected(nxt)
            for nxt, count in transitions if nxt != missing
        )
        return (TOTAL + progress) / (TOTAL - stay)

    @lru_cache(maxsize=None)
    def within(self, missing, horizon):
        if not missing:
            return 1.0
        if horizon == 0:
            return 0.0
        return sum(
            count * self.within(nxt, horizon - 1)
            for nxt, count in self.transitions(missing)
        ) / TOTAL

    def matching(self, required, forbidden):
        return sum(
            count for observed, count in self.counts.items()
            if observed & required == required
            and not observed & forbidden
        )


def history(rows):
    missing, age, synchronized = ALL, 0, False
    durations = []

    for _, numbers in rows:
        age += 1
        missing &= ~mask(numbers)
        if missing == 0:
            if synchronized:
                durations.append(age)
            synchronized = True
            missing, age = ALL, 0

    return missing, age, synchronized, durations


def middle_combination(target, expected_count):
    """Srednja sedmorka u leksikografskom poretku datog obrasca."""
    pool = tuple(
        n for n in range(1, 40)
        if mask((n,)) & ~target == 0
    )
    masks = tuple(mask((n,)) for n in pool)

    @lru_cache(maxsize=None)
    def ways(index, left, covered):
        if left == 0:
            return int(covered == target)
        if len(pool) - index < left:
            return 0
        return (
            ways(index + 1, left - 1, covered | masks[index])
            + ways(index + 1, left, covered)
        )

    total = ways(0, 7, 0)
    if total != expected_count or total == 0:
        raise RuntimeError(
            "Neslaganje nezavisnih kombinatornih racuna."
        )

    rank = (total - 1) // 2
    result, start, covered = [], 0, 0

    for left in range(7, 0, -1):
        for index in range(start, len(pool) - left + 1):
            count = ways(
                index + 1, left - 1, covered | masks[index]
            )
            if rank < count:
                result.append(pool[index])
                covered |= masks[index]
                start = index + 1
                break
            rank -= count
        else:
            raise RuntimeError("Nije pronadjena NEXT kombinacija.")

    assert len(result) == len(set(result)) == 7
    assert mask(result) == target
    return tuple(result)


def predict_next(model, missing, required=0, forbidden=0):
    candidates = {
        observed: count
        for observed, count in model.counts.items()
        if observed & required == required
        and not observed & forbidden
    }
    if not candidates:
        raise ValueError(
            "Nijedna sedmorka ne zadovoljava filter cifara."
        )

    transitions = Counter()
    for observed, count in candidates.items():
        transitions[missing & ~observed] += count

    next_state = min(
        transitions,
        key=lambda state: (-transitions[state], state),
    )
    target = min(
        (
            observed for observed in candidates
            if missing & ~observed == next_state
        ),
        key=lambda observed: (-candidates[observed], observed),
    )

    combo = middle_combination(target, candidates[target])
    return combo, next_state, target, candidates[target]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=DEFAULT_CSV)
    parser.add_argument(
        "--newest", choices=("first", "last"), default="last",
        help="Polozaj najnovijeg izvlacenja: first ili last.",
    )
    parser.add_argument(
        "--require", type=digit_arg, default=0,
        help="Sve navedene cifre moraju biti pokrivene, npr. 478.",
    )
    parser.add_argument(
        "--forbid", type=digit_arg, default=0,
        help="Nijedna navedena cifra ne sme biti pokrivena.",
    )
    args = parser.parse_args()

    if args.require & args.forbid:
        parser.error(
            "Ista cifra ne moze biti obavezna i zabranjena."
        )
    try:
        rows = load_csv(args.csv, args.newest)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    model = Model()
    missing, age, synced, durations = history(rows)

    print(f"Izvlacenja: {len(rows)} | Ukupno sedmorki: {TOTAL:,}")
    print(f"Najnovije (CSV red {rows[-1][0]}): {rows[-1][1]}")
    print("Cifre se citaju iz zapisa 01-39; 01 pokriva i 0 i 1.")
    print(
        f"Trenutni ciklus: {age} kola"
        f" | Nedostaju cifre: {digits(missing)}"
    )
    if not synced:
        print(
            "Prvi ciklus je nepotpun:"
            " pocetak prije CSV-a nije poznat."
        )
    elif age == 0:
        print(
            "Poslednje izvlacenje zatvorilo je ciklus;"
            " novi jos nije zapoceo."
        )

    print(f"Ocekivano preostalo kola: {model.expected(missing):.4f}")
    for horizon in (1, 3, 5):
        probability = model.within(missing, horizon)
        print(
            f"Zavrsetak u narednih {horizon} kola:"
            f" {probability:.4%}"
        )

    print(
        "\nNajverovatniji prelazi:"
        " preostale cifre | broj sedmorki | verovatnoca"
    )
    ranked = sorted(
        model.transitions(missing),
        key=lambda item: (-item[1], item[0]),
    )
    for nxt, count in ranked[:10]:
        print(
            f"{digits(nxt):>10} | {count:>10,}"
            f" | {count / TOTAL:.4%}"
        )
    print("Prikazano najvise 10 prelaza; racun obuhvata sve.")

    print(f"\nPotpuni istorijski ciklusi: {len(durations)}")
    print(
        "Prvi nepotpuni i poslednji otvoreni ciklus"
        " nisu u statistici trajanja."
    )
    if durations:
        print(
            f"Prosek: {sum(durations) / len(durations):.4f} kola"
        )
        print(
            "Trajanje:broj ciklusa =",
            dict(sorted(Counter(durations).items())),
        )
    print(
        f"Teorijski prosek celog ciklusa:"
        f" {model.expected(ALL):.4f} kola"
    )

    count = model.matching(args.require, args.forbid)
    print(
        f"\nFilter cifara: obavezne={digits(args.require)},"
        f" zabranjene={digits(args.forbid)}"
    )
    print(
        f"Tacno {count:,} sedmorki"
        f" ({count / TOTAL:.4%} svih sedmorki)."
    )
    if count == 0:
        parser.error(
            "Nijedna sedmorka ne zadovoljava filter cifara."
        )

    combo, next_state, target, group_count = predict_next(
        model, missing, args.require, args.forbid
    )
    print("\nNEXT:", " ".join(f"{n:02d}" for n in combo))
    print(f"Nedostaju posle NEXT: {digits(next_state)}")
    print(
        f"Obrazac cifara NEXT: {digits(target)}"
        f" | Sedmorki s tim obrascem: {group_count:,}"
    )
    print(
        "Pravilo: najbrojniji prelaz,"
        " najbrojniji obrazac, srednja sedmorka."
    )
    print("Kod izjednacenja prednost ima manja bitmaska cifara.")
    print(
        "Srednja sedmorka je deterministicki izbor"
        " medju jednakim kandidatima."
    )
    print(
        "Brojanja su celobrojna;"
        " verovatnoce i ocekivanja su zaokruzeni."
    )


if __name__ == "__main__":
    main()



"""
Izvlacenja: 4688 | Ukupno sedmorki: 15,380,937
Najnovije (CSV red 4688): (2, 5, 6, 10, 12, 22, 30)
Cifre se citaju iz zapisa 01-39; 01 pokriva i 0 i 1.
Trenutni ciklus: 5 kola | Nedostaju cifre: 4
Ocekivano preostalo kola: 1.7768
Zavrsetak u narednih 1 kola: 56.2802%
Zavrsetak u narednih 3 kola: 91.6433%
Zavrsetak u narednih 5 kola: 98.4027%

Najverovatniji prelazi: preostale cifre | broj sedmorki | verovatnoca
         - |  8,656,417 | 56.2802%
         4 |  6,724,520 | 43.7198%
Prikazano najvise 10 prelaza; racun obuhvata sve.

Potpuni istorijski ciklusi: 1292
Prvi nepotpuni i poslednji otvoreni ciklus nisu u statistici trajanja.
Prosek: 3.6223 kola
Trajanje:broj ciklusa = {1: 5, 2: 285, 3: 427, 4: 290, 5: 151, 6: 83, 7: 26, 8: 14, 9: 3, 10: 5, 11: 2, 18:1}
Teorijski prosek celog ciklusa: 3.5754 kola

Filter cifara: obavezne=-, zabranjene=-
Tacno 15,380,937 sedmorki (100.0000% svih sedmorki).

NEXT: 04 07 10 15 26 35 37
Nedostaju posle NEXT: -
Obrazac cifara NEXT: 01234567 | Sedmorki s tim obrascem: 273,450
Pravilo: najbrojniji prelaz, najbrojniji obrazac, srednja sedmorka.
Kod izjednacenja prednost ima manja bitmaska cifara.
Srednja sedmorka je deterministicki izbor medju jednakim kandidatima.
Brojanja su celobrojna; verovatnoce i ocekivanja su zaokruzeni.
"""



"""
Izvlacenja: 2966 | Ukupno sedmorki: 15,380,937
Najnovije (CSV red 2966): (1, 10, 15, 19, 23, 25, 39)
Cifre se citaju iz zapisa 01-39; 01 pokriva i 0 i 1.
Trenutni ciklus: 2 kola | Nedostaju cifre: 46
Ocekivano preostalo kola: 2.3474
Zavrsetak u narednih 1 kola: 29.6567%
Zavrsetak u narednih 3 kola: 83.7863%
Zavrsetak u narednih 5 kola: 96.8200%

Najverovatniji prelazi: preostale cifre | broj sedmorki | verovatnoca
         - |  4,561,472 | 29.6567%
         4 |  4,094,945 | 26.6235%
         6 |  4,094,945 | 26.6235%
        46 |  2,629,575 | 17.0963%
Prikazano najvise 10 prelaza; racun obuhvata sve.

Potpuni istorijski ciklusi: 834
Prvi nepotpuni i poslednji otvoreni ciklus nisu u statistici trajanja.
Prosek: 3.5504 kola
Trajanje:broj ciklusa = {1: 4, 2: 184, 3: 290, 4: 183, 5: 100, 6: 48, 7: 12, 8: 6, 9: 3, 10: 4}
Teorijski prosek celog ciklusa: 3.5754 kola

Filter cifara: obavezne=-, zabranjene=-
Tacno 15,380,937 sedmorki (100.0000% svih sedmorki).

NEXT: 04 07 10 15 26 35 37
Nedostaju posle NEXT: -
Obrazac cifara NEXT: 01234567 | Sedmorki s tim obrascem: 273,450
Pravilo: najbrojniji prelaz, najbrojniji obrazac, srednja sedmorka.
Kod izjednacenja prednost ima manja bitmaska cifara.
Srednja sedmorka je deterministicki izbor medju jednakim kandidatima.
Brojanja su celobrojna; verovatnoce i ocekivanja su zaokruzeni.
"""



"""
Izvlacenja: 1722 | Ukupno sedmorki: 15,380,937
Najnovije (CSV red 1722): (2, 5, 6, 10, 12, 22, 30)
Cifre se citaju iz zapisa 01-39; 01 pokriva i 0 i 1.
Trenutni ciklus: 2 kola | Nedostaju cifre: 479
Ocekivano preostalo kola: 2.7731
Zavrsetak u narednih 1 kola: 14.3559%
Zavrsetak u narednih 3 kola: 76.4097%
Zavrsetak u narednih 5 kola: 95.2518%

Najverovatniji prelazi: preostale cifre | broj sedmorki | verovatnoca
         4 |  2,353,400 | 15.3008%
         7 |  2,353,400 | 15.3008%
         9 |  2,353,400 | 15.3008%
         - |  2,208,072 | 14.3559%
        47 |  1,741,545 | 11.3227%
        49 |  1,741,545 | 11.3227%
        79 |  1,741,545 | 11.3227%
       479 |    888,030 | 5.7736%
Prikazano najvise 10 prelaza; racun obuhvata sve.

Potpuni istorijski ciklusi: 486
Prvi nepotpuni i poslednji otvoreni ciklus nisu u statistici trajanja.
Prosek: 3.5329 kola
Trajanje:broj ciklusa = {1: 1, 2: 107, 3: 169, 4: 117, 5: 54, 6: 22, 7: 11, 8: 2, 10: 2, 12: 1}
Teorijski prosek celog ciklusa: 3.5754 kola

Filter cifara: obavezne=-, zabranjene=-
Tacno 15,380,937 sedmorki (100.0000% svih sedmorki).

NEXT: 05 09 10 16 27 36 39
Nedostaju posle NEXT: 4
Obrazac cifara NEXT: 01235679 | Sedmorki s tim obrascem: 273,450
Pravilo: najbrojniji prelaz, najbrojniji obrazac, srednja sedmorka.
Kod izjednacenja prednost ima manja bitmaska cifara.
Srednja sedmorka je deterministicki izbor medju jednakim kandidatima.
Brojanja su celobrojna; verovatnoce i ocekivanja su zaokruzeni.
"""
