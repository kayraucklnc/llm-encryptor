"""Build the fixed word list of the short density (run once; the result is part of the format).

secret_words.txt  4096 everyday words: a word of the secret costs 13 bits instead of 4-5 a letter.

usage: build_lists.py OUT_DIR BIP39_FILE EFF_FILE

BIP39_FILE  https://raw.githubusercontent.com/bitcoin/bips/master/bip-0039/english.txt
EFF_FILE    https://www.eff.org/files/2016/07/18/eff_large_wordlist.txt  (CC BY 3.0 US)

The list must stay as it is: running this again with other inputs would
change the format. It is kept to show where the list came from.
"""

from __future__ import annotations

import sys
from pathlib import Path

SECRET_WORDS = """
the a an and or but not no yes so if then than that this these those there here it its it's is are
was were be been being am do does did done have has had will would can could should shall may might
must I I'm I'll I've I'd me my mine we we'll we're us our you you're you'll your yours he him his
she her they them their who what when where why how which at to of in on by for from with without
into onto out up down over under above below behind before after during until till since about
around near next between through across against along inside outside off away back again once
twice too also only just very really still already never always often soon now later today tonight
tomorrow yesterday morning noon afternoon evening night midnight early late time day days week
weeks month months year years hour hours minute minutes second seconds Monday Tuesday Wednesday
Thursday Friday Saturday Sunday January February March April May June July August September October
November December weekend zero one two three four five six seven eight nine ten eleven twelve
twenty thirty forty fifty sixty hundred thousand million first last half all some any none both
each every many much more most less few other another same new old big small good bad best better
right left north south east west front top bottom middle side end start meet meeting met see saw
seen look watch find found get got go going gone come coming came leave leaving stay wait waiting
bring take took give gave send sent call called text message email phone talk tell told say said
ask know knew think thought want need like love miss keep kept put hide hidden open close closed
lock locked buy bought sell sold pay paid cost money cash card bank price free cheap ready done
safe sure okay ok fine please thanks thank sorry hello hi hey bye dear home house flat room door
gate window floor office work job school station airport hotel cafe bar shop store market park
bridge street road corner city town village country car bus train plane flight ticket key keys
password code number name address package box bag letter paper book table bench chair bed kitchen
garden beach lake river sea mountain forest food water coffee tea lunch dinner breakfast drink
friend friends mom dad mother father brother sister son daughter wife husband boss doctor police
man woman boy girl people person someone anyone everyone nobody something anything everything
nothing don't can't won't didn't doesn't isn't aren't wasn't let's that's there's what's he's
she's they're alone together maybe sharp red blue green black white yellow hot cold wet dry happy
sad tired sick well away far close secret plan problem idea news help stop start run walk drive
fly move change try use make made let turn read write wrote hear heard feel felt remember forget
forgot under usual place thing things way part land landed pick picked drop dropped arrive
arrived arriving pm am o'clock here's where's who's you've we've they've couldn't wouldn't
shouldn't haven't hasn't hadn't asap tonight's today's bringing taking waiting leaving
meeting's usual same place spot entrance exit stairs roof basement parking platform
"""


def main() -> None:
    out, bip39_file, eff_file = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    bip39 = bip39_file.read_text().split()
    eff = [line.split()[1] for line in eff_file.read_text().splitlines() if line.strip()]

    words: list[str] = []
    taken: set[str] = set()
    for word in [*SECRET_WORDS.split(), *bip39, *(w for w in eff if w.isalpha())]:
        if word not in taken and len(words) < 4096:
            taken.add(word)
            words.append(word)
    print(f"{len(words)} secret words ({len(SECRET_WORDS.split())} written by hand)")
    if len(words) < 4096:
        raise SystemExit("not enough words")

    out.mkdir(parents=True, exist_ok=True)
    (out / "secret_words.txt").write_text("\n".join(words) + "\n")


main()
