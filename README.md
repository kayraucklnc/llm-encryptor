# llm-encryptor

Hide an encrypted message inside ordinary-looking text written by a local LLM.

Run it, pick Encrypt, type your secret and what the text should be about. The
text lands on your clipboard. Your friend picks Decrypt, and gets the secret
back with the password.

```console
$ llm-encryptor
llm-encryptor  ↑↓ move · Enter choose · q back
❯ 1  Encrypt a message
  2  Decrypt a message
  3  Model: qwen2.5:0.5b
  4  Format: compact (short text; the reader needs the same model)
  5  Density: natural (reads best)
  6  Change password
  7  Quit
Secret to hide: meet at noon
Password (the reader needs the same one):
What should the text be about? (Enter for a friendly note) write about Turkey
╭─ text to send ───────────────────────────────────────────────╮
│ Turkey stretches across two continents, a land where …       │
╰──────────────────────────────────────────────────────────────╯
✓ 107 words · copied to your clipboard
To read it: the password and the model qwen2.5:0.5b.
Ask for a change (e.g. make it friendlier), or Enter when done:
```

Move with the arrow keys and press Enter, or press an item's number.
Decrypt reads the text from your clipboard (or a file, or pasted text). The
password is asked once and kept until you close the app.

## Two formats

| | Compact (default) | Robust (menu item 4, or `--robust`) |
|---|---|---|
| Text for a 12-character secret | about 90–165 words | at least 337 words |
| Time to write (4B model, laptop) | about 20 s | one to two minutes |
| The reader needs | password + the same model | password only |
| Each extra character costs | 3 to 4 words | 12 words |
| Survives | changed spacing and line breaks | also changed case and punctuation marks |

Compact packs the bits into the model's own word choices, so it is as dense
as the model's freedom allows, but the reader must replay the same model.
Robust hides bits in a keyed hash of the words, which any computer can check.

## Install

```console
$ uv tool install git+https://github.com/kayraucklnc/llm-encryptor
$ llm-encryptor
```

That is all. The app looks after [Ollama](https://ollama.com), the program
that runs the models:

- if Ollama is not installed, it offers to install it with Homebrew (on other
  systems it tells you where to get it);
- if Ollama is not running, it starts it in the background and leaves it
  running, so the next start is instant;
- if the model it needs is not installed, it offers to download it.

Everyone starts with the same small model (`qwen2.5:0.5b`, about 400 MB), so
two people who never touched the setting can read each other's texts. Small
models are fast but write clumsier prose; `qwen3:4b` reads much better.

## Use

Run `llm-encryptor` for the menu. For scripts there are one-shot commands:

| Command | What it does |
|---|---|
| `llm-encryptor hide -a "write about Turkey" -o msg.txt` | Hide a secret in one go |
| `llm-encryptor reveal msg.txt` | Recover the secret (or paste the text, then Ctrl-D) |
| `llm-encryptor models` | List the models you can use |

**Choosing a model.** Menu item 3 lists the installed models and has
"Download another model…", where you type any name from
[ollama.com/library](https://ollama.com/library). On the command line use
`--model NAME` (or `$LLM_ENCRYPTOR_MODEL`); a model that is not installed is
downloaded after asking.

For compact texts both sides must use the same model, so tell your friend
which one if you changed it. If they don't know, Decrypt tries their chosen
model and then up to four other installed ones, and says so when none fits.

**Density.** Menu item 5 cycles through three settings, or use `--density`.
The reader detects the density by itself.

| Density | 12-character secret | 39-character secret | The text |
|---|---|---|---|
| natural | about 90–165 words | about 190–230 words | reads best |
| tight | about 100 words | about 95–150 words | less smooth |
| max | about 30 words | about 52–62 words | nonsense |

Tight treats every word the model finds plausible as about equally likely.
Max counts every token the model lists the same, however unlikely, stops the
moment the last bit is in, and opens with 4 free words instead of 16. At that
point the text no longer hides that something is odd; it only keeps the
secret unreadable.

Other options: `--robust`, `--host URL` (or `$OLLAMA_HOST`); the password can
come from `$LLM_ENCRYPTOR_PASSWORD` for scripts. `hide --secret "…"` puts the
secret in your shell history; leave it out and the app asks instead.

## How short can it get?

A text can carry about as many bits as the model has real choices: with
`qwen3:4b` that is roughly 1.4 bits a word. So the length is set by how many
bits must be hidden. For "meet at noon" in the compact format:

| What | Bits |
|---|---|
| The secret, packed (a fixed code: common letters take 3–5 bits) | 56 |
| Random bytes so the same secret never encrypts the same way twice | 16 |
| Tag that tells a real message from noise | 32 |

That is 104 bits, about 75 words, plus 16 free opening words and the words
needed to finish the last sentence. One cover word per secret letter is out
of reach for text that should read naturally: a letter is 3–5 bits and a
natural word carries about 1.4. The tight and max densities trade sense for
more bits per word: max reaches about 4 bits a word, which is close to the
ceiling here, because Ollama reports only a model's 20 likeliest next tokens
(4.3 bits). What is left to gain is compressing the secret with the model
itself (roughly 2 bits a letter instead of 4.3), which is not built yet.

## How it works

Both formats first encrypt the secret: the password goes through scrypt, and
the secret is sealed with AES-256-CCM. That is where the secrecy comes from;
the result looks like random bits. The app always decodes its own text again
before showing it to you.

**Compact.** This is arithmetic coding run backwards, the same idea as using
an LLM for compression. The encrypted bits, read as a binary fraction, pick a
point between 0 and 1. At each step the model's likely next tokens share out
the current interval according to their probability, and the token whose share
contains the point is written. The reader replays the model over the text,
follows the same intervals and reads the bits off the final one.

- The first 16 words (4 at max density) are written freely to your instructions and carry
  nothing. They set topic and tone; the coded part continues them. That is why
  the reader needs no prompt, and why revising works. It also means the text
  can drift from your instructions later on.
- Probabilities are used in five coarse levels, and where one sits close to a
  level boundary the reader tries both sides (the encryption tag tells which
  is right). Writer and reader both start from a freshly loaded model, because
  what the server has cached changes its numbers slightly.
- No run of three words may appear twice in a text. A model that starts
  going in circles becomes certain of every next word, and a certain word
  hides nothing. If a text still makes no progress for 40 tokens, it is
  thrown away and started again.
- The secret is packed with a fixed Huffman code before it is encrypted.
  Overhead is 6 bytes: 2 random bytes and a 4-byte tag. The rest of the nonce
  is a hash of the opening words. Max density sends 4 random bytes, because
  a 4-word opening repeats too easily to keep nonces apart on its own.

**Robust.** The text is split into blocks of three words; HMAC-SHA256 of each
block gives two bits. While the model writes, the app samples its next tokens
as usual but keeps a block only if its bits are the next two bits of the
encrypted secret, and rewrites the previous block when no natural wording
fits. The reader hashes the blocks and decrypts; text after the cover (and up
to 64 words before it) is ignored. Overhead is 16 bytes.

In the robust format only letters and digits count, ignoring case and accents:
re-wrapping and changing capitalisation, quotes or punctuation marks is fine
as long as no words get joined or split (deleting the comma in "1,500" joins
two). In the compact format every character matters except spacing and line
breaks, so "smart quotes" or autocorrect on the way will break it.

## What it does and does not protect

- **The secret is safe without the password.** That rests on scrypt and
  AES-CCM, not on the model. Use a long passphrase: there is no per-message
  salt, so a weak password can be guessed offline by anyone who has a text.
- **Hiding is best-effort, not proven.** The text is sampled from the model
  with a constraint, so it is fluent but not statistically identical to
  unconstrained output. A capable analyst who suspects this tool and compares
  against the same model might tell the difference. Research schemes such as
  [Meteor](https://eprint.iacr.org/2021/686) are provably undetectable, at the
  price that the reader needs the identical model and prompt.
- **A wrong password or a damaged text fails cleanly**; it does not return a
  wrong message.
- **Languages written without spaces** (Chinese, Japanese) are not supported
  by the robust format, which counts space-separated words.
- **It reads like AI text**, because it is. That is only good cover where AI
  text is unremarkable.
- **Compact texts were only tested on one machine.** Writer and reader got
  identical numbers on the same Mac. Another kind of computer or another
  Ollama version may compute slightly different probabilities; the reader
  tolerates small differences, but this is untested. If your friend cannot
  read a compact text, use `--robust`.
- **Compact uses a short tag (4 bytes) and 2 random bytes** to save words.
  The tag tells a message from noise but would not stop a determined forger.
  Two compact texts with the same password, the same first 16 words and the
  same 2 random bytes would leak how their secrets differ; that needs a
  1-in-65,536 coincidence on top of an identical opening.
- Not audited. Do not rely on it where being caught has serious consequences.

## Other runtimes

The app needs the probabilities of a model's next tokens, which Ollama
reports. To add another runtime, implement the `Backend` methods in
`src/llm_encryptor/backend.py`.

## Development

```console
$ uv sync
$ uv run pytest --cov=llm_encryptor
```

MIT licence.
