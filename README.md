# llm-encryptor

Hide an encrypted message inside ordinary-looking text written by a local LLM.

You give it a secret and a password, then chat with a model: "write about
Turkey", "make it friendlier". Every version of the text carries the same
secret. Your friend pastes the text into `llm-encryptor reveal`, types the
password, and gets the secret back.

```console
$ llm-encryptor
Secret to hide: meet at noon
Password (the reader needs the same one):
you › write about Turkey
╭─ text to send ───────────────────────────────────────────────╮
│ Picture this: the morning sun spills over the Bosphorus, …   │
╰──────────────────────────────────────────────────────────────╯
✓ 213 words · checked: it decodes back to your secret
To read it: the password and the model qwen3:4b
you › make it more friendly and casual
…
you › /copy

$ llm-encryptor reveal message.txt --model qwen3:4b
Password (the reader needs the same one):
meet at noon
```

## Two formats

| | Compact (default) | Robust (`--robust`, `/mode robust`) |
|---|---|---|
| Text for a 12-character secret | about 170–220 words | at least 337 words |
| Time to write (4B model, laptop) | about 20 s | one to two minutes |
| The reader needs | password + the same model | password only |
| Survives | changed spacing and line breaks | also changed case and punctuation marks |

Compact packs the bits into the model's own word choices, so it is as dense
as the model's freedom allows, but the reader must replay the same model.
Robust hides bits in a keyed hash of the words, which any computer can check.

## Install

You need Python 3.10+ and [Ollama](https://ollama.com) 0.12.11 or later with
at least one model. (Reading a robust-format text needs neither.)

```console
$ ollama pull qwen3:4b
$ uv sync && uv run llm-encryptor        # from a checkout
```

## Use

| Command | What it does |
|---|---|
| `llm-encryptor` | Chat: write a cover text and keep revising it |
| `llm-encryptor hide -a "write about Turkey" -o msg.txt` | Hide a secret in one go |
| `llm-encryptor reveal msg.txt` | Recover the secret (or paste the text, then Ctrl-D) |
| `llm-encryptor models` | List the models you can use |

In the chat: `/model NAME`, `/models`, `/mode compact|robust`, `/copy`,
`/save FILE`, `/secret` (hide something else), `/new`, `/quit`.

**Choosing a model.** `--model NAME` works for writing and for reading
(default: the first model Ollama lists, or `$LLM_ENCRYPTOR_MODEL`). Any model
Ollama runs will do. Small ones are faster: `qwen2.5:0.5b` writes a text in
about 9 s and reads it in 2 s, but its prose is clumsier. For compact texts
both sides must use the same model, so tell your friend which one.

Other options: `--host URL` (or `$OLLAMA_HOST`); the password can come from
`$LLM_ENCRYPTOR_PASSWORD` for scripts. `hide --secret "…"` puts the secret in
your shell history; leave it out and the app asks instead.

**Use `/copy` or `/save`** rather than selecting the text in the terminal:
the box drawn around it would come along.

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

- The first 24 words are written freely to your instructions and carry
  nothing. They set topic and tone; the coded part continues them. That is why
  the reader needs no prompt, and why revising works. It also means the text
  can drift from your instructions later on.
- Probabilities are used in five coarse levels, and where one sits close to a
  level boundary the reader tries both sides (the encryption tag tells which
  is right). Writer and reader both start from a freshly loaded model, because
  what the server has cached changes its numbers slightly.
- Overhead is 10 bytes: 4 random bytes and a 6-byte tag. The rest of the nonce
  is a hash of the opening words.

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
- **Compact uses a short tag (6 bytes)** to save words. That is plenty to tell
  a message from noise, less than usual against deliberate forgery.
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
