# llm-encryptor

Hide an encrypted message inside ordinary-looking text written by a local LLM.

You give it a secret and a password, then chat with a model: "write about
Turkey", "make it friendlier". Every version of the text carries the same
secret. Your friend pastes the text into `llm-encryptor reveal`, types the
password, and gets the secret back. They do not need the model, the prompt,
or a GPU.

```console
$ llm-encryptor
Secret to hide: meet at noon
Password (the reader needs the same one):
you › write about Turkey
╭─ text to send ───────────────────────────────────────────────╮
│ Imagine yourself standing at the edge of the Aegean Sea, …   │
╰──────────────────────────────────────────────────────────────╯
✓ 464 words · checked: it decodes back to your secret
you › make it more friendly and casual
…
you › /copy

$ llm-encryptor reveal message.txt
Password (the reader needs the same one):
meet at noon
```

## Install

You need Python 3.10+ and, to write texts, [Ollama](https://ollama.com) with
any model (reading needs neither).

```console
$ ollama pull qwen3:4b
$ pipx install git+https://github.com/<you>/llm-encryptor   # or: uv tool install …
```

From a checkout: `uv sync`, then `uv run llm-encryptor`.

## Use

| Command | What it does |
|---|---|
| `llm-encryptor` | Chat: write a cover text and keep revising it |
| `llm-encryptor hide -a "write about Turkey" -o msg.txt` | Hide a secret in one go |
| `llm-encryptor reveal msg.txt` | Recover the secret (or paste the text, then Ctrl-D) |

In the chat: `/copy`, `/save FILE`, `/secret` (hide something else), `/new`
(start a fresh text), `/quit`.

Options: `--model NAME` (default: the first model Ollama lists, or
`$LLM_ENCRYPTOR_MODEL`), `--host URL` (or `$OLLAMA_HOST`), `--temperature`.
For scripts, the password can come from `$LLM_ENCRYPTOR_PASSWORD`.

**Length.** The text has to be long enough to carry the secret: three words for
every two hidden bits, plus 16 bytes of encryption overhead. A 12-character
secret needs at least 337 words, a 50-character one about 800. Keep secrets
short and ask for a kind of text that is naturally that long. With a 4B model
on a laptop, each version takes one to two minutes for a short secret.

`hide --secret "…"` puts the secret in your shell history; leave the option
out and the app asks for it instead.

## How it works

1. **Encrypt.** The password goes through scrypt to make two keys. The secret
   is encrypted with AES-256-CCM (random 8-byte nonce, 8-byte tag). This is
   where the secrecy comes from; the result looks like random bits.
2. **Hide.** The text is split into blocks of three words. A keyed hash
   (HMAC-SHA256) of each block gives two bits. While the model writes, the app
   asks it which tokens could come next and samples among them as usual, but
   keeps a block only if its two bits are the next two bits of the encrypted
   secret. If no natural wording fits, it rewrites the block before it rather
   than forcing an odd word.
3. **Finish.** Once all bits are in, the model ends the text freely. The app
   then decodes its own output to check it before showing it to you.
4. **Reveal.** The reader hashes the blocks of the text with the same key,
   which gives the bit stream, and decrypts it. Authentication tells it where
   the message starts and ends, so text after the cover (and up to 64 words
   before it) is ignored.

Only letters and digits count, ignoring case and accents. Re-wrapping lines
and changing capitalisation, quotes or punctuation marks does not break the
message, as long as no words get joined or split (deleting the comma in
"1,500" joins two). Adding or removing a word in the carrying part breaks it,
and so does changing one (three times out of four; otherwise the message still
decodes correctly).

Because the hidden bits depend only on the finished words, you can steer the
model with any prompt and revise as often as you like: each revision is a new
text carrying a fresh encryption of the same secret.

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
- **Languages written without spaces** (Chinese, Japanese) are not supported:
  the scheme counts space-separated words.
- **It reads like AI text**, because it is. That is only good cover where AI
  text is unremarkable.
- Not audited. Do not rely on it where being caught has serious consequences.

## Other models

Writing needs a model that reports the probabilities of its next tokens.
Ollama does (version 0.12.11 or later), which covers any model it can run.
To add another runtime, implement the two methods of `Backend` in
`src/llm_encryptor/backend.py`.

## Development

```console
$ uv sync
$ uv run pytest --cov=llm_encryptor
```

MIT licence.
