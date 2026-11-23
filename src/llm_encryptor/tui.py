"""The app you get when you run llm-encryptor: chat, hide a message, reveal one.

Everything a screen shows is meant to explain itself: the text anyone may
see on the left, what was really said on the right, and underneath the steps
that turn one into the other, as they happen.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import ClassVar

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message as Event
from textual.reactive import reactive
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button, Footer, Input, Static, TextArea

from .backend import Backend, BackendError
from .codec import RevealError
from .crypto import Keys, derive_keys
from .dense import ReadProgress, read_dense, write_dense
from .embed import EmbedError, Progress
from .hide import validate_secret
from .ollama import OllamaBackend
from .peer import Line, LineError
from .talk import DEFAULT_LENGTH, LENGTHS, request
from .terminal import copy_text, read_clipboard

_BAR = 28
_BITS_SHOWN = 56


# --- the steps panel --------------------------------------------------------


@dataclass(frozen=True)
class Steps:
    """What the steps panel shows. ``hiding`` chooses which of the two stories it tells."""

    hiding: bool = True
    secret: str = ""
    text: str = ""
    read: str = ""
    done_bits: int = 0
    all_bits: int = 0
    bits: str = ""
    draft: int = 0
    finished: bool = False
    problem: str = ""


_STAGES = {"done": "green", "now": "bold yellow", "todo": "dim"}


def steps_view(steps: Steps) -> RenderableType:
    """Three stages joined by a line: secret, locked bits, visible text.

    Hiding runs down from the secret to the text; reading shows the same
    three the other way up. The stage under way is lit.
    """
    idle = not steps.secret and not steps.text
    secret = _stage("secret", Text(steps.secret or "· · ·"), _secret_state(steps, idle))
    locked = _stage("locked", _bits(steps), "todo" if idle else "done" if steps.finished else "now")
    visible = _stage("visible", _visible(steps), _visible_state(steps, idle))
    order = [secret, locked, visible] if steps.hiding else [visible, locked, secret]
    lines: list[RenderableType] = [order[0], Text("  │", style="dim"), order[1], Text("  │", style="dim"), order[2]]
    if steps.problem:
        lines.append(Text(f"  ✗ {steps.problem}", style="bold red"))
    return Group(*lines)


def _stage(name: str, content: Text, state: str) -> RenderableType:
    if state == "todo":
        content.stylize("dim")
    row = Table.grid(padding=(0, 1))
    row.add_column(width=13, no_wrap=True)
    row.add_column(ratio=1)
    row.add_row(Text(f"  ● {name}", style=_STAGES[state]), content)
    return row


def _secret_state(steps: Steps, idle: bool) -> str:
    if idle:
        return "todo"
    return "done" if steps.hiding or steps.finished else "todo"


def _visible_state(steps: Steps, idle: bool) -> str:
    if idle or (steps.hiding and not steps.text):
        return "todo"
    return "done" if steps.finished or not steps.hiding else "now"


def _bits(steps: Steps) -> Text:
    """The locked message as a strip of cells: filled as they go into, or come out of, the text."""
    if steps.hiding:
        if not steps.all_bits:
            return Text("▱" * _BAR)
        filled = _BAR * steps.done_bits // steps.all_bits
        strip = Text("▰" * filled, style="green")
        strip.append("▱" * (_BAR - filled), style="dim")
        strip.append(f"  {steps.done_bits}/{steps.all_bits}", style="dim")
        if steps.draft > 1:
            strip.append(f"  ×{steps.draft}", style="dim")
        return strip
    if not steps.bits:
        return Text("▱" * _BAR)
    strip = Text("".join("▰" if bit == "1" else "▱" for bit in steps.bits[-_BITS_SHOWN:]), style="green")
    strip.append(f"  {len(steps.bits)}", style="dim")
    return strip


def _visible(steps: Steps) -> Text:
    if steps.hiding:
        return Text(steps.text + ("" if steps.finished or not steps.text else "▌") or "· · ·", style="cyan")
    walked = Text(steps.read, style="bold cyan")
    walked.append(steps.text[len(steps.read) :], style="dim")
    return walked


class StepsPanel(Static):
    """Shows the latest ``Steps``."""

    def show(self, steps: Steps) -> None:
        self.steps = steps
        self.update(steps_view(steps))

    def on_mount(self) -> None:
        self.show(Steps())


# --- the length dial --------------------------------------------------------


class LengthDial(Widget):
    """One dial instead of a menu of densities: ← longer and more natural, shorter and stranger →."""

    can_focus = True
    BINDINGS: ClassVar = [Binding("left", "move(-1)", "longer"), Binding("right", "move(1)", "shorter")]
    value = reactive(DEFAULT_LENGTH)

    class Changed(Event):
        def __init__(self, density: str) -> None:
            super().__init__()
            self.density = density

    @property
    def density(self) -> str:
        return LENGTHS[self.value][0]

    def render(self) -> Text:
        dial = Text("longer  ", style="dim")
        for at in range(len(LENGTHS)):
            dial.append("━━━━", style="dim")
            dial.append("●" if at == self.value else "○", style="bold yellow" if at == self.value else "dim")
        dial.append("━━━━  shorter", style="dim")
        dial.append(f"     {LENGTHS[self.value][2]}", style="bold yellow")
        return dial

    def action_move(self, by: int) -> None:
        self.value = max(0, min(len(LENGTHS) - 1, self.value + by))
        self.post_message(self.Changed(self.density))

    def on_click(self, event: events.Click) -> None:
        self.action_move(1 if event.x > self.size.width // 2 else -1)


# --- the work a screen hands to the model -------------------------------------


@dataclass
class Workshop:
    """Runs model work off the screen's thread, one job at a time.

    One at a time matters: writer and reader each start from a freshly
    loaded model, so two jobs at once would spoil each other's numbers.
    """

    make_backend: Callable[[], Backend]
    busy: threading.Lock = field(default_factory=threading.Lock)

    def hide(
        self,
        secret: str,
        keys: Keys,
        asked: list,
        density: str,
        show: Callable[[Steps], None],
        about: str | None = None,
    ) -> str:
        with self.busy:
            steps = Steps(hiding=True, secret=secret)
            show(steps)

            def watch(progress: Progress) -> None:
                nonlocal steps
                restarted = progress.embedded < steps.done_bits or not steps.draft
                steps = replace(
                    steps,
                    text=progress.text,
                    done_bits=progress.embedded,
                    all_bits=progress.total,
                    draft=steps.draft + restarted,
                )
                show(steps)

            text = write_dense(
                secret, keys, self.make_backend(), asked, watch, density=density, about=about
            )
            show(replace(steps, text=text, done_bits=steps.all_bits, finished=True))
            return text

    def reveal(self, text: str, keys: Keys, show: Callable[[Steps], None]) -> str:
        with self.busy:
            steps = Steps(hiding=False, text=" ".join(text.split()))
            show(steps)

            def watch(progress: ReadProgress) -> None:
                nonlocal steps
                steps = replace(steps, read=progress.read, bits=progress.bits)
                show(steps)

            secret = read_dense(text, keys, self.make_backend(), watch)
            show(replace(steps, read=steps.text, secret=secret, finished=True))
            return secret


_PROBLEMS = (EmbedError, RevealError, BackendError, ValueError, LineError)


def _said(error: Exception) -> str:
    return str(error) or error.__class__.__name__


# --- screens ----------------------------------------------------------------


class Home(Screen):
    def compose(self) -> ComposeResult:
        picture = Text(justify="center")
        picture.append("meet at noon", style="green")
        picture.append("   ▸   ", style="dim")
        picture.append("▰▰▱▰▱▱▰▰▱▰", style="yellow")
        picture.append("   ▸   ", style="dim")
        picture.append("Hey, how was your weekend?", style="cyan")
        yield Static(
            Group(Text("l l m - e n c r y p t o r", style="bold", justify="center"), Text(""), picture),
            id="intro",
        )
        with Vertical(id="choices"):
            yield Button("⇄  Chat", id="chat", variant="primary")
            yield Button("▼  Hide", id="hide")
            yield Button("▲  Reveal", id="reveal")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        screens = {"chat": Connect, "hide": Hide, "reveal": Reveal}
        self.app.push_screen(screens[event.button.id or "hide"]())


def box(label: str, name: str, hint: str = "", password: bool = False) -> Input:
    """A text field with its name on the frame, so it stays readable once filled in."""
    made = Input(placeholder=hint, password=password, id=name)
    made.border_title = label
    return made


class Page(Screen):
    """A screen you can leave with Escape."""

    BINDINGS: ClassVar = [Binding("escape", "leave", "back")]

    async def action_leave(self) -> None:
        self.app.pop_screen()

    def say(self, message: str, style: str = "yellow") -> None:
        for status in self.query("#status"):  # Not there yet while the screen is being put up.
            status.update(Text(message, style=style))

    def show_steps(self, steps: Steps) -> None:
        """Called from the worker thread as the model works."""
        self.app.call_from_thread(self.query_one(StepsPanel).show, steps)


class Hide(Page):
    def compose(self) -> ComposeResult:
        yield box("secret", "secret")
        yield box("password", "password", password=True)
        yield box("topic", "about", "dinner on Friday")
        yield LengthDial()
        yield Button("▼  Hide", id="go", variant="primary")
        yield Static("", id="status")
        yield StepsPanel(id="steps")
        yield Footer()

    def on_input_submitted(self) -> None:
        self._start()

    def on_button_pressed(self) -> None:
        self._start()

    def _start(self) -> None:
        secret = self.query_one("#secret", Input).value.strip()
        password = self.query_one("#password", Input).value
        about = self.query_one("#about", Input).value
        density = self.query_one(LengthDial).density
        if not secret or not password:
            return self.say("secret and password needed")
        self.say("")
        self.run_worker(lambda: self._hide(secret, password, about, density), thread=True)

    def _hide(self, secret: str, password: str, about: str, density: str) -> None:
        app = self.app
        try:
            keys = derive_keys(password)
            text = app.workshop.hide(validate_secret(secret), keys, request(about), density, self.show_steps)
        except _PROBLEMS as error:
            return app.call_from_thread(self.say, _said(error), "red")
        copied = " · copied" if copy_text(text) else ""
        app.call_from_thread(self.say, f"✓ {len(text.split())} words{copied}", "green")


class Reveal(Page):
    def compose(self) -> ComposeResult:
        yield TextArea(id="cover")
        yield box("password", "password", password=True)
        with Horizontal(classes="buttons"):
            yield Button("Paste", id="paste")
            yield Button("▲  Reveal", id="go", variant="primary")
        yield Static("", id="status")
        yield StepsPanel(id="steps")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#cover", TextArea).border_title = "visible text"

    def on_input_submitted(self) -> None:
        self._start()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "paste":
            self.query_one("#cover", TextArea).text = read_clipboard() or ""
        else:
            self._start()

    def _start(self) -> None:
        text = self.query_one("#cover", TextArea).text.strip()
        password = self.query_one("#password", Input).value
        if not text or not password:
            return self.say("text and password needed")
        self.say("")
        self.run_worker(lambda: self._reveal(text, password), thread=True)

    def _reveal(self, text: str, password: str) -> None:
        app = self.app
        try:
            secret = app.workshop.reveal(text, derive_keys(password), self.show_steps)
        except _PROBLEMS as error:
            return app.call_from_thread(self.say, _said(error), "red")
        app.call_from_thread(self.say, f"✓ {secret}", "green")


class Connect(Page):
    """Find each other: one of you waits, the other types the ID."""

    def compose(self) -> ComposeResult:
        yield Static("…", id="my-id")
        yield box("name", "name")
        yield box("shared password", "password", password=True)
        yield box("friend's id", "friend", "empty = wait for them")
        yield Button("⇄  Connect", id="go", variant="primary")
        yield Static("", id="status")
        yield Footer()

    async def on_mount(self) -> None:
        self.line = Line("me", self._text_arrived, self._line_changed)
        self.keys: Keys | None = None
        self.early: list[str] = []
        self.chat: Chat | None = None
        try:
            my_id = await self.line.open(self.app.port)
        except OSError as error:
            return self.say(f"Could not start listening: {error.strerror}", "red")
        box = self.query_one("#my-id", Static)
        box.border_title = "your id"
        box.update(Text(my_id, style="bold yellow", justify="center"))

    def on_input_submitted(self) -> None:
        self._start()

    def on_button_pressed(self) -> None:
        self._start()

    def _start(self) -> None:
        password = self.query_one("#password", Input).value
        if not password:
            return self.say("password needed")
        self.line.name = self.query_one("#name", Input).value.strip()[:40] or "someone"
        friend = self.query_one("#friend", Input).value.strip()
        self.say("connecting…" if friend else "waiting for your friend…")
        self.run_worker(self._connect(password, friend))

    async def _connect(self, password: str, friend: str) -> None:
        self.keys = await asyncio.to_thread(derive_keys, password)
        try:
            if friend and not self.line.connected:
                await self.line.call(friend)
        except LineError as error:
            return self.say(_said(error), "red")
        self._line_changed()

    def _line_changed(self) -> None:
        if self.chat is not None:
            return self.chat.line_changed()
        if self.line.connected and self.keys is not None:
            self.chat = Chat(self.line, self.keys, self.early)
            self.app.push_screen(self.chat)
        elif self.line.connected:
            self.say("your friend is here · password needed", "green")

    def _text_arrived(self, text: str) -> None:
        if self.chat is not None:
            self.chat.text_arrived(text)
        else:
            self.early.append(text)

    async def action_leave(self) -> None:
        await self.line.close()
        self.app.pop_screen()


class Turn(Horizontal):
    """One message as two bubbles: what was sent, and what it said."""

    def __init__(self, who: str, mine: bool, cover: str, secret: str) -> None:
        super().__init__(classes="turn mine" if mine else "turn")
        self.who, self.cover, self.secret = who, cover, secret

    def compose(self) -> ComposeResult:
        for side, text in (("seen", self.cover), ("said", self.secret)):
            bubble = Static(Text(text), classes=side)
            bubble.border_title = self.who
            yield bubble

    def set(self, cover: str | None = None, secret: str | None = None, bad: bool = False) -> None:
        self.cover, self.secret = cover or self.cover, secret or self.secret
        self.query_one(".seen", Static).update(Text(self.cover))
        said = self.query_one(".said", Static)
        said.update(Text(self.secret))
        said.set_class(bad, "bad")


class Chat(Page):
    """Two conversations: the visible one that travels, and the real one."""

    def __init__(self, line: Line, keys: Keys, early: list[str]) -> None:
        super().__init__()
        self.line, self.keys, self.early = line, keys, early
        self.last_from_friend: str | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="heads"):
            yield Static("VISIBLE", classes="head seen")
            yield Static("HIDDEN", classes="head said")
        yield VerticalScroll(id="turns")
        yield StepsPanel(id="steps")
        yield box("secret", "secret")
        yield box("topic", "about")
        yield LengthDial()
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._prompt()
        self.line_changed()
        waiting, self.early = self.early, []
        for text in waiting:
            self._show_arrived(text)
        self.query_one("#secret", Input).focus()

    def _prompt(self) -> None:
        about = self.query_one("#about", Input)
        about.placeholder = (
            "↩ replies to their last text" if self.last_from_friend else "dinner on Friday"
        )

    def line_changed(self) -> None:
        friend = self.line.friend or "your friend"
        if self.line.connected:
            self.say(f"● {friend}", "green")
        else:
            self.say(f"○ {friend} left", "red")

    def on_input_submitted(self) -> None:
        secret = self.query_one("#secret", Input)
        about = self.query_one("#about", Input)
        if not secret.value.strip():
            return
        if not self.line.connected:
            return self.say("○ not connected", "red")
        turn = self._add("you", True, "…", secret.value.strip())
        asked = request(about.value, self.last_from_friend)
        density = self.query_one(LengthDial).density
        # A steer decides what the text opens on; without one it answers their text.
        steer = about.value.strip() or self.last_from_friend
        said = secret.value.strip()
        self.run_worker(lambda: self._send(turn, said, asked, density, steer), thread=True)
        secret.value, about.value = "", ""

    def _add(self, who: str, mine: bool, cover: str, secret: str) -> Turn:
        turn = Turn(who, mine, cover, secret)
        turns = self.query_one("#turns", VerticalScroll)
        turns.mount(turn)
        turns.scroll_end(animate=False)
        return turn

    def _send(self, turn: Turn, secret: str, asked: list, density: str, steer: str | None) -> None:
        app = self.app
        try:
            text = app.workshop.hide(
                validate_secret(secret), self.keys, asked, density, self.show_steps, steer
            )
        except _PROBLEMS as error:
            app.call_from_thread(turn.set, "(not sent)", None, True)
            return app.call_from_thread(self.say, _said(error), "red")
        app.call_from_thread(turn.set, text)
        app.call_from_thread(self.run_worker, self._deliver(text))

    async def _deliver(self, text: str) -> None:
        try:
            await self.line.send(text)
        except LineError as error:
            self.say(_said(error), "red")

    def text_arrived(self, text: str) -> None:
        if not self.is_mounted:
            return self.early.append(text)  # Shown as soon as the screen is up.
        self._show_arrived(text)

    def _show_arrived(self, text: str) -> None:
        self.last_from_friend = text
        self._prompt()
        turn = self._add(self.line.friend or "friend", False, " ".join(text.split()), "…")
        self.run_worker(lambda: self._read(turn, text), thread=True)

    def _read(self, turn: Turn, text: str) -> None:
        app = self.app
        try:
            secret = app.workshop.reveal(text, self.keys, self.show_steps)
        except _PROBLEMS:
            return app.call_from_thread(
                turn.set, None, "✗ unreadable · different password?", True
            )
        app.call_from_thread(turn.set, None, secret)


# --- the app ----------------------------------------------------------------


class EncryptorApp(App):
    TITLE = "llm-encryptor"
    BINDINGS: ClassVar = [Binding("ctrl+q", "quit", "quit")]
    CSS = """
    Screen { padding: 1 2; }
    #intro { margin: 3 0 3 0; height: auto; }
    #choices { align: center top; height: auto; }
    #choices Button { width: 30; margin-top: 1; }
    Input { border: round $primary; border-title-color: $text-muted; }
    Input:focus { border: round $accent; }
    LengthDial { height: 1; margin: 1 0; }
    LengthDial:focus { background: $boost; }
    StepsPanel { border: round $surface-lighten-2; padding: 1 1; height: auto; margin-top: 1; }
    #status { height: 1; margin-top: 1; }
    #my-id { border: round $accent; border-title-color: $text-muted; height: 3; margin-bottom: 1; }
    #cover { height: 8; border: round $primary; border-title-color: $text-muted; }
    .buttons { height: auto; margin-top: 1; }
    .buttons Button { margin-right: 2; }
    #heads { height: 1; }
    .head { text-style: bold; width: 1fr; padding: 0 2; }
    .head.seen { color: cyan; }
    .head.said { color: green; }
    #turns { height: 1fr; }
    .turn { height: auto; }
    .turn Static { width: 1fr; padding: 0 1; height: auto; border-title-color: $text-muted; }
    .turn .seen { border: round cyan; color: cyan; }
    .turn .said { border: round green; color: green; }
    .turn.mine .seen, .turn.mine .said { border-title-color: yellow; }
    .turn .bad { border: round red; color: red; }
    """

    def __init__(
        self,
        model: str,
        host: str,
        make_backend: Callable[[str, str], Backend] = OllamaBackend,
        port: int = 0,
    ) -> None:
        super().__init__()
        self.port = port
        self.workshop = Workshop(lambda: make_backend(model, host))
        self.sub_title = f"model {model}"

    def on_mount(self) -> None:
        self.push_screen(Home())
