"""The app you get when you run llm-encryptor: chat, hide a message, reveal one.

Everything a screen shows is meant to explain itself: the text anyone may
see on the left, what was really said on the right, and underneath the steps
that turn one into the other, as they happen.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import ClassVar

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message as Event
from textual.reactive import reactive
from textual.screen import Screen
from textual.theme import Theme
from textual.widget import Widget
from textual.widgets import Input, OptionList, Static

from .backend import Backend, BackendError
from .codec import RevealError
from .crypto import Keys, derive_keys
from .dense import ReadProgress, read_dense, write_dense
from .embed import EmbedError, Progress
from .hide import validate_secret
from .ollama import OllamaBackend, list_models, pull_model
from .peer import Line, LineError
from .talk import DEFAULT_LENGTH, LENGTHS, request
from .terminal import copy_text, read_clipboard

_BAR = 28
_BITS_SHOWN = 56
# How long the finished stages stay on the chat screen before they make room again.
_LINGER = 1.5
# One colour per thing, everywhere: what is visible, what is hidden, what is under way.
VISIBLE, HIDDEN, ACCENT, ACTIVE, MUTED = "#7dd3fc", "#86efac", "#c4b5fd", "#fbbf24", "#6b7280"
BAD = "#f87171"


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


_STAGES = {"done": HIDDEN, "now": f"bold {ACTIVE}", "todo": MUTED}


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
    joint = Text("  │", style=MUTED)
    lines: list[RenderableType] = [order[0], joint, order[1], joint, order[2]]
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
            return Text("━" * _BAR, style=MUTED)
        filled = _BAR * steps.done_bits // steps.all_bits
        strip = Text("━" * filled, style=HIDDEN)
        strip.append("━" * (_BAR - filled), style=MUTED)
        strip.append(f"  {steps.done_bits}/{steps.all_bits}", style=MUTED)
        if steps.draft > 1:
            strip.append(f"  ×{steps.draft}", style=MUTED)
        return strip
    if not steps.bits:
        return Text("━" * _BAR, style=MUTED)
    strip = Text()
    for bit in steps.bits[-_BITS_SHOWN:]:
        strip.append("▮" if bit == "1" else "▯", style=HIDDEN if bit == "1" else MUTED)
    strip.append(f"  {len(steps.bits)}", style=MUTED)
    return strip


def _visible(steps: Steps) -> Text:
    if steps.hiding:
        return Text(steps.text + ("" if steps.finished or not steps.text else "▌") or "· · ·", style=VISIBLE)
    walked = Text(steps.read, style=f"bold {VISIBLE}")
    walked.append(steps.text[len(steps.read) :], style=MUTED)
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
        dial = Text("longer  ", style=MUTED)
        for at in range(len(LENGTHS)):
            dial.append("────", style=MUTED)
            dial.append("◉" if at == self.value else "·", style=f"bold {ACCENT}" if at == self.value else MUTED)
        dial.append("────  shorter", style=MUTED)
        dial.append(f"     {LENGTHS[self.value][2]}", style=ACCENT)
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


# --- pieces every screen is made of -------------------------------------------


class TopBar(Static):
    """Where you are on the left, the model in use on the right."""

    def __init__(self, place: str = "") -> None:
        super().__init__(classes="topbar")
        self.place = place

    def render(self) -> RenderableType:
        bar = Table.grid(expand=True)
        bar.add_column()
        bar.add_column(justify="right")
        left = Text("llm-encryptor", style=f"bold {ACCENT}")
        if self.place:
            left.append(f"  ›  {self.place}", style=MUTED)
        bar.add_row(left, Text(f"◆ {self.app.model}", style=MUTED))
        return bar


class Hints(Static):
    """The keys that do something here, at the foot of the screen."""

    def __init__(self, *keys: tuple[str, str]) -> None:
        line = Text()
        for key, does in keys:
            line.append(f" {key} ", style=f"bold {ACCENT}")
            line.append(f"{does}   ", style=MUTED)
        super().__init__(line, classes="hints")


class Menu(Widget):
    """A short list to move through with ↑ ↓ and choose from with Enter."""

    can_focus = True
    BINDINGS: ClassVar = [
        Binding("up", "move(-1)", show=False),
        Binding("down", "move(1)", show=False),
        Binding("enter", "choose", show=False),
    ]
    at = reactive(0)

    class Chosen(Event):
        def __init__(self, key: str) -> None:
            super().__init__()
            self.key = key

    def __init__(self, items: list[tuple[str, str, str]]) -> None:
        super().__init__()
        self.items = items  # (key, name, a word or two beside it)

    def render(self) -> Text:
        menu = Text()
        for number, (_, name, beside) in enumerate(self.items):
            chosen = number == self.at
            menu.append("  ▌ " if chosen else "    ", style=ACCENT)
            menu.append(f"{name:<10}", style="bold white" if chosen else MUTED)
            menu.append(f"{beside}\n\n", style=ACCENT if chosen else MUTED)
        return menu

    def action_move(self, by: int) -> None:
        self.at = (self.at + by) % len(self.items)

    def action_choose(self) -> None:
        self.post_message(self.Chosen(self.items[self.at][0]))

    def on_click(self, event: events.Click) -> None:
        if event.y // 2 < len(self.items):
            self.at = event.y // 2
            self.action_choose()


class Row(Horizontal):
    """A named line of a form. Its name lights up while you are on it."""

    def __init__(self, label: str, control: Widget) -> None:
        super().__init__(classes="row")
        self.label, self.control = label, control

    def compose(self) -> ComposeResult:
        yield Static(self.label, classes="label")
        yield self.control

    def on_descendant_focus(self) -> None:
        self.add_class("on")

    def on_descendant_blur(self) -> None:
        self.remove_class("on")


def row(label: str, name: str, hint: str = "", password: bool = False) -> Row:
    return Row(label, Input(placeholder=hint, password=password, id=name))


class Page(Screen):
    """A screen worked with the keyboard: ↑ ↓ between its lines, Enter to go, Escape to leave."""

    BINDINGS: ClassVar = [
        Binding("escape", "leave", show=False),
        Binding("up", "app.focus_previous", show=False),
        Binding("down", "app.focus_next", show=False),
    ]

    async def action_leave(self) -> None:
        self.app.pop_screen()

    def on_screen_resume(self) -> None:
        for bar in self.query(TopBar):
            bar.refresh()

    def say(self, message: str, style: str = ACTIVE) -> None:
        for status in self.query("#status"):  # Not there yet while the screen is being put up.
            status.update(Text(message, style=style))

    def show_steps(self, steps: Steps) -> None:
        """Called from the worker thread as the model works."""
        self.app.call_from_thread(self.query_one(StepsPanel).show, steps)


# --- screens ----------------------------------------------------------------


class Home(Screen):
    BINDINGS: ClassVar = [Binding("q", "app.quit", show=False)]

    def compose(self) -> ComposeResult:
        yield TopBar()
        picture = Text(justify="center")
        picture.append("meet at noon", style=HIDDEN)
        picture.append("   →   ", style=MUTED)
        picture.append("▮▯▮▮▯▯▮▯▮▮", style=ACTIVE)
        picture.append("   →   ", style=MUTED)
        picture.append("Hey, how was your weekend?", style=VISIBLE)
        yield Static(picture, id="intro")
        yield Menu([("chat", "Chat", ""), ("hide", "Hide", ""), ("reveal", "Reveal", ""), ("model", "Model", "")])
        yield Hints(("↑↓", "move"), ("enter", "open"), ("q", "quit"))

    def on_mount(self) -> None:
        self.on_screen_resume()

    def on_screen_resume(self) -> None:
        menu = self.query_one(Menu)
        menu.items[-1] = ("model", "Model", self.app.model)
        menu.refresh()
        menu.focus()
        self.query_one(TopBar).refresh()

    def on_menu_chosen(self, event: Menu.Chosen) -> None:
        screens = {"chat": Connect, "hide": Hide, "reveal": Reveal, "model": Models}
        self.app.push_screen(screens[event.key]())


class Hide(Page):
    def compose(self) -> ComposeResult:
        yield TopBar("hide")
        yield row("secret", "secret")
        yield row("password", "password", password=True)
        yield row("topic", "about", "dinner on Friday")
        yield Row("length", LengthDial())
        yield StepsPanel(id="steps")
        yield Static("", id="status")
        yield Hints(("↑↓", "move"), ("←→", "length"), ("enter", "hide"), ("esc", "back"))

    def on_mount(self) -> None:
        self.query_one("#secret", Input).focus()

    def on_input_submitted(self) -> None:
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
            return app.call_from_thread(self.say, _said(error), BAD)
        copied = " · copied" if copy_text(text) else ""
        app.call_from_thread(self.say, f"✓ {len(text.split())} words{copied}", HIDDEN)


class Reveal(Page):
    """Reads the text on the clipboard: copy it, come here, type the password."""

    BINDINGS: ClassVar = [*Page.BINDINGS, Binding("ctrl+r", "paste", show=False)]

    def compose(self) -> ComposeResult:
        yield TopBar("reveal")
        yield Static("", id="cover")
        yield row("password", "password", password=True)
        yield StepsPanel(id="steps")
        yield Static("", id="status")
        yield Hints(("enter", "reveal"), ("ctrl+r", "take the clipboard again"), ("esc", "back"))

    def on_mount(self) -> None:
        self.action_paste()
        self.query_one("#password", Input).focus()

    def action_paste(self) -> None:
        self.cover = " ".join((read_clipboard() or "").split())
        self.query_one("#cover", Static).update(
            Text(self.cover, style=VISIBLE) if self.cover else Text("the clipboard is empty", style=MUTED)
        )

    def on_input_submitted(self) -> None:
        password = self.query_one("#password", Input).value
        if not self.cover or not password:
            return self.say("text and password needed")
        self.say("")
        self.run_worker(lambda: self._reveal(self.cover, password), thread=True)

    def _reveal(self, text: str, password: str) -> None:
        app = self.app
        try:
            secret = app.workshop.reveal(text, derive_keys(password), self.show_steps)
        except _PROBLEMS as error:
            return app.call_from_thread(self.say, _said(error), BAD)
        app.call_from_thread(self.say, f"✓ {secret}", HIDDEN)


class Models(Page):
    """Pick the model, or download another. Both people need the same one."""

    BINDINGS: ClassVar = [Binding("escape", "leave", show=False)]

    def compose(self) -> ComposeResult:
        yield TopBar("model")
        yield OptionList(id="installed")
        yield row("download", "pull", "a name from ollama.com/library, e.g. qwen3:4b")
        yield Static("", id="status")
        yield Hints(("↑↓", "pick"), ("enter", "use"), ("tab", "download another"), ("esc", "back"))

    def on_mount(self) -> None:
        self.run_worker(self._load, thread=True)

    def _load(self) -> None:
        try:
            names = self.app.list_models(self.app.host)
        except _PROBLEMS as error:
            return self.app.call_from_thread(self.say, _said(error), BAD)
        self.app.call_from_thread(self._show, names)

    def _show(self, names: list[str]) -> None:
        installed = self.query_one("#installed", OptionList)
        installed.clear_options()
        installed.add_options(names)
        if self.app.model in names:
            installed.highlighted = names.index(self.app.model)
        installed.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.app.model = str(event.option.prompt)
        self.app.pop_screen()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        name = event.value.strip()
        if name:
            self.run_worker(lambda: self._download(name), thread=True)

    def _download(self, name: str) -> None:
        app = self.app
        try:
            for progress in app.pull_model(name, app.host):
                app.call_from_thread(self.say, f"↓ {name}  {progress}")
        except _PROBLEMS as error:
            return app.call_from_thread(self.say, f"✗ {_said(error)}", BAD)
        app.model = name
        app.call_from_thread(self.say, f"✓ {name}", HIDDEN)
        app.call_from_thread(self.query_one(TopBar).refresh)
        self._load()


class Connect(Page):
    """Find each other: one of you waits, the other types the ID."""

    def compose(self) -> ComposeResult:
        yield TopBar("chat")
        yield Row("your id", Static("…", id="my-id"))
        yield row("name", "name")
        yield row("password", "password", "the one you two share", password=True)
        yield row("friend's id", "friend", "empty = wait for them")
        yield Static("", id="status")
        yield Hints(("↑↓", "move"), ("enter", "connect"), ("esc", "back"))

    async def on_mount(self) -> None:
        self.line = Line("me", self._text_arrived, self._line_changed, self.app.model)
        self.keys: Keys | None = None
        self.early: list[str] = []
        self.chat: Chat | None = None
        self.query_one("#name", Input).focus()
        try:
            my_id = await self.line.open(self.app.port)
        except OSError as error:
            return self.say(f"Could not start listening: {error.strerror}", BAD)
        self.query_one("#my-id", Static).update(Text(my_id, style=f"bold {ACTIVE}"))

    def on_input_submitted(self) -> None:
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
            return self.say(_said(error), BAD)
        self._line_changed()

    def _line_changed(self) -> None:
        if self.chat is not None:
            return self.chat.line_changed()
        if self.line.connected and self.keys is not None:
            self.chat = Chat(self.line, self.keys, self.early)
            self.app.push_screen(self.chat)
        elif self.line.connected:
            self.say("your friend is here · password needed", HIDDEN)

    def _text_arrived(self, text: str) -> None:
        if self.chat is not None:
            self.chat.text_arrived(text)
        else:
            self.early.append(text)

    async def action_leave(self) -> None:
        await self.line.close()
        self.app.pop_screen()


class Turn(Horizontal):
    """One message on one line of the conversation: who, what was sent, what it said."""

    def __init__(self, who: str, mine: bool, cover: str, secret: str) -> None:
        super().__init__(classes="turn mine" if mine else "turn")
        self.who, self.cover, self.secret = who, cover, secret

    def compose(self) -> ComposeResult:
        yield Static(self.who, classes="who")
        yield Static(self.cover, classes="seen")
        yield Static(self.secret, classes="said")

    def set(self, cover: str | None = None, secret: str | None = None, bad: bool = False) -> None:
        self.cover, self.secret = cover or self.cover, secret or self.secret
        self.query_one(".seen", Static).update(self.cover)
        said = self.query_one(".said", Static)
        said.update(self.secret)
        said.set_class(bad, "bad")


class Chat(Page):
    """Two conversations: the visible one that travels, and the real one."""

    BINDINGS: ClassVar = [
        *Page.BINDINGS,
        Binding("pageup", "scroll(-1)", show=False),
        Binding("pagedown", "scroll(1)", show=False),
    ]

    def __init__(self, line: Line, keys: Keys, early: list[str]) -> None:
        super().__init__()
        self.line, self.keys, self.early = line, keys, early
        self.last_from_friend: str | None = None

    def compose(self) -> ComposeResult:
        yield TopBar("chat")
        with Horizontal(id="heads"):
            yield Static("", classes="who")
            yield Static("visible", classes="seen")
            yield Static("hidden", classes="said")
        turns = VerticalScroll(id="turns")
        turns.can_focus = False
        yield turns
        yield StepsPanel(id="steps")
        yield row("secret", "secret")
        yield row("topic", "about")
        yield Row("length", LengthDial())
        yield Static("", id="status")
        yield Hints(("↑↓", "move"), ("←→", "length"), ("enter", "send"), ("pgup pgdn", "scroll"), ("esc", "leave"))

    def show_steps(self, steps: Steps) -> None:
        """The stages are only on screen while a message is on its way in or out."""
        self.app.call_from_thread(self._stages, steps)

    def _stages(self, steps: Steps) -> None:
        panel = self.query_one(StepsPanel)
        panel.show(steps)
        panel.display = True
        if steps.finished or steps.problem:
            self.set_timer(_LINGER, self._put_stages_away)

    def _put_stages_away(self) -> None:
        panel = self.query_one(StepsPanel)
        if panel.steps.finished or panel.steps.problem:  # Not if the next message has started.
            panel.display = False

    def on_mount(self) -> None:
        self.query_one(StepsPanel).display = False
        self._prompt()
        self.line_changed()
        waiting, self.early = self.early, []
        for text in waiting:
            self._show_arrived(text)
        self.query_one("#secret", Input).focus()

    def action_scroll(self, by: int) -> None:
        turns = self.query_one("#turns", VerticalScroll)
        turns.scroll_page_down() if by > 0 else turns.scroll_page_up()

    def _prompt(self) -> None:
        self.query_one("#about", Input).placeholder = (
            "↩ replies to their last text" if self.last_from_friend else "dinner on Friday"
        )

    def line_changed(self) -> None:
        friend = self.line.friend or "your friend"
        theirs = self.line.friend_model
        if not self.line.connected:
            self.say(f"○ {friend} left", BAD)
        elif theirs and theirs != self.app.model:
            # With different models neither side can read the other.
            self.say(f"● {friend}   ≠ model: {theirs}", BAD)
        else:
            self.say(f"● {friend}", HIDDEN)

    def on_input_submitted(self) -> None:
        secret = self.query_one("#secret", Input)
        about = self.query_one("#about", Input)
        said = secret.value.strip()
        if not said:
            return
        if not self.line.connected:
            return self.say("○ not connected", BAD)
        turn = self._add("you", True, "…", said)
        asked = request(about.value, self.last_from_friend)
        density = self.query_one(LengthDial).density
        # A steer decides what the text opens on; without one it answers their text.
        steer = about.value.strip() or self.last_from_friend
        self.run_worker(lambda: self._send(turn, said, asked, density, steer), thread=True)
        secret.value, about.value = "", ""
        secret.focus()

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
            return app.call_from_thread(self.say, _said(error), BAD)
        app.call_from_thread(turn.set, text)
        app.call_from_thread(self.run_worker, self._deliver(text))

    async def _deliver(self, text: str) -> None:
        try:
            await self.line.send(text)
        except LineError as error:
            self.say(_said(error), BAD)

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
            return app.call_from_thread(turn.set, None, "✗ unreadable · different password?", True)
        app.call_from_thread(turn.set, None, secret)


# --- the app ----------------------------------------------------------------


class EncryptorApp(App):
    TITLE = "llm-encryptor"
    BINDINGS: ClassVar = [Binding("ctrl+q", "quit", show=False)]
    CSS = """
    Screen { background: #0f1117; padding: 1 4; align: center top; }
    Screen > * { max-width: 110; }
    .topbar { height: 1; margin-bottom: 2; }
    .hints { dock: bottom; height: 1; margin: 0 4 1 4; }
    #intro { margin: 3 0 4 0; height: auto; width: 100%; text-align: center; }
    Menu { width: 44; height: auto; }
    Menu:focus { background: #0f1117; }
    .row { height: 1; margin-bottom: 1; }
    .row .label { width: 14; color: #6b7280; }
    .row.on .label { color: #c4b5fd; text-style: bold; }
    Input { border: none; height: 1; padding: 0 1; background: #151922; }
    Input:focus { border: none; background: #1c2030; background-tint: transparent; }
    #my-id { width: 1fr; padding: 0 1; }
    LengthDial { height: 1; width: 1fr; padding: 0 1; }
    OptionList { border: none; background: #151922; height: auto; max-height: 14; padding: 1 1; margin-bottom: 2; }
    OptionList:focus { border: none; background: #151922; }
    OptionList > .option-list--option-highlighted { background: #c4b5fd; color: #0f1117; text-style: bold; }
    #cover { background: #151922; padding: 1 2; height: auto; margin-bottom: 2; }
    StepsPanel { background: #151922; padding: 1 2; height: auto; margin: 1 0; }
    #status { height: 1; margin-top: 1; }
    #heads { height: 1; margin-bottom: 1; }
    #heads .seen { color: #7dd3fc; text-style: bold; }
    #heads .said { color: #86efac; text-style: bold; }
    #turns { height: 1fr; scrollbar-size-vertical: 1; }
    .turn { height: auto; margin-bottom: 1; }
    .who { width: 10; color: #6b7280; }
    .turn.mine .who { color: #c4b5fd; text-style: bold; }
    .seen { width: 3fr; padding: 0 2 0 0; color: #7dd3fc; }
    .said { width: 2fr; padding: 0 0 0 2; color: #86efac; border-left: solid #2a3040; }
    #heads .said { border-left: none; padding: 0 0 0 3; }
    .turn .bad { color: #f87171; }
    """

    def __init__(
        self,
        model: str,
        host: str,
        make_backend: Callable[[str, str], Backend] = OllamaBackend,
        port: int = 0,
        list_models: Callable[[str], list[str]] = list_models,
        pull_model: Callable[[str, str], Iterator[str]] = pull_model,
    ) -> None:
        super().__init__()
        self.model, self.host, self.port = model, host, port
        self.list_models, self.pull_model = list_models, pull_model
        # Whatever model is chosen at the time a message is written or read.
        self.workshop = Workshop(lambda: make_backend(self.model, self.host))

    def on_mount(self) -> None:
        self.register_theme(Theme(name="veil", primary=ACCENT, accent=ACCENT, background="#0f1117", dark=True))
        self.theme = "veil"
        self.push_screen(Home())
