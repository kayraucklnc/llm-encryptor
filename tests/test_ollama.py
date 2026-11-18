import io
import json
import math
import urllib.error

import pytest

from llm_encryptor.backend import BackendError, Choice
from llm_encryptor.ollama import OllamaBackend, http_transport, list_models

ASK = [{"role": "user", "content": "Write about the sea."}]


class Recorder:
    def __init__(self, answers):
        self.answers = answers
        self.requests = []

    def __call__(self, path, body):
        self.requests.append((path, body))
        yield from self.answers


def _logprobs(*pairs):
    top = [{"token": token, "logprob": math.log(p)} for token, p in pairs]
    return [{"token": pairs[0][0], "logprob": math.log(pairs[0][1]), "top_logprobs": top}]


def test_next_tokens_sends_the_text_so_far_as_the_reply():
    answer = {"message": {"content": " is"}, "logprobs": _logprobs((" is", 0.75), (" was", 0.25))}
    transport = Recorder([answer])
    choices = OllamaBackend("m", transport=transport).next_tokens(ASK, "The sea")
    assert [choice.text for choice in choices] == [" is", " was"]
    assert [choice.probability for choice in choices] == pytest.approx([0.75, 0.25])
    path, body = transport.requests[0]
    assert path == "/api/chat"
    assert body["messages"][-1] == {"role": "assistant", "content": "The sea"}
    assert body["options"]["num_predict"] == 1
    assert body["logprobs"] is True and body["top_logprobs"] == 20
    assert body["stream"] is False


def test_control_tokens_mean_the_reply_ends():
    answer = {"logprobs": _logprobs((".", 0.4), ("<|im_end|>", 0.3), ("</s>", 0.1), ("<3", 0.1), ("<br>", 0.1))}
    choices = OllamaBackend("m", transport=Recorder([answer])).next_tokens(ASK, "The sea")
    assert [choice.text for choice in choices] == [".", "", "", "<3", "<br>"]


@pytest.mark.parametrize("answer", [{"message": {"content": ""}, "logprobs": []}, {"message": {"content": ""}}])
def test_no_token_means_the_reply_ends(answer):
    backend = OllamaBackend("m", transport=Recorder([answer]))
    assert backend.next_tokens(ASK, "The end.") == [Choice("", 1.0)]


def test_a_server_without_probabilities_is_explained():
    backend = OllamaBackend("m", transport=Recorder([{"message": {"content": " is"}}]))
    with pytest.raises(BackendError, match="Update Ollama"):
        backend.next_tokens(ASK, "The sea")


def test_starting_skips_thinking_and_stops_early():
    chunks = [{"message": {"content": "", "thinking": "hmm"}}]
    chunks += [{"message": {"content": f" w{i}"}} for i in range(20)]
    transport = Recorder(chunks)
    opening = OllamaBackend("m", transport=transport).start(ASK)
    assert opening == "".join(f" w{i}" for i in range(8))
    body = transport.requests[0][1]
    assert body["stream"] is True
    assert body["messages"] == ASK


def test_starting_stops_when_the_model_is_done():
    transport = Recorder([{"message": {"content": "Hi"}, "done": True}])
    assert OllamaBackend("m", transport=transport).start(ASK) == "Hi"


@pytest.mark.parametrize("call", [lambda b: b.next_tokens(ASK, "The sea"), lambda b: b.start(ASK)])
def test_error_in_the_answer_is_raised(call):
    backend = OllamaBackend("m", transport=Recorder([{"error": "model not found"}]))
    with pytest.raises(BackendError, match="model not found"):
        call(backend)


def test_empty_answer_is_raised():
    with pytest.raises(BackendError):
        OllamaBackend("m", transport=Recorder([])).next_tokens(ASK, "The sea")


def test_list_models():
    transport = Recorder([{"models": [{"name": "qwen3:4b"}, {"name": "llama3.2:latest"}]}])
    assert list_models(transport=transport) == ["qwen3:4b", "llama3.2:latest"]
    assert list_models(transport=Recorder([])) == []
    assert list_models(transport=Recorder([{"models": [{"size": 1}]}])) == []


def test_http_transport_reads_json_lines(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return io.BytesIO(b'{"a": 1}\n\n{"a": 2}\n')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert list(http_transport("http://host:1/")("/api/chat", {"x": 1})) == [{"a": 1}, {"a": 2}]
    assert seen == {"url": "http://host:1/api/chat", "body": {"x": 1}}


def test_http_transport_explains_an_unreachable_server(monkeypatch):
    def refuse(request, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(BackendError, match="ollama serve"):
        list(http_transport("http://host:1")("/api/tags", None))


def test_http_transport_explains_a_broken_connection(monkeypatch):
    import http.client

    def cut(request, timeout):
        raise http.client.IncompleteRead(b"")

    monkeypatch.setattr("urllib.request.urlopen", cut)
    with pytest.raises(BackendError, match="Could not reach"):
        list(http_transport("http://host:1")("/api/chat", {}))


def test_http_transport_reports_server_errors(monkeypatch):
    def fail(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "nf", {}, io.BytesIO(b"no such model"))

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(BackendError, match="404.*no such model"):
        list(http_transport("http://host:1")("/api/chat", {}))


def test_http_transport_reports_unreadable_answers(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: io.BytesIO(b"<html>"))
    with pytest.raises(BackendError):
        list(http_transport("http://host:1")("/api/chat", {}))
