"""Tests for aiohttp/protocol.py"""

import asyncio
import zlib
from unittest import mock

import pytest
from multidict import CIMultiDict
from yarl import URL

import aiohttp
from aiohttp import http_exceptions, streams
from aiohttp.http_parser import (
    DeflateBuffer,
    HttpPayloadParser,
    HttpRequestParserPy,
    HttpResponseParserPy,
)

try:
    import brotli
except ImportError:
    brotli = None


REQUEST_PARSERS = [HttpRequestParserPy]
RESPONSE_PARSERS = [HttpResponseParserPy]

try:
    from aiohttp.http_parser import HttpRequestParserC, HttpResponseParserC

    REQUEST_PARSERS.append(HttpRequestParserC)
    RESPONSE_PARSERS.append(HttpResponseParserC)
except ImportError:  # pragma: no cover
    pass


@pytest.fixture
def protocol():
    return mock.Mock()


@pytest.fixture(params=REQUEST_PARSERS)
def parser(loop, protocol, request):
    """Parser implementations"""
    return request.param(
        protocol, loop, max_line_size=8190, max_headers=32768, max_field_size=8190
    )


@pytest.fixture(params=REQUEST_PARSERS)
def request_cls(request):
    """Request Parser class"""
    return request.param


@pytest.fixture(params=RESPONSE_PARSERS)
def response(loop, protocol, request):
    """Parser implementations"""
    return request.param(
        protocol, loop, max_line_size=8190, max_headers=32768, max_field_size=8190
    )


@pytest.fixture(params=RESPONSE_PARSERS)
def response_cls(request):
    """Parser implementations"""
    return request.param


@pytest.fixture
def stream():
    return mock.Mock()


def test_parse_headers(parser) -> None:
    text = b"""GET /test HTTP/1.1\r
test: line\r
 continue\r
test2: data\r
\r
"""
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    msg = messages[0][0]

    assert list(msg.headers.items()) == [("test", "line continue"), ("test2", "data")]
    assert msg.raw_headers == ((b"test", b"line continue"), (b"test2", b"data"))
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade


def test_parse(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    msg, _ = messages[0]
    assert msg.compression is None
    assert not msg.upgrade
    assert msg.method == "GET"
    assert msg.path == "/test"
    assert msg.version == (1, 1)


async def test_parse_body(parser) -> None:
    text = b"GET /test HTTP/1.1\r\nContent-Length: 4\r\n\r\nbody"
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    _, payload = messages[0]
    body = await payload.read(4)
    assert body == b"body"


async def test_parse_body_with_CRLF(parser) -> None:
    text = b"\r\nGET /test HTTP/1.1\r\nContent-Length: 4\r\n\r\nbody"
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    _, payload = messages[0]
    body = await payload.read(4)
    assert body == b"body"


def test_parse_delayed(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 0
    assert not upgrade

    messages, upgrade, tail = parser.feed_data(b"\r\n")
    assert len(messages) == 1
    msg = messages[0][0]
    assert msg.method == "GET"


def test_headers_multi_feed(parser) -> None:
    text1 = b"GET /test HTTP/1.1\r\n"
    text2 = b"test: line\r"
    text3 = b"\n continue\r\n\r\n"

    messages, upgrade, tail = parser.feed_data(text1)
    assert len(messages) == 0

    messages, upgrade, tail = parser.feed_data(text2)
    assert len(messages) == 0

    messages, upgrade, tail = parser.feed_data(text3)
    assert len(messages) == 1

    msg = messages[0][0]
    assert list(msg.headers.items()) == [("test", "line continue")]
    assert msg.raw_headers == ((b"test", b"line continue"),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade


def test_headers_split_field(parser) -> None:
    text1 = b"GET /test HTTP/1.1\r\n"
    text2 = b"t"
    text3 = b"es"
    text4 = b"t: value\r\n\r\n"

    messages, upgrade, tail = parser.feed_data(text1)
    messages, upgrade, tail = parser.feed_data(text2)
    messages, upgrade, tail = parser.feed_data(text3)
    assert len(messages) == 0
    messages, upgrade, tail = parser.feed_data(text4)
    assert len(messages) == 1

    msg = messages[0][0]
    assert list(msg.headers.items()) == [("test", "value")]
    assert msg.raw_headers == ((b"test", b"value"),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade


def test_parse_headers_multi(parser) -> None:
    text = (
        b"GET /test HTTP/1.1\r\n"
        b"Set-Cookie: c1=cookie1\r\n"
        b"Set-Cookie: c2=cookie2\r\n\r\n"
    )

    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    msg = messages[0][0]

    assert list(msg.headers.items()) == [
        ("Set-Cookie", "c1=cookie1"),
        ("Set-Cookie", "c2=cookie2"),
    ]
    assert msg.raw_headers == (
        (b"Set-Cookie", b"c1=cookie1"),
        (b"Set-Cookie", b"c2=cookie2"),
    )
    assert not msg.should_close
    assert msg.compression is None


def test_conn_default_1_0(parser) -> None:
    text = b"GET /test HTTP/1.0\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.should_close


def test_conn_default_1_1(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close


def test_conn_close(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"connection: close\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.should_close


def test_conn_close_1_0(parser) -> None:
    text = b"GET /test HTTP/1.0\r\n" b"connection: close\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.should_close


def test_conn_keep_alive_1_0(parser) -> None:
    text = b"GET /test HTTP/1.0\r\n" b"connection: keep-alive\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close


def test_conn_keep_alive_1_1(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"connection: keep-alive\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close


def test_conn_other_1_0(parser) -> None:
    text = b"GET /test HTTP/1.0\r\n" b"connection: test\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.should_close


def test_conn_other_1_1(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"connection: test\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close


def test_request_chunked(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"transfer-encoding: chunked\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg, payload = messages[0]
    assert msg.chunked
    assert not upgrade
    assert isinstance(payload, streams.StreamReader)


def test_request_te_chunked_with_content_length(parser) -> None:
    text = (
        b"GET /test HTTP/1.1\r\n"
        b"content-length: 1234\r\n"
        b"transfer-encoding: chunked\r\n\r\n"
    )
    if isinstance(parser, HttpRequestParserC):
        with pytest.raises(
            http_exceptions.BadHttpMessage,
            match="unexpected content-length header",
        ):
            parser.feed_data(text)

    if isinstance(parser, HttpRequestParserPy):
        with pytest.raises(
            http_exceptions.BadHttpMessage,
            match="Content-Length can't be present with Transfer-Encoding",
        ):
            parser.feed_data(text)


def test_conn_upgrade(parser) -> None:
    text = (
        b"GET /test HTTP/1.1\r\n"
        b"connection: upgrade\r\n"
        b"upgrade: websocket\r\n\r\n"
    )
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close
    assert msg.upgrade
    assert upgrade


def test_compression_empty(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"content-encoding: \r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression is None


def test_compression_deflate(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"content-encoding: deflate\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression == "deflate"


def test_compression_gzip(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"content-encoding: gzip\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression == "gzip"


@pytest.mark.skipif(brotli is None, reason="brotli is not installed")
def test_compression_brotli(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"content-encoding: br\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression == "br"


def test_compression_unknown(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"content-encoding: compress\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression is None


def test_headers_connect(parser) -> None:
    text = b"CONNECT www.google.com HTTP/1.1\r\n" b"content-length: 0\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg, payload = messages[0]
    assert upgrade
    assert isinstance(payload, streams.StreamReader)


def test_headers_old_websocket_key1(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"SEC-WEBSOCKET-KEY1: line\r\n\r\n"

    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_headers_content_length_err_1(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"content-length: line\r\n\r\n"

    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_headers_content_length_err_2(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"content-length: -1\r\n\r\n"

    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_parse_unusual_request_line(parser) -> None:
    if not isinstance(parser, HttpResponseParserPy):
        return
    text = b"#smol //a HTTP/1.3\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    msg, _ = messages[0]
    assert msg.compression is None
    assert not msg.upgrade
    assert msg.method == "#smol"
    assert msg.path == "//a"
    assert msg.version == (1, 3)


_pad = {
    b"": "empty",
    # not a typo. Python likes triple zero
    b"\000": "NUL",
    b" ": "SP",
    b"  ": "SPSP",
    # not a typo: both 0xa0 and 0x0a in case of 8-bit fun
    b"\n": "LF",
    b"\xa0": "NBSP",
    b"\t ": "TABSP",
}


@pytest.mark.parametrize("hdr", [b"", b"foo"], ids=["name-empty", "with-name"])
@pytest.mark.parametrize("pad2", _pad.keys(), ids=["post-" + n for n in _pad.values()])
@pytest.mark.parametrize("pad1", _pad.keys(), ids=["pre-" + n for n in _pad.values()])
def test_invalid_header_spacing(parser, pad1, pad2, hdr) -> None:
    if not isinstance(parser, HttpResponseParserPy):
        return
    text = b"GET /test HTTP/1.1\r\n" b"%s%s%s: value\r\n\r\n" % (pad1, hdr, pad2)
    should_raise = True
    if pad1 == pad2 == b"" and hdr != b"":
        # one entry in param matrix is correct: non-empty name, not padded
        should_raise = False
    if should_raise:
        with pytest.raises(http_exceptions.BadHttpMessage):
            parser.feed_data(text)
    else:
        parser.feed_data(text)


def test_empty_header_name(parser) -> None:
    if not isinstance(parser, HttpResponseParserPy):
        return
    text = b"GET /test HTTP/1.1\r\n" b":test\r\n\r\n"
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


_num = {
    # dangerous: accepted by Python int()
    # unicodedata.category("\U0001D7D9") == 'Nd'
    "\N{mathematical double-struck digit one}".encode(): "utf8digit",
    # only added for interop tests, refused by Python int()
    # unicodedata.category("\U000000B9") == 'No'
    "\N{superscript one}".encode(): "utf8number",
    "\N{superscript one}".encode("latin-1"): "latin1number",
}


@pytest.mark.parametrize("nonascii_digit", _num.keys(), ids=list(_num.values()))
def test_http_request_bad_status_line_number(parser, nonascii_digit) -> None:
    text = b"GET /digit HTTP/1." + nonascii_digit + b"\r\n\r\n"
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


@pytest.mark.parametrize("nonascii_digit", _num.keys(), ids=list(_num.values()))
def test_http_response_parser_code_not_ascii(response, nonascii_digit) -> None:
    with pytest.raises(http_exceptions.BadStatusLine):
        response.feed_data(b"HTTP/1.1 20" + nonascii_digit + b" test\r\n\r\n")


def test_http_request_bad_status_line_separator(parser) -> None:
    # single code point, old, multibyte NFKC, multibyte NFKD
    utf8sep = "\N{arabic ligature sallallahou alayhe wasallam}".encode()
    text = b"GET /ligature HTTP/1" + utf8sep + b"1\r\n\r\n"
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_http_request_parser_utf8_request_line(parser) -> None:
    if not isinstance(parser, HttpResponseParserPy):
        return
    messages, upgrade, tail = parser.feed_data(
        # note the truncated unicode sequence
        b"GET /P\xc3\xbcnktchen\xa0\xef\xb7 HTTP/1.1\r\n" +
        # for easier grep: ASCII 0xA0 more commonly known as non-breaking space
        # note the leading and trailing spaces
        "sTeP:  \N{latin small letter sharp s}nek\t\N{no-break space}  "
        "\r\n\r\n".encode()
    )
    msg = messages[0][0]

    assert msg.method == "GET"
    assert msg.path == "/Pünktchen\udca0\udcef\udcb7"
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict([("STEP", "ßnek\t\xa0")])
    assert msg.raw_headers == ((b"sTeP", "ßnek\t\xa0".encode()),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url.path == URL("/P%C3%BCnktchen\udca0\udcef\udcb7").path


@pytest.mark.parametrize(
    "rfc9110_5_6_2_token_delim",
    r'"(),/:;<=>?@[\]{}',
)
def test_bad_header_name(parser, rfc9110_5_6_2_token_delim) -> None:
    text = "POST / HTTP/1.1\r\nhead{}er: val\r\n\r\n".format(
        rfc9110_5_6_2_token_delim
    ).encode()
    if rfc9110_5_6_2_token_delim == ":":
        # Inserting colon into header just splits name/value earlier.
        parser.feed_data(text)
    else:
        with pytest.raises(http_exceptions.BadHttpMessage):
            parser.feed_data(text)


def test_invalid_header(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"test line\r\n\r\n"
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_invalid_name(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"test[]: line\r\n\r\n"

    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


@pytest.mark.parametrize(
    "hdr",
    (
        "Content-Length: -5",  # https://www.rfc-editor.org/rfc/rfc9110.html#name-content-length
        "Content-Length: +256",
        "Content-Length: \N{superscript one}",
        "Content-Length: \N{mathematical double-struck digit one}",
        "Foo: abc\rdef",  # https://www.rfc-editor.org/rfc/rfc9110.html#section-5.5-5
        "Bar: abc\ndef",
        "Baz: abc\x00def",
        "Foo : bar",  # https://www.rfc-editor.org/rfc/rfc9112.html#section-5.1-2
        "Foo\t: bar",
    ),
)
def test_bad_headers(parser, hdr) -> None:
    if isinstance(parser, HttpRequestParserC):
        return
    text = "POST / HTTP/1.1\r\n{}\r\n\r\n".format(hdr).encode()
    with pytest.raises(http_exceptions.InvalidHeader):
        parser.feed_data(text)


def test_bad_chunked_py(loop, protocol) -> None:
    """Test that invalid chunked encoding doesn't allow content-length to be used."""
    parser = HttpRequestParserPy(
        protocol=protocol,
        loop=loop,
        max_line_size=8190,
        max_field_size=8190,
    )
    text = (
        b"GET / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n0_2e\r\n\r\n"
        + b"GET / HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\n0\r\n\r\n"
    )
    messages, upgrade, tail = parser.feed_data(text)
    assert isinstance(messages[0][1].exception(), http_exceptions.TransferEncodingError)


@pytest.mark.skipif(
    "HttpRequestParserC" not in dir(aiohttp.http_parser),
    reason="C based HTTP parser not available",
)
def test_bad_chunked_c(loop, protocol) -> None:
    """C parser behaves differently. Maybe we should align them later."""
    parser = HttpRequestParserC(
        protocol,
        loop,
        2**16,
        max_line_size=8190,
        max_field_size=8190,
    )
    text = (
        b"GET / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n0_2e\r\n\r\n"
        + b"GET / HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\n0\r\n\r\n"
    )
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_whitespace_before_header(parser) -> None:
    text = b"GET / HTTP/1.1\r\n\tContent-Length: 1\r\n\r\nX"
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_http_request_parser_bad_version_number(parser) -> None:
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(b"GET /test HTTP/12.3\r\n\r\n")


def test_http_response_parser_bad_version_number(response) -> None:
    with pytest.raises(http_exceptions.BadHttpMessage):
        response.feed_data(b"HTTP/12.3 200 Ok\r\n\r\n")


@pytest.mark.parametrize("size", [40960, 8191])
def test_max_header_field_size(parser, size) -> None:
    name = b"t" * size
    text = b"GET /test HTTP/1.1\r\n" + name + b":data\r\n\r\n"

    match = "400, message='Got more than 8190 bytes \\({}\\) when reading".format(size)
    with pytest.raises(http_exceptions.LineTooLong, match=match):
        parser.feed_data(text)


def test_max_header_field_size_under_limit(parser) -> None:
    name = b"t" * 8190
    text = b"GET /test HTTP/1.1\r\n" + name + b":data\r\n\r\n"

    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.method == "GET"
    assert msg.path == "/test"
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict({name.decode(): "data"})
    assert msg.raw_headers == ((name, b"data"),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL("/test")


@pytest.mark.parametrize("size", [40960, 8191])
def test_max_header_value_size(parser, size) -> None:
    name = b"t" * size
    text = b"GET /test HTTP/1.1\r\n" b"data:" + name + b"\r\n\r\n"

    match = "400, message='Got more than 8190 bytes \\({}\\) when reading".format(size)
    with pytest.raises(http_exceptions.LineTooLong, match=match):
        parser.feed_data(text)


def test_max_header_value_size_under_limit(parser) -> None:
    value = b"A" * 8190
    text = b"GET /test HTTP/1.1\r\n" b"data:" + value + b"\r\n\r\n"

    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.method == "GET"
    assert msg.path == "/test"
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict({"data": value.decode()})
    assert msg.raw_headers == ((b"data", value),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL("/test")


@pytest.mark.parametrize("size", [40965, 8191])
def test_max_header_value_size_continuation(parser, size) -> None:
    name = b"T" * (size - 5)
    text = b"GET /test HTTP/1.1\r\n" b"data: test\r\n " + name + b"\r\n\r\n"

    match = "400, message='Got more than 8190 bytes \\({}\\) when reading".format(size)
    with pytest.raises(http_exceptions.LineTooLong, match=match):
        parser.feed_data(text)


def test_max_header_value_size_continuation_under_limit(parser) -> None:
    value = b"A" * 8185
    text = b"GET /test HTTP/1.1\r\n" b"data: test\r\n " + value + b"\r\n\r\n"

    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.method == "GET"
    assert msg.path == "/test"
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict({"data": "test " + value.decode()})
    assert msg.raw_headers == ((b"data", b"test " + value),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL("/test")


def test_http_request_parser(parser) -> None:
    text = b"GET /path HTTP/1.1\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]

    assert msg.method == "GET"
    assert msg.path == "/path"
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict()
    assert msg.raw_headers == ()
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL("/path")


def test_http_request_bad_status_line(parser) -> None:
    text = b"getpath \r\n\r\n"
    with pytest.raises(http_exceptions.BadStatusLine):
        parser.feed_data(text)


def test_http_request_upgrade(parser) -> None:
    text = (
        b"GET /test HTTP/1.1\r\n"
        b"connection: upgrade\r\n"
        b"upgrade: websocket\r\n\r\n"
        b"some raw data"
    )
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close
    assert msg.upgrade
    assert upgrade
    assert tail == b"some raw data"


def test_http_request_parser_utf8(parser) -> None:
    text = "GET /path HTTP/1.1\r\nx-test:тест\r\n\r\n".encode("utf-8")
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]

    assert msg.method == "GET"
    assert msg.path == "/path"
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict([("X-TEST", "тест")])
    assert msg.raw_headers == ((b"x-test", "тест".encode("utf-8")),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL("/path")


def test_http_request_parser_non_utf8(parser) -> None:
    text = "GET /path HTTP/1.1\r\nx-test:тест\r\n\r\n".encode("cp1251")
    msg = parser.feed_data(text)[0][0][0]

    assert msg.method == "GET"
    assert msg.path == "/path"
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict(
        [("X-TEST", "тест".encode("cp1251").decode("utf8", "surrogateescape"))]
    )
    assert msg.raw_headers == ((b"x-test", "тест".encode("cp1251")),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL("/path")


def test_http_request_parser_two_slashes(parser) -> None:
    text = b"GET //path HTTP/1.1\r\n\r\n"
    msg = parser.feed_data(text)[0][0][0]

    assert msg.method == "GET"
    assert msg.path == "//path"
    assert msg.version == (1, 1)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked


@pytest.mark.parametrize(
    "rfc9110_5_6_2_token_delim",
    [bytes([i]) for i in rb'"(),/:;<=>?@[\]{}'],
)
def test_http_request_parser_bad_method(parser, rfc9110_5_6_2_token_delim) -> None:
    with pytest.raises(http_exceptions.BadStatusLine):
        parser.feed_data(rfc9110_5_6_2_token_delim + b'ET" /get HTTP/1.1\r\n\r\n')


def test_http_request_parser_bad_version(parser) -> None:
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(b"GET //get HT/11\r\n\r\n")


@pytest.mark.parametrize("size", [40965, 8191])
def test_http_request_max_status_line(parser, size) -> None:
    path = b"t" * (size - 5)
    match = "400, message='Got more than 8190 bytes \\({}\\) when reading".format(size)
    with pytest.raises(http_exceptions.LineTooLong, match=match):
        parser.feed_data(b"GET /path" + path + b" HTTP/1.1\r\n\r\n")


def test_http_request_max_status_line_under_limit(parser) -> None:
    path = b"t" * (8190 - 5)
    messages, upgraded, tail = parser.feed_data(
        b"GET /path" + path + b" HTTP/1.1\r\n\r\n"
    )
    msg = messages[0][0]

    assert msg.method == "GET"
    assert msg.path == "/path" + path.decode()
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict()
    assert msg.raw_headers == ()
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL("/path" + path.decode())


def test_http_response_parser_utf8(response) -> None:
    text = "HTTP/1.1 200 Ok\r\nx-test:тест\r\n\r\n".encode("utf-8")

    messages, upgraded, tail = response.feed_data(text)
    assert len(messages) == 1
    msg = messages[0][0]

    assert msg.version == (1, 1)
    assert msg.code == 200
    assert msg.reason == "Ok"
    assert msg.headers == CIMultiDict([("X-TEST", "тест")])
    assert msg.raw_headers == ((b"x-test", "тест".encode("utf-8")),)
    assert not upgraded
    assert not tail


@pytest.mark.parametrize("size", [40962, 8191])
def test_http_response_parser_bad_status_line_too_long(response, size) -> None:
    reason = b"t" * (size - 2)
    match = "400, message='Got more than 8190 bytes \\({}\\) when reading".format(size)
    with pytest.raises(http_exceptions.LineTooLong, match=match):
        response.feed_data(b"HTTP/1.1 200 Ok" + reason + b"\r\n\r\n")


def test_http_response_parser_status_line_under_limit(response) -> None:
    reason = b"O" * 8190
    messages, upgraded, tail = response.feed_data(
        b"HTTP/1.1 200 " + reason + b"\r\n\r\n"
    )
    msg = messages[0][0]
    assert msg.version == (1, 1)
    assert msg.code == 200
    assert msg.reason == reason.decode()


def test_http_response_parser_bad_version(response) -> None:
    with pytest.raises(http_exceptions.BadHttpMessage):
        response.feed_data(b"HT/11 200 Ok\r\n\r\n")


def test_http_response_parser_no_reason(response) -> None:
    msg = response.feed_data(b"HTTP/1.1 200\r\n\r\n")[0][0][0]

    assert msg.version == (1, 1)
    assert msg.code == 200
    assert msg.reason == ""


def test_http_response_parser_bad(response) -> None:
    with pytest.raises(http_exceptions.BadHttpMessage):
        response.feed_data(b"HTT/1\r\n\r\n")


def test_http_response_parser_code_under_100(response) -> None:
    msg = response.feed_data(b"HTTP/1.1 99 test\r\n\r\n")[0][0][0]
    assert msg.code == 99


def test_http_response_parser_code_above_999(response) -> None:
    with pytest.raises(http_exceptions.BadHttpMessage):
        response.feed_data(b"HTTP/1.1 9999 test\r\n\r\n")


def test_http_response_parser_code_not_int(response) -> None:
    with pytest.raises(http_exceptions.BadHttpMessage):
        response.feed_data(b"HTTP/1.1 ttt test\r\n\r\n")


def test_http_request_chunked_payload(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"transfer-encoding: chunked\r\n\r\n"
    msg, payload = parser.feed_data(text)[0][0]

    assert msg.chunked
    assert not payload.is_eof()
    assert isinstance(payload, streams.StreamReader)

    parser.feed_data(b"4\r\ndata\r\n4\r\nline\r\n0\r\n\r\n")

    assert b"dataline" == b"".join(d for d in payload._buffer)
    assert [4, 8] == payload._http_chunk_splits
    assert payload.is_eof()


def test_http_request_chunked_payload_and_next_message(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"transfer-encoding: chunked\r\n\r\n"
    msg, payload = parser.feed_data(text)[0][0]

    messages, upgraded, tail = parser.feed_data(
        b"4\r\ndata\r\n4\r\nline\r\n0\r\n\r\n"
        b"POST /test2 HTTP/1.1\r\n"
        b"transfer-encoding: chunked\r\n\r\n"
    )

    assert b"dataline" == b"".join(d for d in payload._buffer)
    assert [4, 8] == payload._http_chunk_splits
    assert payload.is_eof()

    assert len(messages) == 1
    msg2, payload2 = messages[0]

    assert msg2.method == "POST"
    assert msg2.chunked
    assert not payload2.is_eof()


def test_http_request_chunked_payload_chunks(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"transfer-encoding: chunked\r\n\r\n"
    msg, payload = parser.feed_data(text)[0][0]

    parser.feed_data(b"4\r\ndata\r")
    parser.feed_data(b"\n4")
    parser.feed_data(b"\r")
    parser.feed_data(b"\n")
    parser.feed_data(b"li")
    parser.feed_data(b"ne\r\n0\r\n")
    parser.feed_data(b"test: test\r\n")

    assert b"dataline" == b"".join(d for d in payload._buffer)
    assert [4, 8] == payload._http_chunk_splits
    assert not payload.is_eof()

    parser.feed_data(b"\r\n")
    assert b"dataline" == b"".join(d for d in payload._buffer)
    assert [4, 8] == payload._http_chunk_splits
    assert payload.is_eof()


def test_parse_chunked_payload_chunk_extension(parser) -> None:
    text = b"GET /test HTTP/1.1\r\n" b"transfer-encoding: chunked\r\n\r\n"
    msg, payload = parser.feed_data(text)[0][0]

    parser.feed_data(b"4;test\r\ndata\r\n4\r\nline\r\n0\r\ntest: test\r\n\r\n")

    assert b"dataline" == b"".join(d for d in payload._buffer)
    assert [4, 8] == payload._http_chunk_splits
    assert payload.is_eof()


def _test_parse_no_length_or_te_on_post(loop, protocol, request_cls):
    parser = request_cls(protocol, loop, readall=True)
    text = b"POST /test HTTP/1.1\r\n\r\n"
    msg, payload = parser.feed_data(text)[0][0]

    assert payload.is_eof()


def test_parse_payload_response_without_body(loop, protocol, response_cls) -> None:
    parser = response_cls(protocol, loop, response_with_body=False)
    text = b"HTTP/1.1 200 Ok\r\n" b"content-length: 10\r\n\r\n"
    msg, payload = parser.feed_data(text)[0][0]

    assert payload.is_eof()


def test_parse_length_payload(response) -> None:
    text = b"HTTP/1.1 200 Ok\r\n" b"content-length: 4\r\n\r\n"
    msg, payload = response.feed_data(text)[0][0]
    assert not payload.is_eof()

    response.feed_data(b"da")
    response.feed_data(b"t")
    response.feed_data(b"aHT")

    assert payload.is_eof()
    assert b"data" == b"".join(d for d in payload._buffer)


def test_parse_no_length_payload(parser) -> None:
    text = b"PUT / HTTP/1.1\r\n\r\n"
    msg, payload = parser.feed_data(text)[0][0]
    assert payload.is_eof()


def test_partial_url(parser) -> None:
    messages, upgrade, tail = parser.feed_data(b"GET /te")
    assert len(messages) == 0
    messages, upgrade, tail = parser.feed_data(b"st HTTP/1.1\r\n\r\n")
    assert len(messages) == 1

    msg, payload = messages[0]

    assert msg.method == "GET"
    assert msg.path == "/test"
    assert msg.version == (1, 1)
    assert payload.is_eof()


def test_url_parse_non_strict_mode(parser) -> None:
    payload = "GET /test/тест HTTP/1.1\r\n\r\n".encode("utf-8")
    messages, upgrade, tail = parser.feed_data(payload)
    assert len(messages) == 1

    msg, payload = messages[0]

    assert msg.method == "GET"
    assert msg.path == "/test/тест"
    assert msg.version == (1, 1)
    assert payload.is_eof()


class TestParsePayload:

    async def test_parse_eof_payload(self, stream) -> None:
        out = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        p = HttpPayloadParser(out, readall=True)
        p.feed_data(b"data")
        p.feed_eof()

        assert out.is_eof()
        assert [(bytearray(b"data"), 4)] == list(out._buffer)

    async def test_parse_no_body(self, stream) -> None:
        out = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        p = HttpPayloadParser(out, method="PUT")

        assert out.is_eof()
        assert p.done

    async def test_parse_length_payload_eof(self, stream) -> None:
        out = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())

        p = HttpPayloadParser(out, length=4)
        p.feed_data(b"da")

        with pytest.raises(http_exceptions.ContentLengthError):
            p.feed_eof()

    async def test_parse_chunked_payload_size_error(self, stream) -> None:
        out = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        p = HttpPayloadParser(out, chunked=True)
        with pytest.raises(http_exceptions.TransferEncodingError):
            p.feed_data(b"blah\r\n")
        assert isinstance(out.exception(), http_exceptions.TransferEncodingError)

    async def test_http_payload_parser_length(self, stream) -> None:
        out = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        p = HttpPayloadParser(out, length=2)
        eof, tail = p.feed_data(b"1245")
        assert eof

        assert b"12" == b"".join(d for d, _ in out._buffer)
        assert b"45" == tail

    _comp = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    _COMPRESSED = b"".join([_comp.compress(b"data"), _comp.flush()])

    async def test_http_payload_parser_deflate(self, stream) -> None:
        length = len(self._COMPRESSED)
        out = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        p = HttpPayloadParser(out, length=length, compression="deflate")
        p.feed_data(self._COMPRESSED)
        assert b"data" == b"".join(d for d, _ in out._buffer)
        assert out.is_eof()

    async def test_http_payload_parser_deflate_no_wbits(self, stream) -> None:
        comp = zlib.compressobj()
        COMPRESSED = b"".join([comp.compress(b"data"), comp.flush()])

        length = len(COMPRESSED)
        out = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        p = HttpPayloadParser(out, length=length, compression="deflate")
        p.feed_data(COMPRESSED)
        assert b"data" == b"".join(d for d, _ in out._buffer)
        assert out.is_eof()

    async def test_http_payload_parser_length_zero(self, stream) -> None:
        out = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        p = HttpPayloadParser(out, length=0)
        assert p.done
        assert out.is_eof()

    @pytest.mark.skipif(brotli is None, reason="brotli is not installed")
    async def test_http_payload_brotli(self, stream) -> None:
        compressed = brotli.compress(b"brotli data")
        out = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        p = HttpPayloadParser(out, length=len(compressed), compression="br")
        p.feed_data(compressed)
        assert b"brotli data" == b"".join(d for d, _ in out._buffer)
        assert out.is_eof()


class TestDeflateBuffer:

    async def test_feed_data(self, stream) -> None:
        buf = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        dbuf = DeflateBuffer(buf, "deflate")

        dbuf.decompressor = mock.Mock()
        dbuf.decompressor.decompress.return_value = b"line"

        dbuf.feed_data(b"data", 4)
        assert [b"line"] == list(d for d, _ in buf._buffer)

    async def test_feed_data_err(self, stream) -> None:
        buf = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        dbuf = DeflateBuffer(buf, "deflate")

        exc = ValueError()
        dbuf.decompressor = mock.Mock()
        dbuf.decompressor.decompress.side_effect = exc

        with pytest.raises(http_exceptions.ContentEncodingError):
            dbuf.feed_data(b"data", 4)

    async def test_feed_eof(self, stream) -> None:
        buf = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        dbuf = DeflateBuffer(buf, "deflate")

        dbuf.decompressor = mock.Mock()
        dbuf.decompressor.flush.return_value = b"line"

        dbuf.feed_eof()
        assert [b"line"] == list(d for d, _ in buf._buffer)
        assert buf._eof

    async def test_feed_eof_err_deflate(self, stream) -> None:
        buf = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        dbuf = DeflateBuffer(buf, "deflate")

        dbuf.decompressor = mock.Mock()
        dbuf.decompressor.flush.return_value = b"line"
        dbuf.decompressor.eof = False

        with pytest.raises(http_exceptions.ContentEncodingError):
            dbuf.feed_eof()

    async def test_feed_eof_no_err_gzip(self, stream) -> None:
        buf = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        dbuf = DeflateBuffer(buf, "gzip")

        dbuf.decompressor = mock.Mock()
        dbuf.decompressor.flush.return_value = b"line"
        dbuf.decompressor.eof = False

        dbuf.feed_eof()
        assert [b"line"] == list(d for d, _ in buf._buffer)

    async def test_feed_eof_no_err_brotli(self, stream) -> None:
        buf = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        dbuf = DeflateBuffer(buf, "br")

        dbuf.decompressor = mock.Mock()
        dbuf.decompressor.flush.return_value = b"line"
        dbuf.decompressor.eof = False

        dbuf.feed_eof()
        assert [b"line"] == list(d for d, _ in buf._buffer)

    async def test_empty_body(self, stream) -> None:
        buf = aiohttp.FlowControlDataQueue(stream, loop=asyncio.get_event_loop())
        dbuf = DeflateBuffer(buf, "deflate")
        dbuf.feed_eof()

        assert buf.at_eof()
