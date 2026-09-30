# A tiny JSON parser written in Mojo, exposed to Python as an extension module.
#
# Import it from Python with:
#
#     import mojo.importer   # teaches Python how to import .mojo files
#     import json_parser     # compiles this file on first import (cached after)
#
# Exposed functions:
#   parse(text)    -> Python object (dict / list / str / int / float / bool / None)
#   validate(text) -> dict of stats; walks the document in pure Mojo, no Python objects
#   tokenize(text) -> list of (kind, start, end) tuples, handy for syntax highlighting

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_json_parser() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("json_parser")
        m.def_function[parse](
            "parse", docstring="Parse a JSON string into Python objects."
        )
        m.def_function[validate](
            "validate",
            docstring="Validate a JSON string and return structural stats.",
        )
        m.def_function[tokenize](
            "tokenize", docstring="Split a JSON string into (kind, start, end)."
        )
        return m.finalize()
    except e:
        abort(String("failed to create json_parser module: ", e))


# ---------------------------------------------------------------------------
# Byte constants
# ---------------------------------------------------------------------------

comptime LBRACE: UInt8 = UInt8(ord("{"))
comptime RBRACE: UInt8 = UInt8(ord("}"))
comptime LBRACKET: UInt8 = UInt8(ord("["))
comptime RBRACKET: UInt8 = UInt8(ord("]"))
comptime COLON: UInt8 = UInt8(ord(":"))
comptime COMMA: UInt8 = UInt8(ord(","))
comptime QUOTE: UInt8 = UInt8(ord('"'))
comptime BACKSLASH: UInt8 = UInt8(ord("\\"))
comptime MINUS: UInt8 = UInt8(ord("-"))
comptime PLUS: UInt8 = UInt8(ord("+"))
comptime DOT: UInt8 = UInt8(ord("."))
comptime ZERO: UInt8 = UInt8(ord("0"))
comptime NINE: UInt8 = UInt8(ord("9"))
comptime LOWER_E: UInt8 = UInt8(ord("e"))
comptime UPPER_E: UInt8 = UInt8(ord("E"))
comptime SPACE: UInt8 = UInt8(ord(" "))
comptime TAB: UInt8 = UInt8(ord("\t"))
comptime NEWLINE: UInt8 = UInt8(ord("\n"))
comptime CR: UInt8 = UInt8(ord("\r"))

comptime MAX_DEPTH = 512


def is_digit(c: UInt8) -> Bool:
    return c >= ZERO and c <= NINE


def hex_value(c: UInt8) -> Int:
    if c >= ZERO and c <= NINE:
        return Int(c - ZERO)
    if c >= UInt8(ord("a")) and c <= UInt8(ord("f")):
        return Int(c - UInt8(ord("a"))) + 10
    if c >= UInt8(ord("A")) and c <= UInt8(ord("F")):
        return Int(c - UInt8(ord("A"))) + 10
    return -1


def append_utf8(mut out: List[UInt8], cp: Int):
    """Encode a Unicode code point as UTF-8."""
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


struct Stats:
    var objects: Int
    var arrays: Int
    var strings: Int
    var numbers: Int
    var literals: Int
    var max_depth: Int

    def __init__(out self):
        self.objects = 0
        self.arrays = 0
        self.strings = 0
        self.numbers = 0
        self.literals = 0
        self.max_depth = 0


struct Parser:
    var data: List[UInt8]
    var pos: Int
    var depth: Int
    var stats: Stats

    def __init__(out self, text: String):
        self.data = List[UInt8](text.as_bytes())
        self.pos = 0
        self.depth = 0
        self.stats = Stats()

    # -- helpers ------------------------------------------------------------

    def error(self, msg: String) -> Error:
        """Build an error that tells you exactly where things went wrong."""
        var line = 1
        var col = 1
        for i in range(min(self.pos, len(self.data))):
            if self.data.unsafe_get(i) == NEWLINE:
                line += 1
                col = 1
            else:
                col += 1
        return Error(
            String(
                msg, " at line ", line, ", column ", col, " (offset ", self.pos, ")"
            )
        )

    def at_end(self) -> Bool:
        return self.pos >= len(self.data)

    def peek(self) -> UInt8:
        return self.data.unsafe_get(self.pos)

    def skip_whitespace(mut self):
        while self.pos < len(self.data):
            var c = self.data.unsafe_get(self.pos)
            if c == SPACE or c == NEWLINE or c == TAB or c == CR:
                self.pos += 1
            else:
                return

    def expect(mut self, c: UInt8) raises:
        if self.at_end() or self.peek() != c:
            raise self.error(String("expected '", chr(Int(c)), "'"))
        self.pos += 1

    def expect_word(mut self, word: StaticString) raises:
        var bytes = word.as_bytes()
        for i in range(len(bytes)):
            if self.pos + i >= len(self.data) or self.data.unsafe_get(self.pos + i) != bytes[i]:
                raise self.error(String("invalid literal, expected '", word, "'"))
        self.pos += len(bytes)
        self.stats.literals += 1

    def enter(mut self) raises:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise self.error("nesting too deep")
        if self.depth > self.stats.max_depth:
            self.stats.max_depth = self.depth

    def finish(mut self) raises:
        self.skip_whitespace()
        if not self.at_end():
            raise self.error("unexpected trailing characters")

    # -- scanning (shared by parse and validate) ----------------------------

    def scan_string(mut self, mut out: List[UInt8], decode: Bool) raises:
        """Consume a string literal. When `decode`, append its UTF-8 content."""
        self.expect(QUOTE)
        self.stats.strings += 1
        while True:
            if self.at_end():
                raise self.error("unterminated string")
            var c = self.peek()
            if c == QUOTE:
                self.pos += 1
                return
            if c < 0x20:
                raise self.error("control character in string")
            if c != BACKSLASH:
                if decode:
                    out.append(c)
                self.pos += 1
                continue

            # Escape sequence
            self.pos += 1
            if self.at_end():
                raise self.error("unterminated escape")
            var e = self.peek()
            self.pos += 1
            var decoded: Int
            if e == QUOTE:
                decoded = 0x22
            elif e == BACKSLASH:
                decoded = 0x5C
            elif e == UInt8(ord("/")):
                decoded = 0x2F
            elif e == UInt8(ord("b")):
                decoded = 0x08
            elif e == UInt8(ord("f")):
                decoded = 0x0C
            elif e == UInt8(ord("n")):
                decoded = 0x0A
            elif e == UInt8(ord("r")):
                decoded = 0x0D
            elif e == UInt8(ord("t")):
                decoded = 0x09
            elif e == UInt8(ord("u")):
                decoded = self.scan_hex4()
                # Surrogate pair -> one astral code point (hello, emoji!)
                if decoded >= 0xD800 and decoded <= 0xDBFF:
                    if (
                        self.pos + 1 < len(self.data)
                        and self.data.unsafe_get(self.pos) == BACKSLASH
                        and self.data.unsafe_get(self.pos + 1) == UInt8(ord("u"))
                    ):
                        self.pos += 2
                        var low = self.scan_hex4()
                        if low < 0xDC00 or low > 0xDFFF:
                            raise self.error("invalid low surrogate")
                        decoded = 0x10000 + ((decoded - 0xD800) << 10) + (
                            low - 0xDC00
                        )
                    else:
                        raise self.error("lone high surrogate")
            else:
                self.pos -= 1
                raise self.error("invalid escape")
            if decode:
                append_utf8(out, decoded)

    def scan_hex4(mut self) raises -> Int:
        if self.pos + 4 > len(self.data):
            raise self.error("truncated \\u escape")
        var value = 0
        for _ in range(4):
            var h = hex_value(self.peek())
            if h < 0:
                raise self.error("invalid hex digit in \\u escape")
            value = value * 16 + h
            self.pos += 1
        return value

    def scan_number(mut self) raises -> Bool:
        """Consume a number literal. Returns True if it is an integer."""
        self.stats.numbers += 1
        var is_int = True
        if not self.at_end() and self.peek() == MINUS:
            self.pos += 1
        if self.at_end() or not is_digit(self.peek()):
            raise self.error("invalid number")
        if self.peek() == ZERO:
            self.pos += 1
            if not self.at_end() and is_digit(self.peek()):
                raise self.error("leading zeros are not allowed")
        else:
            while not self.at_end() and is_digit(self.peek()):
                self.pos += 1
        if not self.at_end() and self.peek() == DOT:
            is_int = False
            self.pos += 1
            if self.at_end() or not is_digit(self.peek()):
                raise self.error("expected digit after '.'")
            while not self.at_end() and is_digit(self.peek()):
                self.pos += 1
        if not self.at_end() and (self.peek() == LOWER_E or self.peek() == UPPER_E):
            is_int = False
            self.pos += 1
            if not self.at_end() and (self.peek() == PLUS or self.peek() == MINUS):
                self.pos += 1
            if self.at_end() or not is_digit(self.peek()):
                raise self.error("expected digit in exponent")
            while not self.at_end() and is_digit(self.peek()):
                self.pos += 1
        return is_int

    def slice_string(self, start: Int, end: Int) -> String:
        return String(StringSlice(unsafe_from_utf8=Span(self.data)[start:end]))

    def parse_string(mut self) raises -> String:
        # Fast path: no escapes means we can copy the bytes in one go.
        var i = self.pos + 1
        while i < len(self.data):
            var c = self.data.unsafe_get(i)
            if c == QUOTE:
                var s = self.slice_string(self.pos + 1, i)
                self.pos = i + 1
                self.stats.strings += 1
                return s^
            if c == BACKSLASH or c < 0x20:
                break
            i += 1
        # Slow path: decode escapes byte by byte.
        var buf = List[UInt8]()
        self.scan_string(buf, True)
        return String(unsafe_from_utf8=buf)

    # -- parse: build Python objects ----------------------------------------

    def parse_value(mut self) raises -> PythonObject:
        self.skip_whitespace()
        if self.at_end():
            raise self.error("unexpected end of input")
        var c = self.peek()
        if c == LBRACE:
            return self.parse_object()
        if c == LBRACKET:
            return self.parse_array()
        if c == QUOTE:
            return PythonObject(self.parse_string())
        if c == UInt8(ord("t")):
            self.expect_word("true")
            return PythonObject(True)
        if c == UInt8(ord("f")):
            self.expect_word("false")
            return PythonObject(False)
        if c == UInt8(ord("n")):
            self.expect_word("null")
            return Python.none()
        if c == MINUS or is_digit(c):
            return self.parse_number()
        raise self.error(String("unexpected character '", chr(Int(c)), "'"))

    def parse_number(mut self) raises -> PythonObject:
        var start = self.pos
        var is_int = self.scan_number()
        if is_int and self.pos - start <= 18:
            # Fast path: anything this short fits in 64 bits.
            var negative = self.data.unsafe_get(start) == MINUS
            var value = 0
            for i in range(start + 1 if negative else start, self.pos):
                value = value * 10 + Int(self.data.unsafe_get(i) - ZERO)
            return PythonObject(-value if negative else value)
        var text = self.slice_string(start, self.pos)
        if is_int:
            # Python's arbitrary precision ints for the big ones.
            return Python.int(PythonObject(text))
        return PythonObject(atof(text))

    def parse_array(mut self) raises -> PythonObject:
        self.expect(LBRACKET)
        self.enter()
        self.stats.arrays += 1
        var result = Python.list()
        self.skip_whitespace()
        if not self.at_end() and self.peek() == RBRACKET:
            self.pos += 1
            self.depth -= 1
            return result
        while True:
            result.append(self.parse_value())
            self.skip_whitespace()
            if self.at_end():
                raise self.error("unterminated array")
            if self.peek() == COMMA:
                self.pos += 1
                continue
            self.expect(RBRACKET)
            self.depth -= 1
            return result

    def parse_object(mut self) raises -> PythonObject:
        self.expect(LBRACE)
        self.enter()
        self.stats.objects += 1
        var result = Python.dict()
        self.skip_whitespace()
        if not self.at_end() and self.peek() == RBRACE:
            self.pos += 1
            self.depth -= 1
            return result
        while True:
            self.skip_whitespace()
            if self.at_end() or self.peek() != QUOTE:
                raise self.error("expected string key")
            var key = PythonObject(self.parse_string())
            self.skip_whitespace()
            self.expect(COLON)
            result[key] = self.parse_value()
            self.skip_whitespace()
            if self.at_end():
                raise self.error("unterminated object")
            if self.peek() == COMMA:
                self.pos += 1
                continue
            self.expect(RBRACE)
            self.depth -= 1
            return result

    # -- validate: walk the document without allocating Python objects ------

    def skip_value(mut self) raises:
        self.skip_whitespace()
        if self.at_end():
            raise self.error("unexpected end of input")
        var c = self.peek()
        var scratch = List[UInt8]()
        if c == LBRACE or c == LBRACKET:
            var close = RBRACE if c == LBRACE else RBRACKET
            self.pos += 1
            self.enter()
            if c == LBRACE:
                self.stats.objects += 1
            else:
                self.stats.arrays += 1
            self.skip_whitespace()
            if not self.at_end() and self.peek() == close:
                self.pos += 1
                self.depth -= 1
                return
            while True:
                if c == LBRACE:
                    self.skip_whitespace()
                    if self.at_end() or self.peek() != QUOTE:
                        raise self.error("expected string key")
                    self.scan_string(scratch, False)
                    self.skip_whitespace()
                    self.expect(COLON)
                self.skip_value()
                self.skip_whitespace()
                if self.at_end():
                    raise self.error("unterminated container")
                if self.peek() == COMMA:
                    self.pos += 1
                    continue
                self.expect(close)
                self.depth -= 1
                return
        elif c == QUOTE:
            self.scan_string(scratch, False)
        elif c == UInt8(ord("t")):
            self.expect_word("true")
        elif c == UInt8(ord("f")):
            self.expect_word("false")
        elif c == UInt8(ord("n")):
            self.expect_word("null")
        elif c == MINUS or is_digit(c):
            _ = self.scan_number()
        else:
            raise self.error(String("unexpected character '", chr(Int(c)), "'"))


# ---------------------------------------------------------------------------
# Python-facing functions
# ---------------------------------------------------------------------------


def parse(text: PythonObject) raises -> PythonObject:
    var p = Parser(String(text))
    var value = p.parse_value()
    p.finish()
    return value


def validate(text: PythonObject) raises -> PythonObject:
    var p = Parser(String(text))
    p.skip_value()
    p.finish()
    var d = Python.dict()
    d["bytes"] = len(p.data)
    d["objects"] = p.stats.objects
    d["arrays"] = p.stats.arrays
    d["strings"] = p.stats.strings
    d["numbers"] = p.stats.numbers
    d["literals"] = p.stats.literals
    d["max_depth"] = p.stats.max_depth
    return d


def tokenize(text: PythonObject) raises -> PythonObject:
    """Lexer only: returns [(kind, start, end), ...] with byte offsets."""
    var p = Parser(String(text))
    var tokens = Python.list()
    var scratch = List[UInt8]()
    while True:
        p.skip_whitespace()
        if p.at_end():
            return tokens
        var start = p.pos
        var c = p.peek()
        var kind: String
        if (
            c == LBRACE
            or c == RBRACE
            or c == LBRACKET
            or c == RBRACKET
            or c == COLON
            or c == COMMA
        ):
            p.pos += 1
            kind = "punct"
        elif c == QUOTE:
            p.scan_string(scratch, False)
            # A string followed by ':' is an object key
            var save = p.pos
            p.skip_whitespace()
            kind = "key" if (not p.at_end() and p.peek() == COLON) else "string"
            p.pos = save
        elif c == MINUS or is_digit(c):
            _ = p.scan_number()
            kind = "number"
        elif c == UInt8(ord("t")):
            p.expect_word("true")
            kind = "bool"
        elif c == UInt8(ord("f")):
            p.expect_word("false")
            kind = "bool"
        elif c == UInt8(ord("n")):
            p.expect_word("null")
            kind = "null"
        else:
            raise p.error(String("unexpected character '", chr(Int(c)), "'"))
        tokens.append(Python.tuple(kind, start, p.pos))
