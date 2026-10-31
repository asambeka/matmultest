#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace mg {

namespace {

struct Parser {
    const std::string& t;
    size_t i = 0;

    [[noreturn]] void fail(const char* what) {
        throw std::runtime_error(std::string("json parse error: ") + what + " at offset " +
                                 std::to_string(i));
    }
    void ws() { while (i < t.size() && (t[i] == ' ' || t[i] == '\n' || t[i] == '\t' || t[i] == '\r')) ++i; }
    bool lit(const char* w) {
        size_t n = std::char_traits<char>::length(w);
        if (t.compare(i, n, w) == 0) { i += n; return true; }
        return false;
    }

    std::string string() {
        if (t[i] != '"') fail("expected string");
        ++i;
        std::string out;
        while (i < t.size() && t[i] != '"') {
            char c = t[i++];
            if (c != '\\') { out += c; continue; }
            if (i >= t.size()) fail("bad escape");
            char e = t[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    if (i + 4 > t.size()) fail("bad \\u escape");
                    unsigned cp = (unsigned)std::strtoul(t.substr(i, 4).c_str(), nullptr, 16);
                    i += 4;
                    if (cp < 0x80) out += (char)cp;
                    else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
                    else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
                    break;
                }
                default: fail("bad escape");
            }
        }
        if (i >= t.size()) fail("unterminated string");
        ++i;
        return out;
    }

    Json value() {
        ws();
        if (i >= t.size()) fail("unexpected end");
        char c = t[i];
        if (c == '{') {
            ++i;
            Json j = Json::object();
            ws();
            if (t[i] == '}') { ++i; return j; }
            for (;;) {
                ws();
                std::string k = string();
                ws();
                if (t[i] != ':') fail("expected ':'");
                ++i;
                j.o.emplace_back(std::move(k), value());
                ws();
                if (t[i] == ',') { ++i; continue; }
                if (t[i] == '}') { ++i; return j; }
                fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++i;
            Json j = Json::array();
            ws();
            if (t[i] == ']') { ++i; return j; }
            for (;;) {
                j.a.push_back(value());
                ws();
                if (t[i] == ',') { ++i; continue; }
                if (t[i] == ']') { ++i; return j; }
                fail("expected ',' or ']'");
            }
        }
        if (c == '"') return Json(string());
        if (lit("true")) return Json(true);
        if (lit("false")) return Json(false);
        if (lit("null")) return Json();
        char* endp = nullptr;
        double v = std::strtod(t.c_str() + i, &endp);
        if (endp == t.c_str() + i) fail("bad value");
        i = (size_t)(endp - t.c_str());
        return Json(v);
    }
};

void escape(const std::string& s, std::string& out) {
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", (unsigned)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
}

bool is_flat(const Json& j) {
    if (j.type == Json::Arr) {
        for (auto& v : j.a) if (v.type == Json::Arr || v.type == Json::Obj) return false;
        return true;
    }
    return false;
}

void dump(const Json& j, int indent, int depth, std::string& out) {
    auto nl = [&](int d) {
        if (indent <= 0) return;
        out += '\n';
        out.append((size_t)(indent * d), ' ');
    };
    switch (j.type) {
        case Json::Null: out += "null"; break;
        case Json::Bool: out += j.b ? "true" : "false"; break;
        case Json::Num: {
            char buf[40];
            if (std::isfinite(j.n) && std::fabs(j.n) < 9007199254740992.0 && j.n == std::floor(j.n))
                std::snprintf(buf, sizeof buf, "%lld", (long long)j.n);
            else if (std::isfinite(j.n))
                std::snprintf(buf, sizeof buf, "%.9g", j.n);  // round-trips any float
            else
                std::snprintf(buf, sizeof buf, "null");
            out += buf;
            break;
        }
        case Json::Str: escape(j.s, out); break;
        case Json::Arr: {
            out += '[';
            bool flat = is_flat(j);
            for (size_t k = 0; k < j.a.size(); ++k) {
                if (k) out += flat ? ", " : ",";
                if (!flat) nl(depth + 1);
                dump(j.a[k], indent, depth + 1, out);
            }
            if (!flat && !j.a.empty()) nl(depth);
            out += ']';
            break;
        }
        case Json::Obj: {
            out += '{';
            for (size_t k = 0; k < j.o.size(); ++k) {
                if (k) out += ',';
                nl(depth + 1);
                escape(j.o[k].first, out);
                out += indent > 0 ? ": " : ":";
                dump(j.o[k].second, indent, depth + 1, out);
            }
            if (!j.o.empty()) nl(depth);
            out += '}';
            break;
        }
    }
}

}  // namespace

Json json_parse(const std::string& text) {
    Parser p{text};
    Json j = p.value();
    p.ws();
    if (p.i != text.size()) p.fail("trailing characters");
    return j;
}

std::string json_dump(const Json& j, int indent) {
    std::string out;
    dump(j, indent, 0, out);
    out += '\n';
    return out;
}

}  // namespace mg
