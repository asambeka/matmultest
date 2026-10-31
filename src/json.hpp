// Minimal JSON value, parser and writer. Enough for MatGen's own manifests.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mg {

struct Json {
    enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Json> a;
    std::vector<std::pair<std::string, Json>> o;

    Json() = default;
    Json(bool v) : type(Bool), b(v) {}
    Json(int v) : type(Num), n(v) {}
    Json(int64_t v) : type(Num), n((double)v) {}
    Json(uint32_t v) : type(Num), n(v) {}
    Json(uint64_t v) : type(Num), n((double)v) {}
    Json(double v) : type(Num), n(v) {}
    Json(float v) : type(Num), n(v) {}
    Json(const char* v) : type(Str), s(v) {}
    Json(std::string v) : type(Str), s(std::move(v)) {}

    static Json array() { Json j; j.type = Arr; return j; }
    static Json object() { Json j; j.type = Obj; return j; }

    Json& set(const std::string& k, Json v) {
        for (auto& kv : o) if (kv.first == k) { kv.second = std::move(v); return *this; }
        o.emplace_back(k, std::move(v));
        return *this;
    }
    Json& push(Json v) { a.push_back(std::move(v)); return *this; }

    bool has(const std::string& k) const {
        for (auto& kv : o) if (kv.first == k) return true;
        return false;
    }
    const Json& operator[](const std::string& k) const {
        for (auto& kv : o) if (kv.first == k) return kv.second;
        throw std::runtime_error("json: missing key '" + k + "'");
    }
    const Json& operator[](size_t i) const { return a.at(i); }
    size_t size() const { return type == Arr ? a.size() : o.size(); }

    int64_t i64() const { return (int64_t)n; }
    float f32() const { return (float)n; }
    const std::string& str() const {
        if (type != Str) throw std::runtime_error("json: not a string");
        return s;
    }
};

Json json_parse(const std::string& text);
std::string json_dump(const Json& j, int indent = 2);

}  // namespace mg
