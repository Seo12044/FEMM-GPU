#include "io/strict_json.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace gpu_femm
{

StrictJsonParser::StrictJsonParser(const std::string &text) : text_(text) {}

bool StrictJsonParser::Parse(StrictJson *value, std::string *error)
{
  if (value == nullptr || error == nullptr)
    return false;
  Skip();
  if (!Value(value, error))
    return false;
  Skip();
  if (position_ != text_.size()) {
    *error = "trailing JSON data";
    return false;
  }
  return true;
}

void StrictJsonParser::Skip()
{
  while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_])))
    ++position_;
}

bool StrictJsonParser::Value(StrictJson *value, std::string *error)
{
  if (position_ >= text_.size()) {
    *error = "unexpected JSON end";
    return false;
  }
  const char token = text_[position_];
  if (token == '{')
    return Object(value, error);
  if (token == '[')
    return Array(value, error);
  if (token == '"') {
    value->type = StrictJson::Type::kString;
    return String(&value->string, error);
  }
  if (text_.compare(position_, 4, "true") == 0) {
    position_ += 4;
    value->type = StrictJson::Type::kBool;
    value->boolean = true;
    return true;
  }
  if (text_.compare(position_, 5, "false") == 0) {
    position_ += 5;
    value->type = StrictJson::Type::kBool;
    value->boolean = false;
    return true;
  }
  if (text_.compare(position_, 4, "null") == 0) {
    position_ += 4;
    value->type = StrictJson::Type::kNull;
    return true;
  }
  return Number(value, error);
}

bool StrictJsonParser::String(std::string *output, std::string *error)
{
  if (position_ >= text_.size() || text_[position_++] != '"')
    return false;
  output->clear();
  while (position_ < text_.size()) {
    const unsigned char c = static_cast<unsigned char>(text_[position_++]);
    if (c == '"')
      return true;
    if (c < 0x20) {
      *error = "control character in JSON string";
      return false;
    }
    if (c != '\\') {
      output->push_back(static_cast<char>(c));
      continue;
    }
    if (position_ >= text_.size()) {
      *error = "truncated JSON escape";
      return false;
    }
    const char escaped = text_[position_++];
    switch (escaped) {
    case '"':
      output->push_back('"');
      break;
    case '\\':
      output->push_back('\\');
      break;
    case '/':
      output->push_back('/');
      break;
    case 'b':
      output->push_back('\b');
      break;
    case 'f':
      output->push_back('\f');
      break;
    case 'n':
      output->push_back('\n');
      break;
    case 'r':
      output->push_back('\r');
      break;
    case 't':
      output->push_back('\t');
      break;
    default:
      *error = "unsupported JSON string escape";
      return false;
    }
  }
  *error = "unterminated JSON string";
  return false;
}

bool StrictJsonParser::Number(StrictJson *value, std::string *error)
{
  const size_t begin = position_;
  if (position_ < text_.size() && text_[position_] == '-')
    ++position_;
  if (position_ >= text_.size()) {
    *error = "invalid JSON number";
    return false;
  }
  if (text_[position_] == '0')
    ++position_;
  else if (text_[position_] >= '1' && text_[position_] <= '9')
    while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_])))
      ++position_;
  else {
    *error = "invalid JSON number";
    return false;
  }
  if (position_ < text_.size() && text_[position_] == '.') {
    ++position_;
    const size_t fraction = position_;
    while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_])))
      ++position_;
    if (fraction == position_) {
      *error = "invalid JSON fraction";
      return false;
    }
  }
  if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
    ++position_;
    if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-'))
      ++position_;
    const size_t exponent = position_;
    while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_])))
      ++position_;
    if (exponent == position_) {
      *error = "invalid JSON exponent";
      return false;
    }
  }
  value->number_text = text_.substr(begin, position_ - begin);
  try {
    value->number = std::stod(value->number_text);
  } catch (...) {
    *error = "JSON number conversion failed";
    return false;
  }
  if (!std::isfinite(value->number)) {
    *error = "nonfinite JSON number";
    return false;
  }
  value->type = StrictJson::Type::kNumber;
  return true;
}

bool StrictJsonParser::Array(StrictJson *value, std::string *error)
{
  ++position_;
  value->type = StrictJson::Type::kArray;
  value->array.clear();
  Skip();
  if (position_ < text_.size() && text_[position_] == ']') {
    ++position_;
    return true;
  }
  while (true) {
    StrictJson child;
    if (!Value(&child, error))
      return false;
    value->array.push_back(std::move(child));
    Skip();
    if (position_ >= text_.size()) {
      *error = "unterminated JSON array";
      return false;
    }
    if (text_[position_] == ']') {
      ++position_;
      return true;
    }
    if (text_[position_++] != ',') {
      *error = "JSON array comma expected";
      return false;
    }
    Skip();
  }
}

bool StrictJsonParser::Object(StrictJson *value, std::string *error)
{
  ++position_;
  value->type = StrictJson::Type::kObject;
  value->object.clear();
  Skip();
  if (position_ < text_.size() && text_[position_] == '}') {
    ++position_;
    return true;
  }
  while (true) {
    std::string key;
    if (!String(&key, error)) {
      *error = "JSON object key expected";
      return false;
    }
    Skip();
    if (position_ >= text_.size() || text_[position_++] != ':') {
      *error = "JSON colon expected";
      return false;
    }
    Skip();
    StrictJson child;
    if (!Value(&child, error))
      return false;
    if (!value->object.emplace(std::move(key), std::move(child)).second) {
      *error = "duplicate JSON object key";
      return false;
    }
    Skip();
    if (position_ >= text_.size()) {
      *error = "unterminated JSON object";
      return false;
    }
    if (text_[position_] == '}') {
      ++position_;
      return true;
    }
    if (text_[position_++] != ',') {
      *error = "JSON object comma expected";
      return false;
    }
    Skip();
  }
}

const StrictJson *JsonMember(const StrictJson &object, const char *name, StrictJson::Type type,
                             std::string *error)
{
  if (object.type != StrictJson::Type::kObject) {
    *error = "expected JSON object";
    return nullptr;
  }
  const auto found = object.object.find(name);
  if (found == object.object.end()) {
    *error = std::string("missing JSON field: ") + name;
    return nullptr;
  }
  if (found->second.type != type) {
    *error = std::string("wrong JSON type: ") + name;
    return nullptr;
  }
  return &found->second;
}

bool ExactObject(const StrictJson &value, std::initializer_list<const char *> names,
                 std::string *error)
{
  if (value.type != StrictJson::Type::kObject || value.object.size() != names.size()) {
    *error = "unexpected JSON object fields";
    return false;
  }
  for (const char *name : names)
    if (value.object.find(name) == value.object.end()) {
      *error = std::string("missing JSON field: ") + name;
      return false;
    }
  return true;
}

bool JsonInteger(const StrictJson &value, int32_t *result)
{
  if (value.type != StrictJson::Type::kNumber || value.number != std::floor(value.number) ||
      value.number < static_cast<double>(std::numeric_limits<int32_t>::min()) ||
      value.number > static_cast<double>(std::numeric_limits<int32_t>::max()))
    return false;
  *result = static_cast<int32_t>(value.number);
  return true;
}

bool JsonSha256(const std::string &value)
{
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isxdigit(c); });
}

bool JsonLowerSha256(const std::string &value)
{
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return std::isdigit(c) || (c >= 'a' && c <= 'f');
         });
}

// Small dependency-free SHA-256 implementation.  Artifact fingerprints bind
// the exact bytes on disk, not a parsed/re-serialized JSON representation.
namespace
{

class Sha256
{
public:
  Sha256() { Reset(); }
  void Update(const unsigned char *data, size_t size)
  {
    total_bits_ += static_cast<uint64_t>(size) * 8;
    while (size > 0) {
      const size_t count = std::min(size, static_cast<size_t>(sizeof(buffer_) - buffered_));
      std::memcpy(buffer_ + buffered_, data, count);
      buffered_ += count;
      data += count;
      size -= count;
      if (buffered_ == sizeof(buffer_)) {
        Transform(buffer_);
        buffered_ = 0;
      }
    }
  }
  std::string FinalHex()
  {
    const uint64_t bits = total_bits_;
    const unsigned char one = 0x80;
    Update(&one, 1);
    const unsigned char zero = 0;
    while (buffered_ != 56)
      Update(&zero, 1);
    unsigned char length[8];
    for (int i = 0; i < 8; ++i)
      length[7 - i] = static_cast<unsigned char>(bits >> (8 * i));
    Update(length, sizeof(length));
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (uint32_t word : state_)
      for (int shift = 24; shift >= 0; shift -= 8)
        output << std::setw(2) << ((word >> shift) & 0xffU);
    return output.str();
  }

private:
  static uint32_t RotateRight(uint32_t value, int shift)
  {
    return (value >> shift) | (value << (32 - shift));
  }
  void Reset()
  {
    state_[0] = 0x6a09e667U;
    state_[1] = 0xbb67ae85U;
    state_[2] = 0x3c6ef372U;
    state_[3] = 0xa54ff53aU;
    state_[4] = 0x510e527fU;
    state_[5] = 0x9b05688cU;
    state_[6] = 0x1f83d9abU;
    state_[7] = 0x5be0cd19U;
    total_bits_ = 0;
    buffered_ = 0;
  }
  void Transform(const unsigned char block[64])
  {
    static const uint32_t k[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
        0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
        0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
        0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
        0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
        0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
        0xc67178f2U};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (static_cast<uint32_t>(block[4 * i]) << 24) |
             (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
             (static_cast<uint32_t>(block[4 * i + 2]) << 8) | block[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = RotateRight(w[i - 15], 7) ^ RotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = RotateRight(w[i - 2], 17) ^ RotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3], e = state_[4],
             f = state_[5], g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
      const uint32_t ch = (e & f) ^ ((~e) & g);
      const uint32_t t1 = h + s1 + ch + k[i] + w[i];
      const uint32_t s0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + s0 + maj;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }
  uint32_t state_[8] = {};
  unsigned char buffer_[64] = {};
  size_t buffered_ = 0;
  uint64_t total_bits_ = 0;
};

} // namespace

std::string Sha256Hex(const std::string &bytes)
{
  Sha256 sha;
  sha.Update(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size());
  return sha.FinalHex();
}

namespace
{

void AppendCanonicalJson(const StrictJson &value, std::string *output)
{
  switch (value.type) {
  case StrictJson::Type::kNull:
    output->append("null");
    return;
  case StrictJson::Type::kBool:
    output->append(value.boolean ? "true" : "false");
    return;
  case StrictJson::Type::kNumber:
    output->append(value.number_text);
    return;
  case StrictJson::Type::kString: {
    static const char *hex = "0123456789abcdef";
    output->push_back('"');
    for (unsigned char character : value.string) {
      switch (character) {
      case '"':
        output->append("\\\"");
        break;
      case '\\':
        output->append("\\\\");
        break;
      case '\b':
        output->append("\\b");
        break;
      case '\f':
        output->append("\\f");
        break;
      case '\n':
        output->append("\\n");
        break;
      case '\r':
        output->append("\\r");
        break;
      case '\t':
        output->append("\\t");
        break;
      default:
        if (character < 0x20) {
          output->append("\\u00");
          output->push_back(hex[character >> 4]);
          output->push_back(hex[character & 0x0f]);
        } else {
          output->push_back(static_cast<char>(character));
        }
      }
    }
    output->push_back('"');
    return;
  }
  case StrictJson::Type::kArray:
    output->push_back('[');
    for (size_t index = 0; index < value.array.size(); ++index) {
      if (index != 0)
        output->push_back(',');
      AppendCanonicalJson(value.array[index], output);
    }
    output->push_back(']');
    return;
  case StrictJson::Type::kObject:
    output->push_back('{');
    for (auto iterator = value.object.begin(); iterator != value.object.end(); ++iterator) {
      if (iterator != value.object.begin())
        output->push_back(',');
      StrictJson key;
      key.type = StrictJson::Type::kString;
      key.string = iterator->first;
      AppendCanonicalJson(key, output);
      output->push_back(':');
      AppendCanonicalJson(iterator->second, output);
    }
    output->push_back('}');
    return;
  }
}

} // namespace

std::string CanonicalJsonSha256(const StrictJson &value)
{
  std::string canonical;
  AppendCanonicalJson(value, &canonical);
  return Sha256Hex(canonical);
}

} // namespace gpu_femm
