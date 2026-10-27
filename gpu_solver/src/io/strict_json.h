#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace gpu_femm
{

struct StrictJson {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };
  Type type = Type::kNull;
  bool boolean = false;
  double number = 0.0;
  std::string number_text;
  std::string string;
  std::vector<StrictJson> array;
  std::map<std::string, StrictJson> object;
};

class StrictJsonParser
{
public:
  explicit StrictJsonParser(const std::string &text);
  bool Parse(StrictJson *value, std::string *error);

private:
  void Skip();
  bool Value(StrictJson *value, std::string *error);
  bool String(std::string *output, std::string *error);
  bool Number(StrictJson *value, std::string *error);
  bool Array(StrictJson *value, std::string *error);
  bool Object(StrictJson *value, std::string *error);

  const std::string &text_;
  size_t position_ = 0;
};

const StrictJson *JsonMember(const StrictJson &object, const char *name, StrictJson::Type type,
                             std::string *error);
bool ExactObject(const StrictJson &value, std::initializer_list<const char *> names,
                 std::string *error);
bool JsonInteger(const StrictJson &value, int32_t *result);
bool JsonSha256(const std::string &value);
bool JsonLowerSha256(const std::string &value);
std::string Sha256Hex(const std::string &bytes);
std::string CanonicalJsonSha256(const StrictJson &value);

} // namespace gpu_femm
