#include "io/json_output.h"

#include <sstream>

namespace gpu_femm
{

std::string EscapeJsonString(const std::string &value)
{
  std::ostringstream escaped;
  const char *hex = "0123456789abcdef";
  for (unsigned char character : value) {
    switch (character) {
    case '"':
      escaped << "\\\"";
      break;
    case '\\':
      escaped << "\\\\";
      break;
    case '\b':
      escaped << "\\b";
      break;
    case '\f':
      escaped << "\\f";
      break;
    case '\n':
      escaped << "\\n";
      break;
    case '\r':
      escaped << "\\r";
      break;
    case '\t':
      escaped << "\\t";
      break;
    default:
      if (character < 0x20)
        escaped << "\\u00" << hex[character >> 4] << hex[character & 0x0f];
      else
        escaped << static_cast<char>(character);
    }
  }
  return escaped.str();
}

} // namespace gpu_femm
