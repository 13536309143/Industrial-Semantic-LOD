#include <zstd.h>
#include <cstddef>
extern "C" __declspec(dllexport) size_t decode_frame(void* destination,size_t capacity,const void* source,size_t size)
{
  return ZSTD_decompress(destination,capacity,source,size);
}
