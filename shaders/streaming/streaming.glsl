//==============================================================================
// shaders/streaming/streaming.glsl
// Shared GPU-side streaming helpers.
// Age filtering and request logic update resident groups without requiring the render path to understand CPU task queues.
//==============================================================================
void streamingAgeFilter(uint residentID, uint geometryID, Group_in groupRef)
{
#if STREAMING_DEBUG_ADDRESSES
  if (uint64_t(groupRef) >= STREAMING_INVALID_ADDRESS_START)
  {
    streamingRW.request.errorAgeFilter = residentID;
    return;
  }
#endif


  uint age = streaming.resident.groups.d[residentID].age;

  if (age < 0xFFFF)
  {
    age++;

    streaming.resident.groups.d[residentID].age = uint16_t(age);
  }


  if (age > streaming.ageThreshold)
  {

    uint unloadOffset = atomicAdd(streamingRW.request.unloadCounter, 1);
    if (unloadOffset < streaming.request.maxUnloads) {

      streaming.request.unloadGeometryGroups.d[unloadOffset] = uvec2(geometryID, streaming.resident.groupIDs.d[residentID]);
    }
  }
}
