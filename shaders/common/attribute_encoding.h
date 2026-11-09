//==============================================================================
// shaders/common/attribute_encoding.h
// Cross-language attribute packing helpers for normals and tangents.
// C++ and GLSL compile the same formulas so offline compression and shader decoding stay bit-compatible.
//==============================================================================
#ifndef _ATTRIBUTE_ENCODING_H_


#define _ATTRIBUTE_ENCODING_H_


#define ATTRENC_PI           3.14159265358979323846f


#define ATTRENC_NORMAL_BITS  22


#define ATTRENC_TANGENT_BITS 10


#ifdef __cplusplus


namespace shaderio {


#define ATTRENC_INLINE inline


#define ATTRENC_OUT(a) a&


#define ATTRENC_ATAN2F atan2f


#define ATTRENC_FLOOR  glm::floor


#define ATTRENC_CLAMP  glm::clamp


#define ATTRENC_ABS    glm::abs


static_assert(ATTRENC_NORMAL_BITS % 2 == 0, "Normal bits must be an even number");
#else


#define ATTRENC_INLINE


#define ATTRENC_OUT(a) out a


#define ATTRENC_ATAN2F atan


#define ATTRENC_FLOOR  floor


#define ATTRENC_CLAMP  clamp


#define ATTRENC_ABS    abs
#endif


ATTRENC_INLINE vec2 oct_signNotZero(vec2 v) {
    return vec2((v.x >= 0.0f) ? 1.0f : -1.0f, (v.y >= 0.0f) ? 1.0f : -1.0f);
}


ATTRENC_INLINE vec3 oct_to_vec(vec2 e) {

    vec3 v = vec3(e.x, e.y, 1.0f - ATTRENC_ABS(e.x) - ATTRENC_ABS(e.y));

    if (v.z < 0.0f) {

        vec2 os = oct_signNotZero(e);
        v.x = (1.0f - ATTRENC_ABS(e.y)) * os.x;
        v.y = (1.0f - ATTRENC_ABS(e.x)) * os.y;
    }
    return normalize(v);
}


ATTRENC_INLINE vec2 vec_to_oct(vec3 v) {

    vec2 p = vec2(v.x, v.y) * (1.0f / (ATTRENC_ABS(v.x) + ATTRENC_ABS(v.y) + ATTRENC_ABS(v.z)));

    return (v.z <= 0.0f) ? (vec2(1.0f - ATTRENC_ABS(p.y), 1.0f - ATTRENC_ABS(p.x)) * oct_signNotZero(p)) : p;
}


ATTRENC_INLINE vec2 vec_to_oct_precise(vec3 v, int bits) {

    vec2 s = vec_to_oct(v);
    float M = float(1 << (bits - 1)) - 1.0f;

    s = ATTRENC_FLOOR(ATTRENC_CLAMP(s, -1.0f, 1.0f) * M) * (1.0f / M);
    vec2  bestRepresentation = s;

    float highestCosine = dot(oct_to_vec(s), v);

    for (int i = 0; i <= 1; ++i) {
        for (int j = 0; j <= 1; ++j) {
            if (i != 0 || j != 0) {

                vec2  candidate = s + vec2(i, j) * (1.0f / M);
                float cosine = dot(oct_to_vec(candidate), v);

                if (cosine > highestCosine) {
                    bestRepresentation = candidate;
                    highestCosine = cosine;
                }
            }
        }
    }
    return bestRepresentation;
}


ATTRENC_INLINE uint32_t normal_pack(vec3 normal) {
    const int      halfBits = ATTRENC_NORMAL_BITS / 2;
    const uint32_t mask = (1 << halfBits) - 1;


    vec2 v = vec_to_oct_precise(normal, halfBits);

    v = (v + 1.0f) * 0.5f * float(mask) + 0.5f;

    return (uint32_t(v.x) & mask) | ((uint32_t(v.y) & mask) << halfBits);
}


ATTRENC_INLINE vec3 normal_unpack(uint32_t packed) {
    const int      halfBits = ATTRENC_NORMAL_BITS / 2;
    const uint32_t mask = (1 << halfBits) - 1;


    uvec2 pv = uvec2(packed, (packed >> halfBits)) & uvec2(mask);

    vec2  v = (vec2(pv) / float(mask)) * 2.0f - 1.0f;

    return oct_to_vec(v);
}


ATTRENC_INLINE void tangent_orthonormalBasis(vec3 normal, ATTRENC_OUT(vec3) tangent, ATTRENC_OUT(vec3) bitangent) {

    if (normal.z < -0.99998796f) {

        tangent = vec3(0.0f, -1.0f, 0.0f);

        bitangent = vec3(-1.0f, 0.0f, 0.0f);
        return;
    }

    float a = 1.0f / (1.0f + normal.z);
    float b = -normal.x * normal.y * a;

    tangent = vec3(1.0f - normal.x * normal.x * a, b, -normal.x);

    bitangent = vec3(b, 1.0f - normal.y * normal.y * a, -normal.y);
}


ATTRENC_INLINE uint32_t tangent_pack(vec3 normal, vec4 tangent) {

    const uint32_t mask = (1 << (ATTRENC_TANGENT_BITS - 1)) - 1;
    vec3 autoTangent, autoBitangent;


    tangent_orthonormalBasis(normal, autoTangent, autoBitangent);


    float angle = ATTRENC_ATAN2F(dot(autoTangent, vec3(tangent)), dot(autoBitangent, vec3(tangent))) / ATTRENC_PI;

    float angleUnorm = ATTRENC_CLAMP((angle + 1.0f) * 0.5f, 0.0f, 1.0f);

    uint32_t angleBits = uint32_t(angleUnorm * float(mask) + 0.5f);

    return (angleBits << 1) | (tangent.w > 0.0f ? 1u : 0u);
}


ATTRENC_INLINE vec4 tangent_unpack(vec3 normal, uint32_t encoded) {
    const uint32_t mask = (1 << (ATTRENC_TANGENT_BITS - 1)) - 1;

    uint32_t signBit = encoded & 1;


    float    angleUnorm = float((encoded >> 1) & mask) / float(mask);

    float    angle = (angleUnorm * 2.0f - 1.0f) * ATTRENC_PI;

    vec3 autoTangent, autoBitangent;

    tangent_orthonormalBasis(normal, autoTangent, autoBitangent);

    vec3 tangent = cos(angle) * autoBitangent + sin(angle) * autoTangent;

    return vec4(tangent, (signBit == 1) ? 1.0f : -1.0f);
}


#undef ATTRENC_ABS
#undef ATTRENC_FLOOR
#undef ATTRENC_CLAMP
#undef ATTRENC_INLINE
#undef ATTRENC_ATAN2F
#undef ATTRENC_PI
#undef ATTRENC_OUT
#ifdef __cplusplus
}
#endif
#endif
