//==============================================================================
// shaders/common/culling_hiz.inc
// Hi-Z occlusion helpers for conservative depth-pyramid tests.
// The mip selection favors stable rejection while avoiding over-aggressive culling near the camera.
//==============================================================================
#ifndef CULLING_NO_HIZ


int fastMipmapLevel(float maxsize)
{

    if (maxsize <= 1.0) return 0;


    int miplevel = 0;
    float val = maxsize;


    while (val > 1.0) {
        val *= 0.5;
        miplevel++;
    }

    return miplevel;
}


bool intersectHiz(vec4 clipMin, vec4 clipMax, uint idx)
{
#if defined(USE_HIZ_OCCLUSION) && !USE_HIZ_OCCLUSION
    return true;
#endif
#ifdef ADAPTIVE_TEMPORAL_GUARD
    // Moving single-pass frames use current frusta. A previous-frame Hi-Z is
    // not a conservative occlusion oracle there: keep the candidate instead.
    if(build.adaptiveTemporalBypass != 0u) return true;
#endif

    if (clipMax.w <= 0.0) return false;


    vec2 minUV = clipMin.xy * 0.5 + 0.5;
    vec2 maxUV = clipMax.xy * 0.5 + 0.5;


    minUV *= view.hizSizeFactors.xy;
    maxUV *= view.hizSizeFactors.xy;


    minUV = min(minUV, view.hizSizeFactors.zw);

    maxUV = min(maxUV, view.hizSizeFactors.zw);


    vec2 size = maxUV - minUV;


    if (size.x <= 0.0 || size.y <= 0.0) return false;


    float maxsize = max(size.x, size.y) * view.hizSizeMax;


    if (maxsize < 0.5) return true;


    int miplevel = fastMipmapLevel(maxsize);


    vec2 centerUV = (minUV + maxUV) * 0.5;


    float depth;
#if USE_TWO_PASS_CULLING

    depth = textureLod(texHizFar[idx], centerUV, float(miplevel)).r;
#else

    depth = textureLod(texHizFar, centerUV, float(miplevel)).r;
#endif


    bool visible = clipMax.z >= depth - c_depthNudge;

    return visible;
}

#endif
