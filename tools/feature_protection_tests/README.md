# Feature protection regression checks

Build and run against the installed production headers, without Vulkan:

    cmake -S tools/feature_protection_tests -B .artifacts/feature_protection_tests
    cmake --build .artifacts/feature_protection_tests --config Release
    ctest --test-dir .artifacts/feature_protection_tests -C Release --output-on-failure

The tests cover point-to-triangle distances, degenerate input triangles, caller-provided locks with base simplification, merging feature locks, and the 32-attribute API limit. Additional geometric opening and rotation checks are included in the same executable.

The existing meshoptimizer include and library paths can be overridden with the two CMake cache variables in CMakeLists.txt.

The adaptive-protection revision additionally checks uniform circular anchor gaps,
quality-gate fallback under an infeasible automatic anchor budget, retention of
caller locks during that fallback, BVH distances against exhaustive triangle
search, identical-surface quadrature, and normal-error detection for reversed winding.
