# VST3 and AUv2 bundles around the CLAP that `cargo xtask bundle driftscape-plugin --release` builds.
# clap-wrapper (MIT) compiles a thin wrapper; the CLAP sits inside each bundle at Contents/PlugIns/Driftscape.clap,
# so the bundles are self-contained. The CLAP, VST3 (MIT) and AudioUnit (Apache 2.0) SDKs are downloaded at
# configure time. Build with ./build.sh.
cmake_minimum_required(VERSION 3.21)
project(driftscape-wrap LANGUAGES C CXX OBJC OBJCXX)

set(CMAKE_OSX_DEPLOYMENT_TARGET 11.0 CACHE STRING "")
# The AudioUnit SDK needs C++17 (std::byte) and gets no standard of its own.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)
set(DRIFTSCAPE_CLAP "" CACHE PATH "The Driftscape.clap bundle to wrap")
if (NOT EXISTS "${DRIFTSCAPE_CLAP}")
    message(FATAL_ERROR "DRIFTSCAPE_CLAP must point at Driftscape.clap (run cargo xtask bundle driftscape-plugin --release)")
endif()

set(CLAP_WRAPPER_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "" FORCE)
set(CLAP_WRAPPER_BUILD_AUV2 ON CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(clap-wrapper
    GIT_REPOSITORY https://github.com/free-audio/clap-wrapper.git
    GIT_TAG v0.18.0
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(clap-wrapper)

set(version 0.1.0)

add_library(driftscape_vst3 MODULE)
target_add_vst3_wrapper(
    TARGET driftscape_vst3
    OUTPUT_NAME Driftscape
    BUNDLE_IDENTIFIER io.raskell.driftscape.vst3
    BUNDLE_VERSION ${version}
    MACOS_EMBEDDED_CLAP_LOCATION "${DRIFTSCAPE_CLAP}")

add_library(driftscape_auv2 MODULE)
target_add_auv2_wrapper(
    TARGET driftscape_auv2
    OUTPUT_NAME Driftscape
    BUNDLE_IDENTIFIER io.raskell.driftscape.auv2
    BUNDLE_VERSION ${version}
    MANUFACTURER_NAME "raskell.io"
    MANUFACTURER_CODE "Rskl"
    SUBTYPE_CODE "Drft"
    INSTRUMENT_TYPE "aumu"
    MACOS_EMBEDDED_CLAP_LOCATION "${DRIFTSCAPE_CLAP}")
