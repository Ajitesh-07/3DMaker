cmake_minimum_required(VERSION 3.24)

project(3DMaker LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "89")
set(CMAKE_CUDA_STANDARD 17)

# --- Find Native OpenGL ---
find_package(OpenGL REQUIRED)

# --- FetchContent setup ---
include(FetchContent)

# --- GLFW ---
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# --- FreeType ---
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_Declare(freetype
    GIT_REPOSITORY https://github.com/freetype/freetype.git
    GIT_TAG VER-2-13-2
)
set(FT_DISABLE_ZLIB     ON CACHE BOOL "" FORCE)
set(FT_DISABLE_BZIP2    ON CACHE BOOL "" FORCE)
set(FT_DISABLE_PNG      ON CACHE BOOL "" FORCE)
set(FT_DISABLE_HARFBUZZ ON CACHE BOOL "" FORCE)
set(FT_DISABLE_BROTLI   ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(freetype)
add_library(Freetype::Freetype ALIAS freetype)

# --- RmlUi ---
set(RMLUI_FETCH_DEPENDENCIES ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(rmlui
    GIT_REPOSITORY https://github.com/mikke89/RmlUi.git
    GIT_TAG master
)
FetchContent_MakeAvailable(rmlui)

# --- RmlUi GLFW/GL3 Backend sources ---
set(RMLUI_BACKEND_DIR ${rmlui_SOURCE_DIR}/Backends)
file(GLOB RMLUI_BACKEND_SOURCES
    "${RMLUI_BACKEND_DIR}/RmlUi_Backend_GLFW_GL3.cpp"
    "${RMLUI_BACKEND_DIR}/RmlUi_Platform_GLFW.cpp"
    "${RMLUI_BACKEND_DIR}/RmlUi_Renderer_GL3.cpp"
)

set(TINYMLP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NERF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(modules/NeRF)
add_subdirectory(modules/TinyMLP)

add_executable(${PROJECT_NAME} src/main.cpp ${RMLUI_BACKEND_SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE
    NeRFModule
    TinyMLP
    glfw
    OpenGL::GL
    RmlUi::RmlUi
    RmlUi::Core
    RmlUi::Debugger
    freetype
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
    ${RMLUI_BACKEND_DIR}
)

add_executable(3DViewer src/render.cpp ${RMLUI_BACKEND_SOURCES})

target_link_libraries(3DViewer PRIVATE
    NeRFModule
    TinyMLP
    glfw
    OpenGL::GL
    RmlUi::RmlUi
    RmlUi::Core
    RmlUi::Debugger
    freetype
)

target_include_directories(3DViewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
    ${RMLUI_BACKEND_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)