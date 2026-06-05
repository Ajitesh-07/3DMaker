// #include <iostream>
// #include <filesystem>
// #include <thread>
// #include <chrono>

// // GLAD must come before GLFW
// #include "RmlUi_Include_GL3.h"
// #include <GLFW/glfw3.h>

// #include <cuda_runtime.h>
// #include <cuda_gl_interop.h>

// #include <RmlUi/Core.h>
// #include <RmlUi/Debugger.h>
// #include "RmlUi_Backend.h"
// #include "RmlUi_Renderer_GL3.h"
// #include "../modules/NeRF/NeRFRenderer.h"

// extern "C" void LaunchNeRFKernel(cudaGraphicsResource* cudaPbo, int width, int height, float time, float multiplier);

// int main() {
//     if (!Backend::Initialize("3DMaker", 1280, 720, true)) {
//         std::cerr << "Failed to initialize backend\n";
//         return -1;
//     }

//     glfwSwapInterval(1);

//     // Initialize GLAD — must happen after a GL context exists
    
//     GLFWwindow* window = glfwGetCurrentContext();
//     int fb_width, fb_height;
//     glfwGetFramebufferSize(window, &fb_width, &fb_height);
    
//     int win_w, win_h;
//     glfwGetWindowSize(window, &win_w, &win_h);
//     std::cout << "Window: " << win_w << "x" << win_h << "\n";
//     std::cout << "Framebuffer: " << fb_width << "x" << fb_height << "\n";
    
//     gladLoadGL(glfwGetProcAddress);

//     // Create PBO and register with CUDA
//     GLuint pbo;
//     glGenBuffers(1, &pbo);
//     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
//     glBufferData(GL_PIXEL_UNPACK_BUFFER, fb_width * fb_height * 4, nullptr, GL_DYNAMIC_DRAW);
//     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

//     cudaGraphicsResource* cudaPbo;
//     cudaGraphicsGLRegisterBuffer(&cudaPbo, pbo, cudaGraphicsMapFlagsWriteDiscard);

//     // Create display texture
//     GLuint displayTexture;
//     glGenTextures(1, &displayTexture);
//     glBindTexture(GL_TEXTURE_2D, displayTexture);
//     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, fb_width, fb_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//     glBindTexture(GL_TEXTURE_2D, 0);

//     // Create FBO for blitting
//     GLuint fbo;
//     glGenFramebuffers(1, &fbo);

//     // RmlUi init
//     Rml::SetSystemInterface(Backend::GetSystemInterface());
//     Rml::SetRenderInterface(Backend::GetRenderInterface());
//     Rml::Initialise();

//     if (!Rml::LoadFontFace("assets/OpenSans-Regular.ttf"))
//         std::cerr << "Warning: failed to load font\n";

//     Rml::Context* context = Rml::CreateContext("Main", Rml::Vector2i(1280, 720));
//     if (!context) {
//         std::cerr << "Failed to create RmlUi context\n";
//         return -1;
//     }

//     Rml::Debugger::Initialise(context);
//     std::cout << std::filesystem::current_path() << std::endl;

//     float speed_multiplier = 1.0f;
//     float fps = 0.0f;
//     float last_time = 0.0f;

//     Rml::DataModelConstructor constructor = context->CreateDataModel("NeRFModel");
//     Rml::DataModelHandle modelHandle = constructor.GetModelHandle();
//     if (constructor) {
//         constructor.Bind("speed_multiplier", &speed_multiplier);
//         constructor.Bind("fps", &fps);
//     }

//     Rml::ElementDocument* document = context->LoadDocument("assets/window.rml");
//     if (document)
//         document->Show();
//     else
//         std::cerr << "Warning: failed to load window.rml\n";

//     // Main loop
//     while (Backend::ProcessEvents(context)) {
//         float time = (float)Backend::GetSystemInterface()->GetElapsedTime();
//         float delta_time = time - last_time;
//         last_time = time;

//         fps = 0.9f * fps + 0.1f * (1.0f / delta_time);
//         modelHandle.DirtyVariable("fps");

//         // 1. CUDA writes pixels directly into PBO on GPU
//         LaunchNeRFKernel(cudaPbo, fb_width, fb_height, time, speed_multiplier);

//         // 2. PBO -> Texture (GPU DMA, no CPU)
//         glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
//         glBindTexture(GL_TEXTURE_2D, displayTexture);
//         glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fb_width, fb_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
//         glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
//         glBindTexture(GL_TEXTURE_2D, 0);

//         Backend::BeginFrame();

//         // 3. Blit texture to screen
//         glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
//         glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, displayTexture, 0);
//         glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
//         glBlitFramebuffer(0, 0, fb_width, fb_height,
//                           0, fb_height, fb_width, 0,
//                           GL_COLOR_BUFFER_BIT, GL_NEAREST);
//         glBindFramebuffer(GL_FRAMEBUFFER, 0);

//         // 4. RmlUi on top
//         context->Update();
//         context->Render();

//         Backend::PresentFrame();
//     }

//     // Cleanup
//     cudaGraphicsUnregisterResource(cudaPbo);
//     glDeleteBuffers(1, &pbo);
//     glDeleteTextures(1, &displayTexture);
//     glDeleteFramebuffers(1, &fbo);
//     Rml::Shutdown();
//     Backend::Shutdown();
//     return 0;
// }

#include "TinyMLP/TinyMLP.h"
#include <iostream>

int main() {
    std::cout << "Working" << std::endl;
}