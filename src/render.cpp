#define NOMINMAX
#include <RmlUi_Include_GL3.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <fstream>
#include <cmath>

#include <NeRF/InstantNerf.h>
#include <NeRF/RenderKernels.h>

#include "../third_party/json.hpp"
using json = nlohmann::json;

float g_yaw = 0.0f;
float g_pitch = 30.0f * (3.14159265f / 180.0f);
float g_radius = 1.5f;
float g_targetX = 0.0f;
float g_targetY = 0.0f;
float g_targetZ = 0.0f;

bool g_leftMouseDown = false;
bool g_middleMouseDown = false;
double g_lastMouseX = 0;
double g_lastMouseY = 0;
float g_panDeltaX = 0.0f;
float g_panDeltaY = 0.0f;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) g_leftMouseDown = true;
        else if (action == GLFW_RELEASE) g_leftMouseDown = false;
    }
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        if (action == GLFW_PRESS) g_middleMouseDown = true;
        else if (action == GLFW_RELEASE) g_middleMouseDown = false;
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    float dx = (float)(xpos - g_lastMouseX);
    float dy = (float)(ypos - g_lastMouseY);

    if (g_leftMouseDown) {
        g_yaw -= dx * 0.005f;
        g_pitch += dy * 0.005f;

        if (g_pitch > 1.5f) g_pitch = 1.5f;
        if (g_pitch < -1.5f) g_pitch = -1.5f;
    }
    
    if (g_middleMouseDown) {
        g_panDeltaX += dx;
        g_panDeltaY += dy;
    }
    
    g_lastMouseX = xpos;
    g_lastMouseY = ypos;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    g_radius -= (float)yoffset * 0.1f;
    if (g_radius < 0.1f) g_radius = 0.1f;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: 3DViewer <transforms_train.json> <model.inerf>" << std::endl;
        return 1;
    }

    std::string transforms_path = argv[1];
    std::string model_path = argv[2];

    float camera_angle_x = 0.69f;
    try {
        std::ifstream f(transforms_path);
        json data = json::parse(f);
        if (data.contains("camera_angle_x")) {
            camera_angle_x = data["camera_angle_x"];
        }
        
        if (data.contains("frames") && data["frames"].is_array() && data["frames"].size() > 0) {
            auto matrix = data["frames"][0]["transform_matrix"];
            float px = matrix[0][3];
            float py = matrix[1][3];
            float pz = matrix[2][3];
            
            float zc_x = matrix[0][2];
            float zc_y = matrix[1][2];
            float zc_z = matrix[2][2];
            
            g_radius = px*zc_x + py*zc_y + pz*zc_z;
            if (g_radius < 0.1f) g_radius = 1.0f;
            
            g_targetX = px - g_radius * zc_x;
            g_targetY = py - g_radius * zc_y;
            g_targetZ = pz - g_radius * zc_z;
            
            float rel_x = g_radius * zc_x;
            float rel_y = g_radius * zc_y;
            float rel_z = g_radius * zc_z;
            
            g_pitch = std::asin(rel_z / g_radius);
            g_yaw = std::atan2(rel_y, rel_x);
        }
    } catch (std::exception& e) {
        std::cerr << "Warning: Failed to parse from " << transforms_path << ". Using defaults." << std::endl;
    }

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    int window_width = 800;
    int window_height = 800;
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "3DMaker - Real-time NeRF Viewer", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if (!gladLoaderLoadGL()) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return 1;
    }

    cudaSetDevice(0);
    InstantNerf nerf;
    
    std::cout << "Loading NeRF model: " << model_path << std::endl;
    nerf.load(model_path);
    nerf.setMemoryMode(INFERENCE);
    nerf.setBgColor(make_float3(1.0f, 1.0f, 1.0f));
    nerf.setProfiling(false);

    int pixels = window_width * window_height;
    
    float3* d_rays_o;
    float3* d_rays_d;
    float* d_hdr_rgb;
    cudaMalloc(&d_rays_o, pixels * sizeof(float3));
    cudaMalloc(&d_rays_d, pixels * sizeof(float3));
    cudaMalloc(&d_hdr_rgb, pixels * 3 * sizeof(float));

    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, pixels * 4 * sizeof(uint8_t), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsResource* cuda_pbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    const char* vertexShaderSource = "#version 330 core\n"
        "out vec2 TexCoord;\n"
        "void main() {\n"
        "    float x = -1.0 + float((gl_VertexID & 1) << 2);\n"
        "    float y = -1.0 + float((gl_VertexID & 2) << 1);\n"
        "    TexCoord.x = (x+1.0)*0.5;\n"
        "    TexCoord.y = 1.0 - (y+1.0)*0.5;\n" 
        "    gl_Position = vec4(x, y, 0, 1);\n"
        "}\0";
    
    const char* fragmentShaderSource = "#version 330 core\n"
        "out vec4 FragColor;\n"
        "in vec2 TexCoord;\n"
        "uniform sampler2D screenTexture;\n"
        "void main() {\n"
        "    FragColor = texture(screenTexture, TexCoord);\n"
        "}\n\0";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    GLuint VAO;
    glGenVertexArrays(1, &VAO);

    // 5. Render Loop
    float focal_length = 0.5f * window_width / tanf(0.5f * camera_angle_x);

    double lastTime = glfwGetTime();
    int nbFrames = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        double currentTime = glfwGetTime();
        nbFrames++;
        if (currentTime - lastTime >= 1.0) {
            std::cout << "FPS: " << nbFrames << " | Frame Time: " << 1000.0 / double(nbFrames) << " ms" << std::endl;
            nbFrames = 0;
            lastTime = currentTime;
        }

        float pz = g_radius * sinf(g_pitch);
        float r_xy = g_radius * cosf(g_pitch);
        float px = r_xy * cosf(g_yaw);
        float py = r_xy * sinf(g_yaw);

        float zc_x = px / g_radius, zc_y = py / g_radius, zc_z = pz / g_radius;

        float xc_x = -zc_y, xc_y = zc_x, xc_z = 0.0f;
        float len_x = sqrtf(xc_x*xc_x + xc_y*xc_y);
        if (len_x > 0.0f) { xc_x /= len_x; xc_y /= len_x; }
        else { xc_x = 1.0f; xc_y = 0.0f; xc_z = 0.0f; }

        float yc_x = zc_y * xc_z - zc_z * xc_y;
        float yc_y = zc_z * xc_x - zc_x * xc_z;
        float yc_z = zc_x * xc_y - zc_y * xc_x;

        if (g_panDeltaX != 0.0f || g_panDeltaY != 0.0f) {
            float dx = g_panDeltaX * 0.001f * g_radius;
            float dy = g_panDeltaY * 0.001f * g_radius;
            
            g_targetX -= dx * xc_x - dy * yc_x;
            g_targetY -= dx * xc_y - dy * yc_y;
            g_targetZ -= dx * xc_z - dy * yc_z;
            
            g_panDeltaX = 0.0f;
            g_panDeltaY = 0.0f;
        }

        px += g_targetX;
        py += g_targetY;
        pz += g_targetZ;

        uint8_t* d_rgba_byte;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_rgba_byte, &num_bytes, cuda_pbo_resource);

        wrapper_generate_custom_rays(
            window_width, window_height, focal_length,
            xc_x, yc_x, zc_x, px,
            xc_y, yc_y, zc_y, py,
            xc_z, yc_z, zc_z, pz,
            d_rays_o, d_rays_d, 0
        );
        cudaDeviceSynchronize();

        nerf.renderImage(d_rays_o, d_rays_d, pixels, d_hdr_rgb, 0);
        cudaDeviceSynchronize();

        wrapper_float_to_byte(d_hdr_rgb, d_rgba_byte, pixels, 0);
        cudaDeviceSynchronize();

        cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, window_width, window_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwSwapBuffers(window);
    }

    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
    glDeleteProgram(shaderProgram);
    glDeleteVertexArrays(1, &VAO);

    cudaFree(d_rays_o);
    cudaFree(d_rays_d);
    cudaFree(d_hdr_rgb);

    glfwTerminate();
    return 0;
}
