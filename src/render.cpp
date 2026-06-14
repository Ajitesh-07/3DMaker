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

float g_world_up[3] = {0.0f, 0.0f, 1.0f};
float g_scene_center[3] = {0.0f, 0.0f, 0.0f};

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
            int count = data["frames"].size();

            // Camera centroid (fallback) + average up axis (transform column 1).
            float sum_px = 0.0f, sum_py = 0.0f, sum_pz = 0.0f;
            float sum_ux = 0.0f, sum_uy = 0.0f, sum_uz = 0.0f;

            // Ray-convergence center: the point closest to every camera's view ray (where the
            // cameras actually look). Solve M p = b, M = Sum(I - f f^T), b = Sum(I - f f^T) C.
            // The camera centroid is WRONG for top-down/nadir captures -- it floats in the air
            // above the content; this finds the content itself.
            double M[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
            double bvec[3] = {0,0,0};

            for (auto& frame : data["frames"]) {
                auto matrix = frame["transform_matrix"];
                float Cx = (float)matrix[0][3], Cy = (float)matrix[1][3], Cz = (float)matrix[2][3];
                sum_px += Cx; sum_py += Cy; sum_pz += Cz;
                sum_ux += (float)matrix[0][1]; sum_uy += (float)matrix[1][1]; sum_uz += (float)matrix[2][1];

                // forward = -Z axis (column 2): cameras look along -Z
                float fx = -(float)matrix[0][2], fy = -(float)matrix[1][2], fz = -(float)matrix[2][2];
                float fl = sqrtf(fx*fx + fy*fy + fz*fz);
                if (fl > 0) { fx/=fl; fy/=fl; fz/=fl; }

                double a00 = 1.0 - (double)fx*fx, a01 = -(double)fx*fy, a02 = -(double)fx*fz;
                double a11 = 1.0 - (double)fy*fy, a12 = -(double)fy*fz, a22 = 1.0 - (double)fz*fz;
                M[0][0]+=a00; M[0][1]+=a01; M[0][2]+=a02;
                M[1][0]+=a01; M[1][1]+=a11; M[1][2]+=a12;
                M[2][0]+=a02; M[2][1]+=a12; M[2][2]+=a22;
                bvec[0] += a00*Cx + a01*Cy + a02*Cz;
                bvec[1] += a01*Cx + a11*Cy + a12*Cz;
                bvec[2] += a02*Cx + a12*Cy + a22*Cz;
            }

            float cen_x = sum_px/count, cen_y = sum_py/count, cen_z = sum_pz/count;
            g_scene_center[0] = cen_x; g_scene_center[1] = cen_y; g_scene_center[2] = cen_z;

            float len_u = sqrtf(sum_ux*sum_ux + sum_uy*sum_uy + sum_uz*sum_uz);
            if (len_u > 0) { g_world_up[0]=sum_ux/len_u; g_world_up[1]=sum_uy/len_u; g_world_up[2]=sum_uz/len_u; }

            // Ridge-regularize toward the centroid so M stays invertible even when cameras are
            // near-parallel (the view-axis direction is otherwise unconstrained for pure nadir).
            double eps = 1e-3 * (M[0][0] + M[1][1] + M[2][2]);
            M[0][0]+=eps; M[1][1]+=eps; M[2][2]+=eps;
            bvec[0]+=eps*cen_x; bvec[1]+=eps*cen_y; bvec[2]+=eps*cen_z;

            double det =
                M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
              - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
              + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
            if (fabs(det) > 1e-9) {
                double bx=bvec[0], by=bvec[1], bz=bvec[2];
                double px = ( bx*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
                            - M[0][1]*(by*M[2][2]-M[1][2]*bz)
                            + M[0][2]*(by*M[2][1]-M[1][1]*bz) ) / det;
                double py = ( M[0][0]*(by*M[2][2]-M[1][2]*bz)
                            - bx*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
                            + M[0][2]*(M[1][0]*bz-by*M[2][0]) ) / det;
                double pz = ( M[0][0]*(M[1][1]*bz-by*M[2][1])
                            - M[0][1]*(M[1][0]*bz-by*M[2][0])
                            + bx*(M[1][0]*M[2][1]-M[1][1]*M[2][0]) ) / det;
                g_scene_center[0]=(float)px; g_scene_center[1]=(float)py; g_scene_center[2]=(float)pz;
            }

            // Orbit radius + initial elevation from camera geometry relative to the center, so the
            // view starts framed like the capture (near top-down for nadir, side-on for orbits).
            float sum_r = 0.0f, sum_elev = 0.0f;
            for (auto& frame : data["frames"]) {
                auto matrix = frame["transform_matrix"];
                float dx=(float)matrix[0][3]-g_scene_center[0];
                float dy=(float)matrix[1][3]-g_scene_center[1];
                float dz=(float)matrix[2][3]-g_scene_center[2];
                float r = sqrtf(dx*dx+dy*dy+dz*dz);
                sum_r += r;
                if (r > 1e-6f) {
                    float e = (dx*g_world_up[0]+dy*g_world_up[1]+dz*g_world_up[2]) / r;
                    sum_elev += asinf(fmaxf(-1.0f, fminf(1.0f, e)));
                }
            }
            g_radius = sum_r / count;
            if (g_radius < 0.1f) g_radius = 1.0f;
            g_pitch = sum_elev / count;
            if (g_pitch > 1.5f) g_pitch = 1.5f;
            if (g_pitch < -1.5f) g_pitch = -1.5f;

            std::cout << "[viewer] center=(" << g_scene_center[0] << "," << g_scene_center[1] << ","
                      << g_scene_center[2] << ") radius=" << g_radius
                      << " pitch=" << (g_pitch * 57.2958f) << " deg" << std::endl;
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

        // Orbit basis
        float O_up[3] = {g_world_up[0], g_world_up[1], g_world_up[2]};
        float O_forward[3] = {1.0f, 0.0f, 0.0f};
        if (abs(O_up[0]) > 0.9f) { O_forward[0] = 0.0f; O_forward[1] = 1.0f; O_forward[2] = 0.0f; }
        
        float dot_uf = O_up[0]*O_forward[0] + O_up[1]*O_forward[1] + O_up[2]*O_forward[2];
        O_forward[0] -= dot_uf * O_up[0]; 
        O_forward[1] -= dot_uf * O_up[1]; 
        O_forward[2] -= dot_uf * O_up[2];
        
        float len_f = sqrtf(O_forward[0]*O_forward[0] + O_forward[1]*O_forward[1] + O_forward[2]*O_forward[2]);
        if(len_f > 0){O_forward[0]/=len_f; O_forward[1]/=len_f; O_forward[2]/=len_f;}
        
        float O_right[3] = {
            O_up[1]*O_forward[2] - O_up[2]*O_forward[1],
            O_up[2]*O_forward[0] - O_up[0]*O_forward[2],
            O_up[0]*O_forward[1] - O_up[1]*O_forward[0]
        };

        float r_z = g_radius * sinf(g_pitch);
        float r_xy = g_radius * cosf(g_pitch);
        float local_x = r_xy * cosf(g_yaw);
        float local_y = r_xy * sinf(g_yaw);
        float local_z = r_z;

        float cx = g_scene_center[0] + O_forward[0]*local_x + O_right[0]*local_y + O_up[0]*local_z;
        float cy = g_scene_center[1] + O_forward[1]*local_x + O_right[1]*local_y + O_up[1]*local_z;
        float cz = g_scene_center[2] + O_forward[2]*local_x + O_right[2]*local_y + O_up[2]*local_z;

        cx += g_targetX;
        cy += g_targetY;
        cz += g_targetZ;

        float look_x = g_scene_center[0] + g_targetX;
        float look_y = g_scene_center[1] + g_targetY;
        float look_z = g_scene_center[2] + g_targetZ;
        
        float zc_x = cx - look_x, zc_y = cy - look_y, zc_z = cz - look_z;
        float len_zc = sqrtf(zc_x*zc_x + zc_y*zc_y + zc_z*zc_z);
        if(len_zc > 0) { zc_x/=len_zc; zc_y/=len_zc; zc_z/=len_zc; }

        float xc_x = O_up[1]*zc_z - O_up[2]*zc_y;
        float xc_y = O_up[2]*zc_x - O_up[0]*zc_z;
        float xc_z = O_up[0]*zc_y - O_up[1]*zc_x;
        float len_xc = sqrtf(xc_x*xc_x + xc_y*xc_y + xc_z*xc_z);
        
        if(len_xc > 0) { xc_x/=len_xc; xc_y/=len_xc; xc_z/=len_xc; }
        else { xc_x = O_right[0]; xc_y = O_right[1]; xc_z = O_right[2]; }

        float yc_x = zc_y*xc_z - zc_z*xc_y;
        float yc_y = zc_z*xc_x - zc_x*xc_z;
        float yc_z = zc_x*xc_y - zc_y*xc_x;

        if (g_panDeltaX != 0.0f || g_panDeltaY != 0.0f) {
            float dx = g_panDeltaX * 0.001f * g_radius;
            float dy = g_panDeltaY * 0.001f * g_radius;
            
            g_targetX -= dx * xc_x - dy * yc_x;
            g_targetY -= dx * xc_y - dy * yc_y;
            g_targetZ -= dx * xc_z - dy * yc_z;
            
            g_panDeltaX = 0.0f;
            g_panDeltaY = 0.0f;
        }

        uint8_t* d_rgba_byte;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_rgba_byte, &num_bytes, cuda_pbo_resource);

        wrapper_generate_custom_rays(
            window_width, window_height, focal_length,
            xc_x, yc_x, zc_x, cx,
            xc_y, yc_y, zc_y, cy,
            xc_z, yc_z, zc_z, cz,
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
