#version 330 core

in vec3 position;
in vec2 texcoord;
out vec2 fragTexcoord;

layout(std140) uniform transformMatrices {
    mat4 viewMatrix;
    mat4 projMatrix;
};

uniform mat4 modelMatrix = mat4(1.0);

void main() {
    gl_Position = vec4(position, 1.0) * modelMatrix * viewMatrix * projMatrix;
    fragTexcoord = texcoord;
}
