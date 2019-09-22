#version 330 core

in vec3 position;
in vec4 color;
out vec4 fragColor;

void main() {
    gl_Position = vec4(position, 1.0);
    fragColor = color;
}
