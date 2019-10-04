#version 330 core

in vec3 position;
out vec4 fragColor;

uniform mat4 transModel = mat4(1.0);
uniform mat4 transView = mat4(1.0);
uniform mat4 transProj = mat4(1.0);
uniform vec4 color = vec4(1.0, 1.0, 1.0, 1.0);

void main() {
    gl_Position = vec4(position, 1.0) * transModel * transView * transProj;
    fragColor = color;
}
