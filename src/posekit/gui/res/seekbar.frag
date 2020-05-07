#version 330 core

uniform float progress;
in vec2 fragTexcoord;
out vec4 outFragColor;

void main() {
    if(fragTexcoord.x <= progress) {
        outFragColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        outFragColor = vec4(1.0, 1.0, 1.0, 0.4);
    }
}
