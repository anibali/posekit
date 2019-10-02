#version 330 core

uniform sampler2D tex;
in vec2 fragTexcoord;
out vec4 outFragColor;

void main() {
    outFragColor = texture2D(tex, fragTexcoord);
}
