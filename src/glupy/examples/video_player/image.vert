#version 330 core

in vec2 position;
in vec2 texcoord;
out vec2 fragTexcoord;

uniform mat4 transProj = mat4(1.0);

void main()
{
    fragTexcoord = texcoord;
    gl_Position = vec4(position, 0.0, 1.0) * transProj;
}
