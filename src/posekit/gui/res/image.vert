#version 330 core

in vec2 position;
in vec2 texcoord;
out vec2 fragTexcoord;

uniform mat4 transProj = mat4(1.0);
uniform float depth = 0.999999;

void main()
{
    fragTexcoord = texcoord;
    gl_Position = vec4(position, depth, 1.0) * transProj;
}
