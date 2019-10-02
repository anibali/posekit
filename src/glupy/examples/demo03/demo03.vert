#version 330 core

attribute vec2 position;
attribute vec2 texcoord;
out vec2 fragTexcoord;

void main()
{
    fragTexcoord = texcoord;
    gl_Position = vec4(position, 0.0, 1.0);
}
