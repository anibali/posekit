// This code is based on the anti-aliased checkerboard shader code at:
// http://www.yaldex.com/open-gl/ch17lev1sec5.html

#version 330 core

in vec2 fragTexcoord;
out vec4 outFragColor;

// NOTE: 1 turn = 2 tiles (one whole period of the pattern).
// Controls how many checkerboard tiles will be shown (in turns).
const vec2 frequency = vec2(16, 16);
// Controls the pattern offset (in turns).
const vec2 phase = vec2(0, 0);

// The two tile colours.
const vec4 colour1 = vec4(1.0, 1.0, 1.0, 0.5);
const vec4 colour2 = vec4(0.2, 0.2, 0.2, 0.5);

// Controls how blurred the transitions between tiles are.
const float blurStrength = 1.0;

void main() {
    // Determine the size of the transitional region (in turns).
    vec2 fuzz = fwidth(fragTexcoord) * frequency * blurStrength;

    // Determine the position in the repeating pattern (in turns).
    vec2 angle = fragTexcoord * frequency + phase + fuzz / 2;

    // Clamp boundaries to prevent fuzz on the edges.
    angle = max(angle, (vec2(0) * frequency + phase + fuzz / 2) + fuzz / 2);
    angle = min(angle, (vec2(1) * frequency + phase + fuzz / 2) - fuzz / 2);

    // Normalise the angle to lie within the [0, 1) range.
    vec2 normAngle = fract(angle);

    // Calculate the fragment colour, with a smooth transition between tiles.
    vec2 p = smoothstep(vec2(0.5), fuzz + vec2(0.5), normAngle) +
            (1.0 - smoothstep(vec2(0.0), fuzz, normAngle));
    outFragColor = mix(colour1, colour2, p.x * p.y + (1.0 - p.x) * (1.0 - p.y));
}
