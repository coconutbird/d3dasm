RWTexture2D<float4> output_texture : register(u0);

struct Payload
{
    float4 color;
};

[shader("raygeneration")]
void ray_generation()
{
    output_texture[DispatchRaysIndex().xy] = float4(1.0, 0.5, 0.25, 1.0);
}

[shader("miss")]
void miss(inout Payload payload)
{
    payload.color = 0.0;
}
