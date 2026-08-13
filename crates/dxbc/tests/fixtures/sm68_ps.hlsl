#define ROOT_SIGNATURE \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | " \
    "CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), " \
    "CBV(b0, flags=DATA_STATIC), " \
    "DescriptorTable(SRV(t0, numDescriptors=2, flags=DESCRIPTORS_VOLATILE)), " \
    "StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR)"

cbuffer Constants : register(b0)
{
    float4 tint;
};

Texture2D<float4> texture0 : register(t0);
SamplerState sampler0 : register(s0);

[RootSignature(ROOT_SIGNATURE)]
float4 main(float2 uv : TEXCOORD0) : SV_Target
{
    return texture0.Sample(sampler0, uv) * tint;
}
