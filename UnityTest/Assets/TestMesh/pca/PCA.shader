Shader "Custom/PCA"
{
    Properties
    {
        _MainTex ("TextureA (RGB)", 2D) = "white" {}
        _MainTex2 ("TextureB (RGB)", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 200

        CGPROGRAM
        // Physically based Standard lighting model, and enable shadows on all light types
        #pragma surface surf Standard fullforwardshadows

        // Use shader model 3.0 target, to get nicer looking lighting
        #pragma target 3.0

        sampler2D _MainTex;
        sampler2D _MainTex2;

        struct Input
        {
            float2 uv_MainTex;
        };


        // Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
        // See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
        // #pragma instancing_options assumeuniformscaling
        UNITY_INSTANCING_BUFFER_START(Props)
            // put more per-instance properties here
        UNITY_INSTANCING_BUFFER_END(Props)

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            float4 ta = tex2D (_MainTex, IN.uv_MainTex);
            float4 tb = tex2D (_MainTex2, IN.uv_MainTex);

            ta.rgb = ta.rgb * 8.8314566 + -2.409642484;
            tb.rgb = tb.rgb * 3.249315059 + -1.490176875;

            float b = ta.b * 0.61185951 + ta.g * -0.6185035 + ta.r * 0.00040697 + tb.b * 0.04274134 + tb.g * -0.16927867 + tb.r * 0.24782917 + -1.15684197;
            float g = ta.b * 0.53556616 + ta.g * 0.00278411 + ta.r * 0.01133042 + tb.b * 0.03200419 + tb.g * -0.18844109 + tb.r * -0.19651141 + -1.10568591;
            float r = ta.b * 0.49268253 + ta.g * 0.59531081 + ta.r * 0.00336 + tb.b * -0.00592251 + tb.g * -0.02336042 + tb.r * -0.44877486 + -1.17532121;

            float nx = ta.b * 0.00533332 + ta.g * -0.0079842 + ta.r * 0.61739126 + tb.b * -0.77832434 + tb.g * -0.11216297 + tb.r * 0.01891989 + -0.00350749;
            float ny = ta.b * -0.00219139 + ta.g * 0.00626628 + ta.r * 0.78582084 + tb.b * 0.6044255 + tb.g * 0.13077479 + tb.r * 0.00005425 + 0.00112196;

            float m = ta.b * 0.2981968 + ta.g * 0.37849793 + ta.r * -0.0255512 + tb.b * -0.07698506 + tb.g * 0.49812125 + tb.r * 0.71007133 + -0.73269766;
            float rh = ta.b * -0.08429811 + ta.g * 0.34595811 + ta.r * 0.02282416 + tb.b * 0.14166221 + tb.g * -0.8108524 + tb.r * 0.44046093 + -0.37357262;

            float albedoWeight = 2.5;
            r = clamp(r / albedoWeight, -1, 1) * 127.5 + 127.5;
            g = clamp(g / albedoWeight, -1, 1) * 127.5 + 127.5;
            b = clamp(b / albedoWeight, -1, 1) * 127.5 + 127.5;
            nx = clamp(nx, -1, 1) * 127.5 + 127.5;
            ny = clamp(ny, -1, 1) * 127.5 + 127.5;
            m = clamp(m, -1, 1) * 127.5 + 127.5;
            rh = clamp(rh, -1, 1) * 127.5 + 127.5;

            r = r / 255;
            g = g / 255;
            b = b / 255;
            m = m / 255;
            rh = rh / 255;



            nx = clamp(nx / 127.5 - 1.0, -1, 1);
            ny = clamp(ny / 127.5 - 1.0, -1, 1);
            float nz = sqrt(1 - (nx * nx + ny * ny) + 0.000001);

            float3 normal = float3(nx, ny, nz);
            

            o.Albedo = float3(r, g, b);
            o.Normal = normal;
            o.Emission = float3(0, 0, 0);
            o.Metallic = m;
            o.Smoothness = 1 - rh;
            o.Alpha = 1;
        }
        ENDCG
    }
    FallBack "Diffuse"
}
