Shader "Custom/Tex4"
{
    Properties
    {
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _NormalTex ("Normal", 2D) = "white" {}
        _RoughnessTex ("Roughness", 2D) = "white" {}
        _MetallicTex ("Metallic", 2D) = "white" {}
        _TextureBias ("TexBias", float) = 0.0
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
        sampler2D _NormalTex;
        sampler2D _RoughnessTex;
        sampler2D _MetallicTex;
        uniform float _TextureBias;

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
            // Albedo comes from a texture tinted by color
            fixed4 c = tex2Dbias (_MainTex, float4(IN.uv_MainTex, 0, _TextureBias));
            fixed4 n = tex2Dbias (_NormalTex, float4(IN.uv_MainTex, 0, _TextureBias));
            fixed4 rh = tex2Dbias (_RoughnessTex, float4(IN.uv_MainTex, 0, _TextureBias)).g;
            fixed4 m = tex2Dbias (_MetallicTex, float4(IN.uv_MainTex, 0, _TextureBias)).g;

            o.Albedo = c.rgb;
            o.Normal = UnpackNormal(n);
            // Metallic and smoothness come from slider variables
            o.Metallic = m;
            o.Smoothness = 1 - rh;
            o.Alpha = c.a;
        }
        ENDCG
    }
    FallBack "Diffuse"
}
