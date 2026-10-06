init python:
    # GPU Blinn-Phong: renpy.geometry дает u_transform + a_position + gl_Position,
    # наш шейдер добавляет только свет. Не переобъявляем u_transform/a_position.
    renpy.register_shader("es_blinn_phong", variables="""
        uniform vec3 u_light_dir;
        uniform vec3 u_view_dir;
        uniform vec4 u_base_color;
        uniform float u_shininess;
        uniform float u_spin;
        attribute vec3 a_normal;
        varying vec3 v_normal;
        varying vec3 v_half;
        varying float v_diff;
    """, vertex_200="""
        float es_sa = sin(radians(u_spin));
        float es_ca = cos(radians(u_spin));
        float es_sx = sin(radians(25.0));
        float es_cx = cos(radians(25.0));
        vec3 es_q = a_normal;
        vec3 wnormal = normalize(vec3(
            es_ca * es_q.x + es_sx * es_sa * es_q.y + es_cx * es_sa * es_q.z,
            es_cx * es_q.y - es_sx * es_q.z,
            -es_sa * es_q.x + es_sx * es_ca * es_q.y + es_cx * es_ca * es_q.z));
        v_normal = wnormal;
        vec3 L = normalize(u_light_dir);
        vec3 V = normalize(u_view_dir);
        v_half = normalize(L + V);
        v_diff = max(dot(wnormal, L), 0.0);
        vec4 es_p = u_transform * vec4(a_position.xyz, 1.0);
        es_p.z = clamp(es_p.z, 0.0, 0.99 * es_p.w);
        gl_Position = es_p;
    """, fragment_200="""
        vec3 N = normalize(v_normal);
        vec3 H = normalize(v_half);
        float diff = v_diff;
        float spec = pow(max(dot(N, H), 0.0), u_shininess);
        vec3 col = u_base_color.rgb * (0.12 + 0.88 * diff) + vec3(1.0) * spec * 0.5;
        gl_FragColor = vec4(col, u_base_color.a);
    """)
