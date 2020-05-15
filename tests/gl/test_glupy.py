import numpy as np

from glupy.gl import VAO, VBO


class TestBinding:
    def test_pattern1(self, gl_context):
        vao1 = VAO()
        vao2 = VAO()

        assert VAO._get_currently_bound() == 0
        with vao1:
            assert VAO._get_currently_bound() == vao1._handle
            with vao2:
                assert VAO._get_currently_bound() == vao2._handle
            assert VAO._get_currently_bound() == vao1._handle
        assert VAO._get_currently_bound() == 0

    def test_pattern2(self, gl_context):
        vao1 = VAO()
        vao2 = VAO()

        assert VAO._get_currently_bound() == 0
        vao1.force_bind()
        assert VAO._get_currently_bound() == vao1._handle
        del vao1
        assert VAO._get_currently_bound() == 0
        with vao2:
            assert VAO._get_currently_bound() == vao2._handle
        assert VAO._get_currently_bound() == 0

    def test_pattern3(self, gl_context):
        vao1 = VAO()
        vao2 = VAO()

        assert VAO._get_currently_bound() == 0
        vao1.force_bind()
        assert VAO._get_currently_bound() == vao1._handle
        vao2.force_bind()
        assert VAO._get_currently_bound() == vao2._handle

    def test_pattern4(self, gl_context):
        vbo1 = VBO(np.dtype((np.float32, 4)))
        vbo2 = VBO(np.dtype((np.float32, 4)))
        vbo3 = VBO(np.dtype((np.float32, 4)))
        vao1 = VAO()
        vao2 = VAO()
        with vao2:
            vao2.bind_vbo(vbo2)
            assert vbo2.is_currently_bound()
        assert vbo2.is_currently_bound()
        with vbo3:
            assert vbo3.is_currently_bound()
            with vao1:
                vao1.bind_vbo(vbo1)
                assert vbo1.is_currently_bound()
            assert vbo1.is_currently_bound()
            with vao2:
                assert vbo1.is_currently_bound()
            assert vbo1.is_currently_bound()
