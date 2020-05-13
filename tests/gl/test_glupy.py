from glupy.gl import VAO


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
        vao1.bind()
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
        vao1.bind()
        assert VAO._get_currently_bound() == vao1._handle
        vao2.bind()
        assert VAO._get_currently_bound() == vao2._handle
