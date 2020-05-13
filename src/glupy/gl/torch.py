from contextlib import contextmanager
from typing import ContextManager

import pycuda.autoinit
import pycuda.driver
import torch
from pycuda.gl import graphics_map_flags, RegisteredImage

from glupy.gl import Texture2d


class MappedTexture:
    def __init__(self, height, width):
        """Helper for mapping CUDA tensor data to an OpenGL texture.

        Args:
            height (int): Texture height in pixels.
            width (int): Texture width in pixels.
        """
        channels = 4
        self._gl_texture = Texture2d((height, width, channels))
        self._cuda_buffer = RegisteredImage(int(self.gl_texture._handle), self.gl_texture.target,
                                            graphics_map_flags.WRITE_DISCARD)
        self._tensor = torch.zeros((height, width, channels), dtype=torch.uint8, device='cuda')
        self._tensor_data_ptr = self._tensor.data_ptr()

        memcpy = pycuda.driver.Memcpy2D()
        memcpy.set_src_device(self._tensor_data_ptr)
        mapping = self._cuda_buffer.map()
        memcpy.set_dst_array(mapping.array(0, 0))
        mapping.unmap()
        memcpy.height = height
        memcpy.width_in_bytes = memcpy.src_pitch = memcpy.dst_pitch = width * channels * self._tensor.element_size()
        self._memcpy = memcpy

    @property
    def tensor(self):
        return self._tensor

    @property
    def gl_texture(self):
        return self._gl_texture

    def update(self):
        """Copy data from the PyTorch CUDA tensor into OpenGL texture memory."""
        tensor = self.tensor
        assert tensor.data_ptr() == self._tensor_data_ptr
        assert tensor.numel() * tensor.element_size() == self.gl_texture.nbytes
        assert tensor.is_contiguous()
        self._memcpy(aligned=False)
        torch.cuda.synchronize(tensor.device)

    @contextmanager
    def modify(self) -> ContextManager[torch.Tensor]:
        """Yields a tensor which can be modified to change the texture data.

        The yielded tensor will be copied into texture memory when the context exits.

        Yields:
            torch.Tensor: A mutable CUDA tensor representing the texture data.
        """
        yield self.tensor
        self.update()
