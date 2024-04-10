from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension(
    name="cython_file",
    sources=["cython_file.pyx", "fft.cpp"],
    extra_compile_args=["-std=c++20"],
    language="c++",
)]

setup(
    name='cython_file',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
