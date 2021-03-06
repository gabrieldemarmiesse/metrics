import os
import sys

from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension
import glob

# we'd better have Cython installed, or it's a no-go
try:
    from Cython.Distutils import build_ext
except ImportError:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org and install it")
    sys.exit(1)

if 0:
    cython_gen_files = [x for x in glob.glob('./**/*.c', recursive=True)
                        if x[-5:] == '_cy.c']
    so_files = list(glob.glob('./**/*.so', recursive=True))
    for file in cython_gen_files + so_files:
        os.remove(file)


# scan the 'dvedit' directory for extension files, converting
# them to extension names in dotted notation
def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


# generate an Extension object from its dotted name
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep) + ".pyx"
    return Extension(
        extName,
        [extPath],
        extra_compile_args=["-O3"]
    )


# get the list of extensions
extNames = scandir("metrics")

# and build up the set of Extension objects
extensions = [makeExtension(name) for name in extNames]

setup(
    name='metrics',
    ext_modules=cythonize(extensions, annotate=True),
    requires=['Cython', 'numpy'],
    packages=find_packages(),
)
