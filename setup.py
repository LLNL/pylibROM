import os
import re
import subprocess
import sys
from pathlib import Path

# NOTE: Current setup.py uses "--global-option", which is deprecated and will not be available from pip 23.3.
# It recommends using "--config-settings", though there is no clear documentation about this.
# For now, we enforce to use the versions < 23.3.0.
import pkg_resources
#pkg_resources.require(['pip < 23.3.0'])

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install as _install

# Take the global option for pre-installed librom directory.
librom_dir = None
install_scalapack = False
use_mfem = True
# TODO: fix this.. passing options through pip to setuptools with PEP517 is not working
for arg in sys.argv:
    if (arg[:13] == "--librom_dir=" or arg[:13] == "--librom-dir="):
        librom_dir = arg[13:]
    if (arg[:19] == "--install_scalapack"):
        install_scalapack = True
    if (arg[:9] == "--no-mfem"):
        use_mfem = False
    if (arg[:10] == "--use-mfem"):
        use_mfem = True

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class InstallBuild(_install):

    user_options = _install.user_options + [
        ("librom-dir=", None, "Path to libROM root directory. If specified, pylibROM will use this instead of compiling its own libROM."),
        ("install-scalapack", None, "Enables SCALAPACK when building libROM."),
        ("use-mfem", None, "Compile pylibROM with MFEM enabled."),
        ("no-mfem", None, "Compile pylibROM without MFEM.")
        ]

    negative_opt = dict(_install.negative_opt)
    negative_opt.update({"no-mfem" : "use-mfem"})

    def run(self):
        _install.run(self)


    def initialize_options(self):
        _install.initialize_options(self)
        self.librom_dir = None
        self.install_scalapack = False
        self.use_mfem = True


    def finalize_options(self):
        global librom_dir, install_scalapack, use_mfem
        librom_dir = self.librom_dir
        install_scalapack = self.install_scalapack
        use_mfem = self.use_mfem

        _install.finalize_options(self)


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        global librom_dir, install_scalapack, use_mfem
        cmake_args = []
        if (librom_dir is None):
            librom_dir = os.path.dirname(os.path.realpath(__file__))
            librom_dir += "/extern/libROM/"
            print("Installing libROM library: %s" % librom_dir)
            
            librom_cmd = "cd %s && ./scripts/compile.sh -t ./cmake/toolchains/simple.cmake" % librom_dir
            if (install_scalapack): librom_cmd += " -s"
            if (use_mfem): librom_cmd += " -m -g -l"
            print("libROM installation command: %s" % librom_cmd)
            subprocess.run(
                librom_cmd, shell=True, check=True
            )
        else:
            print("Using pre-installed libROM library: %s" % librom_dir)
            cmake_args += [f"-DLIBROM_DIR=%s" % librom_dir]

        cmake_args += ["-DUSE_MFEM={:s}".format('ON' if use_mfem else 'OFF')]

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args += [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", "-j 8", "-v", *build_args], cwd=build_temp, check=True
        )


    def initialize_options(self):
        build_ext.initialize_options(self)


    def finalize_options(self):
        build_ext.finalize_options(self)


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="pylibROM",
    description="Python Interface for LLNL libROM",
    long_description="",
    packages=find_packages(where='bindings', exclude=['pylibROM.mfem'] if use_mfem == False else ['']),
    package_dir={"":"bindings"},
    ext_modules=[CMakeExtension("_pylibROM")],
    cmdclass={
        "build_ext": CMakeBuild,
        "install": InstallBuild
    },
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.7",
)
