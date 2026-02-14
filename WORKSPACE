workspace(name = "org_deepmind_lab")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@//:python_system.bzl", "python_repo")

http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-1.17.0",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz"],
)

http_archive(
    name = "bazel_skylib",
    strip_prefix = "bazel-skylib-1.9.0",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/refs/tags/1.9.0.tar.gz"],
)

http_archive(
    name = "com_google_absl",
    strip_prefix = "abseil-cpp-20260107.1",
    urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20260107.1.tar.gz"],
)

http_archive(
    name = "com_google_absl_py",
    strip_prefix = "abseil-py-2.4.0",
    urls = ["https://github.com/abseil/abseil-py/archive/refs/tags/v2.4.0.tar.gz"],
)

http_archive(
    name = "eigen_archive",
    build_file = "@//bazel:eigen.BUILD",
    sha256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
    strip_prefix = "eigen-3.4.0",
    urls = [
        "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
    ],
)

http_archive(
    name = "glib_archive",
    build_file = "@//bazel:glib.BUILD",
    sha256 = "0cbb3d31c9d181bbcc97cba3d9dbe3250f75e2da25e5f7c8bf5a993fe54baf6a",
    strip_prefix = "glib-2.55.1",
    urls = [
        "https://ftp.gnome.org/pub/gnome/sources/glib/2.55/glib-2.55.1.tar.xz",
    ],
)

http_archive(
    name = "jpeg_archive",
    build_file = "@//bazel:jpeg.BUILD",
    sha256 = "2303a6acfb6cc533e0e86e8a9d29f7e6079e118b9de3f96e07a71a11c082fa6a",
    strip_prefix = "jpeg-9d",
    urls = ["http://www.ijg.org/files/jpegsrc.v9d.tar.gz"],
)

http_archive(
    name = "libxml_archive",
    build_file = "@//bazel:libxml.BUILD",
    sha256 = "f63c5e7d30362ed28b38bfa1ac6313f9a80230720b7fb6c80575eeab3ff5900c",
    strip_prefix = "libxml2-2.9.7",
    urls = [
        "http://xmlsoft.org/sources/libxml2-2.9.7.tar.gz",
    ],
)

http_archive(
    name = "png_archive",
    build_file = "@//bazel:png.BUILD",
    sha256 = "c2c50c13a727af73ecd3fc0167d78592cf5e0bca9611058ca414b6493339c784",
    strip_prefix = "libpng-1.6.37",
    urls = [
        "https://github.com/glennrp/libpng/archive/v1.6.37.zip",
    ],
)

http_archive(
    name = "zlib_archive",
    build_file = "@//bazel:zlib.BUILD",
    sha256 = "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
    strip_prefix = "zlib-1.3.1",
    urls = [
        "https://zlib.net/zlib-1.3.1.tar.gz",
    ],
)

http_archive(
    name = "six_archive",
    build_file = "@//bazel:six.BUILD",
    sha256 = "30639c035cdb23534cd4aa2dd52c3bf48f06e5f4a941509c8bafd8ce11080259",
    strip_prefix = "six-1.15.0",
    urls = [
        "https://pypi.python.org/packages/source/s/six/six-1.15.0.tar.gz",
    ],
)

http_archive(
    name = "lua_archive",
    build_file = "@//bazel:lua.BUILD",
    sha256 = "2640fc56a795f29d28ef15e13c34a47e223960b0240e8cb0a82d9b0738695333",
    strip_prefix = "lua-5.1.5/src",
    urls = [
        "https://www.lua.org/ftp/lua-5.1.5.tar.gz",
    ],
)

http_archive(
    name = "dm_env_archive",
    build_file = "@//bazel:dm_env.BUILD",
    strip_prefix = "dm_env-1.5",
    urls = ["https://github.com/google-deepmind/dm_env/archive/refs/tags/v1.5.tar.gz"],
)

http_archive(
    name = "tree_archive",
    build_file = "@//bazel:tree.BUILD",
    strip_prefix = "tree-0.1.9",
    urls = ["https://github.com/google-deepmind/tree/archive/refs/tags/0.1.9.tar.gz"],
)

http_archive(
    name = "pybind11_archive",
    build_file = "@//bazel:pybind11.BUILD",
    strip_prefix = "pybind11-2.13.1",
    urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.13.1.tar.gz"],
)

# TODO: Replace with hermetic build
new_local_repository(
    name = "sdl_system",
    build_file = "@//bazel:sdl.BUILD",
    path = "/usr",
)

python_repo(
    name = "python_system",
    py_version = "PY3",
)
