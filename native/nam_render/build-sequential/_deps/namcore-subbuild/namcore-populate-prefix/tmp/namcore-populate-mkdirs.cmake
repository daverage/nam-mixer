# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/andrzejmarczewski/Documents/GitHub/hybrid-nam-builder/native/nam_render/build-sequential/_deps/namcore-src")
  file(MAKE_DIRECTORY "/Users/andrzejmarczewski/Documents/GitHub/hybrid-nam-builder/native/nam_render/build-sequential/_deps/namcore-src")
endif()
file(MAKE_DIRECTORY
  "/Users/andrzejmarczewski/Documents/GitHub/hybrid-nam-builder/native/nam_render/build-sequential/_deps/namcore-build"
  "/Users/andrzejmarczewski/Documents/GitHub/hybrid-nam-builder/native/nam_render/build-sequential/_deps/namcore-subbuild/namcore-populate-prefix"
  "/Users/andrzejmarczewski/Documents/GitHub/hybrid-nam-builder/native/nam_render/build-sequential/_deps/namcore-subbuild/namcore-populate-prefix/tmp"
  "/Users/andrzejmarczewski/Documents/GitHub/hybrid-nam-builder/native/nam_render/build-sequential/_deps/namcore-subbuild/namcore-populate-prefix/src/namcore-populate-stamp"
  "/Users/andrzejmarczewski/Documents/GitHub/hybrid-nam-builder/native/nam_render/build-sequential/_deps/namcore-subbuild/namcore-populate-prefix/src"
  "/Users/andrzejmarczewski/Documents/GitHub/hybrid-nam-builder/native/nam_render/build-sequential/_deps/namcore-subbuild/namcore-populate-prefix/src/namcore-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/andrzejmarczewski/Documents/GitHub/hybrid-nam-builder/native/nam_render/build-sequential/_deps/namcore-subbuild/namcore-populate-prefix/src/namcore-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/andrzejmarczewski/Documents/GitHub/hybrid-nam-builder/native/nam_render/build-sequential/_deps/namcore-subbuild/namcore-populate-prefix/src/namcore-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
