#
#  This is a SAMPLE CMakeLists.txt suitable for direct use with a new PETSc application or
#  you can add fragments of the material below to an existing application's CMakeLists.txt
#
#  Usage:  To build ex1 from ex1.c
#    rm -fr build
#    cmake -B build
#    cmake --build build
#    ./build/ex1
#
#  By default it gets both the compiler information and the library information from PETSc
#
cmake_minimum_required(VERSION 3.1.0)
include(FindPkgConfig REQUIRED)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Dummy project to determine compiler names and version
project(detect_compilers LANGUAGES)

# set root of location to find PETSc's pkg-config
set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)

# Remove the lines below if you do not wish to have PETSc determine the compilers
pkg_get_variable(CMAKE_C_COMPILER PETSc ccompiler)
pkg_get_variable(CXX_COMPILER PETSc cxxcompiler)
if (CXX_COMPILER)
  set(CMAKE_CXX_COMPILER ${CXX_COMPILER})
  enable_language(CXX)
endif (CXX_COMPILER)
pkg_get_variable(FORTRAN_COMPILER PETSc fcompiler)
if (FORTRAN_COMPILER)
  set(CMAKE_Fortran_COMPILER ${FORTRAN_COMPILER})
  enable_language(Fortran)
endif (FORTRAN_COMPILER)

#  tells CMake to build the application ex1 from the source file ex1.c
#  this must appear AFTER the compilers are set
project(ex1)
add_executable(ex1 ex1.c)

find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
target_link_libraries(ex1 PkgConfig::PETSC)
