#pragma once

/* MANSEC = Sys */

/*
    The pragma below silence all compiler warnings coming from code in this header file.
    In particular, it silences `-Wfloat-equal` warnings in `operator==()` and `operator!=` below.
    Other compilers beyond GCC support this pragma.
*/
#if defined(__GNUC__) && (__GNUC__ >= 4) && !defined(__NEC__)
  #pragma GCC system_header
#endif

/*
     Defines additional operator overloading for the C++ complex class that are "missing" in the standard
     include files. For example, the code fragment

     std::complex<double> c = 22.0;
     c = 11 + c;

     will produce a compile time error such as

     error: no match for 'operator+' (operand types are 'int' and 'std::complex<double>')

     The code fragment

     std::complex<float> c = 22.0;
     c = 11.0 + c;

     will produce a compile time error such as

     error: no match for 'operator+' (operand types are 'double' and 'std::complex<float>')

     This deficiency means one may need to write cumbersome code while working with the C++ complex classes.

     This include file defines a few additional operator overload methods for the C++ complex classes to handle
     these cases naturally within PETSc code.

     This file is included in petscsystypes.h when feasible. In the small number of cases where these additional methods
     may conflict with other code one may add '#define PETSC_SKIP_CXX_COMPLEX_FIX 1' before including any PETSc include
     files to prevent these methods from being provided.
*/

#define PETSC_CXX_COMPLEX_FIX(Type) \
  static inline PetscComplex operator+(const PetscComplex &lhs, const Type &rhs) \
  { \
    return lhs + PetscReal(rhs); \
  } \
  static inline PetscComplex operator+(const Type &lhs, const PetscComplex &rhs) \
  { \
    return PetscReal(lhs) + rhs; \
  } \
  static inline PetscComplex operator-(const PetscComplex &lhs, const Type &rhs) \
  { \
    return lhs - PetscReal(rhs); \
  } \
  static inline PetscComplex operator-(const Type &lhs, const PetscComplex &rhs) \
  { \
    return PetscReal(lhs) - rhs; \
  } \
  static inline PetscComplex operator*(const PetscComplex &lhs, const Type &rhs) \
  { \
    return lhs * PetscReal(rhs); \
  } \
  static inline PetscComplex operator*(const Type &lhs, const PetscComplex &rhs) \
  { \
    return PetscReal(lhs) * rhs; \
  } \
  static inline PetscComplex operator/(const PetscComplex &lhs, const Type &rhs) \
  { \
    return lhs / PetscReal(rhs); \
  } \
  static inline PetscComplex operator/(const Type &lhs, const PetscComplex &rhs) \
  { \
    return PetscReal(lhs) / rhs; \
  } \
  static inline bool operator==(const PetscComplex &lhs, const Type &rhs) \
  { \
    return lhs.imag() == PetscReal(0) && lhs.real() == PetscReal(rhs); \
  } \
  static inline bool operator==(const Type &lhs, const PetscComplex &rhs) \
  { \
    return rhs.imag() == PetscReal(0) && rhs.real() == PetscReal(lhs); \
  } \
  static inline bool operator!=(const PetscComplex &lhs, const Type &rhs) \
  { \
    return lhs.imag() != PetscReal(0) || lhs.real() != PetscReal(rhs); \
  } \
  static inline bool operator!=(const Type &lhs, const PetscComplex &rhs) \
  { \
    return rhs.imag() != PetscReal(0) || rhs.real() != PetscReal(lhs); \
  } \
/* PETSC_CXX_COMPLEX_FIX */

// In PETSc, a quad precision PetscComplex is a C type even with clanguage=cxx, therefore no C++ operator overloading needed for it.
#if !defined(PETSC_USE_REAL___FLOAT128)

// Provide operator overloading for 'PetscComplex .op. (an integer type or a real type but not PetscReal)'.
//
// We enumerate all C/C++ POD (Plain Old Data) types to provide exact overload resolution, to keep the precision change
// in the Type to PetscReal conversion intact, as intended by users performing these mixed precision operations.
  #if !defined(PETSC_USE_REAL___FP16) && defined(PETSC_HAVE_REAL___FP16)
PETSC_CXX_COMPLEX_FIX(__fp16)
  #endif

  #if !defined(PETSC_USE_REAL_SINGLE)
PETSC_CXX_COMPLEX_FIX(float)
  #endif

  #if !defined(PETSC_USE_REAL_DOUBLE)
PETSC_CXX_COMPLEX_FIX(double)
  #endif

PETSC_CXX_COMPLEX_FIX(long double)

  #if defined(PETSC_HAVE_REAL___FLOAT128)
PETSC_CXX_COMPLEX_FIX(__float128)
  #endif

PETSC_CXX_COMPLEX_FIX(signed char)
PETSC_CXX_COMPLEX_FIX(short)
PETSC_CXX_COMPLEX_FIX(int)
PETSC_CXX_COMPLEX_FIX(long)
PETSC_CXX_COMPLEX_FIX(long long)

PETSC_CXX_COMPLEX_FIX(unsigned char)
PETSC_CXX_COMPLEX_FIX(unsigned short)
PETSC_CXX_COMPLEX_FIX(unsigned int)
PETSC_CXX_COMPLEX_FIX(unsigned long)
PETSC_CXX_COMPLEX_FIX(unsigned long long)

#endif
