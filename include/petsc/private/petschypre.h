#pragma once

#include <petscsys.h>
#include <petscpkg_version.h>
#include <HYPRE_config.h>
#include <HYPRE_utilities.h>

/* from version 2.16 on, HYPRE_BigInt is 64-bit for 64-bit pointer installations
   and 32-bit for 32-bit installations -> not the best name for a variable */
#if PETSC_PKG_HYPRE_VERSION_LT(2, 16, 0)
typedef PetscInt HYPRE_BigInt;
#endif

#if defined(HYPRE_BIGINT) || defined(HYPRE_MIXEDINT)
  #define PetscHYPRE_BigInt_FMT "lld"
  #ifdef __cplusplus /* make sure our format specifiers line up */
    #include <type_traits>
static_assert(std::is_same<HYPRE_BigInt, long long int>::value, "");
  #endif
#else
  #define PetscHYPRE_BigInt_FMT "d"
  #ifdef __cplusplus /* make sure our format specifiers line up */
    #include <type_traits>
static_assert(std::is_same<HYPRE_BigInt, int>::value, "");
  #endif
#endif

#if defined(PETSC_CLANG_STATIC_ANALYZER)
void PetscCallHYPRE(HYPRE_Int);
#else
  #if PETSC_PKG_HYPRE_VERSION_GT(2, 0, 0) /* HYPRE_DescribeError() added in 2.0.0 */
    #define PetscCallHYPRE(...) \
      do { \
        if (PetscUnlikelyDebug(HYPRE_GetError() != 0)) { \
          char err_str[PETSC_MAX_PATH_LEN]; \
          HYPRE_DescribeError(HYPRE_GetError(), err_str); \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "HYPRE error from previous HYPRE call, error code %" PetscInt_FMT ": %s", (PetscInt)HYPRE_GetError(), err_str); \
        } \
        HYPRE_Int ierr_hypre_ = __VA_ARGS__; \
        if (PetscUnlikely(ierr_hypre_ != 0)) { \
          char err_str[PETSC_MAX_PATH_LEN]; \
          HYPRE_DescribeError(ierr_hypre_, err_str); \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "HYPRE error code %" PetscInt_FMT ": %s", (PetscInt)ierr_hypre_, err_str); \
        } \
      } while (0)
  #elif PETSC_PKG_HYPRE_VERSION_GE(1, 11, 0) /* HYPRE_GetError() added in 1.11.0b */
    #define PetscCallHYPRE(...) \
      do { \
        PetscAssert(HYPRE_GetError() == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "HYPRE error from previous HYPRE call, error code %" PetscInt_FMT, (PetscInt)HYPRE_GetError()); \
        HYPRE_Int ierr_hypre_ = __VA_ARGS__; \
        PetscCheck(ierr_hypre_ == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "HYPRE error code %" PetscInt_FMT, (PetscInt)ierr_hypre_); \
      } while (0)
  #else /* PETSC_PKG_HYPRE_VERSION_LT(1, 11, 0) */
    #define PetscCallHYPRE(...) \
      do { \
        HYPRE_Int ierr_hypre_ = __VA_ARGS__; \
        PetscCheck(ierr_hypre_ == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "HYPRE error code %" PetscInt_FMT, (PetscInt)ierr_hypre_); \
      } while (0)
  #endif /* PETSC_PKG_HYPRE_VERSION */
#endif   /* PETSC_CLANG_STATIC_ANALYZER */

/*
  With scalar type == real, HYPRE_Complex == PetscScalar;
  With scalar type == complex,  HYPRE_Complex is double __complex__ while PetscScalar may be std::complex<double>
*/
static inline PetscErrorCode PetscHYPREScalarCast(PetscScalar a, HYPRE_Complex *b)
{
  PetscFunctionBegin;
#if defined(HYPRE_COMPLEX)
  ((PetscReal *)b)[0] = PetscRealPart(a);
  ((PetscReal *)b)[1] = PetscImaginaryPart(a);
#else
  *b = a;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PETSC_PKG_HYPRE_VERSION_GT(2, 28, 0) || (PETSC_PKG_HYPRE_VERSION_EQ(2, 28, 0) && defined(HYPRE_DEVELOP_NUMBER) && HYPRE_DEVELOP_NUMBER >= 22)
static inline PetscErrorCode PetscHYPREFinalize_Private(void)
{
  PetscFunctionBegin;
  if (HYPRE_Initialized() && !HYPRE_Finalized()) PetscCallHYPRE(HYPRE_Finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscBool use_gpu_aware_mpi;
/*
  Options Database Keys:
. -hypre_umpire_device_pool_size <n>  - set the Umpire device memory pool size (in MiB). HYPRE uses 4 GiB by default.
*/
static inline PetscErrorCode PetscHYPREInitialize(void)
{
  PetscFunctionBegin;
  if (!HYPRE_Initialized()) {
    PetscCallHYPRE(HYPRE_Initialize());
  #if defined(HYPRE_USING_UMPIRE_DEVICE)
    PetscInt  size = 0;
    PetscBool set  = PETSC_FALSE;

    PetscCall(PetscOptionsGetInt(NULL, NULL, "-hypre_umpire_device_pool_size", &size, &set));
    if (set) {
      PetscCheck(size >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "-hypre_umpire_device_pool_size must be non-negative");
      PetscCallHYPRE(HYPRE_SetUmpireDevicePoolSize((size_t)(size << 20))); // converr to size in bytes
    } else if (PetscCIEnabled) {
      // Set smaller pool size for CI tests to avoid OOM, but keep the default for users
      PetscCallHYPRE(HYPRE_SetUmpireDevicePoolSize((size_t)(256 << 20))); // 256 MiB
    }
  #endif
  // hypre-2.31.0 added HYPRE_SetGpuAwareMPI.
  // HYPRE_USING_GPU_AWARE_MPI indicates hypre is configured with --enable-gpu-aware-mpi.
  // HYPRE_SetGpuAwareMPI() controls whether to actually use GPU-aware MPI.
  #if PETSC_PKG_HYPRE_VERSION_GE(2, 31, 0) && defined(HYPRE_USING_GPU_AWARE_MPI)
    PetscCallHYPRE(HYPRE_SetGpuAwareMPI(use_gpu_aware_mpi ? 1 : 0));
  #endif
    PetscCall(PetscRegisterFinalize(PetscHYPREFinalize_Private));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#else
static inline PetscErrorCode PetscHYPREInitialize(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

#if PETSC_PKG_HYPRE_VERSION_LT(2, 19, 0)
typedef int HYPRE_MemoryLocation;
  #define hypre_IJVectorMemoryLocation(a) 0
  #define hypre_IJMatrixMemoryLocation(a) 0
#endif
