#include <petscconf.h>
/* conditionally compile the CPU only code if PETSc is not configured with CUDA or HIP */
#if !defined(PETSC_HAVE_CUDA) && !defined(PETSC_HAVE_HIP)
  #include "../src/mat/impls/h2opus/cuda/math2opusutils.cu"
#endif
