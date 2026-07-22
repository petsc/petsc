#include <petscmacros.h>
/* conditionally compile the CPU only code if PETSc is not configured with CUDA or HIP */
#if !PetscDefined(HAVE_CUDA) && !PetscDefined(HAVE_HIP)
  #include "../src/mat/impls/h2opus/cuda/math2opusutils.cu"
#endif
