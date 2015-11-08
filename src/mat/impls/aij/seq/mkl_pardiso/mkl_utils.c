/* mkl.h could conflict with petscblaslapack.h in mkl_pardiso.c */
/* The current translation unit contains calls specific to mkl library */
#include <mkl.h>

void PetscSetMKL_PARDISOThreads(int threads)
{
  mkl_domain_set_num_threads(threads,MKL_DOMAIN_PARDISO);
}
