#include <../src/mat/impls/adj/mpi/mpiadj.h>
#include <petsc/private/ftnimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matpartitioningsetvertexweights_ MATPARTITIONINGSETVERTEXWEIGHTS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matpartitioningsetvertexweights_ matpartitioningsetvertexweights
#endif

PETSC_EXTERN void matpartitioningsetvertexweights_NOTODAY(MatPartitioning *part, const PetscInt weights[], PetscErrorCode *ierr)
{
  PetscInt  len;
  PetscInt *array;
  *ierr = MatGetLocalSize((*part)->adj, &len, NULL);
  if (*ierr) return;
  *ierr = PetscMalloc1(len, &array);
  if (*ierr) return;
  *ierr = PetscArraycpy(array, weights, len);
  if (*ierr) return;
  *ierr = MatPartitioningSetVertexWeights(*part, array);
}
