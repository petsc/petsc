#include <../src/mat/impls/adj/mpi/mpiadj.h>
#include <petsc-private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matpartitioningsetvertexweights_ MATPARTITIONINGSETVERTEXWEIGHTS
#define matpartitioningview_             MATPARTITIONINGVIEW
#define matpartitioningsettype_          MATPARTITIONINGSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matpartitioningsetvertexweights_ matpartitioningsetvertexweights
#define matpartitioningview_             matpartitioningview
#define matpartitioningsettype_          matpartitioningsettype
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL matpartitioningsetvertexweights_(MatPartitioning *part,const PetscInt weights[],PetscErrorCode *ierr)
{
  PetscInt len;
  PetscInt *array;
  *ierr = MatGetLocalSize((*part)->adj,&len,0); if (*ierr) return;
  *ierr = PetscMalloc(len*sizeof(PetscInt),&array); if (*ierr) return;
  *ierr = PetscMemcpy(array,weights,len*sizeof(PetscInt));if (*ierr) return;
  *ierr = MatPartitioningSetVertexWeights(*part,array);
}
void PETSC_STDCALL matpartitioningview_(MatPartitioning  *part,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = MatPartitioningView(*part,v);
}

void PETSC_STDCALL matpartitioningsettype_(MatPartitioning *part,CHAR type PETSC_MIXED_LEN(len),
                                           PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(type,len,t);
  *ierr = MatPartitioningSetType(*part,t);
  FREECHAR(type,t);
}

EXTERN_C_END
