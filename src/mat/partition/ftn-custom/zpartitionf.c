#include <../src/mat/impls/adj/mpi/mpiadj.h>
#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matpartitioningsetvertexweights_ MATPARTITIONINGSETVERTEXWEIGHTS
#define matpartitioningview_             MATPARTITIONINGVIEW
#define matpartitioningsettype_          MATPARTITIONINGSETTYPE
#define matpartitioningviewfromoptions_  MATPARTITIONINGVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matpartitioningsetvertexweights_ matpartitioningsetvertexweights
#define matpartitioningview_             matpartitioningview
#define matpartitioningsettype_          matpartitioningsettype
#define matpartitioningviewfromoptions_  matpartitioningviewfromoptions
#endif

PETSC_EXTERN void matpartitioningsetvertexweights_(MatPartitioning *part,const PetscInt weights[],PetscErrorCode *ierr)
{
  PetscInt len;
  PetscInt *array;
  *ierr = MatGetLocalSize((*part)->adj,&len,0); if (*ierr) return;
  *ierr = PetscMalloc1(len,&array); if (*ierr) return;
  *ierr = PetscArraycpy(array,weights,len);if (*ierr) return;
  *ierr = MatPartitioningSetVertexWeights(*part,array);
}
PETSC_EXTERN void matpartitioningview_(MatPartitioning *part,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = MatPartitioningView(*part,v);
}

PETSC_EXTERN void matpartitioningsettype_(MatPartitioning *part,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(type,len,t);
  *ierr = MatPartitioningSetType(*part,t);if (*ierr) return;
  FREECHAR(type,t);
}
PETSC_EXTERN void matpartitioningviewfromoptions_(MatPartitioning *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = MatPartitioningViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

