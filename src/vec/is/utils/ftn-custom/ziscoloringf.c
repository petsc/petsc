#include "private/fortranimpl.h"
#include "petscis.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define iscoloringview_        ISCOLORINGVIEW
#define iscoloringcreate_      ISCOLORINGCREATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define iscoloringview_        iscoloringview
#define iscoloringcreate_      iscoloringcreate
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL iscoloringview_(ISColoring *iscoloring,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = ISColoringView(*iscoloring,v);
}



void PETSC_STDCALL iscoloringcreate_(MPI_Comm *comm,PetscInt *n,PetscInt *ncolors,PetscInt *colors,ISColoring *iscoloring,PetscErrorCode *ierr)
{
  ISColoringValue *color;
  PetscInt             i;

  /* copies the colors[] array since that is kept by the ISColoring that is created */
  *ierr = PetscMalloc((*n+1)*sizeof(ISColoringValue),&color);if (*ierr) return;
  for (i=0; i<(*n); i++) {
    if (colors[i] > IS_COLORING_MAX) {
      *ierr = PetscError(__LINE__,"ISColoringCreate_Fortran",__FILE__,__SDIR__,1,1,"Color too large");
      return;
    }
    if (colors[i] < 0) {
      *ierr = PetscError(__LINE__,"ISColoringCreate_Fortran",__FILE__,__SDIR__,1,1,"Color cannot be negative");
      return;
    }
    color[i] = (ISColoringValue)colors[i];
  }
  *ierr = ISColoringCreate(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*n,*ncolors,color,iscoloring);
}

EXTERN_C_END
