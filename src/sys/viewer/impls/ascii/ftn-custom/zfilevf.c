#include "zpetsc.h"
#include "petsc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewersetfilename_    PETSCVIEWERSETFILENAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewersetfilename_    petscviewersetfilename
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petscviewersetfilename_(PetscViewer *viewer,CHAR name PETSC_MIXED_LEN(len),
                                      PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char   *c1;
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerSetFilename(v,c1);
  FREECHAR(name,c1);
}

EXTERN_C_END
