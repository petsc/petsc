#include "zpetsc.h"
#include "petsc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewersetfiletype_    PETSCVIEWERSETFILETYPE
#define petscviewerbinaryopen_     PETSCVIEWERBINARYOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewersetfiletype_    petscviewersetfiletype
#define petscviewerbinaryopen_     petscviewerbinaryopen
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL  petscviewersetfiletype_(PetscViewer *viewer,PetscViewerFileType *type,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerSetFileType(v,*type);
}

void PETSC_STDCALL petscviewerbinaryopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len),PetscViewerFileType *type,
                           PetscViewer *binv,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char   *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerBinaryOpen((MPI_Comm)PetscToPointerComm(*comm),c1,*type,binv);
  FREECHAR(name,c1);
}

EXTERN_C_END
