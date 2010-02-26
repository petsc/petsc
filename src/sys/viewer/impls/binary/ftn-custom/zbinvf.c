#include "private/fortranimpl.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewerfilesetmode_    PETSCVIEWERFILESETMODE
#define petscviewerbinaryopen_     PETSCVIEWERBINARYOPEN
#define petscviewerbinarygetdescriptor_     PETSCVIEWERBINARYGETDESCRIPTOR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerfilesetmode_    petscviewerfilesetmode
#define petscviewerbinaryopen_     petscviewerbinaryopen
#define petscviewerbinarygetdescriptor_     petscviewerbinarygetdescriptor
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL  petscviewerfilesetmode_(PetscViewer *viewer,PetscFileMode *type,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerFileSetMode(v,*type);
}

void PETSC_STDCALL petscviewerbinaryopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len),PetscFileMode *type,
                           PetscViewer *binv,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char   *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerBinaryOpen(MPI_Comm_f2c(*(MPI_Fint *)&*comm),c1,*type,binv);
  FREECHAR(name,c1);
}

void PETSC_STDCALL  petscviewerbinarygetdescriptor_(PetscViewer *viewer,int *fd,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerBinaryGetDescriptor(v,fd);
}

EXTERN_C_END
