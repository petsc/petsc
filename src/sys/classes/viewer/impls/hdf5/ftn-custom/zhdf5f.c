#include <petsc/private/fortranimpl.h>
#include <petscviewerhdf5.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewerhdf5open_        PETSCVIEWERHDF5OPEN
#define petscviewerhdf5pushgroup_   PETSCVIEWERHDF5PUSHGROUP
#define petscviewerhdf5getgroup_    PETSCVIEWERHDF5GETGROUP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerhdf5open_        petscviewerhdf5open
#define petscviewerhdf5pushgroup_   petscviewerhdf5pushgroup
#define petscviewerhdf5getgroup_    petscviewerhdf5getgroup
#endif

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5open_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len),PetscFileMode *type,
                           PetscViewer *binv,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerHDF5Open(MPI_Comm_f2c(*(MPI_Fint*)&*comm),c1,*type,binv);
  FREECHAR(name,c1);
}

PETSC_EXTERN void PETSC_STDCALL  petscviewerhdf5pushgroup_(PetscViewer *viewer, CHAR name PETSC_MIXED_LEN(len),
                           PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerHDF5PushGroup(*viewer,c1);
  FREECHAR(name,c1);
}

PETSC_EXTERN void PETSC_STDCALL  petscviewerhdf5getgroup_(PetscViewer *viewer, CHAR name PETSC_MIXED_LEN(len),
                           PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *c1;
  *ierr = PetscViewerHDF5GetGroup(*viewer,&c1);
  *ierr = PetscStrncpy(name,c1,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}
