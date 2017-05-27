#include <petsc/private/fortranimpl.h>
#include <petscviewerhdf5.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewerhdf5open_            PETSCVIEWERHDF5OPEN
#define petscviewerhdf5pushgroup_       PETSCVIEWERHDF5PUSHGROUP
#define petscviewerhdf5getgroup_        PETSCVIEWERHDF5GETGROUP
#define petscviewerhdf5writeattribute_  PETSCVIEWERHDF5WRITEATTRIBUTE
#define petscviewerhdf5readattribute_   PETSCVIEWERHDF5READATTRIBUTE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerhdf5open_            petscviewerhdf5open
#define petscviewerhdf5pushgroup_       petscviewerhdf5pushgroup
#define petscviewerhdf5getgroup_        petscviewerhdf5getgroup
#define petscviewerhdf5writeattribute_  petscviewerhdf5writeattribute
#define petscviewerhdf5readattribute_   petscviewerhdf5readattribute
#endif

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5open_(MPI_Comm *comm, char* name PETSC_MIXED_LEN(len), PetscFileMode *type,
    PetscViewer *binv, PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(name, len, c1);
  *ierr = PetscViewerHDF5Open(MPI_Comm_f2c(*(MPI_Fint*)&*comm), c1, *type, binv);
  FREECHAR(name, c1);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5pushgroup_(PetscViewer *viewer, char* name PETSC_MIXED_LEN(len),
    PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(name, len, c1);
  *ierr = PetscViewerHDF5PushGroup(*viewer, c1);
  FREECHAR(name, c1);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5getgroup_(PetscViewer *viewer, char* name PETSC_MIXED_LEN(len),
    PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *c1;

  *ierr = PetscViewerHDF5GetGroup(*viewer, &c1);
  *ierr = PetscStrncpy(name, c1, len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5hasattribute_(PetscViewer *viewer, char* parent PETSC_MIXED_LEN(plen),
    char* name PETSC_MIXED_LEN(nlen), PetscBool *has, PetscErrorCode *ierr PETSC_END_LEN(plen) PETSC_END_LEN(nlen))
{
   char *c1, *c2;

   FIXCHAR(parent, plen, c1);
   FIXCHAR(name, nlen, c2);
   *ierr = PetscViewerHDF5HasAttribute(*viewer, c1, c2, has);
   FREECHAR(parent, c1);
   FREECHAR(name, c2);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5writeattribute_(PetscViewer *viewer, char* parent PETSC_MIXED_LEN(plen),
    char* name PETSC_MIXED_LEN(nlen), PetscDataType *datatype, const void *value, PetscErrorCode *ierr PETSC_END_LEN(plen) PETSC_END_LEN(nlen))
{
   char *c1, *c2;

   FIXCHAR(parent, plen, c1);
   FIXCHAR(name, nlen, c2);
   *ierr = PetscViewerHDF5WriteAttribute(*viewer, c1, c2, *datatype, (const void *) value);
   FREECHAR(parent, c1);
   FREECHAR(name, c2);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5readattribute_(PetscViewer *viewer, char* parent PETSC_MIXED_LEN(plen),
    char* name PETSC_MIXED_LEN(nlen), PetscDataType *datatype, void *value, PetscErrorCode *ierr PETSC_END_LEN(plen) PETSC_END_LEN(nlen))
{
   char *c1, *c2;

   FIXCHAR(parent, plen, c1);
   FIXCHAR(name, nlen, c2);
   *ierr = PetscViewerHDF5ReadAttribute(*viewer, c1, c2, *datatype, (void *) value);
   FREECHAR(parent, c1);
   FREECHAR(name, c2);
}
