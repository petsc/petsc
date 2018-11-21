#include <petsc/private/fortranimpl.h>
#include <petscviewerhdf5.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewerhdf5open_            PETSCVIEWERHDF5OPEN
#define petscviewerhdf5pushgroup_       PETSCVIEWERHDF5PUSHGROUP
#define petscviewerhdf5getgroup_        PETSCVIEWERHDF5GETGROUP
#define petscviewerhdf5hasattribute_    PETSCVIEWERHDF5HASATTRIBUTE
#define petscviewerhdf5readsizes_       PETSCVIEWERHDF5READSIZES
#define petscviewerhdf5writeattribute_  PETSCVIEWERHDF5WRITEATTRIBUTE
#define petscviewerhdf5readattribute_   PETSCVIEWERHDF5READATTRIBUTE
#define petscviewerhdf5setaijnames_     PETSCVIEWERHDF5SETAIJNAMES
#define petscviewerhdf5getaijnames_     PETSCVIEWERHDF5GETAIJNAMES
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerhdf5open_            petscviewerhdf5open
#define petscviewerhdf5pushgroup_       petscviewerhdf5pushgroup
#define petscviewerhdf5getgroup_        petscviewerhdf5getgroup
#define petscviewerhdf5hasattribute_    petscviewerhdf5hasattribute
#define petscviewerhdf5readsizes_       petscviewerhdf5readsizes
#define petscviewerhdf5writeattribute_  petscviewerhdf5writeattribute
#define petscviewerhdf5readattribute_   petscviewerhdf5readattribute
#define petscviewerhdf5setaijnames_     petscviewerhdf5setaijnames
#define petscviewerhdf5getaijnames_     petscviewerhdf5getaijnames
#endif

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5open_(MPI_Comm *comm, char* name PETSC_MIXED_LEN(len), PetscFileMode *type,
    PetscViewer *binv, PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(name, len, c1);
  *ierr = PetscViewerHDF5Open(MPI_Comm_f2c(*(MPI_Fint*)&*comm), c1, *type, binv);if (*ierr) return;
  FREECHAR(name, c1);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5pushgroup_(PetscViewer *viewer, char* name PETSC_MIXED_LEN(len),
    PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(name, len, c1);
  *ierr = PetscViewerHDF5PushGroup(*viewer, c1);if (*ierr) return;
  FREECHAR(name, c1);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5getgroup_(PetscViewer *viewer, char* name PETSC_MIXED_LEN(len),
    PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *c1;

  *ierr = PetscViewerHDF5GetGroup(*viewer, &c1);if (*ierr) return;
  *ierr = PetscStrncpy(name, c1, len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5hasattribute_(PetscViewer *viewer, char* parent PETSC_MIXED_LEN(plen),
    char* name PETSC_MIXED_LEN(nlen), PetscBool *has, PetscErrorCode *ierr PETSC_END_LEN(plen) PETSC_END_LEN(nlen))
{
   char *c1, *c2;

   FIXCHAR(parent, plen, c1);
   FIXCHAR(name, nlen, c2);
   *ierr = PetscViewerHDF5HasAttribute(*viewer, c1, c2, has);if (*ierr) return;
   FREECHAR(parent, c1);
   FREECHAR(name, c2);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5readsizes_(PetscViewer *viewer, char* name PETSC_MIXED_LEN(len),
    PetscInt *bs, PetscInt *N, PetscErrorCode *ierr PETSC_END_LEN(len))
{
   char *c1;

   FIXCHAR(name, len, c1);
   *ierr = PetscViewerHDF5ReadSizes(*viewer, c1, bs, N);
   FREECHAR(name, c1);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5writeattribute_(PetscViewer *viewer, char* parent PETSC_MIXED_LEN(plen),
    char* name PETSC_MIXED_LEN(nlen), PetscDataType *datatype, const void *value, PetscErrorCode *ierr PETSC_END_LEN(plen) PETSC_END_LEN(nlen))
{
   char *c1, *c2;

   FIXCHAR(parent, plen, c1);
   FIXCHAR(name, nlen, c2);
   *ierr = PetscViewerHDF5WriteAttribute(*viewer, c1, c2, *datatype, (const void *) value);if (*ierr) return;
   FREECHAR(parent, c1);
   FREECHAR(name, c2);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5readattribute_(PetscViewer *viewer, char* parent PETSC_MIXED_LEN(plen),
    char* name PETSC_MIXED_LEN(nlen), PetscDataType *datatype, void *value, PetscErrorCode *ierr PETSC_END_LEN(plen) PETSC_END_LEN(nlen))
{
   char *c1, *c2;

   FIXCHAR(parent, plen, c1);
   FIXCHAR(name, nlen, c2);
   *ierr = PetscViewerHDF5ReadAttribute(*viewer, c1, c2, *datatype, (void *) value);if (*ierr) return;
   FREECHAR(parent, c1);
   FREECHAR(name, c2);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5setaijnames_(PetscViewer *viewer,
    char* iname PETSC_MIXED_LEN(ilen),
    char* jname PETSC_MIXED_LEN(jlen),
    char* aname PETSC_MIXED_LEN(alen),
    char* cname PETSC_MIXED_LEN(clen),
    PetscErrorCode *ierr PETSC_END_LEN(ilen) PETSC_END_LEN(jlen) PETSC_END_LEN(alen) PETSC_END_LEN(clen))
{
  char *ci, *cj, *ca, *cc;
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(viewer,v);
  FIXCHAR(iname,ilen,ci);
  FIXCHAR(jname,jlen,cj);
  FIXCHAR(aname,alen,ca);
  FIXCHAR(cname,clen,cc);
  *ierr = PetscViewerHDF5SetAIJNames(v,ci,cj,ca,cc);if (*ierr) return;
  FREECHAR(iname,ci);
  FREECHAR(jname,cj);
  FREECHAR(aname,ca);
  FREECHAR(cname,cc);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerhdf5getaijnames_(PetscViewer *viewer,
    char* iname PETSC_MIXED_LEN(ilen),
    char* jname PETSC_MIXED_LEN(jlen),
    char* aname PETSC_MIXED_LEN(alen),
    char* cname PETSC_MIXED_LEN(clen),
    PetscErrorCode *ierr PETSC_END_LEN(ilen) PETSC_END_LEN(jlen) PETSC_END_LEN(alen) PETSC_END_LEN(clen))
{
  const char *ci, *cj, *ca, *cc;

  *ierr = PetscViewerHDF5GetAIJNames(*viewer,&ci,&cj,&ca,&cc);if (*ierr) return;
  *ierr = PetscStrncpy(iname,ci,ilen);if (*ierr) return;
  *ierr = PetscStrncpy(jname,cj,jlen);if (*ierr) return;
  *ierr = PetscStrncpy(aname,ca,alen);if (*ierr) return;
  *ierr = PetscStrncpy(cname,cc,clen);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,iname,ilen);
  FIXRETURNCHAR(PETSC_TRUE,jname,jlen);
  FIXRETURNCHAR(PETSC_TRUE,aname,alen);
  FIXRETURNCHAR(PETSC_TRUE,cname,clen);
}
