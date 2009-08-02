#include "private/fortranimpl.h"
#include "petscmesh.h"
#include "../src/sys/f90-src/f90impl.h"

#ifdef PETSC_USE_POINTER_CONVERSION
#if defined(__cplusplus)
extern "C" { 
#endif 
extern void *PetscToPointer(void*);
extern int PetscFromPointer(void *);
extern void PetscRmPointer(void*);
#if defined(__cplusplus)
} 
#endif 

#else

#define PetscToPointer(a) (*(long *)(a))
#define PetscFromPointer(a) (long)(a)
#define PetscRmPointer(a)
#endif

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define sectionrealrestrictclosure_ SECTIONREALRESTRICTCLOSURE
#define sectionintrestrictclosure_  SECTIONINTRESTRICTCLOSURE
#define sectionrealupdateclosure_   SECTIONREALUPDATECLOSURE
#define sectionintupdateclosure_    SECTIONINTUPDATECLOSURE
#define sectiongetarrayf90_         SECTIONGETARRAYF90
#define sectiongetarray1df90_       SECTIONGETARRAY1DF90
#define bcsectiongetarrayf90_       BCSECTIONGETARRAYF90
#define bcsectiongetarray1df90_     BCSECTIONGETARRAY1DF90
#define bcsectionrealgetarrayf90_   BCSECTIONREALGETARRAYF90
#define bcfuncgetarrayf90_          BCFUNCGETARRAYF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define sectionrealrestrictclosure_ sectionrealrestrictclosure
#define sectionintrestrictclosure_  sectionintrestrictclosure
#define meshupdateclosure_          sectionrealupdateclosure
#define meshupdateclosureint_       sectionintupdateclosure
#define sectiongetarrayf90_         sectiongetarrayf90
#define sectiongetarray1df90_       sectiongetarray1df90
#define bcsectiongetarrayf90_       bcsectiongetarrayf90
#define bcsectiongetarray1df90_     bcsectiongetarray1df90
#define bcsectionrealgetarrayf90_   bcsectionrealgetarrayf90
#define bcfuncgetarrayf90_          bcfuncgetarrayf90
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL sectionrealrestrictclosure_(SectionReal section, Mesh mesh, int *point,int *size,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *c;

  // Should be able to get array size
  *ierr = F90Array1dAccess(ptr, PETSC_SCALAR, (void**) &c PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = SectionRealRestrictClosure((SectionReal) PetscToPointer(section), (Mesh) PetscToPointer(mesh), *point,*size,c); if (*ierr) return;
  // *ierr = F90Array1dCreate(const_cast<PetscScalar *>(c),PETSC_SCALAR,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL sectionintrestrictclosure_(SectionInt section, Mesh mesh, int *point,int *size,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *c;

  // Should be able to get array size
  *ierr = F90Array1dAccess(ptr, PETSC_INT, (void**) &c PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = SectionIntRestrictClosure((SectionInt) PetscToPointer(section),(Mesh) PetscToPointer(mesh), *point,*size,c); if (*ierr) return;
  // *ierr = F90Array1dCreate(const_cast<PetscScalar *>(c),PETSC_SCALAR,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL sectionrealupdateclosure_(SectionReal section, Mesh mesh, int *point,F90Array1d *ptr,InsertMode *mode,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *c;

  *ierr = F90Array1dAccess(ptr, PETSC_SCALAR, (void**) &c PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = SectionRealUpdateClosure((SectionReal) PetscToPointer(section),(Mesh) PetscToPointer(mesh), *point,c,*mode); if (*ierr) return;
}
void PETSC_STDCALL sectionintupdateclosure_(SectionInt section, Mesh mesh, int *point,F90Array1d *ptr,InsertMode *mode,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *c;

  *ierr = F90Array1dAccess(ptr, PETSC_INT, (void**) &c PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = SectionIntUpdateClosure((SectionInt) PetscToPointer(section),(Mesh) PetscToPointer(mesh), *point,c,*mode); if (*ierr) return;
}
void PETSC_STDCALL sectiongetarrayf90_(Mesh *mesh,CHAR name PETSC_MIXED_LEN(len),F90Array2d *ptr,int *ierr PETSC_END_LEN(len) PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *a;
  PetscInt     n, d;
  char        *nF;
  FIXCHAR(name,len,nF);
  *ierr = SectionGetArray(*mesh,nF,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array2dCreate(a,PETSC_SCALAR,1,d,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
  FREECHAR(name,nF);
}
void PETSC_STDCALL sectiongetarray1df90_(Mesh *mesh,CHAR name PETSC_MIXED_LEN(len),F90Array1d *ptr,int *ierr PETSC_END_LEN(len) PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *a;
  PetscInt     n, d;
  char        *nF;
  FIXCHAR(name,len,nF);
  *ierr = SectionGetArray(*mesh,nF,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array1dCreate(a,PETSC_SCALAR,1,n*d,ptr PETSC_F90_2PTR_PARAM(ptrd));
  FREECHAR(name,nF);
}
void PETSC_STDCALL bcsectiongetarrayf90_(Mesh *mesh,CHAR name PETSC_MIXED_LEN(len),F90Array2d *ptr,int *ierr PETSC_END_LEN(len) PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *a;
  PetscInt  n, d;
  char     *nF;
  FIXCHAR(name,len,nF);
  *ierr = BCSectionGetArray(*mesh,nF,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array2dCreate(a,PETSC_INT,1,d,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
  FREECHAR(name,nF);
}
void PETSC_STDCALL bcsectiongetarray1df90_(Mesh *mesh,CHAR name PETSC_MIXED_LEN(len),F90Array1d *ptr,int *ierr PETSC_END_LEN(len) PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *a;
  PetscInt  n, d;
  char     *nF;
  FIXCHAR(name,len,nF);
  *ierr = BCSectionGetArray(*mesh,nF,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array1dCreate(a,PETSC_INT,1,n*d,ptr PETSC_F90_2PTR_PARAM(ptrd));
  FREECHAR(name,nF);
}
void PETSC_STDCALL bcsectionrealgetarrayf90_(Mesh *mesh,CHAR name PETSC_MIXED_LEN(len),F90Array2d *ptr,int *ierr PETSC_END_LEN(len) PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal *a;
  PetscInt   n, d;
  char      *nF;
  FIXCHAR(name,len,nF);
  *ierr = BCSectionRealGetArray(*mesh,nF,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array2dCreate(a,PETSC_SCALAR,1,d,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
  FREECHAR(name,nF);
}
void PETSC_STDCALL bcfuncgetarrayf90_(Mesh *mesh,F90Array2d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *a;
  PetscInt     n, d;
  *ierr = BCFUNCGetArray(*mesh,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array2dCreate(a,PETSC_SCALAR,1,d,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
}

EXTERN_C_END
