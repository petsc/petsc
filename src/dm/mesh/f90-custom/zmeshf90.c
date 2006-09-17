#include "zpetsc.h"
#include "petscmesh.h"
#include "src/sys/f90/f90impl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define meshgetcoordinatesf90_     MESHGETCOORDINATESF90
#define meshrestorecoordinatesf90_ MESHRESTORECOORDINATESF90
#define meshgetelementsf90_        MESHGETELEMENTSF90
#define meshrestoreelementsf90_    MESHRESTOREELEMENTSF90
#define sectiongetarrayf90_        SECTIONGETARRAYF90
#define sectiongetarray1df90_      SECTIONGETARRAY1DF90
#define bcsectiongetarrayf90_      BCSECTIONGETARRAYF90
#define bcfuncgetarrayf90_         BCFUNCGETARRAYF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define meshgetcoordinatesf90_     meshgetcoordinatesf90
#define meshrestorecoordinatesf90_ meshrestorecoordinatesf90
#define meshgetelementsf90_        meshgetelementsf90
#define meshrestoreelementsf90_    meshrestoreelementsf90
#define sectiongetarrayf90_        sectiongetarrayf90
#define sectiongetarray1df90_      sectiongetarray1df90
#define bcsectiongetarrayf90_      bcsectiongetarrayf90
#define bcfuncgetarrayf90_         bcfuncgetarrayf90
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL meshgetcoordinatesf90_(Mesh *mesh,F90Array2d *ptr,int *__ierr)
{
  PetscReal *c;
  PetscInt   n, d;
  *__ierr = MeshGetCoordinates(*mesh,PETSC_TRUE,&n,&d,&c); if (*__ierr) return;
  *__ierr = F90Array2dCreate(c,PETSC_REAL,1,n,1,d,ptr);
}
void PETSC_STDCALL meshrestorecoordinatesf90_(Mesh *x,F90Array2d *ptr,int *__ierr)
{
  PetscReal *c;
  *__ierr = F90Array2dAccess(ptr,(void**)&c);if (*__ierr) return;
  *__ierr = F90Array2dDestroy(ptr);if (*__ierr) return;
  *__ierr = PetscFree(c);
}
void PETSC_STDCALL meshgetelementsf90_(Mesh *mesh,F90Array2d *ptr,int *__ierr)
{
  PetscInt   *v;
  PetscInt   n, c;
  *__ierr = MeshGetElements(*mesh,PETSC_TRUE,&n,&c,&v); if (*__ierr) return;
  *__ierr = F90Array2dCreate(v,PETSC_INT,1,n,1,c,ptr);
}
void PETSC_STDCALL meshrestoreelementsf90_(Mesh *x,F90Array2d *ptr,int *__ierr)
{
  PetscInt   *v;
  *__ierr = F90Array2dAccess(ptr,(void**)&v);if (*__ierr) return;
  *__ierr = F90Array2dDestroy(ptr);if (*__ierr) return;
  *__ierr = PetscFree(v);
}
void PETSC_STDCALL sectiongetarrayf90_(Mesh *mesh,CHAR name PETSC_MIXED_LEN(len),F90Array2d *ptr,int *ierr PETSC_END_LEN(len))
{
  PetscScalar *a;
  PetscInt     n, d;
  char        *nF;
  FIXCHAR(name,len,nF);
  *ierr = SectionGetArray(*mesh,nF,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array2dCreate(a,PETSC_SCALAR,1,d,1,n,ptr);
  FREECHAR(name,nF);
}
void PETSC_STDCALL sectiongetarray1df90_(Mesh *mesh,CHAR name PETSC_MIXED_LEN(len),F90Array1d *ptr,int *ierr PETSC_END_LEN(len))
{
  PetscScalar *a;
  PetscInt     n, d;
  char        *nF;
  FIXCHAR(name,len,nF);
  *ierr = SectionGetArray(*mesh,nF,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array1dCreate(a,PETSC_SCALAR,1,n*d,ptr);
  FREECHAR(name,nF);
}
void PETSC_STDCALL bcsectiongetarrayf90_(Mesh *mesh,CHAR name PETSC_MIXED_LEN(len),F90Array2d *ptr,int *ierr PETSC_END_LEN(len))
{
  PetscInt *a;
  PetscInt  n, d;
  char     *nF;
  FIXCHAR(name,len,nF);
  *ierr = BCSectionGetArray(*mesh,nF,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array2dCreate(a,PETSC_INT,1,d,1,n,ptr);
  FREECHAR(name,nF);
}
void PETSC_STDCALL bcfuncgetarrayf90_(Mesh *mesh,F90Array2d *ptr,int *ierr)
{
  PetscScalar *a;
  PetscInt     n, d;
  *ierr = BCFUNCGetArray(*mesh,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array2dCreate(a,PETSC_SCALAR,1,d,1,n,ptr);
}

EXTERN_C_END
