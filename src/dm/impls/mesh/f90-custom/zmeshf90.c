#include "private/fortranimpl.h"
#include "petscdmmesh.h"
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
#define meshgetcoordinatesf90_     MESHGETCOORDINATESF90
#define meshrestorecoordinatesf90_ MESHRESTORECOORDINATESF90
#define meshgetelementsf90_        MESHGETELEMENTSF90
#define meshrestoreelementsf90_    MESHRESTOREELEMENTSF90
#define meshgetlabelids_           MESHGETLABELIDS
#define meshgetstratum_            MESHGETSTRATUM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define meshgetcoordinatesf90_     meshgetcoordinatesf90
#define meshrestorecoordinatesf90_ meshrestorecoordinatesf90
#define meshgetelementsf90_        meshgetelementsf90
#define meshrestoreelementsf90_    meshrestoreelementsf90
#define meshgetlabelids_           meshgetlabelids
#define meshgetstratum_            meshgetstratum
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL meshgetcoordinatesf90_(DM *dm,F90Array2d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal *c;
  PetscInt   n, d;
  *__ierr = DMMeshGetCoordinates(*dm,PETSC_TRUE,&n,&d,&c); if (*__ierr) return;
  *__ierr = F90Array2dCreate(c,PETSC_REAL,1,n,1,d,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL meshrestorecoordinatesf90_(DM *x,F90Array2d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal *c;
  *__ierr = F90Array2dAccess(ptr,PETSC_REAL,(void**)&c PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array2dDestroy(ptr,PETSC_REAL PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = PetscFree(c);
}
void PETSC_STDCALL meshgetelementsf90_(DM *dm,F90Array2d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt   *v;
  PetscInt   n, c;
  *__ierr = DMMeshGetElements(*dm,PETSC_TRUE,&n,&c,&v); if (*__ierr) return;
  *__ierr = F90Array2dCreate(v,PETSC_INT,1,n,1,c,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL meshrestoreelementsf90_(DM *x,F90Array2d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt   *v;
  *__ierr = F90Array2dAccess(ptr,PETSC_INT,(void**)&v PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array2dDestroy(ptr,PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = PetscFree(v);
}
void PETSC_STDCALL meshgetconef90_(DM *dm,PetscInt *p,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt   *v, n;
  *__ierr = DMMeshGetCone(*dm,*p,&n,&v); if (*__ierr) return;
  *__ierr = F90Array1dCreate(v,PETSC_INT,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL meshrestoreconef90_(DM *dm,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *__ierr = F90Array2dDestroy(ptr,PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
}

#if 0
void PETSC_STDCALL meshrestoreclosuref90_(DM dm,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal *c;
  *__ierr = F90Array1dAccess(ptr,PETSC_REAL,(void**)&c PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr,PETSC_REAL PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = PetscFree(c);
}
#endif
void PETSC_STDCALL meshgetlabelids_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), F90Array1d *ptr, int *ierr PETSC_END_LEN(lenN) PETSC_F90_2PTR_PROTO(ptrd)){
  char     *pN;
  PetscInt *ids;
  FIXCHAR(name,lenN,pN);
  *ierr = F90Array1dAccess(ptr, PETSC_INT, (void**) &ids PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = DMMeshGetLabelIds(*dm,pN, ids);
  FREECHAR(name,pN);
}
void PETSC_STDCALL meshgetstratum_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *value, F90Array1d *ptr, int *ierr PETSC_END_LEN(lenN) PETSC_F90_2PTR_PROTO(ptrd)){
  char     *pN;
  PetscInt *points;
  FIXCHAR(name,lenN,pN);
  *ierr = F90Array1dAccess(ptr, PETSC_INT, (void**) &points PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = DMMeshGetStratum(*dm,pN, *value, points);
  FREECHAR(name,pN);
}

EXTERN_C_END
