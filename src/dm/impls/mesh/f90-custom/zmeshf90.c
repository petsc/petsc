#include <petsc-private/fortranimpl.h>
#include <petscdmmesh.h>
#include <../src/sys/f90-src/f90impl.h>

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dmmeshgetcoordinatesf90_     DMMESHGETCOORDINATESF90
#define dmmeshrestorecoordinatesf90_ DMMESHRESTORECOORDINATESF90
#define dmmeshgetelementsf90_        DMMESHGETELEMENTSF90
#define dmmeshrestoreelementsf90_    DMMESHRESTOREELEMENTSF90
#define dmmeshgetconef90_            DMMESHGETCONEF90
#define dmmeshrestoreconef90_        DMMESHRESTORECONEF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmmeshgetcoordinatesf90_     dmmeshgetcoordinatesf90
#define dmmeshrestorecoordinatesf90_ dmmeshrestorecoordinatesf90
#define dmmeshgetelementsf90_        dmmeshgetelementsf90
#define dmmeshrestoreelementsf90_    dmmeshrestoreelementsf90
#define dmmeshgetconef90_            dmmeshgetconef90
#define dmmeshrestoreconef90_        dmmeshrestoreconef90
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL dmmeshgetcoordinatesf90_(DM *dm,F90Array2d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal *c;
  PetscInt   n, d;
  *__ierr = DMMeshGetCoordinates(*dm,PETSC_TRUE,&n,&d,&c); if (*__ierr) return;
  *__ierr = F90Array2dCreate(c,PETSC_REAL,1,n,1,d,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL dmmeshrestorecoordinatesf90_(DM *x,F90Array2d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal *c;
  *__ierr = F90Array2dAccess(ptr,PETSC_REAL,(void**)&c PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array2dDestroy(ptr,PETSC_REAL PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = PetscFree(c);
}
void PETSC_STDCALL dmmeshgetelementsf90_(DM *dm,F90Array2d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt   *v;
  PetscInt   n, c;
  *__ierr = DMMeshGetElements(*dm,PETSC_TRUE,&n,&c,&v); if (*__ierr) return;
  *__ierr = F90Array2dCreate(v,PETSC_INT,1,n,1,c,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL dmmeshrestoreelementsf90_(DM *x,F90Array2d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt   *v;
  *__ierr = F90Array2dAccess(ptr,PETSC_INT,(void**)&v PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array2dDestroy(ptr,PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = PetscFree(v);
}
void PETSC_STDCALL dmmeshgetconef90_(DM *dm,PetscInt *p,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *v;
  PetscInt        n;
  *__ierr = DMMeshGetConeSize(*dm,*p,&n); if (*__ierr) return;
  *__ierr = DMMeshGetCone(*dm,*p,&v); if (*__ierr) return;
  *__ierr = F90Array1dCreate((void *)v,PETSC_INT,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL dmmeshrestoreconef90_(DM *dm,PetscInt *p,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *__ierr = F90Array1dDestroy(ptr,PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
}

#if 0
void PETSC_STDCALL dmmeshrestoreclosuref90_(DM dm,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal *c;
  *__ierr = F90Array1dAccess(ptr,PETSC_REAL,(void**)&c PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr,PETSC_REAL PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = PetscFree(c);
}
#endif

EXTERN_C_END
