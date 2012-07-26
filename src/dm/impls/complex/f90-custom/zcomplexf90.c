#include <petsc-private/fortranimpl.h>
#include <petscdmcomplex.h>
#include <../src/sys/f90-src/f90impl.h>

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dmcomplexgetcone_                  DMCOMPLEXGETCONE
#define dmcomplexrestorecone_              DMCOMPLEXRESTORECONE
#define dmcomplexgetconeorientation_       DMCOMPLEXGETCONEORIENTATION
#define dmcomplexrestoreconeorientation_   DMCOMPLEXRESTORECONEORIENTATION
#define dmcomplexgetsupport_               DMCOMPLEXGETSUPPORT
#define dmcomplexrestoresupport_           DMCOMPLEXRESTORESUPPORT
#define dmcomplexgettransitiveclosure_     DMCOMPLEXGETTRANSITIVECLOSURE
#define dmcomplexrestoretransitiveclosure_ DMCOMPLEXRESTORETRANSITIVECLOSURE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmcomplexgetcone_                  dmcomplexgetcone
#define dmcomplexrestorecone_              dmcomplexrestorecone
#define dmcomplexgetconeorientation_       dmcomplexgetconeorientation
#define dmcomplexrestoreconeorientation_   dmcomplexrestoreconeorientation
#define dmcomplexgetsupport_               dmcomplexgetsupport
#define dmcomplexrestoresupport_           dmcomplexrestoresupport
#define dmcomplexgettransitiveclosure_     dmcomplexgettransitiveclosure
#define dmcomplexrestoretransitiveclosure_ dmcomplexrestoretransitiveclosure
#endif

/* Definitions of Fortran Wrapper routines */
EXTERN_C_BEGIN

void PETSC_STDCALL dmcomplexgetcone_(DM *dm, PetscInt *p, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *v;
  PetscInt        n;

  *__ierr = DMComplexGetConeSize(*dm, *p, &n);if (*__ierr) return;
  *__ierr = DMComplexGetCone(*dm, *p, &v);if (*__ierr) return;
  *__ierr = F90Array1dCreate((void *) v, PETSC_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

void PETSC_STDCALL dmcomplexrestorecone_(DM *dm, PetscInt *p, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *__ierr = F90Array1dDestroy(ptr, PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
}

void PETSC_STDCALL dmcomplexgetconeorientation_(DM *dm, PetscInt *p, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *v;
  PetscInt        n;

  *__ierr = DMComplexGetConeSize(*dm, *p, &n);if (*__ierr) return;
  *__ierr = DMComplexGetConeOrientation(*dm, *p, &v);if (*__ierr) return;
  *__ierr = F90Array1dCreate((void *) v, PETSC_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

void PETSC_STDCALL dmcomplexrestoreconeorientation_(DM *dm, PetscInt *p, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *__ierr = F90Array1dDestroy(ptr, PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
}

void PETSC_STDCALL dmcomplexgetsupport_(DM *dm, PetscInt *p, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *v;
  PetscInt        n;

  *__ierr = DMComplexGetSupportSize(*dm, *p, &n);if (*__ierr) return;
  *__ierr = DMComplexGetSupport(*dm, *p, &v);if (*__ierr) return;
  *__ierr = F90Array1dCreate((void *) v, PETSC_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

void PETSC_STDCALL dmcomplexrestoresupport_(DM *dm, PetscInt *p, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *__ierr = F90Array1dDestroy(ptr, PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
}

void PETSC_STDCALL dmcomplexgettransitiveclosure_(DM *dm, PetscInt *p, PetscBool *useCone, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *v;
  PetscInt        n;

  *__ierr = DMComplexGetTransitiveClosure(*dm, *p, *useCone, &n, &v);if (*__ierr) return;
  *__ierr = F90Array1dCreate((void *) v, PETSC_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

void PETSC_STDCALL dmcomplexrestoretransitiveclosure_(DM *dm, PetscInt *p, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *__ierr = F90Array1dDestroy(ptr, PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
}

EXTERN_C_END
