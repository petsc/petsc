#include <petsc/private/ftnimpl.h>
#include <petsc/private/tsimpl.h> /*I   "petscts.h"   I*/
#include <../src/ts/impls/explicit/rk/rk.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define tsrkgettableau_     TSRKGETTABLEAU
  #define tsrkrestoretableau_ TSRKRESTORETABLEAU
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define tsrkgettableau_     tsrkgettableau
  #define tsrkrestoretableau_ tsrkrestoretableau
#endif

PETSC_EXTERN void tsrkgettableau_(TS *ts, PetscInt *s, F90Array1d *A, F90Array1d *b, F90Array1d *c, F90Array1d *bembed, PetscInt *p, F90Array1d *binterp, PetscBool *FSAL, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(p1) PETSC_F90_2PTR_PROTO(p2) PETSC_F90_2PTR_PROTO(p3) PETSC_F90_2PTR_PROTO(p4) PETSC_F90_2PTR_PROTO(p5))
{
  TS_RK    *rk  = (TS_RK *)(*ts)->data;
  RKTableau tab = rk->tableau;

  CHKFORTRANNULLINTEGER(s);
  CHKFORTRANNULLINTEGER(p);
  CHKFORTRANNULLBOOL(FSAL);
  if (s) *s = tab->s;
  if (!FORTRANNULLSCALARPOINTER(A)) {
    *ierr = F90Array1dCreate(tab->A, MPIU_REAL, 1, tab->s * tab->s, A PETSC_F90_2PTR_PARAM(p1));
    if (*ierr) return;
  }
  if (!FORTRANNULLSCALARPOINTER(b)) {
    *ierr = F90Array1dCreate(tab->b, MPIU_REAL, 1, tab->s, b PETSC_F90_2PTR_PARAM(p2));
    if (*ierr) return;
  }
  if (!FORTRANNULLSCALARPOINTER(c)) {
    *ierr = F90Array1dCreate(tab->c, MPIU_REAL, 1, tab->s, c PETSC_F90_2PTR_PARAM(p3));
    if (*ierr) return;
  }
  if (!FORTRANNULLSCALARPOINTER(bembed) && tab->bembed) {
    *ierr = F90Array1dCreate(tab->bembed, MPIU_REAL, 1, tab->s, bembed PETSC_F90_2PTR_PARAM(p4));
    if (*ierr) return;
  }
  if (p) *p = tab->p;
  if (!FORTRANNULLSCALARPOINTER(binterp)) {
    *ierr = F90Array1dCreate(tab->binterp, MPIU_REAL, 1, tab->s * tab->p, binterp PETSC_F90_2PTR_PARAM(p5));
    if (*ierr) return;
  }
  if (FSAL) *FSAL = tab->FSAL;
  *ierr = PETSC_SUCCESS;
}

PETSC_EXTERN void tsrkrestoretableau_(TS *ts, PetscInt *s, F90Array1d *A, F90Array1d *b, F90Array1d *c, F90Array1d *bembed, PetscInt *p, F90Array1d *binterp, PetscBool *FSAL, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(p1) PETSC_F90_2PTR_PROTO(p2) PETSC_F90_2PTR_PROTO(p3) PETSC_F90_2PTR_PROTO(p4) PETSC_F90_2PTR_PROTO(p5))
{
  if (!FORTRANNULLSCALARPOINTER(A)) {
    *ierr = F90Array1dDestroy(A, MPIU_REAL PETSC_F90_2PTR_PARAM(p1));
    if (*ierr) return;
  }
  if (!FORTRANNULLSCALARPOINTER(b)) {
    *ierr = F90Array1dDestroy(b, MPIU_REAL PETSC_F90_2PTR_PARAM(p2));
    if (*ierr) return;
  }
  if (!FORTRANNULLSCALARPOINTER(c)) {
    *ierr = F90Array1dDestroy(c, MPIU_REAL PETSC_F90_2PTR_PARAM(p3));
    if (*ierr) return;
  }
  if (!FORTRANNULLSCALARPOINTER(bembed)) {
    *ierr = F90Array1dDestroy(bembed, MPIU_REAL PETSC_F90_2PTR_PARAM(p4));
    if (*ierr) return;
  }
  if (!FORTRANNULLSCALARPOINTER(binterp)) {
    *ierr = F90Array1dDestroy(binterp, MPIU_REAL PETSC_F90_2PTR_PARAM(p5));
    if (*ierr) return;
  }
  *ierr = PETSC_SUCCESS;
}
