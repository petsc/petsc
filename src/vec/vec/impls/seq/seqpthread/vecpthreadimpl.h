
#ifndef __VECPTHREADIMPL
#define __VECPTHREADIMPL

#include <petscsys.h>
#include <private/vecimpl.h>

typedef struct {
  VECHEADER
  PetscScalar   *x,*y;
  PetscInt      n;
  PetscScalar   result;
  PetscScalar   alpha;

  PetscScalar   *ww;
  PetscScalar   *yy;
  PetscScalar   *xx;

  NormType      typeUse;

  PetscScalar*  xvalin;
  Vec*          yavecin;
  PetscInt      nelem;
  PetscInt      ntoproc;
  PetscScalar*  results;

  PetscInt      gind;
  PetscInt      localn;
  PetscInt      localind;
  PetscReal     localmax;
  PetscReal     localmin;

  PetscScalar   *wpin,*xpin,*ypin;
  PetscInt      nlocal;

  PetscRandom   rin;
  PetscScalar*  amult;   //multipliers
  PetscInt      ibase;   //used to properly index into other vectors

  PetscScalar   alphain;
} Vec_SeqPthread;

#endif
