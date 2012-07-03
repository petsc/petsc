#ifndef __LINESEARCHIMPL_H
#define __LINESEARCHIMPL_H

#include <petscsnes.h>

typedef struct _LineSearchOps *LineSearchOps;

struct _LineSearchOps {
  PetscErrorCode (*view)          (SNESLineSearch, PetscViewer);
  SNESLineSearchApplyFunc        apply;
  SNESLineSearchPreCheckFunc     precheckstep;
  SNESLineSearchVIProjectFunc    viproject;
  SNESLineSearchVINormFunc       vinorm;
  SNESLineSearchPostCheckFunc    postcheckstep;
  PetscErrorCode (*setfromoptions)(SNESLineSearch);
  PetscErrorCode (*reset)         (SNESLineSearch);
  PetscErrorCode (*destroy)       (SNESLineSearch);
  PetscErrorCode (*setup)         (SNESLineSearch);
};

struct _p_LineSearch {
  PETSCHEADER(struct _LineSearchOps);

  SNES                snes;     /* temporary -- so we can pull out the function evaluation */

  void                *data;

  PetscBool           setupcalled;

  Vec                 vec_sol;
  Vec                 vec_sol_new;
  Vec                 vec_func;
  Vec                 vec_func_new;
  Vec                 vec_update;

  PetscInt            nwork;
  Vec                 *work;

  PetscReal           lambda;

  PetscBool           norms;
  PetscReal           fnorm;
  PetscReal           ynorm;
  PetscReal           xnorm;
  PetscBool           success;
  PetscBool           keeplambda;

  PetscReal           damping;
  PetscReal           maxstep;
  PetscReal           steptol;
  PetscInt            max_its;
  PetscReal           rtol;
  PetscReal           atol;
  PetscReal           ltol;
  PetscInt            order;

  PetscReal           precheck_picard_angle;

  void *              precheckctx;
  void *              postcheckctx;

  PetscViewer         monitor;

};

#endif
