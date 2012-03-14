#ifndef __LINESEARCHIMPL_H
#define __LINESEARCHIMPL_H

#include <petsclinesearch.h>

struct _LineSearchOps {
  PetscErrorCode (*view)          (LineSearch);
  LineSearchApplyFunc             apply;
  LineSearchPreCheckFunc          precheckstep;
  LineSearchPostCheckFunc         postcheckstep;
  PetscErrorCode (*setfromoptions)(LineSearch);
  PetscErrorCode (*reset)         (LineSearch);
  PetscErrorCode (*destroy)       (LineSearch);
  PetscErrorCode (*setup)         (LineSearch);
};

typedef struct _LineSearchOps *LineSearchOps;

struct _p_LineSearch {
  PETSCHEADER(struct _LineSearchOps);

  SNES          snes;     /* temporary -- so we can pull out the function evaluation */

  void          *data;

  PetscBool     setupcalled;

  Vec           vec_sol;
  Vec           vec_sol_new;
  Vec           vec_func;
  Vec           vec_func_new;
  Vec           vec_update;

  PetscInt      nwork;
  Vec           *work;

  PetscReal     lambda;

  PetscBool     norms;
  PetscReal     fnorm;
  PetscReal     ynorm;
  PetscReal     xnorm;
  PetscBool     success;
  PetscBool     keeplambda;

  PetscReal     damping;
  PetscReal     maxstep;
  PetscReal     steptol;
  PetscInt      max_its;
  PetscReal     rtol;
  PetscReal     atol;
  PetscReal     ltol;


  void *        precheckctx;
  void *        postcheckctx;

  PetscViewer   monitor;

};

#endif
