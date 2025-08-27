#pragma once

#include <petscsnes.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      SNESLineSearchRegisterAllCalled;
PETSC_EXTERN PetscErrorCode SNESLineSearchRegisterAll(void);
PETSC_EXTERN PetscLogEvent  SNESLINESEARCH_Apply;

typedef struct _LineSearchOps *LineSearchOps;

struct _LineSearchOps {
  PetscErrorCode (*view)(SNESLineSearch, PetscViewer);
  SNESLineSearchApplyFn *apply;
  PetscErrorCode (*precheck)(SNESLineSearch, Vec, Vec, PetscBool *, void *);
  SNESLineSearchVIProjectFn  *viproject;
  SNESLineSearchVINormFn     *vinorm;
  SNESLineSearchVIDirDerivFn *vidirderiv;
  PetscErrorCode (*postcheck)(SNESLineSearch, Vec, Vec, Vec, PetscBool *, PetscBool *, void *);
  PetscErrorCode (*setfromoptions)(SNESLineSearch, PetscOptionItems);
  PetscErrorCode (*reset)(SNESLineSearch);
  PetscErrorCode (*destroy)(SNESLineSearch);
  PetscErrorCode (*setup)(SNESLineSearch);
  PetscErrorCode (*snesfunc)(SNES, Vec, Vec);
};

#define MAXSNESLSMONITORS 5

struct _p_LineSearch {
  PETSCHEADER(struct _LineSearchOps);

  SNES snes;

  void *data;

  PetscBool setupcalled;

  Vec vec_sol;
  Vec vec_sol_new;
  Vec vec_func;
  Vec vec_func_new;
  Vec vec_update;

  PetscInt nwork;
  Vec     *work;

  PetscReal lambda;

  PetscBool            norms;
  PetscReal            fnorm;
  PetscReal            ynorm;
  PetscReal            xnorm;
  SNESLineSearchReason result;
  PetscBool            keeplambda;

  PetscReal damping;
  PetscReal maxlambda;
  PetscReal minlambda;
  PetscInt  max_it;
  PetscReal rtol;
  PetscReal atol;
  PetscReal ltol;
  PetscInt  order;

  PetscReal precheck_picard_angle;

  void *precheckctx;
  void *postcheckctx;

  PetscViewer monitor;
  PetscErrorCode (*monitorftns[MAXSNESLSMONITORS])(SNESLineSearch, void *); /* monitor routine */
  PetscCtxDestroyFn *monitordestroy[MAXSNESLSMONITORS];                     /* monitor context destroy routine */
  void              *monitorcontext[MAXSNESLSMONITORS];                     /* monitor context */
  PetscInt           numbermonitors;                                        /* number of monitors */
};
