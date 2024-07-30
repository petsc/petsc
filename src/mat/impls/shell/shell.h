#pragma once

#include <petsc/private/matimpl.h>
#include <petsc/private/vecimpl.h>

struct _MatShellOps {
  /*   3 */ PetscErrorCode (*mult)(Mat, Vec, Vec);
  /*   5 */ PetscErrorCode (*multtranspose)(Mat, Vec, Vec);
  /*  17 */ PetscErrorCode (*getdiagonal)(Mat, Vec);
  /*  32 */ PetscErrorCode (*getdiagonalblock)(Mat, Mat *);
  /*  43 */ PetscErrorCode (*copy)(Mat, Mat, MatStructure);
  /*  60 */ PetscErrorCode (*destroy)(Mat);
  /* 121 */ PetscErrorCode (*multhermitiantranspose)(Mat, Vec, Vec);
};

struct _n_MatShellMatFunctionList {
  PetscErrorCode (*symbolic)(Mat, Mat, Mat, void **);
  PetscErrorCode (*numeric)(Mat, Mat, Mat, void *);
  PetscErrorCode (*destroy)(void *);
  MatProductType ptype;
  char          *composedname; /* string to identify routine with double dispatch */
  char          *resultname;   /* result matrix type */

  struct _n_MatShellMatFunctionList *next;
};
typedef struct _n_MatShellMatFunctionList *MatShellMatFunctionList;

typedef struct {
  struct _MatShellOps ops[1];

  /* The user will manage the scaling and shifts for the MATSHELL, not the default */
  PetscBool managescalingshifts;

  /* support for MatScale, MatShift and MatMultAdd */
  PetscScalar vscale, vshift;
  Vec         dshift;
  Vec         left, right;
  Vec         left_work, right_work;
  Vec         left_add_work, right_add_work;

  /* support for MatAXPY */
  Mat              axpy;
  PetscScalar      axpy_vscale;
  Vec              axpy_left, axpy_right;
  PetscObjectState axpy_state;

  /* support for ZeroRows/Columns operations */
  IS         zrows;
  IS         zcols;
  Vec        zvals;
  Vec        zvals_w;
  VecScatter zvals_sct_r;
  VecScatter zvals_sct_c;

  /* MatMat operations */
  MatShellMatFunctionList matmat;

  /* user defined context */
  PetscContainer ctxcontainer;
} Mat_Shell;

PETSC_INTERN PetscErrorCode MatAssemblyEnd_Shell(Mat X, MatAssemblyType assembly);
