/* $Id: bilinearimpl.h,v 1.2 1999/06/01 16:44:05 knepley Exp $ */

#ifndef __BILINEARIMPL
#define __BILINEARIMPL

#include "bilinear.h"

typedef enum {BILINEAR_FACTOR_NONE, BILINEAR_FACTOR_LU, BILINEAR_FACTOR_CHOLESKY} BilinearFactorizationType;

struct _BilinearOps {
      /* Generic Operations */
  int (*setfromoptions)(Bilinear),
      (*view)(Bilinear, Viewer),
      (*copy)(Bilinear, Bilinear),
      (*convertsametype)(Bilinear, Bilinear *, int),
      (*destroy)(Bilinear),
      (*printhelp)(Bilinear),
      /* Assembly Operations*/
      (*setvalues)(Bilinear, int, int *, int, int *, int, int *, Scalar *, InsertMode),
      (*getarray)(Bilinear, Scalar **),
      (*restorearray)(Bilinear, Scalar **),
      (*assemblybegin)(Bilinear, MatAssemblyType),
      (*assemblyend)(Bilinear, MatAssemblyType),
      (*zeroentries)(Bilinear),
      /* Evaluation Operations */
      (*mult)(Bilinear, Vec, Mat),
      (*fullmult)(Bilinear, Vec, Vec, Vec),
      (*diamond)(Bilinear, Vec, Vec),
      /* Factorization Operations */
      (*lufactor)(Bilinear, IS, IS, IS, double),
      (*choleskyfactor)(Bilinear, IS, double),
      /* Query Functions */
      (*getinfo)(Bilinear, InfoType, BilinearInfo *),
      (*getsize)(Bilinear, int *, int *, int *),
      (*getlocalsize)(Bilinear, int *, int *, int *),
      (*getownershiprange)(Bilinear, int *, int *),
      (*equal)(Bilinear, Bilinear, PetscTruth *),
      (*norm)(Bilinear, NormType, double *),
      (*setoption)(Bilinear, BilinearOption);
};

struct _p_Bilinear {
  PETSCHEADER(struct _BilinearOps)
  void        *data;             /* implementation-specific data */
  /* Size variables */
  int          N_i, N_j, N_k;    /* global numbers of rows, columns, and subcolumns */
  int          n_i, n_j, n_k;    /* local numbers of rows, columns, and subcolumns */
  /* Assembly variables */
  PetscTruth   assembled;        /* is the matrix assembled? */
  PetscTruth   was_assembled;    /* new values inserted into assembled mat */
  int          num_ass;          /* number of times matrix has been assembled */
  PetscTruth   same_nonzero;     /* matrix has same nonzero pattern as previous */
  InsertMode   insertmode;       /* have values been inserted in matrix or added? */
  /* Factorization variables */
  BilinearFactorizationType factor;           /* 0, FACTOR_LU, or FACTOR_CHOLESKY */
  double                    lupivotthreshold; /* threshold for pivoting */
  /* Query variables */
  BilinearInfo info;             /* operator information */
};

/* The final argument for BilinearConvertSameType() */
#define DO_NOT_COPY_VALUES 0
#define COPY_VALUES        1

#endif
