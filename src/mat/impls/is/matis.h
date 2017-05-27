
#if !defined(__is_h)
#define __is_h

#include <petscsf.h>
#include <petsc/private/matimpl.h>

typedef struct {
  Mat        A;                           /* the local matrix */
  VecScatter cctx;                        /* column scatter */
  VecScatter rctx;                        /* row scatter */
  Vec        x,y;                         /* work space for matrix vector product */
  Vec        counter;                     /* counter vector */
  PetscBool  pure_neumann;
  PetscSF    sf,csf;                      /* SFs for rows and cols */
  PetscInt   *sf_rootdata,*sf_leafdata;
  PetscInt   *csf_rootdata,*csf_leafdata;
  IS         getsub_ris,getsub_cis;       /* row and column ISs for MatCreateSubMatrix and MAT_REUSE_MATRIX */
  PetscBool  islocalref;                  /* is a reference to a local submatrix? */
  PetscBool  locempty;                    /* adapt local matrices for empty rows/cols during MatAssemblyEnd_IS */
  PetscBool  usesetlocal;                 /* use MatSetValuesLocal on local matrices */
} Mat_IS;

struct _MatISLocalFields {
  PetscInt nr,nc;
  IS       *rf,*cf;
};
typedef struct _MatISLocalFields *MatISLocalFields;

PETSC_EXTERN PetscErrorCode MatISSetMPIXAIJPreallocation_Private(Mat,Mat,PetscBool);
#endif




