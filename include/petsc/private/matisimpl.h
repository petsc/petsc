#if !defined(__is_h)
#define __is_h

#include <petscsf.h>
#include <petsc/private/matimpl.h>

typedef struct {
  Mat         A;                           /* the local matrix */
  VecScatter  cctx;                        /* column scatter */
  VecScatter  rctx;                        /* row scatter */
  Vec         x,y;                         /* work space for matrix vector product */
  Vec         counter;                     /* counter vector */
  PetscBool   pure_neumann;
  PetscSF     sf,csf;                      /* SFs for rows and cols */
  PetscInt    *sf_rootdata,*sf_leafdata;
  PetscInt    *csf_rootdata,*csf_leafdata;
  IS          getsub_ris,getsub_cis;       /* row and column ISs for MatCreateSubMatrix and MAT_REUSE_MATRIX */
  PetscBool   islocalref;                  /* is a reference to a local submatrix? */
  PetscBool   locempty;                    /* adapt local matrices for empty rows/cols during MatAssemblyEnd_IS */
  PetscBool   storel2l;                    /* carry over local-to-local inherited in MatPtAP */
  char        *lmattype;
  PetscScalar *bdiag;                      /* Used by MatInvertBlockDiagonal_IS */

  PetscObjectState lnnzstate;              /* nonzero state of local matrix */

  PetscBool   keepassembled;               /* store assembled form if needed */
  Mat         assembledA;                  /* assembled operator */
  Mat         dA;                          /* For MatGetDiagonalBlock_IS */

  /* Support for negative or repeated entries in l2map
     These maps can be different than the ones passed in by the user via
     MatSetLocalToGlobalMapping */
  ISLocalToGlobalMapping rmapping, cmapping;
} Mat_IS;

struct _MatISLocalFields {
  PetscInt nr,nc;
  IS       *rf,*cf;
};
typedef struct _MatISLocalFields *MatISLocalFields;

struct _MatISPtAP {
  PetscReal fill;
  IS        cis0,cis1,ris0,ris1;
  Mat       *lP;
};
typedef struct _MatISPtAP *MatISPtAP;

PETSC_EXTERN PetscErrorCode MatISSetMPIXAIJPreallocation_Private(Mat,Mat,PetscBool);
#endif
