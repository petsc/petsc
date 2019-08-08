#if !defined(_KAIJ_H)
#define _KAIJ_H

#include <../src/mat/impls/aij/mpi/mpiaij.h>

#define KAIJHEADER                                \
  PetscInt    p,q;                                \
  Mat         AIJ;                                \
  PetscScalar *S;                                 \
  PetscScalar *T;                                 \
  PetscScalar *ibdiag;                            \
  PetscBool   ibdiagvalid,getrowactive,isTI;      \
  struct {                                        \
    PetscBool setup;                              \
    PetscScalar *w,*work,*t,*arr,*y;              \
  } sor;

typedef struct {
  KAIJHEADER
} Mat_SeqKAIJ;

typedef struct {
  KAIJHEADER
  Mat        OAIJ;
  Mat        A;
  VecScatter ctx;     /* update ghost points for parallel case */
  Vec        w;       /* work space for ghost values for parallel case */
} Mat_MPIKAIJ;

#endif
