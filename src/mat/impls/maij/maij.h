#if !defined(_MAIJ_H)
#define _MAIJ_H

#include "src/mat/impls/aij/mpi/mpiaij.h"

typedef struct {
  int        dof;         /* number of components */
  Mat        AIJ;        /* representation of interpolation for one component */
} Mat_SeqMAIJ;

typedef struct {
  int        dof;         /* number of components */
  Mat        AIJ,OAIJ;    /* representation of interpolation for one component */
  Mat        A;
  VecScatter ctx;         /* update ghost points for parallel case */
  Vec        w;           /* work space for ghost values for parallel case */
} Mat_MPIMAIJ;

#endif
