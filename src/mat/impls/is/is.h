/*$Id: nn.h,v 1.1 2000/06/05 16:59:44 bsmith Exp bsmith $*/

#if !defined(__nn_h)
#define __nn_h

#include "src/mat/matimpl.h"

typedef struct {
  Mat                    A;             /* the local Neumann matrix */
  VecScatter             ctx;           /* update ghost points for matrix vector product */
  Vec                    x,y;           /* work space for ghost values for matrix vector product */
  ISLocalToGlobalMapping mapping;
  int                    rstart,rend;   /* local row ownership */
  int                    *zeroedrows,nzeroedrows;
  Scalar                 diag;
} Mat_NN;

#endif
