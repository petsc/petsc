/*
      Data structure used for Multigrid preconditioner.
*/
#if !defined(__MG_IMPL)
#define __MG_IMPL
#include "pcimpl.h"
#include "mg.h"
#include "sles.h"

typedef struct _MG* MG;

struct _MG
{
    MGMethod am;                     /* Multiplicative, additive or full */
    int      cycles;                 /* Number cycles to run */
    int      level;                  /* level = 0 coarsest level */
    Vec      b;                      /* Right hand side */ 
    Vec      x;                      /* Solution */
    Vec      r;                      /* Residual */
    int      (*residual)(Mat,Vec,Vec,Vec);
    Mat      A;                      /* matrix used in forming residual*/ 
    SLES     smoothd;                /* pre smoother */
    SLES     smoothu;                /* post smoother */
    Mat      interpolate; 
    Mat      restrct;  /* restrict is a reserved word on the Cray!!!*/ 
    SLES     csles;                  /* coarse grid solve */
};

#endif

