/*
      Structure used for Multigrid code 
*/
#if !defined(__MG_IMPL)
#define __MG_IMPL
#include "ptscimpl.h"
#include "pcimpl.h"
#include "mg.h"
#include "sles.h"

typedef struct _MG* MG;

struct _MG
{
    MGMethod am;                     /* Mult,add or full */
    int    cycles;                 /* Number cycles to run */
    int    level;                  /* level = 0 coarsest level */
    Vec    b;                      /* Right hand side */ 
    Vec    x;                      /* Solution */
    Vec    r;                      /* Residual */
    int    (*residual)(Mat,Vec,Vec,Vec);
    Mat    A;                      /* matrix used in forming residual*/ 
    SLES   smoothd; 
    SLES   smoothu; 
    Mat    interpolate; 
    Mat    restrct;  /* restrict is a reserved word on the Cray!!!*/ 
    SLES   csles;
};

#endif

