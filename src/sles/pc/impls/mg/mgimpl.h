/*
      Structure used for Multigrid code 
*/
#if !defined(__MG_IMPL)
#define __MG_IMPL
#include "ptscimpl.h"
#include "mg.h"
#include "sles.h"

struct _MG
{
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
    Mat    restrict; 
    SLES   csles;
} _MG;

#endif

