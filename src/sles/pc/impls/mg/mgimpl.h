/* $Id: mgimpl.h,v 1.14 2001/08/21 21:03:20 bsmith Exp $ */

/*
      Data structure used for Multigrid preconditioner.
*/
#if !defined(__MG_IMPL)
#define __MG_IMPL
#include "src/sles/pc/pcimpl.h"
#include "petscmg.h"
#include "petscksp.h"

typedef struct _MG* MG;

/*
     Structure for abstract multigrid solver. 

     Level (0) is always the coarsest level and Level (levels-1) is the finest.
*/
struct _MG
{
  MGType    am;                           /* Multiplicative, additive or full */
  int       cycles;                       /* Number cycles to run */
  int       level;                        /* level = 0 coarsest level */
  int       levels;                       /* number of active levels used */
  int       maxlevels;                    /* total number of levels allocated */
  Vec       b;                            /* Right hand side */ 
  Vec       x;                            /* Solution */
  Vec       r;                            /* Residual */
  int       (*residual)(Mat,Vec,Vec,Vec);
  Mat       A;                            /* matrix used in forming residual*/ 
  KSP      smoothd;                      /* pre smoother */
  KSP      smoothu;                      /* post smoother */
  Mat       interpolate; 
  Mat       restrct;                      /* restrict is a reserved word on the Cray!!!*/ 
  int       default_smoothu;              /* number of smooths per level if not over-ridden */
  int       default_smoothd;              /*  with calls to KSPSetTolerances() */
  PetscReal rtol,atol,dtol,ttol;          /* tolerances for when running with PCApplyRichardson_MG */
  int       eventsetup;                   /* if logging times for each level */
  int       eventsolve;      
};


#endif

