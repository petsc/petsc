/*
      Data structure used for Multigrid preconditioner.
*/
#if !defined(__MG_IMPL)
#define __MG_IMPL
#include "src/ksp/pc/pcimpl.h"
#include "petscmg.h"
#include "petscksp.h"


/*
     Structure for abstract multigrid solver. 

     Level (0) is always the coarsest level and Level (levels-1) is the finest.
*/
typedef struct
{
  PCMGType   am;                           /* Multiplicative, additive or full */
  int        cycles;                       /* Number cycles to run */
  int        level;                        /* level = 0 coarsest level */
  int        levels;                       /* number of active levels used */
  int        maxlevels;                    /* total number of levels allocated */
  PetscTruth galerkin;                     /* use Galerkin process to compute coarser matrices */
  PetscTruth galerkinused;                 /* destroy the Mat created by the Galerkin process */
  Vec        b;                            /* Right hand side */ 
  Vec        x;                            /* Solution */
  Vec        r;                            /* Residual */
  PetscErrorCode (*residual)(Mat,Vec,Vec,Vec);
  Mat        A;                            /* matrix used in forming residual*/ 
  KSP        smoothd;                      /* pre smoother */
  KSP        smoothu;                      /* post smoother */
  Mat        interpolate; 
  Mat        restrct;                      /* restrict is a reserved word on the Cray!!!*/ 
  int        default_smoothu;              /* number of smooths per level if not over-ridden */
  int        default_smoothd;              /*  with calls to KSPSetTolerances() */
  PetscReal  rtol,abstol,dtol,ttol;        /* tolerances for when running with PCApplyRichardson_MG */
  PetscEvent eventsetup;                   /* if logging times for each level */
  PetscEvent eventsolve;      
}  PC_MG;


#endif

