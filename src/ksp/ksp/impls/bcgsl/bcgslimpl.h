/*  
    Private data structure for BiCGStab(L) solver.
    Allocation takes place before each solve.
*/
#if !defined(__BCGSL)
#define __BCGSL
#include "petscsys.h"

typedef struct {
  PetscInt        ell;		/* Number of search directions. */
  PetscReal	delta;		/* Threshold for recomputing exact residual norm */
  PetscTruth	bConvex;	/* Compute Enhanced BiCGstab polynomial when set to PETSC_TRUE */
  
  /* Workspace Vectors */
  Vec	vB;
  Vec	vRt;
  Vec	vXr;
  Vec	vTm;
  Vec	*vvR;
  Vec	*vvU;

  /* Workspace Arrays */
  PetscScalar	*vY0c, *vYlc, *vYtc;
  PetscScalar	*mZa, *mZb;
} KSP_BCGSL;

/* predefined shorthands */
#define	VX	(ksp->vec_sol)
#define	VB	(bcgsl->vB)
#define	VRT	(bcgsl->vRt)
#define	VXR	(bcgsl->vXr)
#define	VTM	(bcgsl->vTm)
#define	VVR	(bcgsl->vvR)
#define	VVU	(bcgsl->vvU)
#define	AY0c	(bcgsl->vY0c)
#define	AYtc	(bcgsl->vYtc)
#define	AYlc	(bcgsl->vYlc)
#define MZa	(bcgsl->mZa)
#define MZb	(bcgsl->mZb)

#endif
