/*  
    Private data structure for BiCGStab(L) solver.
    Allocation takes place before each solve.
*/
#if !defined(__BCGSL)
#define __BCGSL
#include "petsc.h"

typedef struct {

	int ell;		/* Number of search directions. */
	PetscReal	delta;

	/* Workspace Vectors */
	Vec	vB;
	Vec	vRt;
	Vec	vXr;
	Vec	vTm;
	Vec	*vvR;
	Vec	*vvU;

	/* Workspace Arrays */
	int	ldz;	
	int	ldzc;
	PetscScalar	*vY0t, *vYlt;	
	PetscScalar	*vY0c, *vYlc, *vYtc;
	PetscScalar	*mZ, *mZc;


} KSP_BiCGStabL;

/* predefined shorthands */
#define	VX	(ksp->vec_sol)
#define	VB	(bcgsl->vB)
#define	VRT	(bcgsl->vRt)
#define	VXR	(bcgsl->vXr)
#define	VTM	(bcgsl->vTm)
#define	VVR	(bcgsl->vvR)
#define	VVU	(bcgsl->vvU)
#define	AY0t	(bcgsl->vY0t)
#define	AYlt	(bcgsl->vYlt)
#define	AY0c	(bcgsl->vY0c)
#define	AYtc	(bcgsl->vYtc)
#define	AYlc	(bcgsl->vYlc)
#define MZ	(bcgsl->mZ)
#define MZc	(bcgsl->mZc)

#define LDZ	(bcgsl->ldz)
#define LDZc	(bcgsl->ldzc)

#define	DELTA	(bcgsl->delta)

#endif
