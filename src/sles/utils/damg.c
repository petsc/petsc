/*$Id: damg.c,v 1.1 2000/07/05 15:00:35 bsmith Exp bsmith $*/
 
#include "petscda.h"      /*I      "petscda.h"     I*/
#include "petscsles.h"    /*I      "petscsles.h"    I*/

/*
   Code for almost fully managing multigrid/multi-level linear solvers for DA grids
*/

#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGCreate"></a>*/"DAMGCreate"
/*@C
    DAMGCreate - Creates a DA based multigrid solver object. This allows one to 
      easily implement MG methods on regular grids.

    Collective on MPI_Comm

    Input Parameter:
.   comm - the processors that will share the grids and solution process

    Output Parameters:
.    - the context

    Level: advanced

.seealso DAMGDestroy(), 

@*/
int DAMGCreate(MPI_Comm comm,int nlevels,DAMG **ctx)
{
  int  ierr,i;
  DAMG *p;

  PetscFunctionBegin;
  ierr = OptionsGetInt(0,"-da_mg_nlevels",&nlevels,0);CHKERRQ(ierr);
  p    = (DAMG *)PetscMalloc(nlevels*sizeof(DAMG));CHKPTRQ(p);
  for (i=0; i<nlevels; i++) {
    p[i]          = (DAMG)PetscMalloc(sizeof(struct _p_DAMG));CHKPTRQ(p[i]);
    p[i]->nlevels = nlevels - i;
    p[i]->ratio   = 2;
    p[i]->comm    = comm;
  }
  *ctx = p;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGDestroy"></a>*/"DAMGDestroy"
/*@C
    DAMGCreate - Destroys a DA based multigrid solver object. 

    Collective on DAMG

    Input Parameter:
.    - the context

    Level: advanced

.seealso DAMGCreate()

@*/
int DAMGDestroy(DAMG *ctx)
{
  int     ierr,i,nlevels = ctx[0]->nlevels;

  PetscFunctionBegin;
  for (i=1; i<nlevels; i++) {
    if (ctx[i]->R) {ierr = MatDestroy(ctx[i]->R);CHKERRA(ierr);}
  }
  for (i=0; i<nlevels; i++) {
    if (ctx[i]->da) {ierr = DADestroy(ctx[i]->da);CHKERRQ(ierr);}
    if (ctx[i]->x)  {ierr = VecDestroy(ctx[i]->x);CHKERRQ(ierr);}
    if (ctx[i]->b)  {ierr = VecDestroy(ctx[i]->b);CHKERRQ(ierr);}
    if (ctx[i]->r)  {ierr = VecDestroy(ctx[i]->r);CHKERRQ(ierr);}
    if (ctx[i]->J)  {ierr = MatDestroy(ctx[i]->J);CHKERRQ(ierr);}
    ierr = PetscFree(ctx[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGSetDAInfo3d"></a>*/"DAMGSetDAInfo3d"
/*@C
    DAMGSetDAInfo3d - Sets the grid information for the coarsest grid

    Collective on DAMG

    Input Parameter:
+   ctx - the context
.   pt - DA_NONPERIODIC, DA_XPERIODIC, DA_YPERIODIC, DA_XYPERIODIC, DA_XYZPERIODIC, DA_XZPERIODIC, or DA_YZPERIODIC
.   st - DA_STENCIL_STAR or DA_STENCIL_BOX
.   m,n,p - the number of grid points in the three dimensions
.   sw - 1 or three point stencil, 2 for 5 etc
-   dof - number of degrees of freedom per grid point

    Level: advanced

.seealso DAMGCreate(), DAMGSetDAInfo1d(), DAMGSetDAInfo2d(), DaCreate3d()

@*/
int DAMGSetDAInfo3d(DAMG *ctx,DAPeriodicType pt,DAStencilType st,int m,int n,int p,int sw,int dof)
{
  int      ierr,i,nlevels = ctx[0]->nlevels;
  MPI_Comm comm = ctx[0]->comm;

  PetscFunctionBegin;
  ierr = OptionsGetInt(0,"-da_mg_M",&m,0);CHKERRQ(ierr);
  ierr = OptionsGetInt(0,"-da_mg_N",&n,0);CHKERRQ(ierr);
  ierr = OptionsGetInt(0,"-da_mg_P",&p,0);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    ierr = DACreate3d(comm,pt,st,m,n,p,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,sw,dof,0,0,0,&ctx[i]->da);CHKERRQ(ierr);
    m = ctx[i]->ratio*(m-1) + 1;
    n = ctx[i]->ratio*(n-1) + 1;
    p = ctx[i]->ratio*(p-1) + 1;
  }
  for (i=0; i<nlevels; i++) {
    ierr = DACreateGlobalVector(ctx[i]->da,&ctx[i]->x);CHKERRA(ierr);
    ierr = VecDuplicate(ctx[i]->x,&ctx[i]->b);CHKERRA(ierr);
    ierr = VecDuplicate(ctx[i]->x,&ctx[i]->r);CHKERRA(ierr);
    ierr = DAGetColoring(ctx[i]->da,PETSC_NULL,&ctx[i]->J);CHKERRQ(ierr);
  }
  for (i=1; i<nlevels; i++) {
    ierr = DAGetInterpolation(ctx[i-1]->da,ctx[i]->da,&ctx[i]->R,PETSC_NULL);CHKERRA(ierr);
  }

  PetscFunctionReturn(0);
}





