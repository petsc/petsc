/*$Id: damg.c,v 1.4 2000/07/10 17:52:21 bsmith Exp bsmith $*/
 
#include "petscda.h"      /*I      "petscda.h"     I*/
#include "petscsles.h"    /*I      "petscsles.h"    I*/
#include "petscmg.h"      /*I      "petscmg.h"    I*/

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

.seealso DAMGDestroy() 

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
    p[i]->ratiox  = 2;
    p[i]->ratioy  = 2;
    p[i]->ratioz  = 2;
    p[i]->comm    = comm;
  }
  *ctx = p;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGDestroy"></a>*/"DAMGDestroy"
/*@C
    DAMGDestroy - Destroys a DA based multigrid solver object. 

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
#define __FUNC__ /*<a name="DAMGSetCoarseDA"></a>*/"DAMGSetCoarseDA"
/*@C
    DAMGSetCoarseDA - Sets the grid information for the coarsest grid

    Collective on DAMG and DA

    Input Parameter:
+   ctx - the context
-   da - the DA for the coarsest grid (this routine creates all the other ones)

    Notes:
      This keeps the passed in DA object (in ctx[0]->da) (without increasing the reference count)
    so you should not destroy the da passed in later in your code (the DAMGDestroy() will 
    handle that.

    Level: advanced

.seealso DAMGCreate(), DAMGDestroy

@*/
int DAMGSetCoarseDA(DAMG *ctx,DA da)
{
  int            ierr,i,j,nlevels = ctx[0]->nlevels,M,N,P,m,n,p,sw,dof,dim,flag,Nt;
  MPI_Comm       comm;
  DAPeriodicType pt;
  DAStencilType  st;
  char           *name;

  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_compare(comm,ctx[0]->comm,&flag);CHKERRQ(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) {
    SETERRQ(PETSC_ERR_ARG_NOTSAMECOMM,0,"Different communicators in the DAMG and the DA");
  }

  /* Create DA data structure for all the finer levels */
  ierr       = DAGetInfo(da,&dim,&M,&N,&P,&m,&n,&p,&dof,&sw,&pt,&st);CHKERRQ(ierr);
  ctx[0]->da = da;
  for (i=1; i<nlevels; i++) {
    M = ctx[i-1]->ratiox*(M-1) + 1;
    N = ctx[i-1]->ratioy*(N-1) + 1;
    P = ctx[i-1]->ratioz*(P-1) + 1;
    if (dim == 3) {
      ierr = DACreate3d(comm,pt,st,M,N,P,m,n,p,dof,sw,0,0,0,&ctx[i]->da);CHKERRQ(ierr);
    } else {
      SETERRQ1(1,1,"Cannot handle dimension %d",dim);
    }
  }

  /* Set fieldnames on finer grids */
  for (j=0; j<dof; j++) {
    ierr = DAGetFieldName(da,j,&name);CHKERRQ(ierr);
    if (name) {
      for (i=1; i<nlevels; i++) {
        ierr = DASetFieldName(ctx[i]->da,j,name);CHKERRQ(ierr);
      }
    }
  }




  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGSetSLES"></a>*/"DAMGSetSLES"
/*@C
    DAMGSetSLES - Sets the linear solver object that will use the grid hierarchy

    Collective on DAMG and SLES

    Input Parameter:
+   ctx - the context
.   sles - the linear solver object
-   func - function to compute linear system matrix on each grid level

    Level: advanced

.seealso DAMGCreate(), DAMGDestroy, DAMGSetCoarseDA()

@*/
int DAMGSetSLES(DAMG *ctx,SLES sles,int (*func)(DA,Mat))
{
  int        ierr,i,j,nlevels = ctx[0]->nlevels,flag,m,n,p,Nt,dim;
  MPI_Comm   comm;
  PC         pc;
  Vec        xyz,txyz;
  PetscTruth flg;

  PetscValidHeaderSpecific(sles,SLES_COOKIE);
  ierr = PetscObjectGetComm((PetscObject)sles,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_compare(comm,ctx[0]->comm,&flag);CHKERRQ(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) {
    SETERRQ(PETSC_ERR_ARG_NOTSAMECOMM,0,"Different communicators in the DAMG and the SLES");
  }



  /* Create work vectors and matrix for each level */
  for (i=0; i<nlevels; i++) {
    ierr = DACreateGlobalVector(ctx[i]->da,&ctx[i]->x);CHKERRA(ierr);
    ierr = VecDuplicate(ctx[i]->x,&ctx[i]->b);CHKERRA(ierr);
    ierr = VecDuplicate(ctx[i]->x,&ctx[i]->r);CHKERRA(ierr);
    ierr = DAGetColoring(ctx[i]->da,PETSC_NULL,&ctx[i]->J);CHKERRQ(ierr);
    
  }

  /* Create interpolation/restriction between levels */
  for (i=1; i<nlevels; i++) {
    ierr = DAGetInterpolation(ctx[i-1]->da,ctx[i]->da,&ctx[i]->R,PETSC_NULL);CHKERRA(ierr);
  }

  /* If coarsest grid has coordinate information then interpolate it for finer grids */
  ierr = DAGetCoordinates(ctx[0]->da,&xyz);CHKERRQ(ierr);
  if (xyz) {
    Mat        Rd;
    Vec        xyztest;
    PetscTruth docoors = PETSC_FALSE;

    for (i=1; i<nlevels; i++) {
      ierr = DAGetCoordinates(ctx[i]->da,&xyztest);CHKERRQ(ierr);
      if (!xyztest) {
        docoors = PETSC_TRUE; break;
      }
    }

    if (docoors) {
      ierr = DAGetInfo(ctx[0]->da,&dim,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
      for (i=1; i<nlevels; i++) {
        ierr = DAGetCoordinates(ctx[i]->da,&xyztest);CHKERRQ(ierr);
        ierr = MatMAIJRedimension(ctx[i]->R,dim,&Rd);CHKERRQ(ierr);
        ierr = DAGetCorners(ctx[i]->da,0,0,0,&m,&n,&p);CHKERRQ(ierr);
        Nt   = dim*m*n*p;
        ierr = VecCreateMPI(comm,Nt,PETSC_DECIDE,&txyz);CHKERRQ(ierr);
        ierr = MatInterpolate(Rd,xyz,txyz);CHKERRQ(ierr);
        ierr = MatDestroy(Rd);CHKERRQ(ierr);
        if (!xyztest) {
          ierr = DASetCoordinates(ctx[i]->da,txyz);CHKERRQ(ierr);
        }
        xyz  = txyz;
      }
    }
  }

  ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
  ierr = PCSetType(pc,PCMG);CHKERRA(ierr);
  ierr = MGSetLevels(pc,nlevels);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);

  /* set solvers for each level */
  for (i=0; i<nlevels; i++) {
    ierr = MGGetSmoother(pc,i,&ctx[i]->sles);CHKERRA(ierr);
    ierr = SLESSetFromOptions(ctx[i]->sles);CHKERRA(ierr);
    ierr = SLESSetOperators(ctx[i]->sles,ctx[i]->J,ctx[i]->J,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
    ierr = MGSetX(pc,i,ctx[i]->x);CHKERRA(ierr); 
    ierr = MGSetRhs(pc,i,ctx[i]->b);CHKERRA(ierr); 
    ierr = MGSetR(pc,i,ctx[i]->r);CHKERRA(ierr); 
    ierr = MGSetResidual(pc,i,MGDefaultResidual,ctx[i]->J);CHKERRA(ierr);
  }

  /* Set interpolation/restriction between levels */
  for (i=1; i<nlevels; i++) {
    ierr = MGSetInterpolate(pc,i,ctx[i]->R);CHKERRA(ierr); 
    ierr = MGSetRestriction(pc,i,ctx[i]->R);CHKERRA(ierr); 
  }

  /* set matrix for each level */
  if (func) {
    for (i=0; i<nlevels; i++) {
      ierr = (*func)(ctx[i]->da,ctx[i]->J);CHKERRQ(ierr);
    }
  }

  ierr = OptionsHasName(PETSC_NULL,"-damg_view",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DAMGView(ctx,VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGView"></a>*/"DAMGView"
/*@C
    DAMGView - prints information on a DA based multi-level preconditioner

    Collective on DAMG and Viewer

    Input Parameter:
+   ctx - the context
-   viewer - the viewer

    Level: advanced

.seealso DAMGCreate(), DAMGDestroy

@*/
int DAMGView(DAMG *ctx,Viewer viewer)
{
  int            ierr,i,nlevels = ctx[0]->nlevels,flag;
  MPI_Comm       comm;
  PetscTruth     isascii;

  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_compare(comm,ctx[0]->comm,&flag);CHKERRQ(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) {
    SETERRQ(PETSC_ERR_ARG_NOTSAMECOMM,0,"Different communicators in the DAMG and the Viewer");
  }

  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerASCIIPrintf(viewer,"DAMG Object with %d levels\n",nlevels);CHKERRQ(ierr);
    for (i=0; i<nlevels; i++) {
      ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = DAView(ctx[i]->da,viewer);CHKERRQ(ierr);
      ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported",*((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}














