/*$Id: damgsnes.c,v 1.17 2001/04/19 19:16:00 bsmith Exp bsmith $*/
 
#include "petscda.h"      /*I      "petscda.h"     I*/
#include "petscmg.h"      /*I      "petscmg.h"    I*/

/*
      These evaluate the Jacobian on all of the grids. It is used by DMMG to "replace"
   the user provided Jacobian function. In fact, it calls the user provided one at each level.
*/
/*
          Version for matrix-free Jacobian 
*/
#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobian_MF"
int DMMGComputeJacobian_MF(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DMMG       *dmmg = (DMMG*)ptr;
  int        ierr,i,nlevels = dmmg[0]->nlevels;
  SLES       sles,lsles;
  PC         pc;
  PetscTruth ismg;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as user context which should contain DMMG");

  /* The finest level matrix is "shared" by the corresponding SNES object so we need
     only call MatAssemblyXXX() on it to indicate it is being used in a new solve */
  ierr = MatAssemblyBegin(dmmg[nlevels-1]->J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(dmmg[nlevels-1]->J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     The other levels MUST be told the vector from which we are doing the differencing
  */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    ierr = MGGetSmoother(pc,nlevels-1,&lsles);CHKERRQ(ierr);
    ierr = SLESSetOperators(lsles,DMMGGetFine(dmmg)->J,DMMGGetFine(dmmg)->J,*flag);CHKERRQ(ierr);

    for (i=nlevels-1; i>0; i--) {

      /* restrict X to coarse grid */
      ierr = MatRestrict(dmmg[i]->R,X,dmmg[i-1]->work2);CHKERRQ(ierr);
      X    = dmmg[i-1]->work2;      

      /* scale to "natural" scaling for that grid */
      ierr = VecPointwiseMult(dmmg[i]->Rscale,X,X);CHKERRQ(ierr);

      ierr = MatSNESMFSetBase(dmmg[i-1]->J,X);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(dmmg[i-1]->J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(dmmg[i-1]->J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

      ierr = MGGetSmoother(pc,i-1,&lsles);CHKERRQ(ierr);
      ierr = SLESSetOperators(lsles,dmmg[i-1]->J,dmmg[i-1]->B,*flag);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
    Version for user provided Jacobian
*/
#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobian_User"
int DMMGComputeJacobian_User(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DMMG       *dmmg = (DMMG*)ptr;
  int        ierr,i,nlevels = dmmg[0]->nlevels;
  SLES       sles,lsles;
  PC         pc;
  PetscTruth ismg;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as user context which should contain DMMG");

  ierr = (*DMMGGetFine(dmmg)->computejacobian)(snes,X,J,B,flag,DMMGGetFine(dmmg));CHKERRQ(ierr);

  /* create coarse grid jacobian for preconditioner if multigrid is the preconditioner */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    ierr = MGGetSmoother(pc,nlevels-1,&lsles);CHKERRQ(ierr);
    ierr = SLESSetOperators(lsles,DMMGGetFine(dmmg)->J,DMMGGetFine(dmmg)->J,*flag);CHKERRQ(ierr);

    for (i=nlevels-1; i>0; i--) {

      /* restrict X to coarse grid */
      ierr = MatRestrict(dmmg[i]->R,X,dmmg[i-1]->x);CHKERRQ(ierr);
      X    = dmmg[i-1]->x;      

      /* scale to "natural" scaling for that grid */
      ierr = VecPointwiseMult(dmmg[i]->Rscale,X,X);CHKERRQ(ierr);

      /* form Jacobian on coarse grid */
      ierr = (*dmmg[i-1]->computejacobian)(snes,X,&dmmg[i-1]->J,&dmmg[i-1]->B,flag,dmmg[i-1]);CHKERRQ(ierr);

      ierr = MGGetSmoother(pc,i-1,&lsles);CHKERRQ(ierr);
      ierr = SLESSetOperators(lsles,dmmg[i-1]->J,dmmg[i-1]->B,*flag);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
/*
    Version for Jacobian computed via PETSc finite differencing. This is the same 
  as DMMGComputeJacobian_User() except passes in the fdcoloring as the private context
*/
#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobian_FD"
int DMMGComputeJacobian_FD(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DMMG       *dmmg = (DMMG*)ptr;
  int        ierr,i,nlevels = dmmg[0]->nlevels;
  SLES       sles,lsles;
  PC         pc;
  PetscTruth ismg;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as user context which should contain DMMG");

  ierr = (*DMMGGetFine(dmmg)->computejacobian)(snes,X,J,B,flag,DMMGGetFine(dmmg)->fdcoloring);CHKERRQ(ierr);

  /* create coarse grid jacobian for preconditioner if multigrid is the preconditioner */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    ierr = MGGetSmoother(pc,nlevels-1,&lsles);CHKERRQ(ierr);
    ierr = SLESSetOperators(lsles,DMMGGetFine(dmmg)->J,DMMGGetFine(dmmg)->J,*flag);CHKERRQ(ierr);

    for (i=nlevels-1; i>0; i--) {

      /* restrict X to coarse grid */
      ierr = MatRestrict(dmmg[i]->R,X,dmmg[i-1]->x);CHKERRQ(ierr);
      X    = dmmg[i-1]->x;      

      /* scale to "natural" scaling for that grid */
      ierr = VecPointwiseMult(dmmg[i]->Rscale,X,X);CHKERRQ(ierr);

      /* form Jacobian on coarse grid */
      ierr = (*dmmg[i-1]->computejacobian)(snes,X,&dmmg[i-1]->J,&dmmg[i-1]->B,flag,dmmg[i-1]->fdcoloring);CHKERRQ(ierr);

      ierr = MGGetSmoother(pc,i-1,&lsles);CHKERRQ(ierr);
      ierr = SLESSetOperators(lsles,dmmg[i-1]->J,dmmg[i-1]->B,*flag);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

extern int DMMGFormJacobianWithAD(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveSNES"
int DMMGSolveSNES(DMMG *dmmg,int level)
{
  int  ierr,nlevels = dmmg[0]->nlevels,its;

  PetscFunctionBegin;
  dmmg[0]->nlevels = level+1;
  ierr = SNESSolve(dmmg[level]->snes,dmmg[level]->x,&its);CHKERRQ(ierr);
  dmmg[0]->nlevels = nlevels;
  PetscFunctionReturn(0);
}

EXTERN int DMMGSetUpLevel(DMMG*,SLES,int);

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNES"
/*@C
    DMMGSetSNES - Sets the nonlinear function that defines the nonlinear set of equations
      to be solved will use the grid hierarchy

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
.   function - the function that defines the nonlinear system
-   jacobian - optional function to compute Jacobian

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy, DMMGSetSLES()

@*/
int DMMGSetSNES(DMMG *dmmg,int (*function)(SNES,Vec,Vec,void*),int (*jacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*))
{
  int         ierr,i,nlevels = dmmg[0]->nlevels;
  PetscTruth  usefd,snesmonitor;
  SLES        sles;
  PetscViewer ascii;
  MPI_Comm    comm;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as DMMG");

  ierr = PetscOptionsHasName(PETSC_NULL,"-dmmg_snes_monitor",&snesmonitor);CHKERRQ(ierr);
  /* create solvers for each level */
  for (i=0; i<nlevels; i++) {
    ierr = SNESCreate(dmmg[i]->comm,SNES_NONLINEAR_EQUATIONS,&dmmg[i]->snes);CHKERRQ(ierr);
    if (snesmonitor) {
      ierr = PetscObjectGetComm((PetscObject)dmmg[i]->snes,&comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIOpen(comm,"stdout",&ascii);CHKERRQ(ierr);
      ierr = PetscViewerASCIISetTab(ascii,nlevels-i);CHKERRQ(ierr);
      ierr = SNESSetMonitor(dmmg[i]->snes,SNESDefaultMonitor,ascii,(int(*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
    }
    if (dmmg[0]->matrixfree) {
      ierr = MatCreateSNESMF(dmmg[i]->snes,dmmg[i]->x,&dmmg[i]->J);CHKERRQ(ierr);
      if (!dmmg[i]->B) dmmg[i]->B = dmmg[i]->J;
      if (i != nlevels-1) {
        ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->work1);CHKERRQ(ierr);
        ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->work2);CHKERRQ(ierr);
        ierr = MatSNESMFSetFunction(dmmg[i]->J,dmmg[i]->work1,function,dmmg[i]);CHKERRQ(ierr);
      }
    }

    ierr = SNESGetSLES(dmmg[i]->snes,&sles);CHKERRQ(ierr);
    ierr = DMMGSetUpLevel(dmmg,sles,i+1);CHKERRQ(ierr);
    
    /*
       if the number of levels is > 1 then we want the coarse solve in the grid sequencing to use LU
       when possible 
    */
    if (nlevels > 1 && i == 0) {
      PCType     stype;
      PC         pc;
      SLES       csles;
      PetscTruth flg1,flg2,flg3;

      ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
      ierr = MGGetCoarseSolve(pc,&csles);CHKERRQ(ierr);
      ierr = SLESGetPC(csles,&pc);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PCILU,&flg1);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PCSOR,&flg2);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PETSC_NULL,&flg3);CHKERRQ(ierr);
      if (flg1 || flg2 || flg3) {
        ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
      }
    }

    ierr = SNESSetFromOptions(dmmg[i]->snes);CHKERRQ(ierr);
    dmmg[i]->solve = DMMGSolveSNES;
    dmmg[i]->computejacobian = jacobian;
    dmmg[i]->computefunction = function;
  }

  ierr = PetscOptionsHasName(PETSC_NULL,"-dmmg_fd",&usefd);CHKERRQ(ierr);
  if ((!jacobian && !dmmg[0]->matrixfree) || usefd) {
    ISColoring iscoloring;
    for (i=0; i<nlevels; i++) {
      ierr = DMGetColoring(dmmg[i]->dm,IS_COLORING_LOCAL,MATMPIAIJ,&iscoloring,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(dmmg[i]->J,iscoloring,&dmmg[i]->fdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(dmmg[i]->fdcoloring,(int(*)(void))function,dmmg[i]);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(dmmg[i]->fdcoloring);CHKERRQ(ierr);
      dmmg[i]->computejacobian = SNESDefaultComputeJacobianColor;
    }
#if defined(PETSC_HAVE_ADIC)
  } else if (jacobian == DMMGFormJacobianWithAD) {
    for (i=0; i<nlevels; i++) {
      ierr = DMGetColoring(dmmg[i]->dm,IS_COLORING_GHOSTED,MATMPIAIJ,&dmmg[i]->iscoloring,PETSC_NULL);CHKERRQ(ierr);
    }
#endif
  }

  for (i=0; i<nlevels; i++) {
    if (dmmg[i]->matrixfree) {
      ierr = SNESSetJacobian(dmmg[i]->snes,dmmg[i]->J,dmmg[i]->B,DMMGComputeJacobian_MF,dmmg);CHKERRQ(ierr);
    } else if (dmmg[i]->computejacobian == SNESDefaultComputeJacobianColor) {
      ierr = SNESSetJacobian(dmmg[i]->snes,dmmg[i]->J,dmmg[i]->B,DMMGComputeJacobian_FD,dmmg);CHKERRQ(ierr);
    } else {
      ierr = SNESSetJacobian(dmmg[i]->snes,dmmg[i]->J,dmmg[i]->B,DMMGComputeJacobian_User,dmmg);CHKERRQ(ierr);
    }
    ierr = SNESSetFunction(dmmg[i]->snes,dmmg[i]->b,function,dmmg[i]);CHKERRQ(ierr);
  }

  /* Create interpolation scaling */
  for (i=1; i<nlevels; i++) {
    ierr = DMGetInterpolationScale(dmmg[i-1]->dm,dmmg[i]->dm,dmmg[i]->R,&dmmg[i]->Rscale);CHKERRQ(ierr);
  }

  for (i=0; i<nlevels-1; i++) {
    ierr = SNESSetOptionsPrefix(dmmg[i]->snes,"dmmg_levels_");CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetInitialGuess"
/*@C
    DMMGSetInitialGuess - Sets the function that computes an initial guess, if not given
         uses 0.

    Collective on DMMG and SNES

    Input Parameter:
+   dmmg - the context
-   guess - the function

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy, DMMGSetSLES()

@*/
int DMMGSetInitialGuess(DMMG *dmmg,int (*guess)(SNES,Vec,void*))
{
  int i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  for (i=0; i<nlevels; i++) {
    dmmg[i]->initialguess = guess;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMGFormFunction"
/* 
   DMMGFormFunction - This is a universal global FormFunction used by the DMMG code
     when the user provides a local function.

   Input Parameters:
.  snes - the SNES context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector

 */
int DMMGFormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  DMMG        dmmg = (DMMG)ptr;
  int         ierr;
  Scalar      **x,**f;
  Vec         localX;
  DA          da = (DA)dmmg->dm;
  DALocalInfo info;

  PetscFunctionBegin;
  ierr = DAGetLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
  */
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DAVecGetArray((DA)dmmg->dm,localX,(void**)&x);CHKERRQ(ierr);
  ierr = DAVecGetArray((DA)dmmg->dm,F,(void**)&f);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
   ierr = (*dmmg->computefunctionlocal)(x,f,&info,dmmg->user);CHKERRQ(ierr); 

  /*
     Restore vectors
  */
  ierr = DAVecRestoreArray((DA)dmmg->dm,localX,(void**)&x);CHKERRQ(ierr);
  ierr = DAVecRestoreArray((DA)dmmg->dm,F,(void**)&f);CHKERRQ(ierr);

  ierr = DARestoreLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);

  return 0; 
} 

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNESLocal"
/*@C
    DMMGSetSNESLocal - Sets the local user function that defines the nonlinear set of equations
          that will use the grid hierarchy

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
.   function - the function that defines the nonlinear system
-   jacobian - optional function to compute Jacobian


    Level: advanced

.seealso DMMGCreate(), DMMGDestroy, DMMGSetSLES()

@*/
int DMMGSetSNESLocal(DMMG *dmmg,int (*function)(Scalar **,Scalar**,DALocalInfo*,void*),int (*jacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*))
{
  int         ierr,i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  ierr = DMMGSetSNES(dmmg,DMMGFormFunction,jacobian);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    dmmg[i]->computefunctionlocal = function;
  }
  PetscFunctionReturn(0);
}


/* ---------------------------------------------------------------------------------------------------------------------------*/

#if defined(PETSC_HAVE_ADIC)

typedef struct {
	double value;
	double grad[ad_GRAD_MAX];
} DERIV_TYPE;

#define DERIV_val(a) ((a).value)
#define DERIV_grad(a) ((a).grad)
void ad_AD_Init();
void ad_AD_Final();
#include "adic_utils.h"
#include "ad_grad.h"

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PetscGetStructArray2d"
static int PetscGetStructArray2d(int xs,int ys,int xm,int ym,int structsize,void ***ptr,void **array_start)
{
  int  ierr,j;
  void *tmpptr;

  PetscFunctionBegin;
  ierr  = PetscMalloc((ym+1)*sizeof(void *)+xm*ym*structsize,(void **)array_start);CHKERRQ(ierr);
  *ptr  = (void**)(*array_start + xm*ym*structsize - ys*sizeof(void*));

  for(j=ys;j<ys+ym;j++) {
    (*ptr)[j] = *array_start + structsize*(xm*(j-ys) - xs);
  }
  ierr = PetscMemzero(*array_start,xm*ym*structsize);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMGFormJacobianWithAD"
/*
    DMMGFormJacobianWithAD - Evaluates the Jacobian via AD when the user has provide
        a local form function
*/
int DMMGFormJacobianWithAD(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DMMG           dmmg = (DMMG) ptr;
  int            ierr,i,j,k,l,*colors = dmmg->iscoloring->colors;
  int            xs,ys,xm,ym;
  int            gxs,gys,gxm,gym;
  int            mx,my,*colorptr;
  Vec            localX;
  Scalar         **x;
  DALocalInfo    info;
  DERIV_TYPE     **ad_x,**ad_f,*derivptr;
  int            dim,dof,stencil_size,stencil_width;
  DAStencilType  stencil_type;
  DAPeriodicType periodicity;
  DA             da= (DA) dmmg->dm;
  int            deriv_type_size;
  void           *ad_xstart,*ad_fstart;

  PetscFunctionBegin;
  ad_AD_Init(dmmg->iscoloring->n);
  deriv_type_size = my_AD_GetDerivTypeSize();

  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);


  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = DAGetInfo(da,&dim,&mx,&my,0,0,0,0,&dof,&stencil_width,&periodicity,&stencil_type);CHKERRQ(ierr);

  /* Verify that this DA type is supported */
  if ((dim != 2) || (stencil_width != 1) || (stencil_type != DA_STENCIL_STAR)
      || (periodicity != DA_NONPERIODIC)) {
    SETERRQ(0,"This DA type is not yet supported. Sorry.\n");
  }

  stencil_size = 5;

  /*
     Get local grid boundaries
  */
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL);CHKERRQ(ierr);

  /*  printf("myid = %d x[%d,%d], y[%d,%d], gx[%d,%d], gy[%d,%d]\n",myid); */



  /*
     Get pointer to vector data
  */
  ierr = DAVecGetArray(da,localX,(void**)&x);CHKERRQ(ierr);

  /* allocate space for derivative objects.  */
  ierr = PetscGetStructArray2d(gxs*dof,gys,gxm*dof,gym,deriv_type_size,(void ***)&ad_x,&ad_xstart); CHKERRQ(ierr);
  for(j=gys;j<gys+gym;j++) {
    for(i=dof*gxs;i<dof*(gxs+gxm);i++) {
      DERIV_val(ad_x[j][i]) = x[j][i];
    }
  }
  ierr = PetscGetStructArray2d(xs*dof,ys,xm*dof,ym,deriv_type_size,(void ***)&ad_f,&ad_fstart); CHKERRQ(ierr);


  ad_AD_ResetIndep();
  for(j=gys;j<gys+gym;j++) {
    derivptr = &(ad_x[j][dof*gxs]);
    colorptr = &(colors[j*gxm*dof]);
    ad_AD_SetIndepArrayColored(derivptr,dof*gxm,&colors[(j-gys)*gxm*dof]);
  }
  ad_AD_IncrementTotalGradSize(dmmg->iscoloring->n);
  ad_AD_SetIndepDone();

  /* 
     Compute entries for the locally owned part of the Jacobian.
  */
  ierr = (*dmmg->ad_computefunctionlocal)((Scalar**)ad_x,(Scalar**)ad_f,&info,dmmg->user);CHKERRQ(ierr); 

  ierr = DAVecRestoreArray(da,localX,(void**)&x);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);

  /* stick the values into the matrix */
  ierr = MatADSetColoring_MPIAIJ(*B,dmmg->iscoloring);CHKERRQ(ierr);
  ierr = MatADSetValues_MPIAIJ(*B,(Scalar**)&ad_f[ys][xs],ad_GRAD_MAX);CHKERRQ(ierr);

  /* Assemble true Jacobian; if it is different */
  ierr  = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr  = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr  = MatSetOption(*B,MAT_NEW_NONZERO_LOCATION_ERR);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNESLocalWithAD"
/*@C
    DMMGSetSNESLocalWithAD - Sets the local user function that defines the nonlinear set of equations
          that will use the grid hierarchy and its AD derivate function

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
.   function - the function that defines the nonlinear system
-   jacobian - AD function to compute Jacobian

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy, DMMGSetSLES()

@*/
int DMMGSetSNESLocalWithAD(DMMG *dmmg,int (*function)(Scalar **,Scalar**,DALocalInfo*,void*),int (*ad_function)(Scalar **,Scalar**,DALocalInfo*,void*))
{
  int ierr,i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  ierr = DMMGSetSNES(dmmg,DMMGFormFunction,DMMGFormJacobianWithAD);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    dmmg[i]->computefunctionlocal    = function;
    dmmg[i]->ad_computefunctionlocal = ad_function;
  }
  PetscFunctionReturn(0);
}

#endif





