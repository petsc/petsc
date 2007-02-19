
static char help[] = "Model multi-physics solver\n\n";

/*
     A model "multi-physics" solver based on the Vincent Mousseau's reactor core pilot code.

     There are three grids:

             ---------------------
            |                    |
            |                    |         DA 3
   nyfv+2   |                    |
            |                    |
            ---------------------

             ---------------------
            |                    |
   nyv+2    |                    |         DA 2
            |                    |
            |                    |
            ---------------------

            ---------------------          DA 1
                   nxv

*/

#include "petscdmmg.h"

typedef struct {                 /* Fluid unknowns */
  PetscScalar prss;
  PetscScalar ergg;
  PetscScalar ergf;
  PetscScalar alfg;
  PetscScalar velg;
  PetscScalar velf;
} FluidField;

typedef struct {                 /* Fuel unknowns */
  PetscScalar phii;
  PetscScalar prei;
} FuelField;

extern PetscErrorCode FormInitialGuess(DMMG,Vec);
extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg;               /* multilevel grid structure */
  PetscErrorCode ierr;
  MPI_Comm       comm;
  DA             da;
  DMComposite    pack;

  PetscInt       nxv = 3, nyv = 3, nyfv = 3;  

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;


  PreLoadBegin(PETSC_TRUE,"SetUp");

    /*
       Create the DMComposite object to manage the three grids/physics. 
       We use a 1d decomposition along the y direction (since one of the grids is 1d).

    */
    ierr = DMCompositeCreate(comm,&pack);CHKERRQ(ierr);
    ierr = DACreate1d(comm,DA_NONPERIODIC,nxv,6,1,0,&da);CHKERRQ(ierr);
    ierr = DMCompositeAddDA(pack,da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);
    ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,nxv,nyv+2,PETSC_DETERMINE,1,1,1,0,0,&da);CHKERRQ(ierr);
    ierr = DMCompositeAddDA(pack,da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);
    ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,nxv,nyfv+2,PETSC_DETERMINE,1,2,1,0,0,&da);CHKERRQ(ierr);
    ierr = DMCompositeAddDA(pack,da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);
   
    /*
       Create the solver object and attach the grid/physics info 
    */
    ierr = DMMGCreate(comm,2,0,&dmmg);CHKERRQ(ierr);
    ierr = DMMGSetDM(dmmg,(DM)pack);CHKERRQ(ierr);
    ierr = DMCompositeDestroy(pack);CHKERRQ(ierr);
    CHKMEMQ;


    ierr = DMMGSetInitialGuess(dmmg,FormInitialGuess);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMMGSetSNES(dmmg,FormFunction,0);CHKERRQ(ierr);
    CHKMEMQ;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve the nonlinear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PreLoadStage("Solve");
    ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  PreLoadEnd();

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */


/* 
   FormInitialGuessLocal* Forms the initial SOLUTION for the fluid, cladding and fuel

 */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessLocalFluid"
PetscErrorCode FormInitialGuessLocalFluid(DALocalInfo *info1,FluidField *f)
{
  PetscInt       i;

  PetscFunctionBegin;

  for (i=info1->xs; i<info1->xs+info1->xm; i++) {
    f[i].prss = 22.3;
    f[i].ergg = 12;
    f[i].ergf = 11;;
    f[i].alfg = 9;
    f[i].velg = 12;
    f[i].velf = 3;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessLocalCladding"
PetscErrorCode FormInitialGuessLocalCladding(DALocalInfo *info2,PetscScalar **T)
{
  PetscInt i,j;

  PetscFunctionBegin;

  for (i=info2->xs; i<info2->xs+info2->xm; i++) {
    for (j=info2->ys;j<info2->ys+info2->ym; j++) {
      T[j][i] = 0.0;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessLocalFuel"
PetscErrorCode FormInitialGuessLocalFuel(DALocalInfo *info2,FuelField **F)
{
  PetscInt i,j;

  PetscFunctionBegin;

  for (i=info2->xs; i<info2->xs+info2->xm; i++) {
    for (j=info2->ys;j<info2->ys+info2->ym; j++) {
      F[j][i].phii = 0.0;
      F[j][i].prei = 0.0;
    }
  }
  PetscFunctionReturn(0);
}

/* 
   FormFunctionLocal* - Forms user provided function

*/
#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocalFluid"
PetscErrorCode FormFunctionLocalFluid(DALocalInfo *info1,FluidField *u,FluidField *f)
{
  PetscInt       i;

  PetscFunctionBegin;

  for (i=info1->xs; i<info1->xs+info1->xm; i++) {
    f[i].prss = u[i].prss;
    f[i].ergg = u[i].ergg;
    f[i].ergf = u[i].ergf;
    f[i].alfg = u[i].alfg;
    f[i].velg = u[i].velg;
    f[i].velf = u[i].velf;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocalCladding"
PetscErrorCode FormFunctionLocalCladding(DALocalInfo *info2,PetscScalar **T,PetscScalar **f)
{
  PetscInt i,j;

  PetscFunctionBegin;

  for (i=info2->xs; i<info2->xs+info2->xm; i++) {
    for (j=info2->ys;j<info2->ys+info2->ym; j++) {
      f[j][i] = T[j][i];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocalFuel"
PetscErrorCode FormFunctionLocalFuel(DALocalInfo *info2,FuelField **U,FuelField **F)
{
  PetscInt i,j;

  PetscFunctionBegin;

  for (i=info2->xs; i<info2->xs+info2->xm; i++) {
    for (j=info2->ys;j<info2->ys+info2->ym; j++) {
      F[j][i].phii = U[j][i].phii;
      F[j][i].prei = U[j][i].prei;
    }
  }
  PetscFunctionReturn(0);
}

 
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/* 
   FormInitialGuess  - Unwraps the global solution vector and passes its local peices into the user function

 */
PetscErrorCode FormInitialGuess(DMMG dmmg,Vec X)
{
  DMComposite    dm = (DMComposite)dmmg->dm;
  DALocalInfo    info1,info2,info3;
  DA             da1,da2,da3;
  FluidField     *x1;
  PetscScalar    **x2;
  FuelField      **x3;
  Vec            X1,X2,X3;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCompositeGetEntries(dm,&da1,&da2,&da3);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da2,&info2);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da3,&info3);CHKERRQ(ierr);

  /* Access the three subvectors in X */
  ierr = DMCompositeGetAccess(dm,X,&X1,&X2,&X3);CHKERRQ(ierr);

  /* Access the arrays inside the subvectors of X */
  ierr = DAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DAVecGetArray(da3,X3,(void**)&x3);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormInitialGuessLocalFluid(&info1,x1);CHKERRQ(ierr);
  ierr = FormInitialGuessLocalCladding(&info2,x2);CHKERRQ(ierr);
  ierr = FormInitialGuessLocalFuel(&info3,x3);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da3,X3,(void**)&x3);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(dm,X,&X1,&X2,&X3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormFunction"
/* 
   FormFunction  - Unwraps the input vector and passes its local ghosted pieces into the user function

 */
PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *ctx)
{
  DMMG           dmmg = (DMMG)ctx;
  DMComposite    dm = (DMComposite)dmmg->dm;
  DALocalInfo    info1,info2,info3;
  DA             da1,da2,da3;
  FluidField     *x1,*f1;
  PetscScalar    **x2,**f2;
  FuelField      **x3,**f3;
  Vec            X1,X2,X3,F1,F2,F3;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCompositeGetEntries(dm,&da1,&da2,&da3);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da2,&info2);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da3,&info3);CHKERRQ(ierr);

  /* Get local vectors to hold ghosted parts of X */
  ierr = DMCompositeGetLocalVectors(dm,&X1,&X2,&X3);CHKERRQ(ierr);
  ierr = DMCompositeScatter(dm,X,X1,X2,X3);CHKERRQ(ierr);

  /* Access the arrays inside the subvectors of X */
  ierr = DAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DAVecGetArray(da3,X3,(void**)&x3);CHKERRQ(ierr);

  /* Access the three subvectors in F */
  ierr = DMCompositeGetAccess(dm,F,&F1,&F2,&F3);CHKERRQ(ierr);

  /* Access the arrays inside the subvectors of F */
  ierr = DAVecGetArray(da1,F1,(void**)&f1);CHKERRQ(ierr);
  ierr = DAVecGetArray(da2,F2,(void**)&f2);CHKERRQ(ierr);
  ierr = DAVecGetArray(da3,F3,(void**)&f3);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormFunctionLocalFluid(&info1,x1,f1);CHKERRQ(ierr);
  ierr = FormFunctionLocalCladding(&info2,x2,f2);CHKERRQ(ierr);
  ierr = FormFunctionLocalFuel(&info3,x3,f3);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da3,X3,(void**)&x3);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(dm,&X1,&X2,&X3);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da1,F1,(void**)&f1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,F2,(void**)&f2);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da3,F3,(void**)&f3);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(dm,F,&F1,&F2,&F3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
