
static char help[] = "Model multi-physics solver. Modified from src/snes/examples/tutorials/ex19.c \n\\n";

/* ------------------------------------------------------------------------
    See ex19.c for discussion of the problem 

    Examples of command line options:
      ./mp -dmmg_jacobian_mf_fd_operator
      ./mp -dmcomposite_dense_jacobian #inefficient, but compute entire Jacobian for testing
      ./mp -couple -snes_monitor_short -pc_type fieldsplit -ksp_monitor_short -pc_fieldsplit_type schur -fieldsplit_ksp_monitor_short -fieldsplit_1_ksp_type fgmres -fieldsplit_0_ksp_type gmres -fieldsplit_0_ksp_monitor_short -pc_fieldsplit_schur_precondition self 
  ----------------------------------------------------------------------------------------- */
#include "mp.h"

extern PetscErrorCode FormInitialGuessComp(DMMG,Vec);
extern PetscErrorCode FormFunctionComp(SNES,Vec,Vec,void*); 
extern PetscLogEvent  EVENT_FORMFUNCTIONLOCAL1, EVENT_FORMFUNCTIONLOCAL2;
extern PetscErrorCode FormCoupleLocations(DM,Mat,PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt,PetscInt);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg_comp;          /* multilevel grid structure */
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,my,its;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  SNES           snes;
  DM             da1,da2;
  DM             pack;
  PetscBool      couple = PETSC_FALSE;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscLogEventRegister("FormFunc1", 0,&EVENT_FORMFUNCTIONLOCAL1);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("FormFunc2", 0,&EVENT_FORMFUNCTIONLOCAL2);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create user context, set problem data, create vector data structures.
     Also, compute the initial guess.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup Physics 1: 
        - Lap(U) - Grad_y(Omega) = 0
	- Lap(V) + Grad_x(Omega) = 0
	- Lap(Omega) + Div([U*Omega,V*Omega]) - GR*Grad_x(T) = 0
        where T is given by the given x.temp
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(comm,DMDA_NONPERIODIC,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,3,1,0,0,&da1);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da1,0,"x-velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da1,1,"y-velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da1,2,"Omega");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup Physics 2: 
        - Lap(T) + PR*Div([U*T,V*T]) = 0        
        where U and V are given by the given x.u and x.v
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(comm,DMDA_NONPERIODIC,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da2);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da2,0,"temperature");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create the DMComposite object to manage the two grids/physics. 
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCompositeCreate(comm,&pack);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(pack,da1);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(pack,da2);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-couple",&couple);CHKERRQ(ierr);
  if (couple) {
    ierr = DMCompositeSetCoupling(pack,FormCoupleLocations);CHKERRQ(ierr);
  }

  /* Create the solver object and attach the grid/physics info */
  ierr = DMMGCreate(comm,1,&user,&dmmg_comp);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg_comp,pack);CHKERRQ(ierr);
  ierr = DMMGSetISColoringType(dmmg_comp,IS_COLORING_GLOBAL);CHKERRQ(ierr);

  ierr = DMMGSetInitialGuess(dmmg_comp,FormInitialGuessComp);CHKERRQ(ierr);
  ierr = DMMGSetSNES(dmmg_comp,FormFunctionComp,0);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg_comp);CHKERRQ(ierr);
  ierr = DMMGSetUp(dmmg_comp);CHKERRQ(ierr);

  /* Problem parameters (velocity of lid, prandtl, and grashof numbers) */
  ierr = DMDAGetInfo(da1,PETSC_NULL,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  user.lidvelocity = 1.0/(mx*my);
  user.prandtl     = 1.0;
  user.grashof     = 1000.0; 
  ierr = PetscOptionsGetReal(PETSC_NULL,"-lidvelocity",&user.lidvelocity,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-prandtl",&user.prandtl,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-grashof",&user.grashof,PETSC_NULL);CHKERRQ(ierr);

  /* Solve the nonlinear system */
  ierr = DMMGSolve(dmmg_comp);CHKERRQ(ierr); 
  snes = DMMGGetSNES(dmmg_comp);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Composite Physics: Number of Newton iterations = %D\n\n", its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free spaces 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDestroy(pack);CHKERRQ(ierr);
  ierr = DMDestroy(da1);CHKERRQ(ierr);
  ierr = DMDestroy(da2);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg_comp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessComp"
/* 
   FormInitialGuessComp - 
              Forms the initial guess for the composite model
              Unwraps the global solution vector and passes its local pieces into the user functions
 */
PetscErrorCode FormInitialGuessComp(DMMG dmmg,Vec X)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DM             dm = dmmg->dm;
  Vec            X1,X2;
  Field1         **x1;
  Field2         **x2;
  DMDALocalInfo    info1,info2;
  DM             da1,da2;

  PetscFunctionBegin;
  ierr = DMCompositeGetEntries(dm,&da1,&da2);CHKERRQ(ierr);
  /* Access the subvectors in X */
  ierr = DMCompositeGetAccess(dm,X,&X1,&X2);CHKERRQ(ierr);
  /* Access the arrays inside the subvectors of X */
  ierr = DMDAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da2,&info2);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormInitialGuessLocal1(&info1,x1);CHKERRQ(ierr);
  ierr = FormInitialGuessLocal2(&info2,x2,user);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(dm,X,&X1,&X2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionComp"
/* 
   FormFunctionComp  - Unwraps the input vector and passes its local ghosted pieces into the user function
*/
PetscErrorCode FormFunctionComp(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DMMG           dmmg = (DMMG)ctx;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DM             dm = dmmg->dm;
  DMDALocalInfo    info1,info2;
  DM             da1,da2;
  Field1         **x1,**f1;
  Field2         **x2,**f2;
  Vec            X1,X2,F1,F2;

  PetscFunctionBegin;

  ierr = DMCompositeGetEntries(dm,&da1,&da2);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da2,&info2);CHKERRQ(ierr);

  /* Get local vectors to hold ghosted parts of X */
  ierr = DMCompositeGetLocalVectors(dm,&X1,&X2);CHKERRQ(ierr);
  ierr = DMCompositeScatter(dm,X,X1,X2);CHKERRQ(ierr); 

  /* Access the arrays inside the subvectors of X */
  ierr = DMDAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);

  /* Access the subvectors in F. 
     These are not ghosted so directly access the memory locations in F */
  ierr = DMCompositeGetAccess(dm,F,&F1,&F2);CHKERRQ(ierr);

  /* Access the arrays inside the subvectors of F */  
  ierr = DMDAVecGetArray(da1,F1,(void**)&f1);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,F2,(void**)&f2);CHKERRQ(ierr);

  /* Evaluate local user provided function */    
  ierr = FormFunctionLocal1(&info1,x1,x2,f1,(void**)user);CHKERRQ(ierr);
  ierr = FormFunctionLocal2(&info2,x1,x2,f2,(void**)user);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da1,F1,(void**)&f1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,F2,(void**)&f2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(dm,F,&F1,&F2);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(dm,&X1,&X2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormCoupleLocations"
/* 
   Computes the coupling between DMDA1 and DMDA2. This determines the location of each coupling between DMDA1 and DMDA2.
   Input Parameters:
+     dmcomposite -
.     A - Jacobian matrix
.     dnz - number of nonzeros per row in DIAGONAL portion of local submatrix
.     onz - number of nonzeros per row in the OFF-DIAGONAL portion of local submatrix
.     __rstart - the global index of the first local row of A
.     __nrows - number of loacal rows
.     __start - the global index of the first local column
.     __end - the global index of the last local column + 1
*/
PetscErrorCode FormCoupleLocations(DM dmcomposite,Mat A,PetscInt *dnz,PetscInt *onz,PetscInt __rstart,PetscInt __nrows,PetscInt __start,PetscInt __end)
{
  PetscInt       i,j,cols[2],istart,jstart,in,jn,row,col,M;
  PetscErrorCode ierr;
  DM             da1,da2;

  PetscFunctionBegin;
  PetscMPIInt rank;
  ierr = MPI_Comm_rank(((PetscObject)dmcomposite)->comm,&rank);CHKERRQ(ierr);
  /* printf("[%d] __rstart %d, __nrows %d, __start %d, __end %d,\n",rank,__rstart,__nrows,__start,__end);*/
  ierr =  DMCompositeGetEntries(dmcomposite,&da1,&da2);CHKERRQ(ierr);
  ierr =  DMDAGetInfo(da1,0,&M,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr  = DMDAGetCorners(da1,&istart,&jstart,PETSC_NULL,&in,&jn,PETSC_NULL);CHKERRQ(ierr);

  /* coupling from physics 1 to physics 2 */
  row = __rstart + 2;  /* global location of first omega on this process */
  col = __rstart + 3*in*jn;  /* global location of first temp on this process */
  for (j=jstart; j<jstart+jn; j++) {
    for (i=istart; i<istart+in; i++) {

      /* each omega is coupled to the temp to the left and right */
      if (i == 0) {
        cols[0] = col + 1;
        ierr = MatPreallocateLocation(A,row,1,cols,dnz,onz);CHKERRQ(ierr);
      } else if (i == M-1) {
        cols[0] = col - 1;
        ierr = MatPreallocateLocation(A,row,1,cols,dnz,onz);CHKERRQ(ierr);
      } else {
        cols[0] = col - 1;
        cols[1] = col + 1;
        ierr = MatPreallocateLocation(A,row,2,cols,dnz,onz);CHKERRQ(ierr);
      }
      row += 3;
      col += 1;
    }
  }

  /* coupling from physics 2 to physics 1 */
  col = __rstart;  /* global location of first u on this process */
  row = __rstart + 3*in*jn;  /* global location of first temp on this process */
  for (j=jstart; j<jstart+jn; j++) {
    for (i=istart; i<istart+in; i++) {

      /* temp is coupled to both u and v at each point */
      cols[0] = col;
      cols[1] = col + 1;
      ierr = MatPreallocateLocation(A,row,2,cols,dnz,onz);CHKERRQ(ierr); 
      row += 1;
      col += 3;
    }
  }

  PetscFunctionReturn(0);
}


