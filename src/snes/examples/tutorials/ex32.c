
static char help[] = "Model multi-physics solver. Modified from ex19.c \n\\n\
The 2D driven cavity problem is solved in a velocity-vorticity formulation.\n\
The flow can be driven with the lid or with bouyancy or both:\n\
  -lidvelocity <lid>, where <lid> = dimensionless velocity of lid\n\
  -grashof <gr>, where <gr> = dimensionless temperature gradent\n\
  -prandtl <pr>, where <pr> = dimensionless thermal/momentum diffusity ratio\n\
  -contours : draw contour plots of solution\n\n";

/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DA^using distributed arrays;
   Concepts: multicomponent
   Processors: n
T*/

/* ------------------------------------------------------------------------

    We thank David E. Keyes for contributing the driven cavity discretization
    within this example code.

    This problem is modeled by the partial differential equation system
  
	- Lap(U) - Grad_y(Omega) = 0
	- Lap(V) + Grad_x(Omega) = 0
	- Lap(Omega) + Div([U*Omega,V*Omega]) - GR*Grad_x(T) = 0
	- Lap(T) + PR*Div([U*T,V*T]) = 0

    in the unit square, which is uniformly discretized in each of x and
    y in this simple encoding.

    No-slip, rigid-wall Dirichlet conditions are used for [U,V].
    Dirichlet conditions are used for Omega, based on the definition of
    vorticity: Omega = - Grad_y(U) + Grad_x(V), where along each
    constant coordinate boundary, the tangential derivative is zero.
    Dirichlet conditions are used for T on the left and right walls,
    and insulation homogeneous Neumann conditions are used for T on
    the top and bottom walls. 

    A finite difference approximation with the usual 5-point stencil 
    is used to discretize the boundary value problem to obtain a 
    nonlinear system of equations.  Upwinding is used for the divergence
    (convective) terms and central for the gradient (source) terms.
    
    The Jacobian can be either
      * formed via finite differencing using coloring (the default), or
      * applied matrix-free via the option -snes_mf 
        (for larger grid problems this variant may not converge 
        without a preconditioner due to ill-conditioning).

  ------------------------------------------------------------------------- */
#include "petscsnes.h"
#include "petscda.h"
#include "petscdmmg.h"

/* User-defined routines and data structure */
typedef struct {
  PetscScalar u,v,omega,temp;
} Field;

typedef struct {
  PetscScalar u,v,omega;
} Field1;

typedef struct {
  PetscScalar temp;
} Field2;

typedef struct {
  PassiveReal  lidvelocity,prandtl,grashof;  /* physical parameters */
  PetscTruth   draw_contours;                /* flag - 1 indicates drawing contours */
  DMMG         *dmmg,*dmmg_comp;             /* used by MySolutionView() */
  DMMG         *dmmg1,*dmmg2; /* passing objects of sub-physics into the composite physics - used by FormInitialGuessComp() */
  DMComposite  pack;
  PetscTruth   COMPOSITE_MODEL;
  Field        **x;   /* passing DMMGGetx(dmmg) - final solution of original coupled physics */
  Field1       **x1;  /* passing local ghosted vector array of Physics 1 */
  Field2       **x2;  /* passing local ghosted vector array of Physics 2 */
} AppCtx;

extern PetscErrorCode FormInitialGuessLocal(DMMG,Vec);
extern PetscErrorCode FormInitialGuessLocal1(DMMG,Vec);
extern PetscErrorCode FormInitialGuessLocal2(DMMG,Vec);
extern PetscErrorCode FormInitialGuessComp(DMMG,Vec);

extern PetscErrorCode FormFunctionLocal(DALocalInfo*,Field**,Field**,void*);
extern PetscErrorCode FormFunctionLocal1(DALocalInfo*,Field1**,Field1**,void*);
extern PetscErrorCode FormFunctionLocal2(DALocalInfo*,Field2**,Field2**,void*);
extern PetscErrorCode FormFunctionComp(SNES,Vec,Vec,void*);

extern PetscErrorCode MySolutionView(MPI_Comm,PetscInt,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg,*dmmg1,*dmmg2,*dmmg_comp; /* multilevel grid structure */
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,my,its,dof,nlevels=1; 
  PetscErrorCode ierr;
  MPI_Comm       comm;
  SNES           snes;
  DA             da;
  PetscTruth     ViewSolu=PETSC_FALSE,CompSolu=PETSC_TRUE,DoSubPhysics=PETSC_FALSE;
  Vec            solu_true,solu_local;

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;

  ierr = PetscOptionsGetInt(PETSC_NULL,"-nlevels",&nlevels,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMMGCreate(comm,nlevels,&user,&dmmg);CHKERRQ(ierr);

  /*
    Create distributed array multigrid object (DMMG) to manage parallel grid and vectors
    for principal unknowns (x) and governing residuals (f)
  */
  dof  = 4;
  ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,dof,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);

  ierr = DAGetInfo(DMMGGetDA(dmmg),0,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  /* Problem parameters (velocity of lid, prandtl, and grashof numbers) */
  user.lidvelocity = 1.0/(mx*my);
  user.prandtl     = 1.0;
  user.grashof     = 1.0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-lidvelocity",&user.lidvelocity,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-prandtl",&user.prandtl,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-grashof",&user.grashof,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-contours",&user.draw_contours);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-view_solu",&ViewSolu);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-do_subphysics",&DoSubPhysics);CHKERRQ(ierr);

  ierr = DASetFieldName(DMMGGetDA(dmmg),0,"x-velocity");CHKERRQ(ierr);
  ierr = DASetFieldName(DMMGGetDA(dmmg),1,"y-velocity");CHKERRQ(ierr);
  ierr = DASetFieldName(DMMGGetDA(dmmg),2,"Omega");CHKERRQ(ierr);
  ierr = DASetFieldName(DMMGGetDA(dmmg),3,"temperature");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create user context, set problem data, create vector data structures.
     Also, compute the initial guess.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Create nonlinear solver context */
  ierr = DMMGSetSNESLocal(dmmg,FormFunctionLocal,0,ad_FormFunctionLocal,admf_FormFunctionLocal);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"lid velocity = %G, prandtl # = %G, grashof # = %G\n",
		       user.lidvelocity,user.prandtl,user.grashof);CHKERRQ(ierr);
  ierr = DMMGSetInitialGuess(dmmg,FormInitialGuessLocal);CHKERRQ(ierr);

  /* Solve the nonlinear system */
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 

  snes = DMMGGetSNES(dmmg);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Origianl Physics: Number of Newton iterations = %D\n\n", its);CHKERRQ(ierr);
 
  /* Save the ghosted local solu_true to be used by Physics 1 and Physics 2 */
  da        = DMMGGetDA(dmmg);
  solu_true = DMMGGetx(dmmg);
  ierr = DACreateLocalVector(da,&solu_local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,solu_true,INSERT_VALUES,solu_local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,solu_true,INSERT_VALUES,solu_local);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,solu_local,(Field **)&user.x);CHKERRQ(ierr);

  /* Visualize solution */
  if (user.draw_contours) {
    ierr = VecView(DMMGGetx(dmmg),PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }
  if (ViewSolu){
    user.dmmg = dmmg;
    ierr = MySolutionView(comm,0,&user);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup Physics 1: 
        - Lap(U) - Grad_y(Omega) = 0
	- Lap(V) + Grad_x(Omega) = 0
	- Lap(Omega) + Div([U*Omega,V*Omega]) - GR*Grad_x(T) = 0
        where T is given by the computed x.temp
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMMGCreate(comm,nlevels,&user,&dmmg1);CHKERRQ(ierr);
  dof  = 3;
  user.COMPOSITE_MODEL = PETSC_FALSE;
  ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,dof,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg1,(DM)da);CHKERRQ(ierr);
  ierr = DASetFieldName(da,0,"x-velocity");CHKERRQ(ierr);
  ierr = DASetFieldName(da,1,"y-velocity");CHKERRQ(ierr);
  ierr = DASetFieldName(da,2,"Omega");CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);

  if (DoSubPhysics){
    ierr = DMMGSetSNESLocal(dmmg1,FormFunctionLocal1,0,ad_FormFunctionLocal,admf_FormFunctionLocal);CHKERRQ(ierr);
    ierr = DMMGSetFromOptions(dmmg1);CHKERRQ(ierr);
    ierr = DMMGSetInitialGuess(dmmg1,FormInitialGuessLocal1);CHKERRQ(ierr);

    ierr = DMMGSolve(dmmg1);CHKERRQ(ierr); 
    snes = DMMGGetSNES(dmmg1);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"SubPhysics 1, Number of Newton iterations = %D\n\n", its);CHKERRQ(ierr);
  
    if (ViewSolu){ /* View individial componets of the solution */   
      user.dmmg1 = dmmg1;
      ierr = MySolutionView(comm,1,&user);CHKERRQ(ierr);
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup Physics 2: 
        - Lap(T) + PR*Div([U*T,V*T]) = 0        
        where U and V are given by the computed x.u and x.v
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMMGCreate(comm,nlevels,&user,&dmmg2);CHKERRQ(ierr);
  dof  = 1;
  user.COMPOSITE_MODEL = PETSC_FALSE;
  ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,dof,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg2,(DM)da);CHKERRQ(ierr);
  ierr = DASetFieldName(da,0,"temperature");CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);

  if (DoSubPhysics){
    ierr = DMMGSetSNESLocal(dmmg2,FormFunctionLocal2,0,ad_FormFunctionLocal,admf_FormFunctionLocal);CHKERRQ(ierr);
    ierr = DMMGSetFromOptions(dmmg2);CHKERRQ(ierr);
    ierr = DMMGSetInitialGuess(dmmg2,FormInitialGuessLocal2);CHKERRQ(ierr);

    ierr = DMMGSolve(dmmg2);CHKERRQ(ierr); 
    snes = DMMGGetSNES(dmmg2);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"SubPhysics 2, Number of Newton iterations = %D\n\n", its);CHKERRQ(ierr);

    if (ViewSolu){ /* View individial componets of the solution */
      user.dmmg2 = dmmg2;
      ierr = MySolutionView(comm,2,&user);CHKERRQ(ierr);
    }
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create the DMComposite object to manage the two grids/physics. 
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCompositeCreate(comm,&user.pack);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.pack,(DM)DMMGGetDA(dmmg1));CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.pack,(DM)DMMGGetDA(dmmg2));CHKERRQ(ierr);

  /* Create the solver object and attach the grid/physics info */
  ierr = DMMGCreate(comm,nlevels,&user,&dmmg_comp);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg_comp,(DM)user.pack);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = DMMGSetISColoringType(dmmg_comp,IS_COLORING_GLOBAL);CHKERRQ(ierr);

  user.dmmg1 = dmmg1;
  user.dmmg2 = dmmg2;
  user.COMPOSITE_MODEL = PETSC_TRUE;
  ierr = DMMGSetInitialGuess(dmmg_comp,FormInitialGuessComp);CHKERRQ(ierr);
  ierr = DMMGSetSNES(dmmg_comp,FormFunctionComp,0);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg_comp);CHKERRQ(ierr);

  /* Solve the nonlinear system */
  ierr = DMMGSolve(dmmg_comp);CHKERRQ(ierr); 
  snes = DMMGGetSNES(dmmg_comp);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Composite Physics: Number of Newton iterations = %D\n\n", its);CHKERRQ(ierr);

  /* Compare the solutions obtained from dmmg and dmmg_comp */
  if (ViewSolu || CompSolu){ 
    /* View the solution of composite physics (phy_num = 3)
       or Compare solutions (phy_num > 3) */
    user.dmmg      = dmmg;
    user.dmmg_comp = dmmg_comp;
    ierr = MySolutionView(comm,4,&user);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free spaces 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DAVecRestoreArray(DMMGGetDA(dmmg),solu_local,(Field **)&user.x);CHKERRQ(ierr);
  ierr = VecDestroy(solu_local);CHKERRQ(ierr);
  ierr = DMCompositeDestroy(user.pack);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg1);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg2);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg_comp);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessLocal"
/* 
   FormInitialGuessLocal - Forms initial approximation for this process

   Input Parameters:
     user - user-defined application context
     X    - vector (DA local vector)

   Output Parameter:
     X - vector with the local values set
 */
PetscErrorCode FormInitialGuessLocal(DMMG dmmg,Vec X)
{
  AppCtx         *user = (AppCtx*)dmmg->user;
  DA             da = (DA)dmmg->dm;
  PetscInt       i,j,mx,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscReal      grashof,dx;
  Field          **x;

  grashof = user->grashof;
  ierr = DAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dx  = 1.0/(mx-1);

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)
  */
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecResetArraystoreArray() when you no longer need access to
         the array.
  */
  ierr = DAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
     Initial condition is motionless fluid and equilibrium temperature
     U = V = Omega = 0.0; Temp[x_i, y_j] = x_i;
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x[j][i].u     = 0.0;
      x[j][i].v     = 0.0;
      x[j][i].omega = 0.0;
      x[j][i].temp  = (grashof>0)*i*dx;  
    }
  }

  /* Restore vector */
  ierr = DAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessLocal1"
/* Form initial guess for Physic 1 */
PetscErrorCode FormInitialGuessLocal1(DMMG dmmg,Vec X)
{
  AppCtx         *user = (AppCtx*)dmmg->user;
  DA             da = (DA)dmmg->dm;
  PetscInt       i,j,mx,xs,ys,xm,ym;
  PetscErrorCode ierr;
  Field1         **x;

  ierr = DAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  ierr = DAVecGetArray(da,X,&x);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x[j][i].u     = 0.0;
      x[j][i].v     = 0.0;
      x[j][i].omega = 0.0;
    }
  }
  ierr = DAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessLocal2"
/* Form initial guess for Physic 2 */
PetscErrorCode FormInitialGuessLocal2(DMMG dmmg,Vec X)
{
  AppCtx         *user = (AppCtx*)dmmg->user;
  DA             da = (DA)dmmg->dm;
  PetscInt       i,j,mx,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscReal      grashof,dx;
  Field2         **x;

  grashof = user->grashof;
  ierr = DAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dx  = 1.0/(mx-1);

  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  ierr = DAVecGetArray(da,X,&x);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x[j][i].temp  = (grashof>0)*i*dx;  
    }
  }
  ierr = DAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
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
  DMMG           *dmmg1 = user->dmmg1,*dmmg2=user->dmmg2;
  DMComposite    dm = (DMComposite)dmmg->dm;
  Vec            X1,X2;

  PetscFunctionBegin;
  /* Access the subvectors in X */
  ierr = DMCompositeGetAccess(dm,X,&X1,&X2);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormInitialGuessLocal1(*dmmg1,X1);CHKERRQ(ierr);
  ierr = FormInitialGuessLocal2(*dmmg2,X2);CHKERRQ(ierr);

  ierr = DMCompositeRestoreAccess(dm,X,&X1,&X2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/* 
   FormFunctionLocal - Function evaluation for this process

   Input Parameters:
     info - DALocalInfo context
     x    - (DA local) vector array including ghost points
     f    - (DA local) vector array to be evaluated 
     ptr  - user-defined application context

   Output Parameter:
     f - array holds local function values 
     
     f.u     = - Lap(U) - Grad_y(Omega)
     f.v     = - Lap(V) + Grad_x(Omega)
     f.omega = - Lap(Omega) + Div([U*Omega,V*Omega]) - GR*Grad_x(T)
     f.temp  = - Lap(T) + PR*Div([U*T,V*T])
*/
PetscErrorCode FormFunctionLocal(DALocalInfo *info,Field **x,Field **f,void *ptr)
 {
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       xints,xinte,yints,yinte,i,j;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal      grashof,prandtl,lid;
  PetscScalar    u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;

  PetscFunctionBegin;
  grashof = user->grashof;  
  prandtl = user->prandtl;
  lid     = user->lidvelocity;

  /* 
     Define mesh intervals ratios for uniform grid.

     Note: FD formulae below are normalized by multiplying through by
     local volume element (i.e. hx*hy) to obtain coefficients O(1) in two dimensions.

     
  */
  dhx = (PetscReal)(info->mx-1);  dhy = (PetscReal)(info->my-1);
  hx = 1.0/dhx;                   hy = 1.0/dhy;
  hxdhy = hx*dhy; /* hx/hy */     hydhx = hy*dhx; /* hy/hx */

  xints = info->xs; xinte = info->xs+info->xm; yints = info->ys; yinte = info->ys+info->ym;

  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    j = 0;
    yints = yints + 1;
    /* bottom edge: 
         U = V = 0; Omega = -Grad_y(U)+Grad_x(V); dT/dn = 0; */
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega + (x[j+1][i].u - x[j][i].u)*dhy; 
      f[j][i].temp  = x[j][i].temp-x[j+1][i].temp;
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == info->my) {
    j = info->my - 1;
    yinte = yinte - 1;
    /* top edge:
         U = lid; V = 0; Omega = -Grad_y(U)+Grad_x(V); dT/dn = 0; */
    for (i=info->xs; i<info->xs+info->xm; i++) {
        f[j][i].u     = x[j][i].u - lid;
        f[j][i].v     = x[j][i].v;
        f[j][i].omega = x[j][i].omega + (x[j][i].u - x[j-1][i].u)*dhy; 
	f[j][i].temp  = x[j][i].temp-x[j-1][i].temp;
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i = 0;
    xints = xints + 1;
    /* left edge:
         U = V = 0; Omega = -Grad_y(U)+Grad_x(V); T = 0; */
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i+1].v - x[j][i].v)*dhx; 
      f[j][i].temp  = x[j][i].temp;
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info->mx) {
    i = info->mx - 1;
    xinte = xinte - 1;
    /* right edge:
         U = V = 0; Omega = -Grad_y(U)+Grad_x(V); T = grashof; */ 
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i].v - x[j][i-1].v)*dhx; 
      f[j][i].temp  = x[j][i].temp - (PetscReal)(grashof>0);
    }
  }

  /* Compute over the interior points */
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      /* convective coefficients for upwinding */
      vx = x[j][i].u; avx = PetscAbsScalar(vx);
      vxp = .5*(vx+avx); vxm = .5*(vx-avx);
      vy = x[j][i].v; avy = PetscAbsScalar(vy);
      vyp = .5*(vy+avy); vym = .5*(vy-avy);

      /* U velocity */
      u          = x[j][i].u;
      uxx        = (2.0*u - x[j][i-1].u - x[j][i+1].u)*hydhx;
      uyy        = (2.0*u - x[j-1][i].u - x[j+1][i].u)*hxdhy;
      f[j][i].u  = uxx + uyy - .5*(x[j+1][i].omega-x[j-1][i].omega)*hx;

      /* V velocity */
      u          = x[j][i].v;
      uxx        = (2.0*u - x[j][i-1].v - x[j][i+1].v)*hydhx;
      uyy        = (2.0*u - x[j-1][i].v - x[j+1][i].v)*hxdhy;
      f[j][i].v  = uxx + uyy + .5*(x[j][i+1].omega-x[j][i-1].omega)*hy;

      /* Omega */
      u          = x[j][i].omega;
      uxx        = (2.0*u - x[j][i-1].omega - x[j][i+1].omega)*hydhx;
      uyy        = (2.0*u - x[j-1][i].omega - x[j+1][i].omega)*hxdhy;
      f[j][i].omega = uxx + uyy  
                      + (vxp*(u - x[j][i-1].omega) + vxm*(x[j][i+1].omega - u)) * hy 
		      + (vyp*(u - x[j-1][i].omega) + vym*(x[j+1][i].omega - u)) * hx 
		      -	.5 * grashof * (x[j][i+1].temp - x[j][i-1].temp) * hy;

      /* Temperature */
      u            = x[j][i].temp;
      uxx          = (2.0*u - x[j][i-1].temp - x[j][i+1].temp)*hydhx;
      uyy          = (2.0*u - x[j-1][i].temp - x[j+1][i].temp)*hxdhy;
      f[j][i].temp = uxx + uyy  
		     + prandtl * ((vxp*(u - x[j][i-1].temp) + vxm*(x[j][i+1].temp - u)) * hy
		     + (vyp*(u - x[j-1][i].temp) + vym*(x[j+1][i].temp - u)) * hx);
    }
  }

  /* Flop count (multiply-adds are counted as 2 operations) */
  ierr = PetscLogFlops(84.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal1"
/* 
    Form function for Physics 1: 
      same as FormFunctionLocal() except without f.temp and x.temp.
      the input x.temp comes from the solu_true 
*/
PetscErrorCode FormFunctionLocal1(DALocalInfo *info,Field1 **x,Field1 **f,void *ptr)
 {
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       xints,xinte,yints,yinte,i,j;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal      grashof,prandtl,lid;
  PetscScalar    u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;
  Field          **solu=user->x;
  Field2         **solu2=user->x2;

  PetscFunctionBegin;
  grashof = user->grashof;  
  prandtl = user->prandtl;
  lid     = user->lidvelocity;

  dhx = (PetscReal)(info->mx-1);  dhy = (PetscReal)(info->my-1);
  hx = 1.0/dhx;                   hy = 1.0/dhy;
  hxdhy = hx*dhy;                 hydhx = hy*dhx;

  xints = info->xs; xinte = info->xs+info->xm; yints = info->ys; yinte = info->ys+info->ym;

  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    j = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega + (x[j+1][i].u - x[j][i].u)*dhy;  
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == info->my) {
    j = info->my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
        f[j][i].u     = x[j][i].u - lid;
        f[j][i].v     = x[j][i].v;
        f[j][i].omega = x[j][i].omega + (x[j][i].u - x[j-1][i].u)*dhy; 
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i = 0;
    xints = xints + 1;
    /* left edge */
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i+1].v - x[j][i].v)*dhx; 
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info->mx) {
    i = info->mx - 1;
    xinte = xinte - 1;
    /* right edge */ 
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i].v - x[j][i-1].v)*dhx; 
    }
  }

  /* Compute over the interior points */
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {      
      /* convective coefficients for upwinding */
	vx = x[j][i].u; avx = PetscAbsScalar(vx);
        vxp = .5*(vx+avx); vxm = .5*(vx-avx);
	vy = x[j][i].v; avy = PetscAbsScalar(vy);
        vyp = .5*(vy+avy); vym = .5*(vy-avy);

	/* U velocity */
        u          = x[j][i].u;
        uxx        = (2.0*u - x[j][i-1].u - x[j][i+1].u)*hydhx;
        uyy        = (2.0*u - x[j-1][i].u - x[j+1][i].u)*hxdhy;
        f[j][i].u  = uxx + uyy - .5*(x[j+1][i].omega-x[j-1][i].omega)*hx;

	/* V velocity */
        u          = x[j][i].v;
        uxx        = (2.0*u - x[j][i-1].v - x[j][i+1].v)*hydhx;
        uyy        = (2.0*u - x[j-1][i].v - x[j+1][i].v)*hxdhy;
        f[j][i].v  = uxx + uyy + .5*(x[j][i+1].omega-x[j][i-1].omega)*hy;

	/* Omega */
        u          = x[j][i].omega;
        uxx        = (2.0*u - x[j][i-1].omega - x[j][i+1].omega)*hydhx;
        uyy        = (2.0*u - x[j-1][i].omega - x[j+1][i].omega)*hxdhy;   
	f[j][i].omega = uxx + uyy 
			+ (vxp*(u - x[j][i-1].omega) + vxm*(x[j][i+1].omega - u)) * hy 
          + (vyp*(u - x[j-1][i].omega) + vym*(x[j+1][i].omega - u)) * hx;
        if (user->COMPOSITE_MODEL){
          f[j][i].omega += - .5 * grashof * (solu2[j][i+1].temp - solu2[j][i-1].temp) * hy;
        } else {
          f[j][i].omega += - .5 * grashof * (solu[j][i+1].temp - solu[j][i-1].temp) * hy;
        }
    }
  }
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal2"
/* 
    Form function for Physics 2: 
      same as FormFunctionLocal() but only has f.temp and x.temp.
      the input x.u and x.v come from solu_true 
*/
PetscErrorCode FormFunctionLocal2(DALocalInfo *info,Field2 **x,Field2 **f,void *ptr)
 {
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       xints,xinte,yints,yinte,i,j;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal      grashof,prandtl,lid;
  PetscScalar    u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;
  Field          **solu=user->x;
  Field1         **solu1=user->x1;

  PetscFunctionBegin;
  grashof = user->grashof;  
  prandtl = user->prandtl;
  lid     = user->lidvelocity;

  dhx = (PetscReal)(info->mx-1);  dhy = (PetscReal)(info->my-1);
  hx = 1.0/dhx;                   hy = 1.0/dhy;
  hxdhy = hx*dhy;                 hydhx = hy*dhx;

  xints = info->xs; xinte = info->xs+info->xm; yints = info->ys; yinte = info->ys+info->ym;

  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    j = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].temp  = x[j][i].temp-x[j+1][i].temp;
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == info->my) {
    j = info->my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].temp  = x[j][i].temp-x[j-1][i].temp;
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i = 0;
    xints = xints + 1;
    /* left edge */
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].temp  = x[j][i].temp;
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info->mx) {
    i = info->mx - 1;
    xinte = xinte - 1;
    /* right edge */ 
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].temp  = x[j][i].temp - (PetscReal)(grashof>0);
    }
  }

  /* Compute over the interior points */
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      /* convective coefficients for upwinding */
      if (user->COMPOSITE_MODEL){
        vx = solu1[j][i].u; vy = solu1[j][i].v; 
      } else {
        vx = solu[j][i].u; vy = solu[j][i].v; 
      }
      avx = PetscAbsScalar(vx); 
      vxp = .5*(vx+avx); vxm = .5*(vx-avx);
      avy = PetscAbsScalar(vy); 
      vyp = .5*(vy+avy); 
      vym = .5*(vy-avy);

      /* Temperature */
      u             = x[j][i].temp;
      uxx           = (2.0*u - x[j][i-1].temp - x[j][i+1].temp)*hydhx;
      uyy           = (2.0*u - x[j-1][i].temp - x[j+1][i].temp)*hxdhy;
      f[j][i].temp  =  uxx + uyy  
		      + prandtl * ((vxp*(u - x[j][i-1].temp) + vxm*(x[j][i+1].temp - u)) * hy 
		      + (vyp*(u - x[j-1][i].temp) + vym*(x[j+1][i].temp - u)) * hx);
    }
  }
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
  DMComposite    dm = (DMComposite)dmmg->dm;
  DALocalInfo    info1,info2;
  DA             da1,da2;
  Field1         **x1,**f1;
  Field2         **x2,**f2;
  Vec            X1,X2,F1,F2;

  PetscFunctionBegin;
  ierr = DMCompositeGetEntries(dm,&da1,&da2);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da2,&info2);CHKERRQ(ierr);

  /* Get local vectors to hold ghosted parts of X */
  ierr = DMCompositeGetLocalVectors(dm,&X1,&X2);CHKERRQ(ierr);
  ierr = DMCompositeScatter(dm,X,X1,X2);CHKERRQ(ierr); 

  /* Access the arrays inside the subvectors of X */
  ierr = DAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);

  /* Access the subvectors in F. 
     These are not ghosted and directly access the memory locations in F */
  ierr = DMCompositeGetAccess(dm,F,&F1,&F2);CHKERRQ(ierr);

  /* Access the arrays inside the subvectors of F */  
  ierr = DAVecGetArray(da1,F1,(void**)&f1);CHKERRQ(ierr);
  ierr = DAVecGetArray(da2,F2,(void**)&f2);CHKERRQ(ierr);

  /* Evaluate local user provided function */    
  user->x2 = x2; /* used by FormFunctionLocal1() */
  ierr = FormFunctionLocal1(&info1,x1,f1,(void**)user);CHKERRQ(ierr);
  user->x1 = x1; /* used by FormFunctionLocal2() */
  ierr = FormFunctionLocal2(&info2,x2,f2,(void**)user);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da1,F1,(void**)&f1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,F2,(void**)&f2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(dm,F,&F1,&F2);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(dm,&X1,&X2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MySolutionView"
/* 
   Visualize solutions
*/
PetscErrorCode MySolutionView(MPI_Comm comm,PetscInt phy_num,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)ctx;
  DMMG           *dmmg = user->dmmg;
  DA             da=DMMGGetDA(dmmg);
  Field          **x = user->x;
  Field1         **x1 = user->x1;
  PetscInt       i,j,mx,xs,ys,xm,ym;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);
  
  switch (phy_num){
  case 0:
    if (size == 1){
      ierr = PetscPrintf(PETSC_COMM_SELF,"Original Physics %d U,V,Omega,Temp: \n",phy_num);
      ierr = PetscPrintf(PETSC_COMM_SELF,"-----------------------------------\n");
      for (j=ys; j<ys+ym; j++) {
        for (i=xs; i<xs+xm; i++) {
          ierr = PetscPrintf(PETSC_COMM_SELF,"x[%d,%d] = %g, %g, %g, %g\n",j,i,x[j][i].u,x[j][i].v,x[j][i].omega,x[j][i].temp);
        }
      }    
    }
    break;
  case 1:
    if (size == 1){
      ierr = PetscPrintf(PETSC_COMM_SELF,"SubPhysics %d: U,V,Omega: \n",phy_num);
      ierr = PetscPrintf(PETSC_COMM_SELF,"------------------------\n");
      DMMG *dmmg1=user->dmmg1;
      Vec  solu_true = DMMGGetx(dmmg1);
      DA   da=DMMGGetDA(dmmg1);
      Field1 **x1;
      ierr = DAVecGetArray(da,solu_true,&x1);CHKERRQ(ierr);
      for (j=ys; j<ys+ym; j++) {
        for (i=xs; i<xs+xm; i++) {
          ierr = PetscPrintf(PETSC_COMM_SELF,"x[%d,%d] = %g, %g, %g\n",j,i,x1[j][i].u,x1[j][i].v,x1[j][i].omega);
        }
      } 
      ierr = DAVecRestoreArray(da,solu_true,&x1);CHKERRQ(ierr);
    }
    break;
  case 2:
    if (size == 1){
      ierr = PetscPrintf(PETSC_COMM_SELF,"SubPhysics %d: Temperature: \n",phy_num);
      ierr = PetscPrintf(PETSC_COMM_SELF,"--------------------------\n");
      DMMG *dmmg2=user->dmmg2;
      Vec  solu_true = DMMGGetx(dmmg2);
      DA   da=DMMGGetDA(dmmg2);
      Field2 **x2;
      ierr = DAVecGetArray(da,solu_true,&x2);CHKERRQ(ierr);
      for (j=ys; j<ys+ym; j++) {
        for (i=xs; i<xs+xm; i++) {
          ierr = PetscPrintf(PETSC_COMM_SELF,"x[%d,%d] = %g\n",j,i,x2[j][i].temp);
        }
      } 
      ierr = DAVecRestoreArray(da,solu_true,&x2);CHKERRQ(ierr);
    }
    break;
  default:
    if (size == 1){
      DMMG        *dmmg_comp=user->dmmg_comp;
      DA          da1,da2,da=DMMGGetDA(dmmg);
      Vec         X1,X2,solu_true = DMMGGetx(dmmg);
      Field       **x;
      Field1      **x1;
      Field2      **x2;
      DMComposite dm = (DMComposite)(*dmmg_comp)->dm;
      PetscReal   err,err_tmp;
      if (phy_num == 3){
        ierr = PetscPrintf(PETSC_COMM_SELF,"Composite physics %d, U,V,Omega,Temp: \n",phy_num);
        ierr = PetscPrintf(PETSC_COMM_SELF,"------------------------------------\n");
        ierr = PetscPrintf(PETSC_COMM_SELF,"Composite physics, U,V,Omega,Temp: \n");CHKERRQ(ierr);
      }
      ierr = DAVecGetArray(da,solu_true,&x);CHKERRQ(ierr);
      ierr = DMCompositeGetEntries(dm,&da1,&da2);CHKERRQ(ierr);
      ierr = DMCompositeGetLocalVectors(dm,&X1,&X2);CHKERRQ(ierr);
      ierr = DAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
      ierr = DAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);

      err = 0.0;
      for (j=ys; j<ys+ym; j++) {
        for (i=xs; i<xs+xm; i++) {
          err_tmp = PetscAbs(x[j][i].u-x1[j][i].u) + PetscAbs(x[j][i].v-x1[j][i].v) + PetscAbs(x[j][i].omega-x1[j][i].omega);
          err_tmp += PetscAbs(x[j][i].temp-x2[j][i].temp);
          if (err < err_tmp) err = err_tmp; 
          if (phy_num == 3){
            ierr = PetscPrintf(PETSC_COMM_SELF,"x[%d,%d] = %g, %g, %g, %g\n",j,i,x1[j][i].u,x1[j][i].v,x1[j][i].omega,x2[j][i].temp);
          }
        }
      }
      ierr = PetscPrintf(PETSC_COMM_SELF,"|solu - solu_comp| =  %g\n",err);CHKERRQ(ierr);
      ierr = DAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
      ierr = DAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
      ierr = DMCompositeRestoreLocalVectors(dm,&X1,&X2);CHKERRQ(ierr);
      ierr = DAVecRestoreArray(da,solu_true,&x);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
