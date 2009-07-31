
static char help[] = "Grad-Shafranov solver for one dimensional CHI equilibrium.\n\
The command line options include:\n\
  -n <n> number of grid points\n\
  -psi_axis <axis> \n\
  -r_min <min> \n\
  -param <param> \n\n";

/*T
   Concepts: SNES^parallel CHI equilibrium
   Concepts: DA^using distributed arrays;
   Processors: n
T*/

/* ------------------------------------------------------------------------

   Grad-Shafranov solver for one dimensional CHI equilibrium
  
    A finite difference approximation with the usual 3-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.

    Contributed by Xianzhu Tang, LANL

    An interesting feature of this example is that as you refine the grid
    (with a larger -n <n> you cannot force the residual norm as small. This
    appears to be due to "NOISE" in the function, the FormFunctionLocal() cannot
    be computed as accurately with a finer grid.
  ------------------------------------------------------------------------- */

#include "petscda.h"
#include "petscsnes.h"

/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines, FormJacobian() and
   FormFunction().
*/
typedef struct {
  DA            da;               /* distributed array data structure */
  Vec           psi,r;            /* solution, residual vectors */
  Mat           A,J;              /* Jacobian matrix */
  Vec           coordinates;      /* grid coordinates */
  PassiveReal   psi_axis,psi_bdy;
  PassiveReal   r_min;
  PassiveReal   param;            /* test problem parameter */
} AppCtx;

#define GdGdPsi(r,u)      (((r) < 0.05) ? 0.0 : (user->param*((r)-0.05)*(1.0-(u)*(u))*(1.0-(u)*(u))))
#define CurrentWire(r)    (((r) < .05) ? -3.E2 : 0.0)
#define GdGdPsiPrime(r,u) (((r) < 0.05) ? 0.0 : -4.*(user->param*((r)-0.05)*(1.0-(u)*(u)))*u)

/* 
   User-defined routines
*/
extern PetscErrorCode FormInitialGuess(AppCtx*,Vec);
extern PetscErrorCode FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode FormFunctionLocal(DALocalInfo*,PetscScalar*,PetscScalar*,AppCtx*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES                   snes;                 /* nonlinear solver */
  AppCtx                 user;                 /* user-defined work context */
  PetscInt               its;                  /* iterations for convergence */
  PetscTruth             fd_jacobian = PETSC_FALSE;
  PetscTruth             adicmf_jacobian = PETSC_FALSE;
  PetscInt               grids = 100, dof = 1, stencil_width = 1; 
  PetscErrorCode         ierr;
  PetscReal              fnorm;
  MatFDColoring          matfdcoloring = 0;
  ISColoring             iscoloring;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscInitialize(&argc,&argv,(char *)0,help);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize problem parameters
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  user.psi_axis=0.0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-psi_axis",&user.psi_axis,PETSC_NULL);CHKERRQ(ierr);
  user.psi_bdy=1.0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-psi_bdy",&user.psi_bdy,PETSC_NULL);CHKERRQ(ierr);
  user.r_min=0.0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-r_min",&user.r_min,PETSC_NULL);CHKERRQ(ierr);
  user.param=-1.E1;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-param",&user.param,PETSC_NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DA) to manage parallel grid and vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&grids,PETSC_NULL);CHKERRQ(ierr);
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,grids,dof,stencil_width,PETSC_NULL,&user.da);CHKERRQ(ierr);
  
  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DACreateGlobalVector(user.da,&user.psi);CHKERRQ(ierr);
  ierr = VecDuplicate(user.psi,&user.r);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set local function evaluation routine
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = DASetLocalFunction(user.da,(DALocalFunction1)FormFunctionLocal);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,user.r,SNESDAFormFunction,&user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine

     Set Jacobian matrix data structure and default Jacobian evaluation
     routine. User can override with:
     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                (unless user explicitly sets preconditioner) 
     -snes_mf_operator : form preconditioning matrix as set by the user,
                         but use matrix-free approx for Jacobian-vector
                         products within Newton-Krylov method

     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-fd_jacobian",&fd_jacobian,0);CHKERRQ(ierr);
  /*
       Note that fd_jacobian DOES NOT compute the finite difference approximation to 
    the ENTIRE Jacobian. Rather it removes the global coupling from the Jacobian and
    computes the finite difference approximation only for the "local" coupling.

       Thus running with fd_jacobian and not -snes_mf_operator or -adicmf_jacobian
    won't converge.
  */
  if (!fd_jacobian) {
    ierr      = MatCreate(PETSC_COMM_WORLD,&user.J);CHKERRQ(ierr);
    ierr      = MatSetSizes(user.J,PETSC_DECIDE,PETSC_DECIDE,grids,grids);CHKERRQ(ierr);
    ierr      = MatSetType(user.J,MATAIJ);CHKERRQ(ierr);
    ierr      = MatSetFromOptions(user.J);CHKERRQ(ierr);
    ierr      = MatSeqAIJSetPreallocation(user.J,5,PETSC_NULL);CHKERRQ(ierr);
    ierr      = MatMPIAIJSetPreallocation(user.J,5,PETSC_NULL,3,PETSC_NULL);CHKERRQ(ierr);
    user.A    = user.J;
  } else {
    ierr      = DAGetMatrix(user.da,MATAIJ,&user.J);CHKERRQ(ierr);
    user.A    = user.J;
  }

  ierr = PetscOptionsGetTruth(PETSC_NULL,"-adicmf_jacobian",&adicmf_jacobian,0);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ADIC)
  if (adicmf_jacobian) {
    ierr = DASetLocalAdicMFFunction(user.da,admf_FormFunctionLocal);CHKERRQ(ierr);
    ierr = MatRegisterDAAD();CHKERRQ(ierr);
    ierr = MatCreateDAAD(user.da,&user.A);CHKERRQ(ierr);
    ierr = MatDAADSetSNES(user.A,snes);CHKERRQ(ierr);
    ierr = MatDAADSetCtx(user.A,&user);CHKERRQ(ierr);
  }    
#endif

  if (fd_jacobian) {
    ierr = DAGetColoring(user.da,IS_COLORING_GLOBAL,MATAIJ,&iscoloring);CHKERRQ(ierr);
    ierr = MatFDColoringCreate(user.J,iscoloring,&matfdcoloring);CHKERRQ(ierr);
    ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
    ierr = MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))SNESDAFormFunction,&user);CHKERRQ(ierr);
    ierr = MatFDColoringSetFromOptions(matfdcoloring);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,user.A,user.J,SNESDefaultComputeJacobianColor,matfdcoloring);CHKERRQ(ierr);
  } else {
    ierr = SNESSetJacobian(snes,user.A,user.J,FormJacobian,&user);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = FormInitialGuess(&user,user.psi);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSolve(snes,PETSC_NULL,user.psi);CHKERRQ(ierr); 
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Explicitly check norm of the residual of the solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESDAFormFunction(snes,user.psi,user.r,(void*)&user);CHKERRQ(ierr);
  ierr = VecNorm(user.r,NORM_MAX,&fnorm);CHKERRQ(ierr); 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %D fnorm %G\n",its,fnorm);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     Output the solution vector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  {
    PetscViewer view_out;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"psi.binary",FILE_MODE_WRITE,&view_out);CHKERRQ(ierr);
    ierr = VecView(user.psi,view_out);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(view_out);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"psi.out",&view_out);CHKERRQ(ierr);
    ierr = VecView(user.psi,view_out);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(view_out);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (user.A != user.J) {
    ierr = MatDestroy(user.A);CHKERRQ(ierr);
  }
  ierr = MatDestroy(user.J);CHKERRQ(ierr);
  if (matfdcoloring) {
    ierr = MatFDColoringDestroy(matfdcoloring);CHKERRQ(ierr);
  }
  ierr = VecDestroy(user.psi);CHKERRQ(ierr);
  ierr = VecDestroy(user.r);CHKERRQ(ierr);      
  ierr = SNESDestroy(snes);CHKERRQ(ierr);
  ierr = DADestroy(user.da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
PetscErrorCode FormInitialGuess(AppCtx *user,Vec X)
{
  PetscErrorCode ierr;
  PetscInt       i,Mx,xs,xm;
  PetscScalar    *x;

  PetscFunctionBegin;
  ierr = DAGetInfo(user->da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DAVecGetArray(user->da,X,&x);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)

  */
  ierr = DAGetCorners(user->da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
  */
  for (i=xs; i<xs+xm; i++) {
    x[i] = user->psi_axis + i*(user->psi_bdy - user->psi_axis)/(PetscReal)(Mx-1); 
  }

  /*
     Restore vector
  */
  ierr = DAVecRestoreArray(user->da,X,&x);CHKERRQ(ierr);

  /* 
     Check to see if we can import an initial guess from disk
  */
  {
    char         filename[PETSC_MAX_PATH_LEN];
    PetscTruth   flg;
    PetscViewer  view_in;
    PetscReal    fnorm;
    Vec          Y;
    ierr = PetscOptionsGetString(PETSC_NULL,"-initialGuess",filename,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&view_in);CHKERRQ(ierr);
      ierr = VecLoad(view_in,PETSC_NULL,&Y);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(view_in);CHKERRQ(ierr);
      ierr = VecMax(Y,PETSC_NULL,&user->psi_bdy);CHKERRQ(ierr);
      ierr = SNESDAFormFunction(PETSC_NULL,Y,user->r,(void*)user);CHKERRQ(ierr);
      ierr = VecNorm(user->r,NORM_2,&fnorm);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"In initial guess: psi_bdy = %f, fnorm = %G.\n",user->psi_bdy,fnorm);CHKERRQ(ierr);
      ierr = VecCopy(Y,X);CHKERRQ(ierr);
      ierr = VecDestroy(Y);CHKERRQ(ierr); 
    }
  }

  PetscFunctionReturn(0);
} 
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/* 
   FormFunctionLocal - Evaluates nonlinear function, F(x).
   
   Process adiC(36): FormFunctionLocal
   
*/
PetscErrorCode FormFunctionLocal(DALocalInfo *info,PetscScalar *x,PetscScalar *f,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       i,xint,xend;
  PetscReal      hx,dhx,r;
  PetscScalar    u,uxx,min = 100000.0,max = -10000.0;
  PetscScalar    psi_0=0.0, psi_a=1.0;
  
  PetscFunctionBegin;
  for (i=info->xs; i<info->xs + info->xm; i++) {
    if (x[i] > max) max = x[i];
    if (x[i] < min) min = x[i];
  }
  PetscGlobalMax(&max,&psi_a,PETSC_COMM_WORLD);
  PetscGlobalMin(&min,&psi_0,PETSC_COMM_WORLD); 

  hx     = 1.0/(PetscReal)(info->mx-1);  dhx    = 1.0/hx;
  
  /*
    Compute function over the locally owned part of the grid
  */
  if (info->xs == 0) {
    xint = info->xs + 1; f[0] = (4.*x[1] - x[2] - 3.*x[0])*dhx; /* f[0] = x[0] - user->psi_axis; */
  } 
  else {
    xint = info->xs;
  }
  if ((info->xs+info->xm) == info->mx) {
    xend = info->mx - 1; f[info->mx-1] = -(x[info->mx-1] - user->psi_bdy)*dhx;
  }
  else {
    xend = info->xs + info->xm;
  }
  
  for (i=xint; i<xend; i++) {
    r       = i*hx + user->r_min;   /* r coordinate */
    u       = (x[i]-psi_0)/(psi_a - psi_0);
    uxx     = ((r+0.5*hx)*(x[i+1]-x[i]) - (r-0.5*hx)*(x[i]-x[i-1]))*dhx; /* r*nabla^2\psi */
    f[i] = uxx  + r*GdGdPsi(r,u)*hx  + r*CurrentWire(r)*hx ;
  }

  ierr = PetscLogFlops(11.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
/*
   FormJacobian - Evaluates Jacobian matrix.

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  ptr - optional user-defined context, as set by SNESSetJacobian()

   Output Parameters:
.  A - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure

*/
PetscErrorCode FormJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;  /* user-defined application context */
  Mat            jac = *B;                /* Jacobian matrix */
  Vec            localX;
  PetscErrorCode ierr;
  PetscInt       col[6],row,i,xs,xm,Mx,xint,xend,imin, imax;
  PetscScalar    v[6],hx,dhx,*x;
  PetscReal      r, u, psi_0=0.0, psi_a=1.0;
  PetscTruth     assembled;

  PetscFunctionBegin;
  ierr = MatAssembled(*B,&assembled);CHKERRQ(ierr);
  if (assembled) {
    ierr = MatZeroEntries(*B);CHKERRQ(ierr);
  }

  ierr = DAGetLocalVector(user->da,&localX);CHKERRQ(ierr);
  ierr = DAGetInfo(user->da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx     = 1.0/(PetscReal)(Mx-1);  dhx    = 1.0/hx;

  imin = 0; imax = Mx-1;
  ierr = VecMin(X,&imin,&psi_0);CHKERRQ(ierr);
  ierr = VecMax(X,&imax,&psi_a);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"psi_0(%D)=%G, psi_a(%D)=%G.\n",imin,psi_0,imax,psi_a);CHKERRQ(ierr);

  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /*
     Get pointer to vector data
  */
  ierr = DAVecGetArray(user->da,localX,&x);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DAGetCorners(user->da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /* 
     Compute entries for the locally owned part of the Jacobian.
      - Currently, all PETSc parallel matrix formats are partitioned by
        contiguous chunks of rows across the processors. 
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly). 
      - Here, we set all entries for a particular row at once.
      - We can set matrix entries either using either
        MatSetValuesLocal() or MatSetValues(), as discussed above.
  */
  if (xs == 0) {
    xint = xs + 1; /* f[0] = 4.*x[1] - x[2] - 3.*x[0]; */
    row  = 0;     /* first row */
    v[0] = -3.0*dhx;                                              col[0]=row;
    v[1] = 4.0*dhx;                                               col[1]=row+1;
    v[2] = -1.0*dhx;                                              col[2]=row+2;
    ierr = MatSetValues(jac,1,&row,3,col,v,ADD_VALUES);CHKERRQ(ierr);
  } 
  else {
    xint = xs;
  }
  if ((xs+xm) == Mx) {
    xend = Mx - 1;   /* f[Mx-1] = x[Mx-1] - user->psi_bdy; */
    row  = Mx - 1;  /* last row */
    v[0] = -1.0*dhx;
    ierr = MatSetValue(jac,row,row,v[0],ADD_VALUES);CHKERRQ(ierr);
  }
  else {
    xend = xs + xm;
  }

  for (i=xint; i<xend; i++) {
    r       = i*hx + user->r_min;   /* r coordinate */
    u       = (x[i]-psi_0)/(psi_a - psi_0);
    /* uxx     = ((r+0.5*hx)*(x[i+1]-x[i]) - (r-0.5*hx)*(x[i]-x[i-1]))*dhx*dhx; */ /* r*nabla^2\psi */
    row  = i;
    v[0] = (r-0.5*hx)*dhx;                                                              col[0] = i-1;
    v[1] = -2.*r*dhx + hx*r*GdGdPsiPrime(r,u)/(psi_a - psi_0);                          col[1] = i;
    v[2] = (r+0.5*hx)*dhx;                                                              col[2] = i+1;
    v[3] = hx*r*GdGdPsiPrime(r,u)*(x[i] - psi_a)/((psi_a - psi_0)*(psi_a - psi_0));     col[3] = imin;
    v[4] = hx*r*GdGdPsiPrime(r,u)*(psi_0 - x[i])/((psi_a - psi_0)*(psi_a - psi_0));     col[4] = imax;
    ierr = MatSetValues(jac,1,&row,5,col,v,ADD_VALUES);CHKERRQ(ierr);
  }
  
  ierr = DAVecRestoreArray(user->da,localX,&x);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(user->da,&localX);CHKERRQ(ierr);

  /* 
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
  */
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     Normally since the matrix has already been assembled above; this
     would do nothing. But in the matrix free mode -snes_mf_operator
     this tells the "matrix-free" matrix that a new linear system solve
     is about to be done.
  */
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     Set flag to indicate that the Jacobian matrix retains an identical
     nonzero structure throughout all nonlinear iterations (although the
     values of the entries change). Thus, we can save some work in setting
     up the preconditioner (e.g., no need to redo symbolic factorization for
     ILU/ICC preconditioners).
      - If the nonzero structure of the matrix is different during
        successive linear solves, then the flag DIFFERENT_NONZERO_PATTERN
        must be used instead.  If you are unsure whether the matrix
        structure has changed or not, use the flag DIFFERENT_NONZERO_PATTERN.
      - Caution:  If you specify SAME_NONZERO_PATTERN, PETSc
        believes your assertion and does not check the structure
        of the matrix.  If you erroneously claim that the structure
        is the same when it actually is not, the new preconditioner
        will not function correctly.  Thus, use this optimization
        feature with caution!
  */
  *flag = SAME_NONZERO_PATTERN;

  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */

  PetscFunctionReturn(0);
}

