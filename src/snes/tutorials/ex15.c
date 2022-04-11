static const char help[] = "p-Bratu nonlinear PDE in 2d.\n\
We solve the  p-Laplacian (nonlinear diffusion) combined with\n\
the Bratu (solid fuel ignition) nonlinearity in a 2D rectangular\n\
domain, using distributed arrays (DAs) to partition the parallel grid.\n\
The command line options include:\n\
  -p <2>: `p' in p-Laplacian term\n\
  -epsilon <1e-05>: Strain-regularization in p-Laplacian\n\
  -lambda <6>: Bratu parameter\n\
  -blocks <bx,by>: number of coefficient interfaces in x and y direction\n\
  -kappa <1e-3>: diffusivity in odd regions\n\
\n";

/*F
    The $p$-Bratu problem is a combination of the $p$-Laplacian (nonlinear diffusion) and the Brutu solid fuel ignition problem.
    This problem is modeled by the partial differential equation

\begin{equation*}
        -\nabla\cdot (\eta \nabla u) - \lambda \exp(u) = 0
\end{equation*}

    on $\Omega = (-1,1)^2$ with closure

\begin{align*}
        \eta(\gamma) &= (\epsilon^2 + \gamma)^{(p-2)/2} & \gamma &= \frac 1 2 |\nabla u|^2
\end{align*}

    and boundary conditions $u = 0$ for $(x,y) \in \partial \Omega$

    A 9-point finite difference stencil is used to discretize
    the boundary value problem to obtain a nonlinear system of equations.
    This would be a 5-point stencil if not for the $p$-Laplacian's nonlinearity.
F*/

/*
      mpiexec -n 2 ./ex15 -snes_monitor -ksp_monitor log_summary
*/

/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

typedef enum {JAC_BRATU,JAC_PICARD,JAC_STAR,JAC_NEWTON} JacType;
static const char *const JacTypes[] = {"BRATU","PICARD","STAR","NEWTON","JacType","JAC_",0};

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormJacobianLocal() and
   FormFunctionLocal().
*/
typedef struct {
  PetscReal   lambda;         /* Bratu parameter */
  PetscReal   p;              /* Exponent in p-Laplacian */
  PetscReal   epsilon;        /* Regularization */
  PetscReal   source;         /* Source term */
  JacType     jtype;          /* What type of Jacobian to assemble */
  PetscBool   picard;
  PetscInt    blocks[2];
  PetscReal   kappa;
  PetscInt    initial;        /* initial conditions type */
} AppCtx;

/*
   User-defined routines
*/
static PetscErrorCode FormRHS(AppCtx*,DM,Vec);
static PetscErrorCode FormInitialGuess(AppCtx*,DM,Vec);
static PetscErrorCode FormFunctionLocal(DMDALocalInfo*,PetscScalar**,PetscScalar**,AppCtx*);
static PetscErrorCode FormFunctionPicardLocal(DMDALocalInfo*,PetscScalar**,PetscScalar**,AppCtx*);
static PetscErrorCode FormJacobianLocal(DMDALocalInfo*,PetscScalar**,Mat,Mat,AppCtx*);
static PetscErrorCode NonlinearGS(SNES,Vec,Vec,void*);

typedef struct _n_PreCheck *PreCheck;
struct _n_PreCheck {
  MPI_Comm    comm;
  PetscReal   angle;
  Vec         Ylast;
  PetscViewer monitor;
};
PetscErrorCode PreCheckCreate(MPI_Comm,PreCheck*);
PetscErrorCode PreCheckDestroy(PreCheck*);
PetscErrorCode PreCheckFunction(SNESLineSearch,Vec,Vec,PetscBool*,void*);
PetscErrorCode PreCheckSetFromOptions(PreCheck);

int main(int argc,char **argv)
{
  SNES                snes;                    /* nonlinear solver */
  Vec                 x,r,b;                   /* solution, residual, rhs vectors */
  AppCtx              user;                    /* user-defined work context */
  PetscInt            its;                     /* iterations for convergence */
  SNESConvergedReason reason;                  /* Check convergence */
  PetscBool           alloc_star;              /* Only allocate for the STAR stencil  */
  PetscBool           write_output;
  char                filename[PETSC_MAX_PATH_LEN] = "ex15.vts";
  PetscReal           bratu_lambda_max = 6.81,bratu_lambda_min = 0.;
  DM                  da;                      /* distributed array data structure */
  PreCheck            precheck = NULL;         /* precheck context for version in this file */
  PetscInt            use_precheck;            /* 0=none, 1=version in this file, 2=SNES-provided version */
  PetscReal           precheck_angle;          /* When manually setting the SNES-provided precheck function */
  SNESLineSearch      linesearch;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize problem parameters
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.lambda    = 0.0; user.p = 2.0; user.epsilon = 1e-5; user.source = 0.1; user.jtype = JAC_NEWTON;user.initial=-1;
  user.blocks[0] = 1; user.blocks[1] = 1; user.kappa = 1e-3;
  alloc_star     = PETSC_FALSE;
  use_precheck   = 0; precheck_angle = 10.;
  user.picard    = PETSC_FALSE;
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"p-Bratu options",__FILE__);
  {
    PetscInt two=2;
    PetscCall(PetscOptionsReal("-lambda","Bratu parameter","",user.lambda,&user.lambda,NULL));
    PetscCall(PetscOptionsReal("-p","Exponent `p' in p-Laplacian","",user.p,&user.p,NULL));
    PetscCall(PetscOptionsReal("-epsilon","Strain-regularization in p-Laplacian","",user.epsilon,&user.epsilon,NULL));
    PetscCall(PetscOptionsReal("-source","Constant source term","",user.source,&user.source,NULL));
    PetscCall(PetscOptionsEnum("-jtype","Jacobian approximation to assemble","",JacTypes,(PetscEnum)user.jtype,(PetscEnum*)&user.jtype,NULL));
    PetscCall(PetscOptionsName("-picard","Solve with defect-correction Picard iteration","",&user.picard));
    if (user.picard) {
      user.jtype = JAC_PICARD;
      PetscCheck(user.p == 3,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Picard iteration is only supported for p == 3");
      /* the Picard linearization only requires a 5 point stencil, while the Newton linearization requires a 9 point stencil */
      /* hence allocating the 5 point stencil gives the same convergence as the 9 point stencil since the extra stencil points are not used */
      PetscCall(PetscOptionsBool("-alloc_star","Allocate for STAR stencil (5-point)","",alloc_star,&alloc_star,NULL));
    }
    PetscCall(PetscOptionsInt("-precheck","Use a pre-check correction intended for use with Picard iteration 1=this version, 2=library","",use_precheck,&use_precheck,NULL));
    PetscCall(PetscOptionsInt("-initial","Initial conditions type (-1: default, 0: zero-valued, 1: peaked guess)","",user.initial,&user.initial,NULL));
    if (use_precheck == 2) {    /* Using library version, get the angle */
      PetscCall(PetscOptionsReal("-precheck_angle","Angle in degrees between successive search directions necessary to activate step correction","",precheck_angle,&precheck_angle,NULL));
    }
    PetscCall(PetscOptionsIntArray("-blocks","number of coefficient interfaces in x and y direction","",user.blocks,&two,NULL));
    PetscCall(PetscOptionsReal("-kappa","diffusivity in odd regions","",user.kappa,&user.kappa,NULL));
    PetscCall(PetscOptionsString("-o","Output solution in vts format","",filename,filename,sizeof(filename),&write_output));
  }
  PetscOptionsEnd();
  if (user.lambda > bratu_lambda_max || user.lambda < bratu_lambda_min) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"WARNING: lambda %g out of range for p=2\n",(double)user.lambda));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,alloc_star ? DMDA_STENCIL_STAR : DMDA_STENCIL_BOX,4,4,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DM; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da,&x));
  PetscCall(VecDuplicate(x,&r));
  PetscCall(VecDuplicate(x,&b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     User can override with:
     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                (unless user explicitly sets preconditioner)
     -snes_mf_operator : form preconditioning matrix as set by the user,
                         but use matrix-free approx for Jacobian-vector
                         products within Newton-Krylov method

     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set local function evaluation routine
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSetApplicationContext(da, &user));
  PetscCall(SNESSetDM(snes,da));
  if (user.picard) {
    /*
        This is not really right requiring the user to call SNESSetFunction/Jacobian but the DMDASNESSetPicardLocal() cannot access
        the SNES to set it
    */
    PetscCall(DMDASNESSetPicardLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionPicardLocal,
                                     (PetscErrorCode (*)(DMDALocalInfo*,void*,Mat,Mat,void*))FormJacobianLocal,&user));
    PetscCall(SNESSetFunction(snes,NULL,SNESPicardComputeFunction,&user));
    PetscCall(SNESSetJacobian(snes,NULL,NULL,SNESPicardComputeJacobian,&user));
  } else {
    PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,&user));
    PetscCall(DMDASNESSetJacobianLocal(da,(PetscErrorCode (*)(DMDALocalInfo*,void*,Mat,Mat,void*))FormJacobianLocal,&user));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSetNGS(snes,NonlinearGS,&user));
  PetscCall(SNESGetLineSearch(snes, &linesearch));
  /* Set up the precheck context if requested */
  if (use_precheck == 1) {      /* Use the precheck routines in this file */
    PetscCall(PreCheckCreate(PETSC_COMM_WORLD,&precheck));
    PetscCall(PreCheckSetFromOptions(precheck));
    PetscCall(SNESLineSearchSetPreCheck(linesearch,PreCheckFunction,precheck));
  } else if (use_precheck == 2) { /* Use the version provided by the library */
    PetscCall(SNESLineSearchSetPreCheck(linesearch,SNESLineSearchPreCheckPicard,&precheck_angle));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */

  PetscCall(FormInitialGuess(&user,da,x));
  PetscCall(FormRHS(&user,da,b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESSolve(snes,b,x));
  PetscCall(SNESGetIterationNumber(snes,&its));
  PetscCall(SNESGetConvergedReason(snes,&reason));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%s Number of nonlinear iterations = %D\n",SNESConvergedReasons[reason],its));

  if (write_output) {
    PetscViewer viewer;
    PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer));
    PetscCall(VecView(x,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&b));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));
  PetscCall(PreCheckDestroy(&precheck));
  PetscCall(PetscFinalize());
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
static PetscErrorCode FormInitialGuess(AppCtx *user,DM da,Vec X)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      temp1,temp,hx,hy;
  PetscScalar    **x;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx    = 1.0/(PetscReal)(Mx-1);
  hy    = 1.0/(PetscReal)(My-1);
  temp1 = user->lambda / (user->lambda + 1.);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  PetscCall(DMDAVecGetArray(da,X,&x));

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)

  */
  PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

  /*
     Compute initial guess over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    temp = (PetscReal)(PetscMin(j,My-j-1))*hy;
    for (i=xs; i<xs+xm; i++) {
      if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
        /* boundary conditions are all zero Dirichlet */
        x[j][i] = 0.0;
      } else {
        if (user->initial == -1) {
          if (user->lambda != 0) {
            x[j][i] = temp1*PetscSqrtReal(PetscMin((PetscReal)(PetscMin(i,Mx-i-1))*hx,temp));
          } else {
            /* The solution above is an exact solution for lambda=0, this avoids "accidentally" starting
             * with an exact solution. */
            const PetscReal
              xx = 2*(PetscReal)i/(Mx-1) - 1,
              yy = 2*(PetscReal)j/(My-1) - 1;
            x[j][i] = (1 - xx*xx) * (1-yy*yy) * xx * yy;
          }
        } else if (user->initial == 0) {
          x[j][i] = 0.;
        } else if (user->initial == 1) {
          const PetscReal
            xx = 2*(PetscReal)i/(Mx-1) - 1,
            yy = 2*(PetscReal)j/(My-1) - 1;
          x[j][i] = (1 - xx*xx) * (1-yy*yy) * xx * yy;
        } else {
          if (user->lambda != 0) {
            x[j][i] = temp1*PetscSqrtReal(PetscMin((PetscReal)(PetscMin(i,Mx-i-1))*hx,temp));
          } else {
            x[j][i] = 0.5*PetscSqrtReal(PetscMin((PetscReal)(PetscMin(i,Mx-i-1))*hx,temp));
          }
        }
      }
    }
  }
  /*
     Restore vector
  */
  PetscCall(DMDAVecRestoreArray(da,X,&x));
  PetscFunctionReturn(0);
}

/*
   FormRHS - Forms constant RHS for the problem.

   Input Parameters:
   user - user-defined application context
   B - RHS vector

   Output Parameter:
   B - vector
 */
static PetscErrorCode FormRHS(AppCtx *user,DM da,Vec B)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy;
  PetscScalar    **b;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx    = 1.0/(PetscReal)(Mx-1);
  hy    = 1.0/(PetscReal)(My-1);
  PetscCall(DMDAVecGetArray(da,B,&b));
  PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
        b[j][i] = 0.0;
      } else {
        b[j][i] = hx*hy*user->source;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da,B,&b));
  PetscFunctionReturn(0);
}

static inline PetscReal kappa(const AppCtx *ctx,PetscReal x,PetscReal y)
{
  return (((PetscInt)(x*ctx->blocks[0])) + ((PetscInt)(y*ctx->blocks[1]))) % 2 ? ctx->kappa : 1.0;
}
/* p-Laplacian diffusivity */
static inline PetscScalar eta(const AppCtx *ctx,PetscReal x,PetscReal y,PetscScalar ux,PetscScalar uy)
{
  return kappa(ctx,x,y) * PetscPowScalar(PetscSqr(ctx->epsilon)+0.5*(ux*ux + uy*uy),0.5*(ctx->p-2.));
}
static inline PetscScalar deta(const AppCtx *ctx,PetscReal x,PetscReal y,PetscScalar ux,PetscScalar uy)
{
  return (ctx->p == 2)
         ? 0
         : kappa(ctx,x,y)*PetscPowScalar(PetscSqr(ctx->epsilon)+0.5*(ux*ux + uy*uy),0.5*(ctx->p-4)) * 0.5 * (ctx->p-2.);
}

/* ------------------------------------------------------------------- */
/*
   FormFunctionLocal - Evaluates nonlinear function, F(x).
 */
static PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,PetscScalar **x,PetscScalar **f,AppCtx *user)
{
  PetscReal      hx,hy,dhx,dhy,sc;
  PetscInt       i,j;
  PetscScalar    eu;

  PetscFunctionBeginUser;
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  sc     = hx*hy*user->lambda;
  dhx    = 1/hx;
  dhy    = 1/hy;
  /*
     Compute function over the locally owned part of the grid
  */
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      PetscReal xx = i*hx,yy = j*hy;
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
        f[j][i] = x[j][i];
      } else {
        const PetscScalar
          u    = x[j][i],
          ux_E = dhx*(x[j][i+1]-x[j][i]),
          uy_E = 0.25*dhy*(x[j+1][i]+x[j+1][i+1]-x[j-1][i]-x[j-1][i+1]),
          ux_W = dhx*(x[j][i]-x[j][i-1]),
          uy_W = 0.25*dhy*(x[j+1][i-1]+x[j+1][i]-x[j-1][i-1]-x[j-1][i]),
          ux_N = 0.25*dhx*(x[j][i+1]+x[j+1][i+1]-x[j][i-1]-x[j+1][i-1]),
          uy_N = dhy*(x[j+1][i]-x[j][i]),
          ux_S = 0.25*dhx*(x[j-1][i+1]+x[j][i+1]-x[j-1][i-1]-x[j][i-1]),
          uy_S = dhy*(x[j][i]-x[j-1][i]),
          e_E  = eta(user,xx,yy,ux_E,uy_E),
          e_W  = eta(user,xx,yy,ux_W,uy_W),
          e_N  = eta(user,xx,yy,ux_N,uy_N),
          e_S  = eta(user,xx,yy,ux_S,uy_S),
          uxx  = -hy * (e_E*ux_E - e_W*ux_W),
          uyy  = -hx * (e_N*uy_N - e_S*uy_S);
        if (sc) eu = PetscExpScalar(u);
        else    eu = 0.;
        /* For p=2, these terms decay to:
         uxx = (2.0*u - x[j][i-1] - x[j][i+1])*hydhx
         uyy = (2.0*u - x[j-1][i] - x[j+1][i])*hxdhy
        */
        f[j][i] = uxx + uyy - sc*eu;
      }
    }
  }
  PetscCall(PetscLogFlops(info->xm*info->ym*(72.0)));
  PetscFunctionReturn(0);
}

/*
    This is the opposite sign of the part of FormFunctionLocal that excludes the A(x) x part of the operation,
    that is FormFunction applies A(x) x - b(x) while this applies b(x) because for Picard we think of it as solving A(x) x = b(x)

*/
static PetscErrorCode FormFunctionPicardLocal(DMDALocalInfo *info,PetscScalar **x,PetscScalar **f,AppCtx *user)
{
  PetscReal hx,hy,sc;
  PetscInt  i,j;

  PetscFunctionBeginUser;
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  sc     = hx*hy*user->lambda;
  /*
     Compute function over the locally owned part of the grid
  */
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      if (!(i == 0 || j == 0 || i == info->mx-1 || j == info->my-1)) {
        const PetscScalar u = x[j][i];
        f[j][i] = sc*PetscExpScalar(u);
      } else {
        f[j][i] = 0.0; /* this is zero because the A(x) x term forces the x to be zero on the boundary */
      }
    }
  }
  PetscCall(PetscLogFlops(info->xm*info->ym*2.0));
  PetscFunctionReturn(0);
}

/*
   FormJacobianLocal - Evaluates Jacobian matrix.
*/
static PetscErrorCode FormJacobianLocal(DMDALocalInfo *info,PetscScalar **x,Mat J,Mat B,AppCtx *user)
{
  PetscInt       i,j;
  MatStencil     col[9],row;
  PetscScalar    v[9];
  PetscReal      hx,hy,hxdhy,hydhx,dhx,dhy,sc;

  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(B));
  hx    = 1.0/(PetscReal)(info->mx-1);
  hy    = 1.0/(PetscReal)(info->my-1);
  sc    = hx*hy*user->lambda;
  dhx   = 1/hx;
  dhy   = 1/hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  /*
     Compute entries for the locally owned part of the Jacobian.
      - PETSc parallel matrix formats are partitioned by
        contiguous chunks of rows across the processors.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Here, we set all entries for a particular row at once.
  */
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      PetscReal xx = i*hx,yy = j*hy;
      row.j = j; row.i = i;
      /* boundary points */
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
        v[0] = 1.0;
        PetscCall(MatSetValuesStencil(B,1,&row,1,&row,v,INSERT_VALUES));
      } else {
        /* interior grid points */
        const PetscScalar
          ux_E     = dhx*(x[j][i+1]-x[j][i]),
          uy_E     = 0.25*dhy*(x[j+1][i]+x[j+1][i+1]-x[j-1][i]-x[j-1][i+1]),
          ux_W     = dhx*(x[j][i]-x[j][i-1]),
          uy_W     = 0.25*dhy*(x[j+1][i-1]+x[j+1][i]-x[j-1][i-1]-x[j-1][i]),
          ux_N     = 0.25*dhx*(x[j][i+1]+x[j+1][i+1]-x[j][i-1]-x[j+1][i-1]),
          uy_N     = dhy*(x[j+1][i]-x[j][i]),
          ux_S     = 0.25*dhx*(x[j-1][i+1]+x[j][i+1]-x[j-1][i-1]-x[j][i-1]),
          uy_S     = dhy*(x[j][i]-x[j-1][i]),
          u        = x[j][i],
          e_E      = eta(user,xx,yy,ux_E,uy_E),
          e_W      = eta(user,xx,yy,ux_W,uy_W),
          e_N      = eta(user,xx,yy,ux_N,uy_N),
          e_S      = eta(user,xx,yy,ux_S,uy_S),
          de_E     = deta(user,xx,yy,ux_E,uy_E),
          de_W     = deta(user,xx,yy,ux_W,uy_W),
          de_N     = deta(user,xx,yy,ux_N,uy_N),
          de_S     = deta(user,xx,yy,ux_S,uy_S),
          skew_E   = de_E*ux_E*uy_E,
          skew_W   = de_W*ux_W*uy_W,
          skew_N   = de_N*ux_N*uy_N,
          skew_S   = de_S*ux_S*uy_S,
          cross_EW = 0.25*(skew_E - skew_W),
          cross_NS = 0.25*(skew_N - skew_S),
          newt_E   = e_E+de_E*PetscSqr(ux_E),
          newt_W   = e_W+de_W*PetscSqr(ux_W),
          newt_N   = e_N+de_N*PetscSqr(uy_N),
          newt_S   = e_S+de_S*PetscSqr(uy_S);
        /* interior grid points */
        switch (user->jtype) {
        case JAC_BRATU:
          /* Jacobian from p=2 */
          v[0] = -hxdhy;                                           col[0].j = j-1;   col[0].i = i;
          v[1] = -hydhx;                                           col[1].j = j;     col[1].i = i-1;
          v[2] = 2.0*(hydhx + hxdhy) - sc*PetscExpScalar(u);       col[2].j = row.j; col[2].i = row.i;
          v[3] = -hydhx;                                           col[3].j = j;     col[3].i = i+1;
          v[4] = -hxdhy;                                           col[4].j = j+1;   col[4].i = i;
          PetscCall(MatSetValuesStencil(B,1,&row,5,col,v,INSERT_VALUES));
          break;
        case JAC_PICARD:
          /* Jacobian arising from Picard linearization */
          v[0] = -hxdhy*e_S;                                           col[0].j = j-1;   col[0].i = i;
          v[1] = -hydhx*e_W;                                           col[1].j = j;     col[1].i = i-1;
          v[2] = (e_W+e_E)*hydhx + (e_S+e_N)*hxdhy;                    col[2].j = row.j; col[2].i = row.i;
          v[3] = -hydhx*e_E;                                           col[3].j = j;     col[3].i = i+1;
          v[4] = -hxdhy*e_N;                                           col[4].j = j+1;   col[4].i = i;
          PetscCall(MatSetValuesStencil(B,1,&row,5,col,v,INSERT_VALUES));
          break;
        case JAC_STAR:
          /* Full Jacobian, but only a star stencil */
          col[0].j = j-1; col[0].i = i;
          col[1].j = j;   col[1].i = i-1;
          col[2].j = j;   col[2].i = i;
          col[3].j = j;   col[3].i = i+1;
          col[4].j = j+1; col[4].i = i;
          v[0]     = -hxdhy*newt_S + cross_EW;
          v[1]     = -hydhx*newt_W + cross_NS;
          v[2]     = hxdhy*(newt_N + newt_S) + hydhx*(newt_E + newt_W) - sc*PetscExpScalar(u);
          v[3]     = -hydhx*newt_E - cross_NS;
          v[4]     = -hxdhy*newt_N - cross_EW;
          PetscCall(MatSetValuesStencil(B,1,&row,5,col,v,INSERT_VALUES));
          break;
        case JAC_NEWTON:
          /* The Jacobian is

           -div [ eta (grad u) + deta (grad u0 . grad u) grad u0 ] - (eE u0) u

          */
          col[0].j = j-1; col[0].i = i-1;
          col[1].j = j-1; col[1].i = i;
          col[2].j = j-1; col[2].i = i+1;
          col[3].j = j;   col[3].i = i-1;
          col[4].j = j;   col[4].i = i;
          col[5].j = j;   col[5].i = i+1;
          col[6].j = j+1; col[6].i = i-1;
          col[7].j = j+1; col[7].i = i;
          col[8].j = j+1; col[8].i = i+1;
          v[0]     = -0.25*(skew_S + skew_W);
          v[1]     = -hxdhy*newt_S + cross_EW;
          v[2]     =  0.25*(skew_S + skew_E);
          v[3]     = -hydhx*newt_W + cross_NS;
          v[4]     = hxdhy*(newt_N + newt_S) + hydhx*(newt_E + newt_W) - sc*PetscExpScalar(u);
          v[5]     = -hydhx*newt_E - cross_NS;
          v[6]     =  0.25*(skew_N + skew_W);
          v[7]     = -hxdhy*newt_N - cross_EW;
          v[8]     = -0.25*(skew_N + skew_E);
          PetscCall(MatSetValuesStencil(B,1,&row,9,col,v,INSERT_VALUES));
          break;
        default:
          SETERRQ(PetscObjectComm((PetscObject)info->da),PETSC_ERR_SUP,"Jacobian type %d not implemented",user->jtype);
        }
      }
    }
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
  */
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  if (J != B) {
    PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  }

  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  if (user->jtype == JAC_NEWTON) {
    PetscCall(PetscLogFlops(info->xm*info->ym*(131.0)));
  }
  PetscFunctionReturn(0);
}

/***********************************************************
 * PreCheck implementation
 ***********************************************************/
PetscErrorCode PreCheckSetFromOptions(PreCheck precheck)
{
  PetscBool      flg;

  PetscFunctionBeginUser;
  PetscOptionsBegin(precheck->comm,NULL,"PreCheck Options","none");
  PetscCall(PetscOptionsReal("-precheck_angle","Angle in degrees between successive search directions necessary to activate step correction","",precheck->angle,&precheck->angle,NULL));
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-precheck_monitor","Monitor choices made by precheck routine","",flg,&flg,NULL));
  if (flg) {
    PetscCall(PetscViewerASCIIOpen(precheck->comm,"stdout",&precheck->monitor));
  }
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*
  Compare the direction of the current and previous step, modify the current step accordingly
*/
PetscErrorCode PreCheckFunction(SNESLineSearch linesearch,Vec X,Vec Y,PetscBool *changed, void *ctx)
{
  PreCheck       precheck;
  Vec            Ylast;
  PetscScalar    dot;
  PetscInt       iter;
  PetscReal      ynorm,ylastnorm,theta,angle_radians;
  SNES           snes;

  PetscFunctionBeginUser;
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  precheck = (PreCheck)ctx;
  if (!precheck->Ylast) PetscCall(VecDuplicate(Y,&precheck->Ylast));
  Ylast = precheck->Ylast;
  PetscCall(SNESGetIterationNumber(snes,&iter));
  if (iter < 1) {
    PetscCall(VecCopy(Y,Ylast));
    *changed = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  PetscCall(VecDot(Y,Ylast,&dot));
  PetscCall(VecNorm(Y,NORM_2,&ynorm));
  PetscCall(VecNorm(Ylast,NORM_2,&ylastnorm));
  /* Compute the angle between the vectors Y and Ylast, clip to keep inside the domain of acos() */
  theta         = PetscAcosReal((PetscReal)PetscClipInterval(PetscAbsScalar(dot) / (ynorm * ylastnorm),-1.0,1.0));
  angle_radians = precheck->angle * PETSC_PI / 180.;
  if (PetscAbsReal(theta) < angle_radians || PetscAbsReal(theta - PETSC_PI) < angle_radians) {
    /* Modify the step Y */
    PetscReal alpha,ydiffnorm;
    PetscCall(VecAXPY(Ylast,-1.0,Y));
    PetscCall(VecNorm(Ylast,NORM_2,&ydiffnorm));
    alpha = ylastnorm / ydiffnorm;
    PetscCall(VecCopy(Y,Ylast));
    PetscCall(VecScale(Y,alpha));
    if (precheck->monitor) {
      PetscCall(PetscViewerASCIIPrintf(precheck->monitor,"Angle %E degrees less than threshold %g, corrected step by alpha=%g\n",(double)(theta*180./PETSC_PI),(double)precheck->angle,(double)alpha));
    }
  } else {
    PetscCall(VecCopy(Y,Ylast));
    *changed = PETSC_FALSE;
    if (precheck->monitor) {
      PetscCall(PetscViewerASCIIPrintf(precheck->monitor,"Angle %E degrees exceeds threshold %g, no correction applied\n",(double)(theta*180./PETSC_PI),(double)precheck->angle));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PreCheckDestroy(PreCheck *precheck)
{
  PetscFunctionBeginUser;
  if (!*precheck) PetscFunctionReturn(0);
  PetscCall(VecDestroy(&(*precheck)->Ylast));
  PetscCall(PetscViewerDestroy(&(*precheck)->monitor));
  PetscCall(PetscFree(*precheck));
  PetscFunctionReturn(0);
}

PetscErrorCode PreCheckCreate(MPI_Comm comm,PreCheck *precheck)
{
  PetscFunctionBeginUser;
  PetscCall(PetscNew(precheck));

  (*precheck)->comm  = comm;
  (*precheck)->angle = 10.;     /* only active if angle is less than 10 degrees */
  PetscFunctionReturn(0);
}

/*
      Applies some sweeps on nonlinear Gauss-Seidel on each process

 */
PetscErrorCode NonlinearGS(SNES snes,Vec X, Vec B, void *ctx)
{
  PetscInt       i,j,k,xs,ys,xm,ym,its,tot_its,sweeps,l,m;
  PetscReal      hx,hy,hxdhy,hydhx,dhx,dhy,sc;
  PetscScalar    **x,**b,bij,F,F0=0,J,y,u,eu;
  PetscReal      atol,rtol,stol;
  DM             da;
  AppCtx         *user = (AppCtx*)ctx;
  Vec            localX,localB;
  DMDALocalInfo  info;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes,&da));
  PetscCall(DMDAGetLocalInfo(da,&info));

  hx     = 1.0/(PetscReal)(info.mx-1);
  hy     = 1.0/(PetscReal)(info.my-1);
  sc     = hx*hy*user->lambda;
  dhx    = 1/hx;
  dhy    = 1/hy;
  hxdhy  = hx/hy;
  hydhx  = hy/hx;

  tot_its = 0;
  PetscCall(SNESNGSGetSweeps(snes,&sweeps));
  PetscCall(SNESNGSGetTolerances(snes,&atol,&rtol,&stol,&its));
  PetscCall(DMGetLocalVector(da,&localX));
  if (B) {
    PetscCall(DMGetLocalVector(da,&localB));
  }
  if (B) {
    PetscCall(DMGlobalToLocalBegin(da,B,INSERT_VALUES,localB));
    PetscCall(DMGlobalToLocalEnd(da,B,INSERT_VALUES,localB));
  }
  if (B) PetscCall(DMDAVecGetArrayRead(da,localB,&b));
  PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX));
  PetscCall(DMDAVecGetArray(da,localX,&x));
  for (l=0; l<sweeps; l++) {
    /*
     Get local grid boundaries (for 2-dimensional DMDA):
     xs, ys   - starting grid indices (no ghost points)
     xm, ym   - widths of local grid (no ghost points)
     */
    PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));
    for (m=0; m<2; m++) {
      for (j=ys; j<ys+ym; j++) {
        for (i=xs+(m+j)%2; i<xs+xm; i+=2) {
          PetscReal xx = i*hx,yy = j*hy;
          if (B) bij = b[j][i];
          else   bij = 0.;

          if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1) {
            /* boundary conditions are all zero Dirichlet */
            x[j][i] = 0.0 + bij;
          } else {
            const PetscScalar
              u_E = x[j][i+1],
              u_W = x[j][i-1],
              u_N = x[j+1][i],
              u_S = x[j-1][i];
            const PetscScalar
              uy_E   = 0.25*dhy*(x[j+1][i]+x[j+1][i+1]-x[j-1][i]-x[j-1][i+1]),
              uy_W   = 0.25*dhy*(x[j+1][i-1]+x[j+1][i]-x[j-1][i-1]-x[j-1][i]),
              ux_N   = 0.25*dhx*(x[j][i+1]+x[j+1][i+1]-x[j][i-1]-x[j+1][i-1]),
              ux_S   = 0.25*dhx*(x[j-1][i+1]+x[j][i+1]-x[j-1][i-1]-x[j][i-1]);
            u = x[j][i];
            for (k=0; k<its; k++) {
              const PetscScalar
                ux_E   = dhx*(u_E-u),
                ux_W   = dhx*(u-u_W),
                uy_N   = dhy*(u_N-u),
                uy_S   = dhy*(u-u_S),
                e_E    = eta(user,xx,yy,ux_E,uy_E),
                e_W    = eta(user,xx,yy,ux_W,uy_W),
                e_N    = eta(user,xx,yy,ux_N,uy_N),
                e_S    = eta(user,xx,yy,ux_S,uy_S),
                de_E   = deta(user,xx,yy,ux_E,uy_E),
                de_W   = deta(user,xx,yy,ux_W,uy_W),
                de_N   = deta(user,xx,yy,ux_N,uy_N),
                de_S   = deta(user,xx,yy,ux_S,uy_S),
                newt_E = e_E+de_E*PetscSqr(ux_E),
                newt_W = e_W+de_W*PetscSqr(ux_W),
                newt_N = e_N+de_N*PetscSqr(uy_N),
                newt_S = e_S+de_S*PetscSqr(uy_S),
                uxx    = -hy * (e_E*ux_E - e_W*ux_W),
                uyy    = -hx * (e_N*uy_N - e_S*uy_S);

              if (sc) eu = PetscExpScalar(u);
              else    eu = 0;

              F = uxx + uyy - sc*eu - bij;
              if (k == 0) F0 = F;
              J  = hxdhy*(newt_N + newt_S) + hydhx*(newt_E + newt_W) - sc*eu;
              y  = F/J;
              u -= y;
              tot_its++;
              if (atol > PetscAbsReal(PetscRealPart(F)) ||
                  rtol*PetscAbsReal(PetscRealPart(F0)) > PetscAbsReal(PetscRealPart(F)) ||
                  stol*PetscAbsReal(PetscRealPart(u)) > PetscAbsReal(PetscRealPart(y))) {
                break;
              }
            }
            x[j][i] = u;
          }
        }
      }
    }
    /*
x     Restore vector
     */
  }
  PetscCall(DMDAVecRestoreArray(da,localX,&x));
  PetscCall(DMLocalToGlobalBegin(da,localX,INSERT_VALUES,X));
  PetscCall(DMLocalToGlobalEnd(da,localX,INSERT_VALUES,X));
  PetscCall(PetscLogFlops(tot_its*(118.0)));
  PetscCall(DMRestoreLocalVector(da,&localX));
  if (B) {
    PetscCall(DMDAVecRestoreArrayRead(da,localB,&b));
    PetscCall(DMRestoreLocalVector(da,&localB));
  }
  PetscFunctionReturn(0);
}

/*TEST

   test:
      nsize: 2
      args: -snes_monitor_short -da_grid_x 20 -da_grid_y 20 -p 1.3 -lambda 1 -jtype NEWTON
      requires: !single

   test:
      suffix: 2
      nsize: 2
      args: -snes_monitor_short -da_grid_x 20 -da_grid_y 20 -p 1.3 -lambda 1 -jtype PICARD -precheck 1
      requires: !single

   test:
      suffix: 3
      nsize: 2
      args: -snes_monitor_short -da_grid_x 20 -da_grid_y 20 -p 1.3 -lambda 1 -jtype PICARD -picard -precheck 1 -p 3
      requires: !single

   test:
      suffix: 3_star
      nsize: 2
      args: -snes_monitor_short -da_grid_x 20 -da_grid_y 20 -p 1.3 -lambda 1 -jtype PICARD -picard -precheck 1 -p 3 -alloc_star
      output_file: output/ex15_3.out
      requires: !single

   test:
      suffix: 4
      args: -snes_monitor_short -snes_type newtonls -npc_snes_type ngs -snes_npc_side left -da_grid_x 20 -da_grid_y 20 -p 1.3 -lambda 1 -ksp_monitor_short -pc_type none
      requires: !single

   test:
      suffix: lag_jac
      nsize: 4
      args: -snes_monitor_short -da_grid_x 20 -da_grid_y 20 -p 6.0 -lambda 0 -jtype NEWTON -snes_type ngmres -npc_snes_type newtonls -npc_snes_lag_jacobian 5 -npc_pc_type asm -npc_ksp_converged_reason -npc_snes_lag_jacobian_persists
      requires: !single

   test:
      suffix: lag_pc
      nsize: 4
      args: -snes_monitor_short -da_grid_x 20 -da_grid_y 20 -p 6.0 -lambda 0 -jtype NEWTON -snes_type ngmres -npc_snes_type newtonls -npc_snes_lag_preconditioner 5 -npc_pc_type asm -npc_ksp_converged_reason -npc_snes_lag_preconditioner_persists
      requires: !single

   test:
      suffix: nleqerr
      args: -snes_monitor_short -snes_type newtonls -da_grid_x 20 -da_grid_y 20 -p 1.3 -lambda 1 -snes_linesearch_monitor -pc_type lu -snes_linesearch_type nleqerr
      requires: !single

   test:
      suffix: mf
      args: -snes_monitor_short -pc_type lu -da_refine 4  -p 3 -ksp_rtol 1.e-12  -snes_mf_operator
      requires: !single

   test:
      suffix: mf_picard
      args: -snes_monitor_short -pc_type lu -da_refine 4  -p 3 -ksp_rtol 1.e-12  -snes_mf_operator -picard
      requires: !single
      output_file: output/ex15_mf.out

   test:
      suffix: fd_picard
      args: -snes_monitor_short -pc_type lu -da_refine 2  -p 3 -ksp_rtol 1.e-12  -snes_fd -picard
      requires: !single

   test:
      suffix: fd_color_picard
      args: -snes_monitor_short -pc_type lu -da_refine 4  -p 3 -ksp_rtol 1.e-12  -snes_fd_color -picard
      requires: !single
      output_file: output/ex15_mf.out

TEST*/
