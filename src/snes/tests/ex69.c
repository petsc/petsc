
static char help[] = "Tests recovery from domain errors in MatMult() and PCApply()\n\n";

/*
      See src/ksp/ksp/tutorials/ex19.c from which this was copied
*/

#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>

/*
   User-defined routines and data structures
*/
typedef struct {
  PetscScalar u,v,omega,temp;
} Field;

PetscErrorCode FormFunctionLocal(DMDALocalInfo*,Field**,Field**,void*);

typedef struct {
  PetscReal   lidvelocity,prandtl,grashof;  /* physical parameters */
  PetscBool   draw_contours;                /* flag - 1 indicates drawing contours */
  PetscBool   errorindomain;
  PetscBool   errorindomainmf;
  SNES        snes;
} AppCtx;

typedef struct {
  Mat Jmf;
} MatShellCtx;

extern PetscErrorCode FormInitialGuess(AppCtx*,DM,Vec);
extern PetscErrorCode MatMult_MyShell(Mat,Vec,Vec);
extern PetscErrorCode MatAssemblyEnd_MyShell(Mat,MatAssemblyType);
extern PetscErrorCode PCApply_MyShell(PC,Vec,Vec);
extern PetscErrorCode SNESComputeJacobian_MyShell(SNES,Vec,Mat,Mat,void*);

int main(int argc,char **argv)
{
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,my;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  DM             da;
  Vec            x;
  Mat            J = NULL,Jmf = NULL;
  MatShellCtx    matshellctx;
  PetscInt       mlocal,nlocal;
  PC             pc;
  KSP            ksp;
  PetscBool      errorinmatmult = PETSC_FALSE,errorinpcapply = PETSC_FALSE,errorinpcsetup = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetBool(NULL,NULL,"-error_in_matmult",&errorinmatmult,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-error_in_pcapply",&errorinpcapply,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-error_in_pcsetup",&errorinpcsetup,NULL);CHKERRQ(ierr);
  user.errorindomain = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-error_in_domain",&user.errorindomain,NULL);CHKERRQ(ierr);
  user.errorindomainmf = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-error_in_domainmf",&user.errorindomainmf,NULL);CHKERRQ(ierr);

  comm = PETSC_COMM_WORLD;
  ierr = SNESCreate(comm,&user.snes);CHKERRQ(ierr);

  /*
      Create distributed array object to manage parallel grid and vectors
      for principal unknowns (x) and governing residuals (f)
  */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,4,4,PETSC_DECIDE,PETSC_DECIDE,4,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = SNESSetDM(user.snes,da);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,0,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  /*
     Problem parameters (velocity of lid, prandtl, and grashof numbers)
  */
  user.lidvelocity = 1.0/(mx*my);
  user.prandtl     = 1.0;
  user.grashof     = 1.0;

  ierr = PetscOptionsGetReal(NULL,NULL,"-lidvelocity",&user.lidvelocity,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-prandtl",&user.prandtl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-grashof",&user.grashof,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-contours",&user.draw_contours);CHKERRQ(ierr);

  ierr = DMDASetFieldName(da,0,"x_velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"y_velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,2,"Omega");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,3,"temperature");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create user context, set problem data, create vector data structures.
     Also, compute the initial guess.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context

     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,&user);CHKERRQ(ierr);

  if (errorinmatmult) {
    ierr = MatCreateSNESMF(user.snes,&Jmf);CHKERRQ(ierr);
    ierr = MatSetFromOptions(Jmf);CHKERRQ(ierr);
    ierr = MatGetLocalSize(Jmf,&mlocal,&nlocal);CHKERRQ(ierr);
    matshellctx.Jmf = Jmf;
    ierr = MatCreateShell(PetscObjectComm((PetscObject)Jmf),mlocal,nlocal,PETSC_DECIDE,PETSC_DECIDE,&matshellctx,&J);CHKERRQ(ierr);
    ierr = MatShellSetOperation(J,MATOP_MULT,(void (*)(void))MatMult_MyShell);CHKERRQ(ierr);
    ierr = MatShellSetOperation(J,MATOP_ASSEMBLY_END,(void (*)(void))MatAssemblyEnd_MyShell);CHKERRQ(ierr);
    ierr = SNESSetJacobian(user.snes,J,J,MatMFFDComputeJacobian,NULL);CHKERRQ(ierr);
  }

  ierr = SNESSetFromOptions(user.snes);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"lid velocity = %g, prandtl # = %g, grashof # = %g\n",(double)user.lidvelocity,(double)user.prandtl,(double)user.grashof);CHKERRQ(ierr);

  if (errorinpcapply) {
    ierr = SNESGetKSP(user.snes,&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCSHELL);CHKERRQ(ierr);
    ierr = PCShellSetApply(pc,PCApply_MyShell);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = FormInitialGuess(&user,da,x);CHKERRQ(ierr);

  if (errorinpcsetup) {
    ierr = SNESSetUp(user.snes);CHKERRQ(ierr);
    ierr = SNESSetJacobian(user.snes,NULL,NULL,SNESComputeJacobian_MyShell,NULL);CHKERRQ(ierr);
  }
  ierr = SNESSolve(user.snes,NULL,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = MatDestroy(&Jmf);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = SNESDestroy(&user.snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
PetscErrorCode FormInitialGuess(AppCtx *user,DM da,Vec X)
{
  PetscInt       i,j,mx,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscReal      grashof,dx;
  Field          **x;

  PetscFunctionBeginUser;
  grashof = user->grashof;

  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dx   = 1.0/(mx-1);

  /*
     Get local grid boundaries (for 2-dimensional DMDA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
     Initial condition is motionless fluid and equilibrium temperature
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x[j][i].u     = 0.0;
      x[j][i].v     = 0.0;
      x[j][i].omega = 0.0;
      x[j][i].temp  = (grashof>0)*i*dx;
    }
  }

  /*
     Restore vector
  */
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,Field **x,Field **f,void *ptr)
{
  AppCtx          *user = (AppCtx*)ptr;
  PetscErrorCode  ierr;
  PetscInt        xints,xinte,yints,yinte,i,j;
  PetscReal       hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal       grashof,prandtl,lid;
  PetscScalar     u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;
  static PetscInt fail = 0;

  PetscFunctionBeginUser;
  if ((fail++ > 7 && user->errorindomainmf) || (fail++ > 36 && user->errorindomain)){
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)user->snes),&rank);CHKERRMPI(ierr);
    if (!rank) {
      ierr = SNESSetFunctionDomainError(user->snes);CHKERRQ(ierr);
    }
  }
  grashof = user->grashof;
  prandtl = user->prandtl;
  lid     = user->lidvelocity;

  /*
     Define mesh intervals ratios for uniform grid.

     Note: FD formulae below are normalized by multiplying through by
     local volume element (i.e. hx*hy) to obtain coefficients O(1) in two dimensions.

  */
  dhx   = (PetscReal)(info->mx-1);  dhy = (PetscReal)(info->my-1);
  hx    = 1.0/dhx;                   hy = 1.0/dhy;
  hxdhy = hx*dhy;                 hydhx = hy*dhx;

  xints = info->xs; xinte = info->xs+info->xm; yints = info->ys; yinte = info->ys+info->ym;

  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    j     = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega + (x[j+1][i].u - x[j][i].u)*dhy;
      f[j][i].temp  = x[j][i].temp-x[j+1][i].temp;
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == info->my) {
    j     = info->my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].u     = x[j][i].u - lid;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega + (x[j][i].u - x[j-1][i].u)*dhy;
      f[j][i].temp  = x[j][i].temp-x[j-1][i].temp;
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i     = 0;
    xints = xints + 1;
    /* left edge */
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i+1].v - x[j][i].v)*dhx;
      f[j][i].temp  = x[j][i].temp;
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info->mx) {
    i     = info->mx - 1;
    xinte = xinte - 1;
    /* right edge */
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

      /*
       convective coefficients for upwinding
      */
      vx  = x[j][i].u; avx = PetscAbsScalar(vx);
      vxp = .5*(vx+avx); vxm = .5*(vx-avx);
      vy  = x[j][i].v; avy = PetscAbsScalar(vy);
      vyp = .5*(vy+avy); vym = .5*(vy-avy);

      /* U velocity */
      u         = x[j][i].u;
      uxx       = (2.0*u - x[j][i-1].u - x[j][i+1].u)*hydhx;
      uyy       = (2.0*u - x[j-1][i].u - x[j+1][i].u)*hxdhy;
      f[j][i].u = uxx + uyy - .5*(x[j+1][i].omega-x[j-1][i].omega)*hx;

      /* V velocity */
      u         = x[j][i].v;
      uxx       = (2.0*u - x[j][i-1].v - x[j][i+1].v)*hydhx;
      uyy       = (2.0*u - x[j-1][i].v - x[j+1][i].v)*hxdhy;
      f[j][i].v = uxx + uyy + .5*(x[j][i+1].omega-x[j][i-1].omega)*hy;

      /* Omega */
      u             = x[j][i].omega;
      uxx           = (2.0*u - x[j][i-1].omega - x[j][i+1].omega)*hydhx;
      uyy           = (2.0*u - x[j-1][i].omega - x[j+1][i].omega)*hxdhy;
      f[j][i].omega = uxx + uyy + (vxp*(u - x[j][i-1].omega) + vxm*(x[j][i+1].omega - u))*hy +
                      (vyp*(u - x[j-1][i].omega) + vym*(x[j+1][i].omega - u))*hx -
                      .5*grashof*(x[j][i+1].temp - x[j][i-1].temp)*hy;

      /* Temperature */
      u            = x[j][i].temp;
      uxx          = (2.0*u - x[j][i-1].temp - x[j][i+1].temp)*hydhx;
      uyy          = (2.0*u - x[j-1][i].temp - x[j+1][i].temp)*hxdhy;
      f[j][i].temp =  uxx + uyy  + prandtl*((vxp*(u - x[j][i-1].temp) + vxm*(x[j][i+1].temp - u))*hy +
                                            (vyp*(u - x[j-1][i].temp) + vym*(x[j+1][i].temp - u))*hx);
    }
  }

  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  ierr = PetscLogFlops(84.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MyShell(Mat A,Vec x,Vec y)
{
  PetscErrorCode  ierr;
  MatShellCtx     *matshellctx;
  static PetscInt fail = 0;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&matshellctx);CHKERRQ(ierr);
  ierr = MatMult(matshellctx->Jmf,x,y);CHKERRQ(ierr);
  if (fail++ > 5) {
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank);CHKERRMPI(ierr);
    if (!rank) {ierr = VecSetInf(y);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MyShell(Mat A,MatAssemblyType tp)
{
  PetscErrorCode ierr;
  MatShellCtx    *matshellctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&matshellctx);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(matshellctx->Jmf,tp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCApply_MyShell(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  static PetscInt fail = 0;

  PetscFunctionBegin;
  ierr = VecCopy(x,y);CHKERRQ(ierr);
  if (fail++ > 3) {
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank);CHKERRMPI(ierr);
    if (!rank) {ierr = VecSetInf(y);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode SNESComputeJacobian_DMDA(SNES,Vec,Mat,Mat,void*);

PetscErrorCode SNESComputeJacobian_MyShell(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  static PetscInt fail = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = SNESComputeJacobian_DMDA(snes,X,A,B,ctx);CHKERRQ(ierr);
  if (fail++ > 0) {
    ierr = MatZeroEntries(A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -snes_converged_reason -ksp_converged_reason

   test:
      suffix: 2
      args: -snes_converged_reason -ksp_converged_reason -error_in_matmult

   test:
      suffix: 3
      args: -snes_converged_reason -ksp_converged_reason -error_in_pcapply

   test:
      suffix: 4
      args: -snes_converged_reason -ksp_converged_reason -error_in_pcsetup

   test:
      suffix: 5
      args: -snes_converged_reason -ksp_converged_reason -error_in_pcsetup -pc_type bjacobi

   test:
      suffix: 5_fieldsplit
      args: -snes_converged_reason -ksp_converged_reason -error_in_pcsetup -pc_type fieldsplit
      output_file: output/ex69_5.out

   test:
      suffix: 6
      args: -snes_converged_reason -ksp_converged_reason -error_in_domainmf -snes_mf -pc_type none

   test:
      suffix: 7
      args: -snes_converged_reason -ksp_converged_reason -error_in_domain

   test:
      suffix: 8
      args: -snes_converged_reason -ksp_converged_reason -error_in_domain -snes_mf -pc_type none

TEST*/
