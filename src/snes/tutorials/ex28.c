static const char help[] = "1D multiphysics prototype with analytic Jacobians to solve individual problems and a coupled problem.\n\n";

/* Solve a PDE coupled to an algebraic system in 1D
 *
 * PDE (U):
 *     -(k u_x)_x = 1 on (0,1), subject to u(0) = 0, u(1) = 1
 * Algebraic (K):
 *     exp(k-1) + k = 1/(1/(1+u) + 1/(1+u_x^2))
 *
 * The discretization places k at staggered points, and a separate DMDA is used for each "physics".
 *
 * This example is a prototype for coupling in multi-physics problems, therefore residual evaluation and assembly for
 * each problem (referred to as U and K) are written separately.  This permits the same "physics" code to be used for
 * solving each uncoupled problem as well as the coupled system.  In particular, run with -problem_type 0 to solve only
 * problem U (with K fixed), -problem_type 1 to solve only K (with U fixed), and -problem_type 2 to solve both at once.
 *
 * In all cases, a fully-assembled analytic Jacobian is available, so the systems can be solved with a direct solve or
 * any other standard method.  Additionally, by running with
 *
 *   -pack_dm_mat_type nest
 *
 * The same code assembles a coupled matrix where each block is stored separately, which allows the use of PCFieldSplit
 * without copying values to extract submatrices.
 */

#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmcomposite.h>

typedef struct _UserCtx *User;
struct _UserCtx {
  PetscInt ptype;
  DM       pack;
  Vec      Uloc,Kloc;
};

static PetscErrorCode FormFunctionLocal_U(User user,DMDALocalInfo *info,const PetscScalar u[],const PetscScalar k[],PetscScalar f[])
{
  PetscReal hx = 1./info->mx;
  PetscInt  i;

  PetscFunctionBeginUser;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    if (i == 0) f[i] = 1./hx*u[i];
    else if (i == info->mx-1) f[i] = 1./hx*(u[i] - 1.0);
    else f[i] = hx*((k[i-1]*(u[i]-u[i-1]) - k[i]*(u[i+1]-u[i]))/(hx*hx) - 1.0);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormFunctionLocal_K(User user,DMDALocalInfo *info,const PetscScalar u[],const PetscScalar k[],PetscScalar f[])
{
  PetscReal hx = 1./info->mx;
  PetscInt  i;

  PetscFunctionBeginUser;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    const PetscScalar
      ubar  = 0.5*(u[i+1]+u[i]),
      gradu = (u[i+1]-u[i])/hx,
      g     = 1. + gradu*gradu,
      w     = 1./(1.+ubar) + 1./g;
    f[i] = hx*(PetscExpScalar(k[i]-1.0) + k[i] - 1./w);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormFunction_All(SNES snes,Vec X,Vec F,void *ctx)
{
  User           user = (User)ctx;
  DM             dau,dak;
  DMDALocalInfo  infou,infok;
  PetscScalar    *u,*k;
  PetscScalar    *fu,*fk;
  PetscErrorCode ierr;
  Vec            Uloc,Kloc,Fu,Fk;

  PetscFunctionBeginUser;
  ierr = DMCompositeGetEntries(user->pack,&dau,&dak);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dau,&infou);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dak,&infok);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(user->pack,&Uloc,&Kloc);CHKERRQ(ierr);
  switch (user->ptype) {
  case 0:
    ierr = DMGlobalToLocalBegin(dau,X,INSERT_VALUES,Uloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd  (dau,X,INSERT_VALUES,Uloc);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dau,Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dak,user->Kloc,&k);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dau,F,&fu);CHKERRQ(ierr);
    ierr = FormFunctionLocal_U(user,&infou,u,k,fu);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dau,F,&fu);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dau,Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dak,user->Kloc,&k);CHKERRQ(ierr);
    break;
  case 1:
    ierr = DMGlobalToLocalBegin(dak,X,INSERT_VALUES,Kloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd  (dak,X,INSERT_VALUES,Kloc);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dau,user->Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dak,Kloc,&k);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dak,F,&fk);CHKERRQ(ierr);
    ierr = FormFunctionLocal_K(user,&infok,u,k,fk);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dak,F,&fk);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dau,user->Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dak,Kloc,&k);CHKERRQ(ierr);
    break;
  case 2:
    ierr = DMCompositeScatter(user->pack,X,Uloc,Kloc);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dau,Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dak,Kloc,&k);CHKERRQ(ierr);
    ierr = DMCompositeGetAccess(user->pack,F,&Fu,&Fk);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dau,Fu,&fu);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dak,Fk,&fk);CHKERRQ(ierr);
    ierr = FormFunctionLocal_U(user,&infou,u,k,fu);CHKERRQ(ierr);
    ierr = FormFunctionLocal_K(user,&infok,u,k,fk);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dau,Fu,&fu);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dak,Fk,&fk);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(user->pack,F,&Fu,&Fk);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dau,Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dak,Kloc,&k);CHKERRQ(ierr);
    break;
  }
  ierr = DMCompositeRestoreLocalVectors(user->pack,&Uloc,&Kloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormJacobianLocal_U(User user,DMDALocalInfo *info,const PetscScalar u[],const PetscScalar k[],Mat Buu)
{
  PetscReal      hx = 1./info->mx;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBeginUser;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    PetscInt    row = i-info->gxs,cols[] = {row-1,row,row+1};
    PetscScalar val = 1./hx;
    if (i == 0) {
      ierr = MatSetValuesLocal(Buu,1,&row,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);
    } else if (i == info->mx-1) {
      ierr = MatSetValuesLocal(Buu,1,&row,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      PetscScalar vals[] = {-k[i-1]/hx,(k[i-1]+k[i])/hx,-k[i]/hx};
      ierr = MatSetValuesLocal(Buu,1,&row,3,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormJacobianLocal_K(User user,DMDALocalInfo *info,const PetscScalar u[],const PetscScalar k[],Mat Bkk)
{
  PetscReal      hx = 1./info->mx;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBeginUser;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    PetscInt    row    = i-info->gxs;
    PetscScalar vals[] = {hx*(PetscExpScalar(k[i]-1.)+ (PetscScalar)1.)};
    ierr = MatSetValuesLocal(Bkk,1,&row,1,&row,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormJacobianLocal_UK(User user,DMDALocalInfo *info,DMDALocalInfo *infok,const PetscScalar u[],const PetscScalar k[],Mat Buk)
{
  PetscReal      hx = 1./info->mx;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscInt       row,cols[2];
  PetscScalar    vals[2];

  PetscFunctionBeginUser;
  if (!Buk) PetscFunctionReturn(0); /* Not assembling this block */
  for (i=info->xs; i<info->xs+info->xm; i++) {
    if (i == 0 || i == info->mx-1) continue;
    row     = i-info->gxs;
    cols[0] = i-1-infok->gxs;  vals[0] = (u[i]-u[i-1])/hx;
    cols[1] = i-infok->gxs;    vals[1] = (u[i]-u[i+1])/hx;
    ierr    = MatSetValuesLocal(Buk,1,&row,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormJacobianLocal_KU(User user,DMDALocalInfo *info,DMDALocalInfo *infok,const PetscScalar u[],const PetscScalar k[],Mat Bku)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      hx = 1./(info->mx-1);

  PetscFunctionBeginUser;
  if (!Bku) PetscFunctionReturn(0); /* Not assembling this block */
  for (i=infok->xs; i<infok->xs+infok->xm; i++) {
    PetscInt    row = i-infok->gxs,cols[2];
    PetscScalar vals[2];
    const PetscScalar
      ubar     = 0.5*(u[i]+u[i+1]),
      ubar_L   = 0.5,
      ubar_R   = 0.5,
      gradu    = (u[i+1]-u[i])/hx,
      gradu_L  = -1./hx,
      gradu_R  = 1./hx,
      g        = 1. + PetscSqr(gradu),
      g_gradu  = 2.*gradu,
      w        = 1./(1.+ubar) + 1./g,
      w_ubar   = -1./PetscSqr(1.+ubar),
      w_gradu  = -g_gradu/PetscSqr(g),
      iw       = 1./w,
      iw_ubar  = -w_ubar * PetscSqr(iw),
      iw_gradu = -w_gradu * PetscSqr(iw);
    cols[0] = i-info->gxs;         vals[0] = -hx*(iw_ubar*ubar_L + iw_gradu*gradu_L);
    cols[1] = i+1-info->gxs;       vals[1] = -hx*(iw_ubar*ubar_R + iw_gradu*gradu_R);
    ierr    = MatSetValuesLocal(Bku,1,&row,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormJacobian_All(SNES snes,Vec X,Mat J,Mat B,void *ctx)
{
  User           user = (User)ctx;
  DM             dau,dak;
  DMDALocalInfo  infou,infok;
  PetscScalar    *u,*k;
  PetscErrorCode ierr;
  Vec            Uloc,Kloc;

  PetscFunctionBeginUser;
  ierr = DMCompositeGetEntries(user->pack,&dau,&dak);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dau,&infou);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dak,&infok);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(user->pack,&Uloc,&Kloc);CHKERRQ(ierr);
  switch (user->ptype) {
  case 0:
    ierr = DMGlobalToLocalBegin(dau,X,INSERT_VALUES,Uloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd  (dau,X,INSERT_VALUES,Uloc);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dau,Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dak,user->Kloc,&k);CHKERRQ(ierr);
    ierr = FormJacobianLocal_U(user,&infou,u,k,B);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dau,Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dak,user->Kloc,&k);CHKERRQ(ierr);
    break;
  case 1:
    ierr = DMGlobalToLocalBegin(dak,X,INSERT_VALUES,Kloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd  (dak,X,INSERT_VALUES,Kloc);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dau,user->Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dak,Kloc,&k);CHKERRQ(ierr);
    ierr = FormJacobianLocal_K(user,&infok,u,k,B);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dau,user->Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dak,Kloc,&k);CHKERRQ(ierr);
    break;
  case 2: {
    Mat       Buu,Buk,Bku,Bkk;
    PetscBool nest;
    IS        *is;
    ierr = DMCompositeScatter(user->pack,X,Uloc,Kloc);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dau,Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dak,Kloc,&k);CHKERRQ(ierr);
    ierr = DMCompositeGetLocalISs(user->pack,&is);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(B,is[0],is[0],&Buu);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(B,is[0],is[1],&Buk);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(B,is[1],is[0],&Bku);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(B,is[1],is[1],&Bkk);CHKERRQ(ierr);
    ierr = FormJacobianLocal_U(user,&infou,u,k,Buu);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)B,MATNEST,&nest);CHKERRQ(ierr);
    if (!nest) {
      /*
         DMCreateMatrix_Composite()  for a nested matrix does not generate off-block matrices that one can call MatSetValuesLocal() on, it just creates dummy
         matrices with no entries; there cannot insert values into them. Commit b6480e041dd2293a65f96222772d68cdb4ed6306
         changed Mat_Nest() from returning NULL pointers for these submatrices to dummy matrices because PCFIELDSPLIT could not
         handle the returned null matrices.
      */
      ierr = FormJacobianLocal_UK(user,&infou,&infok,u,k,Buk);CHKERRQ(ierr);
      ierr = FormJacobianLocal_KU(user,&infou,&infok,u,k,Bku);CHKERRQ(ierr);
    }
    ierr = FormJacobianLocal_K(user,&infok,u,k,Bkk);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(B,is[0],is[0],&Buu);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(B,is[0],is[1],&Buk);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(B,is[1],is[0],&Bku);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(B,is[1],is[1],&Bkk);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dau,Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dak,Kloc,&k);CHKERRQ(ierr);

    ierr = ISDestroy(&is[0]);CHKERRQ(ierr);
    ierr = ISDestroy(&is[1]);CHKERRQ(ierr);
    ierr = PetscFree(is);CHKERRQ(ierr);
  } break;
  }
  ierr = DMCompositeRestoreLocalVectors(user->pack,&Uloc,&Kloc);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != B) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormInitial_Coupled(User user,Vec X)
{
  PetscErrorCode ierr;
  DM             dau,dak;
  DMDALocalInfo  infou,infok;
  Vec            Xu,Xk;
  PetscScalar    *u,*k,hx;
  PetscInt       i;

  PetscFunctionBeginUser;
  ierr = DMCompositeGetEntries(user->pack,&dau,&dak);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(user->pack,X,&Xu,&Xk);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dau,Xu,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dak,Xk,&k);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dau,&infou);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dak,&infok);CHKERRQ(ierr);
  hx   = 1./(infok.mx);
  for (i=infou.xs; i<infou.xs+infou.xm; i++) u[i] = (PetscScalar)i*hx * (1.-(PetscScalar)i*hx);
  for (i=infok.xs; i<infok.xs+infok.xm; i++) k[i] = 1.0 + 0.5*PetscSinScalar(2*PETSC_PI*i*hx);
  ierr = DMDAVecRestoreArray(dau,Xu,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dak,Xk,&k);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(user->pack,X,&Xu,&Xk);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user->pack,X,user->Uloc,user->Kloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  DM             dau,dak,pack;
  const PetscInt *lxu;
  PetscInt       *lxk,m,sizes;
  User           user;
  SNES           snes;
  Vec            X,F,Xu,Xk,Fu,Fk;
  Mat            B;
  IS             *isg;
  PetscBool      pass_dm;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,10,1,1,NULL,&dau);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(dau,"u_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(dau);CHKERRQ(ierr);
  ierr = DMSetUp(dau);CHKERRQ(ierr);
  ierr = DMDAGetOwnershipRanges(dau,&lxu,0,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dau,0, &m,0,0, &sizes,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscMalloc1(sizes,&lxk);CHKERRQ(ierr);
  ierr = PetscArraycpy(lxk,lxu,sizes);CHKERRQ(ierr);
  lxk[0]--;
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,m-1,1,1,lxk,&dak);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(dak,"k_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(dak);CHKERRQ(ierr);
  ierr = DMSetUp(dak);CHKERRQ(ierr);
  ierr = PetscFree(lxk);CHKERRQ(ierr);

  ierr = DMCompositeCreate(PETSC_COMM_WORLD,&pack);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(pack,"pack_");CHKERRQ(ierr);
  ierr = DMCompositeAddDM(pack,dau);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(pack,dak);CHKERRQ(ierr);
  ierr = DMDASetFieldName(dau,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(dak,0,"k");CHKERRQ(ierr);
  ierr = DMSetFromOptions(pack);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(pack,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);

  ierr = PetscNew(&user);CHKERRQ(ierr);

  user->pack = pack;

  ierr = DMCompositeGetGlobalISs(pack,&isg);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(pack,&user->Uloc,&user->Kloc);CHKERRQ(ierr);
  ierr = DMCompositeScatter(pack,X,user->Uloc,user->Kloc);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Coupled problem options","SNES");CHKERRQ(ierr);
  {
    user->ptype = 0; pass_dm = PETSC_TRUE;

    ierr = PetscOptionsInt("-problem_type","0: solve for u only, 1: solve for k only, 2: solve for both",0,user->ptype,&user->ptype,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-pass_dm","Pass the packed DM to SNES to use when determining splits and forward into splits",0,pass_dm,&pass_dm,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = FormInitial_Coupled(user,X);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  switch (user->ptype) {
  case 0:
    ierr = DMCompositeGetAccess(pack,X,&Xu,0);CHKERRQ(ierr);
    ierr = DMCompositeGetAccess(pack,F,&Fu,0);CHKERRQ(ierr);
    ierr = DMCreateMatrix(dau,&B);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,Fu,FormFunction_All,user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,B,B,FormJacobian_All,user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,dau);CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,Xu);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(pack,X,&Xu,0);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(pack,F,&Fu,0);CHKERRQ(ierr);
    break;
  case 1:
    ierr = DMCompositeGetAccess(pack,X,0,&Xk);CHKERRQ(ierr);
    ierr = DMCompositeGetAccess(pack,F,0,&Fk);CHKERRQ(ierr);
    ierr = DMCreateMatrix(dak,&B);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,Fk,FormFunction_All,user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,B,B,FormJacobian_All,user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,dak);CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,Xk);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(pack,X,0,&Xk);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(pack,F,0,&Fk);CHKERRQ(ierr);
    break;
  case 2:
    ierr = DMCreateMatrix(pack,&B);CHKERRQ(ierr);
    /* This example does not correctly allocate off-diagonal blocks. These options allows new nonzeros (slow). */
    ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,F,FormFunction_All,user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,B,B,FormJacobian_All,user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    if (!pass_dm) {             /* Manually provide index sets and names for the splits */
      KSP ksp;
      PC  pc;
      ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCFieldSplitSetIS(pc,"u",isg[0]);CHKERRQ(ierr);
      ierr = PCFieldSplitSetIS(pc,"k",isg[1]);CHKERRQ(ierr);
    } else {
      /* The same names come from the options prefix for dau and dak. This option can support geometric multigrid inside
       * of splits, but it requires using a DM (perhaps your own implementation). */
      ierr = SNESSetDM(snes,pack);CHKERRQ(ierr);
    }
    ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
    break;
  }
  ierr = VecViewFromOptions(X,NULL,"-view_sol");CHKERRQ(ierr);

  if (0) {
    PetscInt  col      = 0;
    PetscBool mult_dup = PETSC_FALSE,view_dup = PETSC_FALSE;
    Mat       D;
    Vec       Y;

    ierr = PetscOptionsGetInt(NULL,0,"-col",&col,0);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,0,"-mult_dup",&mult_dup,0);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,0,"-view_dup",&view_dup,0);CHKERRQ(ierr);

    ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
    /* ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); */
    /* ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); */
    ierr = MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
    ierr = VecZeroEntries(X);CHKERRQ(ierr);
    ierr = VecSetValue(X,col,1.0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
    ierr = MatMult(mult_dup ? D : B,X,Y);CHKERRQ(ierr);
    ierr = MatView(view_dup ? D : B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    /* ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
    ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
    ierr = VecDestroy(&Y);CHKERRQ(ierr);
  }

  ierr = DMCompositeRestoreLocalVectors(pack,&user->Uloc,&user->Kloc);CHKERRQ(ierr);
  ierr = PetscFree(user);CHKERRQ(ierr);

  ierr = ISDestroy(&isg[0]);CHKERRQ(ierr);
  ierr = ISDestroy(&isg[1]);CHKERRQ(ierr);
  ierr = PetscFree(isg);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = DMDestroy(&dau);CHKERRQ(ierr);
  ierr = DMDestroy(&dak);CHKERRQ(ierr);
  ierr = DMDestroy(&pack);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 0
      nsize: 3
      args: -u_da_grid_x 20 -snes_converged_reason -snes_monitor_short -problem_type 0

   test:
      suffix: 1
      nsize: 3
      args: -u_da_grid_x 20 -snes_converged_reason -snes_monitor_short -problem_type 1

   test:
      suffix: 2
      nsize: 3
      args: -u_da_grid_x 20 -snes_converged_reason -snes_monitor_short -problem_type 2

   test:
      suffix: 3
      nsize: 3
      args: -u_da_grid_x 20 -snes_converged_reason -snes_monitor_short -ksp_monitor_short -problem_type 2 -snes_mf_operator -pack_dm_mat_type {{aij nest}} -pc_type fieldsplit -pc_fieldsplit_dm_splits -pc_fieldsplit_type additive -fieldsplit_u_ksp_type gmres -fieldsplit_k_pc_type jacobi

   test:
      suffix: 4
      nsize: 6
      args: -u_da_grid_x 257 -snes_converged_reason -snes_monitor_short -ksp_monitor_short -problem_type 2 -snes_mf_operator -pack_dm_mat_type aij -pc_type fieldsplit -pc_fieldsplit_type multiplicative -fieldsplit_u_ksp_type gmres -fieldsplit_u_ksp_pc_side right -fieldsplit_u_pc_type mg -fieldsplit_u_pc_mg_levels 4 -fieldsplit_u_mg_levels_ksp_type richardson -fieldsplit_u_mg_levels_ksp_max_it 1 -fieldsplit_u_mg_levels_pc_type sor -fieldsplit_u_pc_mg_galerkin pmat -fieldsplit_u_ksp_converged_reason -fieldsplit_k_pc_type jacobi
      requires: !single

   test:
      suffix: glvis_composite_da_1d
      args: -u_da_grid_x 20 -problem_type 0 -snes_monitor_solution glvis:

   test:
      suffix: cuda
      nsize: 1
      requires: cuda
      args: -u_da_grid_x 20 -snes_converged_reason -snes_monitor_short -problem_type 2 -pack_dm_mat_type aijcusparse -pack_dm_vec_type cuda

   test:
      suffix: viennacl
      nsize: 1
      requires: viennacl
      args: -u_da_grid_x 20 -snes_converged_reason -snes_monitor_short -problem_type 2 -pack_dm_mat_type aijviennacl -pack_dm_vec_type viennacl

TEST*/
