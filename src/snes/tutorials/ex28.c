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
  Vec            Uloc,Kloc,Fu,Fk;

  PetscFunctionBeginUser;
  PetscCall(DMCompositeGetEntries(user->pack,&dau,&dak));
  PetscCall(DMDAGetLocalInfo(dau,&infou));
  PetscCall(DMDAGetLocalInfo(dak,&infok));
  PetscCall(DMCompositeGetLocalVectors(user->pack,&Uloc,&Kloc));
  switch (user->ptype) {
  case 0:
    PetscCall(DMGlobalToLocalBegin(dau,X,INSERT_VALUES,Uloc));
    PetscCall(DMGlobalToLocalEnd  (dau,X,INSERT_VALUES,Uloc));
    PetscCall(DMDAVecGetArray(dau,Uloc,&u));
    PetscCall(DMDAVecGetArray(dak,user->Kloc,&k));
    PetscCall(DMDAVecGetArray(dau,F,&fu));
    PetscCall(FormFunctionLocal_U(user,&infou,u,k,fu));
    PetscCall(DMDAVecRestoreArray(dau,F,&fu));
    PetscCall(DMDAVecRestoreArray(dau,Uloc,&u));
    PetscCall(DMDAVecRestoreArray(dak,user->Kloc,&k));
    break;
  case 1:
    PetscCall(DMGlobalToLocalBegin(dak,X,INSERT_VALUES,Kloc));
    PetscCall(DMGlobalToLocalEnd  (dak,X,INSERT_VALUES,Kloc));
    PetscCall(DMDAVecGetArray(dau,user->Uloc,&u));
    PetscCall(DMDAVecGetArray(dak,Kloc,&k));
    PetscCall(DMDAVecGetArray(dak,F,&fk));
    PetscCall(FormFunctionLocal_K(user,&infok,u,k,fk));
    PetscCall(DMDAVecRestoreArray(dak,F,&fk));
    PetscCall(DMDAVecRestoreArray(dau,user->Uloc,&u));
    PetscCall(DMDAVecRestoreArray(dak,Kloc,&k));
    break;
  case 2:
    PetscCall(DMCompositeScatter(user->pack,X,Uloc,Kloc));
    PetscCall(DMDAVecGetArray(dau,Uloc,&u));
    PetscCall(DMDAVecGetArray(dak,Kloc,&k));
    PetscCall(DMCompositeGetAccess(user->pack,F,&Fu,&Fk));
    PetscCall(DMDAVecGetArray(dau,Fu,&fu));
    PetscCall(DMDAVecGetArray(dak,Fk,&fk));
    PetscCall(FormFunctionLocal_U(user,&infou,u,k,fu));
    PetscCall(FormFunctionLocal_K(user,&infok,u,k,fk));
    PetscCall(DMDAVecRestoreArray(dau,Fu,&fu));
    PetscCall(DMDAVecRestoreArray(dak,Fk,&fk));
    PetscCall(DMCompositeRestoreAccess(user->pack,F,&Fu,&Fk));
    PetscCall(DMDAVecRestoreArray(dau,Uloc,&u));
    PetscCall(DMDAVecRestoreArray(dak,Kloc,&k));
    break;
  }
  PetscCall(DMCompositeRestoreLocalVectors(user->pack,&Uloc,&Kloc));
  PetscFunctionReturn(0);
}

static PetscErrorCode FormJacobianLocal_U(User user,DMDALocalInfo *info,const PetscScalar u[],const PetscScalar k[],Mat Buu)
{
  PetscReal      hx = 1./info->mx;
  PetscInt       i;

  PetscFunctionBeginUser;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    PetscInt    row = i-info->gxs,cols[] = {row-1,row,row+1};
    PetscScalar val = 1./hx;
    if (i == 0) {
      PetscCall(MatSetValuesLocal(Buu,1,&row,1,&row,&val,INSERT_VALUES));
    } else if (i == info->mx-1) {
      PetscCall(MatSetValuesLocal(Buu,1,&row,1,&row,&val,INSERT_VALUES));
    } else {
      PetscScalar vals[] = {-k[i-1]/hx,(k[i-1]+k[i])/hx,-k[i]/hx};
      PetscCall(MatSetValuesLocal(Buu,1,&row,3,cols,vals,INSERT_VALUES));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormJacobianLocal_K(User user,DMDALocalInfo *info,const PetscScalar u[],const PetscScalar k[],Mat Bkk)
{
  PetscReal      hx = 1./info->mx;
  PetscInt       i;

  PetscFunctionBeginUser;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    PetscInt    row    = i-info->gxs;
    PetscScalar vals[] = {hx*(PetscExpScalar(k[i]-1.)+ (PetscScalar)1.)};
    PetscCall(MatSetValuesLocal(Bkk,1,&row,1,&row,vals,INSERT_VALUES));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormJacobianLocal_UK(User user,DMDALocalInfo *info,DMDALocalInfo *infok,const PetscScalar u[],const PetscScalar k[],Mat Buk)
{
  PetscReal      hx = 1./info->mx;
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
    PetscCall(MatSetValuesLocal(Buk,1,&row,2,cols,vals,INSERT_VALUES));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormJacobianLocal_KU(User user,DMDALocalInfo *info,DMDALocalInfo *infok,const PetscScalar u[],const PetscScalar k[],Mat Bku)
{
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
    PetscCall(MatSetValuesLocal(Bku,1,&row,2,cols,vals,INSERT_VALUES));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormJacobian_All(SNES snes,Vec X,Mat J,Mat B,void *ctx)
{
  User           user = (User)ctx;
  DM             dau,dak;
  DMDALocalInfo  infou,infok;
  PetscScalar    *u,*k;
  Vec            Uloc,Kloc;

  PetscFunctionBeginUser;
  PetscCall(DMCompositeGetEntries(user->pack,&dau,&dak));
  PetscCall(DMDAGetLocalInfo(dau,&infou));
  PetscCall(DMDAGetLocalInfo(dak,&infok));
  PetscCall(DMCompositeGetLocalVectors(user->pack,&Uloc,&Kloc));
  switch (user->ptype) {
  case 0:
    PetscCall(DMGlobalToLocalBegin(dau,X,INSERT_VALUES,Uloc));
    PetscCall(DMGlobalToLocalEnd  (dau,X,INSERT_VALUES,Uloc));
    PetscCall(DMDAVecGetArray(dau,Uloc,&u));
    PetscCall(DMDAVecGetArray(dak,user->Kloc,&k));
    PetscCall(FormJacobianLocal_U(user,&infou,u,k,B));
    PetscCall(DMDAVecRestoreArray(dau,Uloc,&u));
    PetscCall(DMDAVecRestoreArray(dak,user->Kloc,&k));
    break;
  case 1:
    PetscCall(DMGlobalToLocalBegin(dak,X,INSERT_VALUES,Kloc));
    PetscCall(DMGlobalToLocalEnd  (dak,X,INSERT_VALUES,Kloc));
    PetscCall(DMDAVecGetArray(dau,user->Uloc,&u));
    PetscCall(DMDAVecGetArray(dak,Kloc,&k));
    PetscCall(FormJacobianLocal_K(user,&infok,u,k,B));
    PetscCall(DMDAVecRestoreArray(dau,user->Uloc,&u));
    PetscCall(DMDAVecRestoreArray(dak,Kloc,&k));
    break;
  case 2: {
    Mat       Buu,Buk,Bku,Bkk;
    PetscBool nest;
    IS        *is;
    PetscCall(DMCompositeScatter(user->pack,X,Uloc,Kloc));
    PetscCall(DMDAVecGetArray(dau,Uloc,&u));
    PetscCall(DMDAVecGetArray(dak,Kloc,&k));
    PetscCall(DMCompositeGetLocalISs(user->pack,&is));
    PetscCall(MatGetLocalSubMatrix(B,is[0],is[0],&Buu));
    PetscCall(MatGetLocalSubMatrix(B,is[0],is[1],&Buk));
    PetscCall(MatGetLocalSubMatrix(B,is[1],is[0],&Bku));
    PetscCall(MatGetLocalSubMatrix(B,is[1],is[1],&Bkk));
    PetscCall(FormJacobianLocal_U(user,&infou,u,k,Buu));
    PetscCall(PetscObjectTypeCompare((PetscObject)B,MATNEST,&nest));
    if (!nest) {
      /*
         DMCreateMatrix_Composite()  for a nested matrix does not generate off-block matrices that one can call MatSetValuesLocal() on, it just creates dummy
         matrices with no entries; there cannot insert values into them. Commit b6480e041dd2293a65f96222772d68cdb4ed6306
         changed Mat_Nest() from returning NULL pointers for these submatrices to dummy matrices because PCFIELDSPLIT could not
         handle the returned null matrices.
      */
      PetscCall(FormJacobianLocal_UK(user,&infou,&infok,u,k,Buk));
      PetscCall(FormJacobianLocal_KU(user,&infou,&infok,u,k,Bku));
    }
    PetscCall(FormJacobianLocal_K(user,&infok,u,k,Bkk));
    PetscCall(MatRestoreLocalSubMatrix(B,is[0],is[0],&Buu));
    PetscCall(MatRestoreLocalSubMatrix(B,is[0],is[1],&Buk));
    PetscCall(MatRestoreLocalSubMatrix(B,is[1],is[0],&Bku));
    PetscCall(MatRestoreLocalSubMatrix(B,is[1],is[1],&Bkk));
    PetscCall(DMDAVecRestoreArray(dau,Uloc,&u));
    PetscCall(DMDAVecRestoreArray(dak,Kloc,&k));

    PetscCall(ISDestroy(&is[0]));
    PetscCall(ISDestroy(&is[1]));
    PetscCall(PetscFree(is));
  } break;
  }
  PetscCall(DMCompositeRestoreLocalVectors(user->pack,&Uloc,&Kloc));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd  (B,MAT_FINAL_ASSEMBLY));
  if (J != B) {
    PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd  (J,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormInitial_Coupled(User user,Vec X)
{
  DM             dau,dak;
  DMDALocalInfo  infou,infok;
  Vec            Xu,Xk;
  PetscScalar    *u,*k,hx;
  PetscInt       i;

  PetscFunctionBeginUser;
  PetscCall(DMCompositeGetEntries(user->pack,&dau,&dak));
  PetscCall(DMCompositeGetAccess(user->pack,X,&Xu,&Xk));
  PetscCall(DMDAVecGetArray(dau,Xu,&u));
  PetscCall(DMDAVecGetArray(dak,Xk,&k));
  PetscCall(DMDAGetLocalInfo(dau,&infou));
  PetscCall(DMDAGetLocalInfo(dak,&infok));
  hx   = 1./(infok.mx);
  for (i=infou.xs; i<infou.xs+infou.xm; i++) u[i] = (PetscScalar)i*hx * (1.-(PetscScalar)i*hx);
  for (i=infok.xs; i<infok.xs+infok.xm; i++) k[i] = 1.0 + 0.5*PetscSinScalar(2*PETSC_PI*i*hx);
  PetscCall(DMDAVecRestoreArray(dau,Xu,&u));
  PetscCall(DMDAVecRestoreArray(dak,Xk,&k));
  PetscCall(DMCompositeRestoreAccess(user->pack,X,&Xu,&Xk));
  PetscCall(DMCompositeScatter(user->pack,X,user->Uloc,user->Kloc));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  DM             dau,dak,pack;
  const PetscInt *lxu;
  PetscInt       *lxk,m,sizes;
  User           user;
  SNES           snes;
  Vec            X,F,Xu,Xk,Fu,Fk;
  Mat            B;
  IS             *isg;
  PetscBool      pass_dm;

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,10,1,1,NULL,&dau));
  PetscCall(DMSetOptionsPrefix(dau,"u_"));
  PetscCall(DMSetFromOptions(dau));
  PetscCall(DMSetUp(dau));
  PetscCall(DMDAGetOwnershipRanges(dau,&lxu,0,0));
  PetscCall(DMDAGetInfo(dau,0, &m,0,0, &sizes,0,0, 0,0,0,0,0,0));
  PetscCall(PetscMalloc1(sizes,&lxk));
  PetscCall(PetscArraycpy(lxk,lxu,sizes));
  lxk[0]--;
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,m-1,1,1,lxk,&dak));
  PetscCall(DMSetOptionsPrefix(dak,"k_"));
  PetscCall(DMSetFromOptions(dak));
  PetscCall(DMSetUp(dak));
  PetscCall(PetscFree(lxk));

  PetscCall(DMCompositeCreate(PETSC_COMM_WORLD,&pack));
  PetscCall(DMSetOptionsPrefix(pack,"pack_"));
  PetscCall(DMCompositeAddDM(pack,dau));
  PetscCall(DMCompositeAddDM(pack,dak));
  PetscCall(DMDASetFieldName(dau,0,"u"));
  PetscCall(DMDASetFieldName(dak,0,"k"));
  PetscCall(DMSetFromOptions(pack));

  PetscCall(DMCreateGlobalVector(pack,&X));
  PetscCall(VecDuplicate(X,&F));

  PetscCall(PetscNew(&user));

  user->pack = pack;

  PetscCall(DMCompositeGetGlobalISs(pack,&isg));
  PetscCall(DMCompositeGetLocalVectors(pack,&user->Uloc,&user->Kloc));
  PetscCall(DMCompositeScatter(pack,X,user->Uloc,user->Kloc));

  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Coupled problem options","SNES");
  {
    user->ptype = 0; pass_dm = PETSC_TRUE;

    PetscCall(PetscOptionsInt("-problem_type","0: solve for u only, 1: solve for k only, 2: solve for both",0,user->ptype,&user->ptype,NULL));
    PetscCall(PetscOptionsBool("-pass_dm","Pass the packed DM to SNES to use when determining splits and forward into splits",0,pass_dm,&pass_dm,NULL));
  }
  PetscOptionsEnd();

  PetscCall(FormInitial_Coupled(user,X));

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  switch (user->ptype) {
  case 0:
    PetscCall(DMCompositeGetAccess(pack,X,&Xu,0));
    PetscCall(DMCompositeGetAccess(pack,F,&Fu,0));
    PetscCall(DMCreateMatrix(dau,&B));
    PetscCall(SNESSetFunction(snes,Fu,FormFunction_All,user));
    PetscCall(SNESSetJacobian(snes,B,B,FormJacobian_All,user));
    PetscCall(SNESSetFromOptions(snes));
    PetscCall(SNESSetDM(snes,dau));
    PetscCall(SNESSolve(snes,NULL,Xu));
    PetscCall(DMCompositeRestoreAccess(pack,X,&Xu,0));
    PetscCall(DMCompositeRestoreAccess(pack,F,&Fu,0));
    break;
  case 1:
    PetscCall(DMCompositeGetAccess(pack,X,0,&Xk));
    PetscCall(DMCompositeGetAccess(pack,F,0,&Fk));
    PetscCall(DMCreateMatrix(dak,&B));
    PetscCall(SNESSetFunction(snes,Fk,FormFunction_All,user));
    PetscCall(SNESSetJacobian(snes,B,B,FormJacobian_All,user));
    PetscCall(SNESSetFromOptions(snes));
    PetscCall(SNESSetDM(snes,dak));
    PetscCall(SNESSolve(snes,NULL,Xk));
    PetscCall(DMCompositeRestoreAccess(pack,X,0,&Xk));
    PetscCall(DMCompositeRestoreAccess(pack,F,0,&Fk));
    break;
  case 2:
    PetscCall(DMCreateMatrix(pack,&B));
    /* This example does not correctly allocate off-diagonal blocks. These options allows new nonzeros (slow). */
    PetscCall(MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
    PetscCall(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
    PetscCall(SNESSetFunction(snes,F,FormFunction_All,user));
    PetscCall(SNESSetJacobian(snes,B,B,FormJacobian_All,user));
    PetscCall(SNESSetFromOptions(snes));
    if (!pass_dm) {             /* Manually provide index sets and names for the splits */
      KSP ksp;
      PC  pc;
      PetscCall(SNESGetKSP(snes,&ksp));
      PetscCall(KSPGetPC(ksp,&pc));
      PetscCall(PCFieldSplitSetIS(pc,"u",isg[0]));
      PetscCall(PCFieldSplitSetIS(pc,"k",isg[1]));
    } else {
      /* The same names come from the options prefix for dau and dak. This option can support geometric multigrid inside
       * of splits, but it requires using a DM (perhaps your own implementation). */
      PetscCall(SNESSetDM(snes,pack));
    }
    PetscCall(SNESSolve(snes,NULL,X));
    break;
  }
  PetscCall(VecViewFromOptions(X,NULL,"-view_sol"));

  if (0) {
    PetscInt  col      = 0;
    PetscBool mult_dup = PETSC_FALSE,view_dup = PETSC_FALSE;
    Mat       D;
    Vec       Y;

    PetscCall(PetscOptionsGetInt(NULL,0,"-col",&col,0));
    PetscCall(PetscOptionsGetBool(NULL,0,"-mult_dup",&mult_dup,0));
    PetscCall(PetscOptionsGetBool(NULL,0,"-view_dup",&view_dup,0));

    PetscCall(VecDuplicate(X,&Y));
    /* PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY)); */
    /* PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY)); */
    PetscCall(MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&D));
    PetscCall(VecZeroEntries(X));
    PetscCall(VecSetValue(X,col,1.0,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(X));
    PetscCall(VecAssemblyEnd(X));
    PetscCall(MatMult(mult_dup ? D : B,X,Y));
    PetscCall(MatView(view_dup ? D : B,PETSC_VIEWER_STDOUT_WORLD));
    /* PetscCall(VecView(X,PETSC_VIEWER_STDOUT_WORLD)); */
    PetscCall(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(MatDestroy(&D));
    PetscCall(VecDestroy(&Y));
  }

  PetscCall(DMCompositeRestoreLocalVectors(pack,&user->Uloc,&user->Kloc));
  PetscCall(PetscFree(user));

  PetscCall(ISDestroy(&isg[0]));
  PetscCall(ISDestroy(&isg[1]));
  PetscCall(PetscFree(isg));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&F));
  PetscCall(MatDestroy(&B));
  PetscCall(DMDestroy(&dau));
  PetscCall(DMDestroy(&dak));
  PetscCall(DMDestroy(&pack));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
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
