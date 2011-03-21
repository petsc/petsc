static const char help[] = "Uses analytic Jacobians to solve individual problems and a coupled problem\n\n";

/* Solve a PDE coupled to an algebraic system in 1D
 *
 * PDE (U):
 *     -(k u_x)_x = 1 on (0,1), subject to u(0) = 0, u(1) = 1
 * Algebraic (K):
 *     exp(k-1) + k = u + 1/(1/(1+u) + 1/(1+u_x^2))
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
#include <petscdmmesh.h>
#include <petscdmcomposite.h>

  #include <petscdmda.h>

#include "ex10_quadrature.h"

typedef enum {NEUMANN, DIRICHLET} BCType;

PetscErrorCode DMDACreateOwnershipRanges(DM); /* Import an internal function */

typedef struct _UserCtx *User;
struct _UserCtx {
  PetscInt ptype;
  DM       pack;
  Vec      Uloc,Kloc;

  BCType        bcType;                      // The type of boundary conditions
  PetscScalar (*exactFunc)(const double []); // The exact solution function
  double (*integrate)(const double *, const double *, const int, double (*)(const double *)); // Basis functional application
};

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal_U"
static PetscErrorCode FormFunctionLocal_U(User user,DMDALocalInfo *info,const PetscScalar u[],const PetscScalar k[],PetscScalar f[])
{
  PetscReal hx = 1./info->mx;
  PetscInt i;

  PetscFunctionBegin;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    if (i == 0) f[i] = hx*u[i];
    else if (i == info->mx-1) f[i] = hx*(u[i] - 1.0);
    else f[i] = hx*((k[i-1]*(u[i]-u[i-1]) - k[i]*(u[i+1]-u[i]))/(hx*hx) - 1.0);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal_K"
static PetscErrorCode FormFunctionLocal_K(User user,DMDALocalInfo *info,const PetscScalar u[],const PetscScalar k[],PetscScalar f[])
{
  PetscReal hx = 1./info->mx;
  PetscInt  i;

  PetscFunctionBegin;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    const PetscScalar
      ubar = 0.5*(u[i+1]+u[i]),
      gradu = (u[i+1]-u[i])/hx,
      g = 1. + gradu*gradu,
      w = 1./(1.+ubar) + 1./g;
    f[i] = hx*(PetscExpScalar(k[i]-1.0) + k[i] - 1./w);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction_All"
static PetscErrorCode FormFunction_All(SNES snes,Vec X,Vec F,void *ctx)
{
  User              user = (User)ctx;
  DM                dau,dak;
  DMDALocalInfo     infou,infok;
  const PetscScalar *u,*k;
  PetscScalar       *fu,*fk;
  PetscErrorCode    ierr;
  Vec               Uloc,Kloc,Fu,Fk;

  PetscFunctionBegin;
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

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal_U"
static PetscErrorCode FormJacobianLocal_U(User user,DMDALocalInfo *info,const PetscScalar u[],const PetscScalar k[],Mat Buu)
{
  PetscReal      hx = 1./info->mx;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    PetscInt row = i-info->gxs,cols[] = {row-1,row,row+1};
    PetscScalar val = hx;
    if (i == 0) {ierr = MatSetValuesLocal(Buu,1,&row,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);}
    else if (i == info->mx-1) {ierr = MatSetValuesLocal(Buu,1,&row,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);}
    else {
      PetscScalar vals[] = {-k[i-1]/hx,(k[i-1]+k[i])/hx,-k[i]/hx};
      ierr = MatSetValuesLocal(Buu,1,&row,3,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal_K"
static PetscErrorCode FormJacobianLocal_K(User user,DMDALocalInfo *info,const PetscScalar u[],const PetscScalar k[],Mat Bkk)
{
  PetscReal      hx = 1./info->mx;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    PetscInt row = i-info->gxs;
    PetscScalar vals[] = {hx*(PetscExpScalar(k[i]-1.)+1.)};
    ierr = MatSetValuesLocal(Bkk,1,&row,1,&row,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal_UK"
static PetscErrorCode FormJacobianLocal_UK(User user,DMDALocalInfo *info,DMDALocalInfo *infok,const PetscScalar u[],const PetscScalar k[],Mat Buk)
{
  PetscReal hx = 1./info->mx;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (!Buk) PetscFunctionReturn(0); /* Not assembling this block */
  for (i=info->xs; i<info->xs+info->xm; i++) {
    if (i == 0 || i == info->mx-1) continue;
    PetscInt row = i-info->gxs,cols[] = {i-1-infok->gxs,i-infok->gxs};
    PetscScalar vals[] = {(u[i]-u[i-1])/hx,(u[i]-u[i+1])/hx};
    ierr = MatSetValuesLocal(Buk,1,&row,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal_KU"
static PetscErrorCode FormJacobianLocal_KU(User user,DMDALocalInfo *info,DMDALocalInfo *infok,const PetscScalar u[],const PetscScalar k[],Mat Bku)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      hx = 1./(info->mx-1);

  PetscFunctionBegin;
  if (!Bku) PetscFunctionReturn(0); /* Not assembling this block */
  for (i=infok->xs; i<infok->xs+infok->xm; i++) {
    PetscInt row = i-infok->gxs,cols[] = {i-info->gxs,i+1-info->gxs};
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
      iw_gradu = -w_gradu * PetscSqr(iw),
      vals[]   = {-hx*(iw_ubar*ubar_L + iw_gradu*gradu_L),
                  -hx*(iw_ubar*ubar_R + iw_gradu*gradu_R)};
    ierr = MatSetValuesLocal(Bku,1,&row,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobian_All"
static PetscErrorCode FormJacobian_All(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *mstr,void *ctx)
{
  User              user = (User)ctx;
  DM                dau,dak;
  DMDALocalInfo     infou,infok;
  const PetscScalar *u,*k;
  PetscErrorCode    ierr;
  Vec               Uloc,Kloc;

  PetscFunctionBegin;
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
    ierr = FormJacobianLocal_U(user,&infou,u,k,*B);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dau,Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dak,user->Kloc,&k);CHKERRQ(ierr);
    break;
  case 1:
    ierr = DMGlobalToLocalBegin(dak,X,INSERT_VALUES,Kloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd  (dak,X,INSERT_VALUES,Kloc);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dau,user->Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dak,Kloc,&k);CHKERRQ(ierr);
    ierr = FormJacobianLocal_K(user,&infok,u,k,*B);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dau,user->Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dak,Kloc,&k);CHKERRQ(ierr);
    break;
  case 2: {
    Mat Buu,Buk,Bku,Bkk;
    IS  *is;
    ierr = DMCompositeScatter(user->pack,X,Uloc,Kloc);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dau,Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dak,Kloc,&k);CHKERRQ(ierr);
    ierr = DMCompositeGetLocalISs(user->pack,&is);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(*B,is[0],is[0],&Buu);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(*B,is[0],is[1],&Buk);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(*B,is[1],is[0],&Bku);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(*B,is[1],is[1],&Bkk);CHKERRQ(ierr);
    ierr = FormJacobianLocal_U(user,&infou,u,k,Buu);CHKERRQ(ierr);
    ierr = FormJacobianLocal_UK(user,&infou,&infok,u,k,Buk);CHKERRQ(ierr);
    ierr = FormJacobianLocal_KU(user,&infou,&infok,u,k,Bku);CHKERRQ(ierr);
    ierr = FormJacobianLocal_K(user,&infok,u,k,Bkk);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(*B,is[0],is[0],&Buu);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(*B,is[0],is[1],&Buk);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(*B,is[1],is[0],&Bku);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(*B,is[1],is[1],&Bkk);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dau,Uloc,&u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dak,Kloc,&k);CHKERRQ(ierr);

    ierr = ISDestroy(is[0]);CHKERRQ(ierr);
    ierr = ISDestroy(is[1]);CHKERRQ(ierr);
    ierr = PetscFree(is);CHKERRQ(ierr);
  } break;
  }
  ierr = DMCompositeRestoreLocalVectors(user->pack,&Uloc,&Kloc);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != *B) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *mstr = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormInitial_Coupled"
static PetscErrorCode FormInitial_Coupled(User user,Vec X)
{
  ALE::Obj<PETSC_MESH_TYPE> meshU;
  ALE::Obj<PETSC_MESH_TYPE> meshK;
  DM             dmu,          dmk;
  SectionReal    coordinatesU, coordinatesK;
  SectionReal    sectionU,     sectionK;
  Vec            vecU,         vecK;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCompositeGetEntries(user->pack, &dmu, &dmk);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dmu, meshU);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dmk, meshK);CHKERRQ(ierr);
  ierr = DMMeshGetSectionReal(dmu, "coordinates", &coordinatesU);CHKERRQ(ierr);
  ierr = DMMeshGetSectionReal(dmk, "coordinates", &coordinatesK);CHKERRQ(ierr);
  ierr = DMMeshGetSectionReal(dmu, "default", &sectionU);CHKERRQ(ierr);
  ierr = DMMeshGetSectionReal(dmk, "default", &sectionK);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& verticesU = meshU->depthStratum(0);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& verticesK = meshK->depthStratum(0);

  for(PETSC_MESH_TYPE::label_sequence::iterator v_iter = verticesU->begin(); v_iter != verticesU->end(); ++v_iter) {
    PetscScalar  values[1];
    PetscScalar *coords;

    ierr = SectionRealRestrict(coordinatesU, *v_iter, &coords);CHKERRQ(ierr);
    values[0] = coords[0]*(1.0 - coords[0]);
    ierr = SectionRealUpdate(sectionU, *v_iter, values, INSERT_VALUES);CHKERRQ(ierr);
  }
  for(PETSC_MESH_TYPE::label_sequence::iterator v_iter = verticesK->begin(); v_iter != verticesK->end(); ++v_iter) {
    PetscScalar  values[1];
    PetscScalar *coords;

    ierr = SectionRealRestrict(coordinatesK, *v_iter, &coords);CHKERRQ(ierr);
    values[0] = (PetscScalar) (1.0 + 0.5*sin((double) 2*PETSC_PI*coords[0]));
    ierr = SectionRealUpdate(sectionK, *v_iter, values, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = SectionRealCreateLocalVector(sectionU, &vecU);CHKERRQ(ierr);
  ierr = SectionRealCreateLocalVector(sectionK, &vecK);CHKERRQ(ierr);
  ierr = VecCopy(vecU, user->Uloc);CHKERRQ(ierr);
  ierr = VecCopy(vecK, user->Kloc);CHKERRQ(ierr);
  ierr = VecDestroy(vecU);CHKERRQ(ierr);
  ierr = VecDestroy(vecK);CHKERRQ(ierr);
  ierr = SectionRealDestroy(coordinatesU);CHKERRQ(ierr);
  ierr = SectionRealDestroy(coordinatesK);CHKERRQ(ierr);
  ierr = SectionRealDestroy(sectionU);CHKERRQ(ierr);
  ierr = SectionRealDestroy(sectionK);CHKERRQ(ierr);
  ierr = VecView(user->Uloc,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(user->Kloc,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMCompositeGather(user->pack, X, INSERT_VALUES, user->Uloc, user->Kloc);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  DM              dau, dak, dmu, dmk, pack;
  User            user;
  SNES            snes;
  KSP             ksp;
  PC              pc;
  Mat             B;
  Vec             X,F,Xu,Xk,Fu,Fk;
  IS             *isg;
  const PetscInt *lxu;
  PetscInt       *lxk, m, nprocs;
  PetscBool       view_draw;
  PetscMPIInt     size;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This example does not yet work in parallel");
  /* Create meshes */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,-10,1,1,PETSC_NULL,&dau);CHKERRQ(ierr);
  ierr = DMDACreateOwnershipRanges(dau);CHKERRQ(ierr); /* Ensure that the ownership ranges agree so that we can get a compatible grid for the coefficient */
  ierr = DMDAGetOwnershipRanges(dau,&lxu,0,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dau,0, &m,0,0, &nprocs,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscMalloc(nprocs*sizeof(*lxk),&lxk);CHKERRQ(ierr);
  ierr = PetscMemcpy(lxk,lxu,nprocs*sizeof(*lxk));CHKERRQ(ierr);
  lxk[0]--;
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,m-1,1,1,lxk,&dak);CHKERRQ(ierr);
  ierr = PetscFree(lxk);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dau, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dak, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
  ierr = DMConvert(dau, DMMESH, &dmu);CHKERRQ(ierr);
  ierr = DMConvert(dak, DMMESH, &dmk);CHKERRQ(ierr);
  ierr = DMDestroy(dau);CHKERRQ(ierr);
  ierr = DMDestroy(dak);CHKERRQ(ierr);

  ierr = PetscNew(struct _UserCtx,&user);CHKERRQ(ierr);
  user->bcType = NEUMANN;
  /* Setup dof layout.
   For a DMDA, this is automatic given the number of dof at each vertex. However,
   for a DMMesh, we need to specify this.
  */
  {
    /* There is perhaps a better way to do this that does not rely on the Discretization/BoundaryCondition objects in Mesh.hh */
    int      numBC      = (user->bcType == DIRICHLET) ? 1 : 0;
    int      markers[1] = {1};
    double (*funcs[1])(const double *coords) = {user->exactFunc};

    ierr = CreateProblem_gen_0(dmu, "u", numBC, markers, funcs, user->exactFunc);CHKERRQ(ierr);
    ierr = CreateProblem_gen_0(dmk, "k", numBC, markers, funcs, user->exactFunc);CHKERRQ(ierr);
    user->integrate = IntegrateDualBasis_gen_0;
  }
  SectionReal defaultSection;

  ierr = DMMeshGetSectionReal(dmu, "default", &defaultSection);CHKERRQ(ierr);
  ierr = DMMeshSetupSection(dmu, defaultSection);CHKERRQ(ierr);
  ierr = SectionRealDestroy(defaultSection);CHKERRQ(ierr);
  ierr = DMMeshGetSectionReal(dmk, "default", &defaultSection);CHKERRQ(ierr);
  ierr = DMMeshSetupSection(dmk, defaultSection);CHKERRQ(ierr);
  ierr = SectionRealDestroy(defaultSection);CHKERRQ(ierr);

  ierr = DMCompositeCreate(PETSC_COMM_WORLD, &pack);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(pack, dmu);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(pack, dmk);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) dmu, "u_");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) dmk, "k_");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) pack, "pack_");CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(pack, &X);CHKERRQ(ierr);
  ierr = VecDuplicate(X, &F);CHKERRQ(ierr);

  user->pack = pack;
  ierr = DMCompositeGetGlobalISs(pack, &isg);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(pack, &user->Uloc, &user->Kloc);CHKERRQ(ierr);
  ierr = DMCompositeScatter(pack, X, user->Uloc, user->Kloc);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Coupled problem options","SNES");CHKERRQ(ierr);
  {
    user->ptype = 0; view_draw = PETSC_FALSE;
    ierr = PetscOptionsInt("-problem_type","0: solve for u only, 1: solve for k only, 2: solve for both",0,user->ptype,&user->ptype,0);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view_draw","Draw the final coupled solution regardless of whether only one physics was solved",0,view_draw,&view_draw,0);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = FormInitial_Coupled(user,X);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  switch (user->ptype) {
  case 0:
    ierr = DMCompositeGetAccess(pack,X,&Xu,0);CHKERRQ(ierr);
    ierr = DMCompositeGetAccess(pack,F,&Fu,0);CHKERRQ(ierr);
    ierr = DMGetMatrix(dmu,PETSC_NULL,&B);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,Fu,FormFunction_All,user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,B,B,FormJacobian_All,user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    ierr = SNESSolve(snes,PETSC_NULL,Xu);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(pack,X,&Xu,0);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(pack,F,&Fu,0);CHKERRQ(ierr);
    break;
  case 1:
    ierr = DMCompositeGetAccess(pack,X,0,&Xk);CHKERRQ(ierr);
    ierr = DMCompositeGetAccess(pack,F,0,&Fk);CHKERRQ(ierr);
    ierr = DMGetMatrix(dmk,PETSC_NULL,&B);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,Fk,FormFunction_All,user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,B,B,FormJacobian_All,user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    ierr = SNESSolve(snes,PETSC_NULL,Xk);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(pack,X,0,&Xk);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(pack,F,0,&Fk);CHKERRQ(ierr);
    break;
  case 2:
    ierr = DMGetMatrix(pack,PETSC_NULL,&B);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,F,FormFunction_All,user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,B,B,FormJacobian_All,user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"u",isg[0]);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"k",isg[1]);CHKERRQ(ierr);
    ierr = SNESSolve(snes,PETSC_NULL,X);CHKERRQ(ierr);
    break;
  }
  if (view_draw) {ierr = VecView(X,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  if (0) {
    PetscInt col = 0;
    PetscBool mult_dup = PETSC_FALSE,view_dup = PETSC_FALSE;
    Mat D;
    Vec Y;

    ierr = PetscOptionsGetInt(0,"-col",&col,0);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(0,"-mult_dup",&mult_dup,0);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(0,"-view_dup",&view_dup,0);CHKERRQ(ierr);

    ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
    /* ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); */
    /* ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); */
    ierr = MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
    ierr = VecZeroEntries(X);CHKERRQ(ierr);
    ierr = VecSetValue(X,col,1.0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
    ierr = MatMult(mult_dup?D:B,X,Y);CHKERRQ(ierr);
    ierr = MatView(view_dup?D:B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    /* ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
    ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatDestroy(D);CHKERRQ(ierr);
    ierr = VecDestroy(Y);CHKERRQ(ierr);
  }

  ierr = DMCompositeRestoreLocalVectors(pack,&user->Uloc,&user->Kloc);CHKERRQ(ierr);
  ierr = PetscFree(user);CHKERRQ(ierr);

  ierr = ISDestroy(isg[0]);CHKERRQ(ierr);
  ierr = ISDestroy(isg[1]);CHKERRQ(ierr);
  ierr = PetscFree(isg);CHKERRQ(ierr);
  ierr = VecDestroy(X);CHKERRQ(ierr);
  ierr = VecDestroy(F);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  ierr = DMDestroy(dmu);CHKERRQ(ierr);
  ierr = DMDestroy(dmk);CHKERRQ(ierr);
  ierr = DMDestroy(pack);CHKERRQ(ierr);
  ierr = SNESDestroy(snes);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
