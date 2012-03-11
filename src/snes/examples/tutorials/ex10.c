static const char help[] = "Uses analytic Jacobians to solve individual problems and a coupled problem.\n\n";

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

#undef __FUNCT__
#define __FUNCT__ "CreateProblem_gen_0"
PetscErrorCode CreateProblem_gen_0(DM dm, const char *name)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr = 0;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  {
    const ALE::Obj<ALE::Discretization>& d = new ALE::Discretization(m->comm(), m->debug());

    d->setNumDof(0, 1);
    d->setNumDof(1, 0);
    m->setDiscretization(name, d);
  }
  PetscFunctionReturn(0);
}

typedef enum {NEUMANN, DIRICHLET} BCType;

typedef struct _UserCtx *User;
struct _UserCtx {
  PetscInt  ptype;
  DM        pack;
  Vec       Uloc,Kloc;
  PetscReal hxu, hxk;
};

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal_U"
static PetscErrorCode FormFunctionLocal_U(DM dmu, DM dmk, SectionReal sectionU, SectionReal sectionK, SectionReal sectionF, User user)
{
  ALE::Obj<PETSC_MESH_TYPE> meshU;
  ALE::Obj<PETSC_MESH_TYPE> meshK;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dmu, meshU);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dmk, meshK);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& verticesU = meshU->depthStratum(0);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& verticesK = meshK->depthStratum(0);
  PETSC_MESH_TYPE::label_sequence::iterator vur_iter = verticesU->begin();
  PETSC_MESH_TYPE::point_type ulp = -1;
  PETSC_MESH_TYPE::point_type urp = *(++vur_iter);
  PETSC_MESH_TYPE::point_type klp = -1;
  PetscReal hx = user->hxu;

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Starting U residual\n");CHKERRQ(ierr);
  for(PETSC_MESH_TYPE::label_sequence::iterator vu_iter = verticesU->begin(), vk_iter = verticesK->begin(); vu_iter != verticesU->end(); ++vu_iter,  ++vk_iter) {
    PETSC_MESH_TYPE::point_type up = *vu_iter;
    PETSC_MESH_TYPE::point_type kp = *vk_iter;
    const PetscInt marker = meshU->getValue(meshU->getLabel("marker"), *vu_iter, 0);
    PetscScalar    values[1];
    PetscScalar   *u;

    ierr = SectionRealRestrict(sectionU, up, &u);CHKERRQ(ierr);
    if (marker == 1) { /* Left end */
      values[0] = hx*u[0];
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "  Left  End vu %d hx: %g f %g\n", up, hx, values[0]);CHKERRQ(ierr);
      urp  = *(++vur_iter);
    } else if (marker == 2) { /* Right end */
      values[0] = hx*(u[0] - 1.0);
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "  Right End vu %d hx: %g f %g\n", up, hx, values[0]);CHKERRQ(ierr);
    } else if (marker == 3) { /* Left Ghost */
      urp  = *(++vur_iter);
    } else if (marker == 4) { /* Right Ghost */
    } else {
      PetscScalar *ul, *ur, *k, *kl;

      ierr = SectionRealRestrict(sectionU, urp, &ur);CHKERRQ(ierr);
      ierr = SectionRealRestrict(sectionU, ulp, &ul);CHKERRQ(ierr);
      ierr = SectionRealRestrict(sectionK, kp,  &k);CHKERRQ(ierr);
      ierr = SectionRealRestrict(sectionK, klp, &kl);CHKERRQ(ierr);
      values[0] = hx*((kl[0]*(u[0]-ul[0]) - k[0]*(ur[0]-u[0]))/(hx*hx) - 1.0);
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "  vu %d hx: %g ul %g u %g ur %g kl %g k %g f %g\n", up, hx, ul[0], u[0], ur[0], kl[0], k[0], values[0]);CHKERRQ(ierr);
      urp  = *(++vur_iter);
    }
    ierr = SectionRealUpdate(sectionF, up, values, INSERT_VALUES);CHKERRQ(ierr);
    ulp  = up;
    klp  = kp;
  }
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Ending U residual\n");CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal_K"
static PetscErrorCode FormFunctionLocal_K(DM dmu, DM dmk, SectionReal sectionU, SectionReal sectionK, SectionReal sectionF, User user)
{
  ALE::Obj<PETSC_MESH_TYPE> meshU;
  ALE::Obj<PETSC_MESH_TYPE> meshK;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dmu, meshU);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dmk, meshK);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& verticesU = meshU->depthStratum(0);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& verticesK = meshK->depthStratum(0);
  PETSC_MESH_TYPE::label_sequence::iterator vu_iter = verticesU->begin();
  PETSC_MESH_TYPE::point_type               up      = *vu_iter;
  PETSC_MESH_TYPE::point_type               urp;
  PetscReal hx = user->hxk;

  //ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting K residual\n");CHKERRQ(ierr);
  for(PETSC_MESH_TYPE::label_sequence::iterator vk_iter = verticesK->begin(); vk_iter != verticesK->end(); ++vk_iter) {
    PetscScalar    values[1];
    PetscScalar   *u, *ur, *k;

    urp  = *(++vu_iter);
    ierr = SectionRealRestrict(sectionU, up, &u);CHKERRQ(ierr);
    ierr = SectionRealRestrict(sectionU, urp, &ur);CHKERRQ(ierr);
    ierr = SectionRealRestrict(sectionK, *vk_iter, &k);CHKERRQ(ierr);
    const PetscScalar ubar  = 0.5*(ur[0] + u[0]);
    const PetscScalar gradu = (ur[0] - u[0])/hx;
    const PetscScalar g     = 1.0 + gradu*gradu;
    const PetscScalar w     = 1.0/(1.0 + ubar) + 1.0/g;

    values[0] = hx*(PetscExpScalar(k[0]-1.0) + k[0] - 1.0/w);
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "  vk %d vu %d vur %d: ubar %g gradu %g g %g w %g f %g\n", *vk_iter, up, urp, ubar, gradu, g, w, values[0]);CHKERRQ(ierr);
    ierr = SectionRealUpdate(sectionF, *vk_iter, values, INSERT_VALUES);CHKERRQ(ierr);

    up = urp;
  }
  //ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending K residual\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction_All"
static PetscErrorCode FormFunction_All(SNES snes, Vec X, Vec F, void *ctx)
{
  User           user = (User) ctx;
  DM             dmu, dmk;
  Vec            Uloc, Kloc, Fu, Fk, vecFu, vecFk, vecU, vecK;
  SectionReal    sectionU, sectionK, sectionFu, sectionFk;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCompositeGetEntries(user->pack, &dmu, &dmk);CHKERRQ(ierr);
  switch (user->ptype) {
  case 0:
    ierr = DMMeshGetSectionReal(dmu, "default", &sectionU);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionU, &vecU);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dmu, X, INSERT_VALUES, vecU);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmu, X, INSERT_VALUES, vecU);CHKERRQ(ierr);
    ierr = VecDestroy(&vecU);CHKERRQ(ierr);

    ierr = DMMeshGetSectionReal(dmk, "default", &sectionK);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionK, &vecK);CHKERRQ(ierr);
    ierr = VecCopy(user->Kloc, vecK);CHKERRQ(ierr);
    ierr = VecDestroy(&vecK);CHKERRQ(ierr);

    ierr = SectionRealDuplicate(sectionU, &sectionFu);CHKERRQ(ierr);

    ierr = FormFunctionLocal_U(dmu, dmk, sectionU, sectionK, sectionFu, user);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionFu, &vecFu);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dmu, vecFu, INSERT_VALUES, F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dmu, vecFu, INSERT_VALUES, F);CHKERRQ(ierr);
    ierr = VecDestroy(&vecFu);CHKERRQ(ierr);
    ierr = SectionRealDestroy(&sectionFu);CHKERRQ(ierr);
    break;
  case 1:
    ierr = DMMeshGetSectionReal(dmk, "default", &sectionK);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionK, &vecK);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dmk, X, INSERT_VALUES, vecK);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmk, X, INSERT_VALUES, vecK);CHKERRQ(ierr);
    ierr = VecDestroy(&vecK);CHKERRQ(ierr);

    ierr = DMMeshGetSectionReal(dmu, "default", &sectionU);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionU, &vecU);CHKERRQ(ierr);
    ierr = VecCopy(user->Uloc, vecU);CHKERRQ(ierr);
    ierr = VecDestroy(&vecU);CHKERRQ(ierr);

    ierr = SectionRealDuplicate(sectionK, &sectionFk);CHKERRQ(ierr);

    ierr = FormFunctionLocal_K(dmu, dmk, sectionU, sectionK, sectionFk, user);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionFk, &vecFk);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dmk, vecFk, INSERT_VALUES, F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dmk, vecFk, INSERT_VALUES, F);CHKERRQ(ierr);
    ierr = VecDestroy(&vecFk);CHKERRQ(ierr);
    ierr = SectionRealDestroy(&sectionFk);CHKERRQ(ierr);
    break;
  case 2:
    ierr = DMCompositeGetLocalVectors(user->pack, &Uloc, &Kloc);CHKERRQ(ierr);
    ierr = DMCompositeScatter(user->pack, X, Uloc, Kloc);CHKERRQ(ierr);

    ierr = DMMeshGetSectionReal(dmu, "default", &sectionU);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionU, &vecU);CHKERRQ(ierr);
    ierr = VecCopy(Uloc, vecU);CHKERRQ(ierr);
    ierr = VecDestroy(&vecU);CHKERRQ(ierr);

    ierr = DMMeshGetSectionReal(dmk, "default", &sectionK);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionK, &vecK);CHKERRQ(ierr);
    ierr = VecCopy(Kloc, vecK);CHKERRQ(ierr);
    ierr = VecDestroy(&vecK);CHKERRQ(ierr);

    ierr = DMCompositeRestoreLocalVectors(user->pack, &Uloc, &Kloc);CHKERRQ(ierr);
    ierr = SectionRealDuplicate(sectionU, &sectionFu);CHKERRQ(ierr);
    ierr = SectionRealDuplicate(sectionK, &sectionFk);CHKERRQ(ierr);

    ierr = FormFunctionLocal_U(dmu, dmk, sectionU, sectionK, sectionFu, user);CHKERRQ(ierr);
    ierr = FormFunctionLocal_K(dmu, dmk, sectionU, sectionK, sectionFk, user);CHKERRQ(ierr);
    ierr = DMCompositeGetLocalVectors(user->pack, &Fu, &Fk);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionFu, &vecFu);CHKERRQ(ierr);
    ierr = VecCopy(vecFu, Fu);CHKERRQ(ierr);
    ierr = VecDestroy(&vecFu);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionFk, &vecFk);CHKERRQ(ierr);
    ierr = VecCopy(vecFk, Fk);CHKERRQ(ierr);
    ierr = VecDestroy(&vecFk);CHKERRQ(ierr);
    ierr = DMCompositeGather(user->pack, F, INSERT_VALUES, Fu, Fk);CHKERRQ(ierr);
    ierr = SectionRealDestroy(&sectionFu);CHKERRQ(ierr);
    ierr = SectionRealDestroy(&sectionFk);CHKERRQ(ierr);
    break;
  }
  ierr = SectionRealDestroy(&sectionU);CHKERRQ(ierr);
  ierr = SectionRealDestroy(&sectionK);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal_U"
static PetscErrorCode FormJacobianLocal_U(DM dmu, DM dmk, SectionReal sectionU, SectionReal sectionK, Mat Buu, User user)
{
  ALE::Obj<PETSC_MESH_TYPE> meshU;
  ALE::Obj<PETSC_MESH_TYPE> meshK;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dmu, meshU);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dmk, meshK);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& verticesU = meshU->depthStratum(0);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& verticesK = meshK->depthStratum(0);
  PETSC_MESH_TYPE::label_sequence::iterator vur_iter = verticesU->begin();
  PETSC_MESH_TYPE::point_type klp = -1;
  PETSC_MESH_TYPE::point_type ulp = -1;
  PETSC_MESH_TYPE::point_type urp = *(++vur_iter);
  PetscReal hx = user->hxu;

  for(PETSC_MESH_TYPE::label_sequence::iterator vu_iter = verticesU->begin(), vk_iter = verticesK->begin(); vu_iter != verticesU->end(); ++vu_iter,  ++vk_iter) {
    PETSC_MESH_TYPE::point_type up = *vu_iter;
    PETSC_MESH_TYPE::point_type kp = *vk_iter;
    const PetscInt marker = meshU->getValue(meshU->getLabel("marker"), *vu_iter, 0);
    PetscScalar    values[3];

    if (marker == 1) {
      values[0] = hx;
      std::cout << "["<<meshU->commRank()<<"]: row " << up << " Left BC" << std::endl;
      ierr = MatSetValuesTopology(Buu, dmu, 1, &up, dmu, 1, &up, values, INSERT_VALUES);CHKERRQ(ierr);
      urp  = *(++vur_iter);
    } else if (marker == 2) {
      values[0] = hx;
      std::cout << "["<<meshU->commRank()<<"]: row " << up << " Right BC" << std::endl;
      ierr = MatSetValuesTopology(Buu, dmu, 1, &up, dmu, 1, &up, values, INSERT_VALUES);CHKERRQ(ierr);
    } else {
      PetscScalar *k, *kl;
      PetscInt     cols[3] = {ulp, up, urp};

      ierr = SectionRealRestrict(sectionK, kp,  &k);CHKERRQ(ierr);
      ierr = SectionRealRestrict(sectionK, klp, &kl);CHKERRQ(ierr);
      values[0] = -kl[0]/hx;
      values[1] = (kl[0]+k[0])/hx;
      values[2] = -k[0]/hx;
      std::cout << "["<<meshU->commRank()<<"]: row " << up << " cols " << cols[0] <<", "<< cols[1] <<", "<< cols[2] << std::endl;
      ierr = MatSetValuesTopology(Buu, dmu, 1, &up, dmu, 3, cols, values, INSERT_VALUES);CHKERRQ(ierr);
      urp  = *(++vur_iter);
    }
    ulp = up;
    klp = kp;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal_K"
static PetscErrorCode FormJacobianLocal_K(DM dmu, DM dmk, SectionReal sectionU, SectionReal sectionK, Mat Bkk, User user)
{
  ALE::Obj<PETSC_MESH_TYPE> meshU;
  ALE::Obj<PETSC_MESH_TYPE> meshK;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dmu, meshU);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dmk, meshK);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& verticesK = meshK->depthStratum(0);
  PetscReal hx = user->hxk;

  for(PETSC_MESH_TYPE::label_sequence::iterator vk_iter = verticesK->begin(); vk_iter != verticesK->end(); ++vk_iter) {
    PETSC_MESH_TYPE::point_type kp = *vk_iter;
    PetscScalar                 values[1];
    PetscScalar                *k;

    ierr = SectionRealRestrict(sectionK, kp,  &k);CHKERRQ(ierr);
    values[0] = hx*(PetscExpScalar(k[0] - 1.0) + 1.0);
    ierr = MatSetValuesTopology(Bkk, dmk, 1, &kp, dmk, 1, &kp, values, INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal_UK"
static PetscErrorCode FormJacobianLocal_UK(DM dmu, DM dmk, SectionReal sectionU, SectionReal sectionK, Mat Buk, User user)
{
  ALE::Obj<PETSC_MESH_TYPE> meshU;
  ALE::Obj<PETSC_MESH_TYPE> meshK;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!Buk) PetscFunctionReturn(0); /* Not assembling this block */
  ierr = DMMeshGetMesh(dmu, meshU);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dmk, meshK);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& verticesU = meshU->depthStratum(0);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& verticesK = meshK->depthStratum(0);
  PETSC_MESH_TYPE::label_sequence::iterator vur_iter = verticesU->begin();
  PETSC_MESH_TYPE::point_type ulp = -1;
  PETSC_MESH_TYPE::point_type urp = *(++vur_iter);
  PETSC_MESH_TYPE::point_type klp = -1;
  PetscReal hx = user->hxu;

  for(PETSC_MESH_TYPE::label_sequence::iterator vu_iter = verticesU->begin(), vk_iter = verticesK->begin(); vu_iter != verticesU->end(); ++vu_iter, ++vk_iter) {
    PETSC_MESH_TYPE::point_type up = *vu_iter;
    PETSC_MESH_TYPE::point_type kp = *vk_iter;
    const PetscInt marker = meshU->getValue(meshU->getLabel("marker"), *vu_iter, 0);
    PetscScalar    values[3];

    if (marker == 1) {
      ulp = up;
      urp = *(++vur_iter);
      klp = kp;
      continue;
    }
    if (marker == 2) continue;
    PetscInt     cols[3] = {klp, kp};
    PetscScalar *u, *ul, *ur;

    ierr = SectionRealRestrict(sectionU, up,  &u);CHKERRQ(ierr);
    ierr = SectionRealRestrict(sectionU, ulp, &ul);CHKERRQ(ierr);
    ierr = SectionRealRestrict(sectionU, urp, &ur);CHKERRQ(ierr);
    values[0] = (u[0]-ul[0])/hx;
    values[1] = (u[0]-ur[0])/hx;
    ierr = MatSetValuesTopology(Buk, dmu, 1, &up, dmk, 2, cols, values, INSERT_VALUES);CHKERRQ(ierr);
    ulp  = up;
    urp  = *(++vur_iter);
    klp  = kp;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal_KU"
static PetscErrorCode FormJacobianLocal_KU(DM dmu, DM dmk, SectionReal sectionU, SectionReal sectionK, Mat Bku, User user)
{
  ALE::Obj<PETSC_MESH_TYPE> meshU;
  ALE::Obj<PETSC_MESH_TYPE> meshK;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!Bku) PetscFunctionReturn(0); /* Not assembling this block */
  ierr = DMMeshGetMesh(dmu, meshU);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dmk, meshK);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& verticesU = meshU->depthStratum(0);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& verticesK = meshK->depthStratum(0);
  PETSC_MESH_TYPE::label_sequence::iterator vur_iter = verticesU->begin();
  PETSC_MESH_TYPE::label_sequence::iterator vkr_iter = verticesK->begin();
  PETSC_MESH_TYPE::point_type urp = *(++vur_iter);
  PetscReal hx = user->hxk;

  for(PETSC_MESH_TYPE::label_sequence::iterator vk_iter = verticesK->begin(), vu_iter = verticesU->begin(); vk_iter != verticesK->end(); ++vk_iter,  ++vu_iter) {
    PETSC_MESH_TYPE::point_type up      = *vu_iter;
    PETSC_MESH_TYPE::point_type kp      = *vk_iter;
    PetscInt                    cols[2] = {up, urp};
    PetscScalar                 values[2];
    PetscScalar                *u, *ur;

    ierr = SectionRealRestrict(sectionU, up,  &u);CHKERRQ(ierr);
    ierr = SectionRealRestrict(sectionU, urp, &ur);CHKERRQ(ierr);
    const PetscScalar
      ubar     = 0.5*(u[0]+ur[0]),
      ubar_L   = 0.5,
      ubar_R   = 0.5,
      gradu    = (ur[0]-u[0])/hx,
      gradu_L  = -1.0/hx,
      gradu_R  = 1.0/hx,
      g        = 1.0 + PetscSqr(gradu),
      g_gradu  = 2.0*gradu,
      w        = 1.0/(1.0+ubar) + 1.0/g,
      w_ubar   = -1./PetscSqr(1.+ubar),
      w_gradu  = -g_gradu/PetscSqr(g),
      iw       = 1.0/w,
      iw_ubar  = -w_ubar * PetscSqr(iw),
      iw_gradu = -w_gradu * PetscSqr(iw);

    values[0]  = -hx*(iw_ubar*ubar_L + iw_gradu*gradu_L);
    values[1]  = -hx*(iw_ubar*ubar_R + iw_gradu*gradu_R);
    ierr = MatSetValuesTopology(Bku, dmk, 1, &kp, dmu, 2, cols, values, INSERT_VALUES);CHKERRQ(ierr);
    urp  = *(++vur_iter);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobian_All"
static PetscErrorCode FormJacobian_All(SNES snes, Vec X, Mat *J, Mat *B, MatStructure *mstr, void *ctx)
{
  User           user = (User) ctx;
  DM             dmu, dmk;
  Vec            Uloc, Kloc, vecU, vecK;
  SectionReal    sectionU, sectionK;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCompositeGetEntries(user->pack, &dmu, &dmk);CHKERRQ(ierr);
  switch (user->ptype) {
  case 0:
    ierr = DMMeshGetSectionReal(dmu, "default", &sectionU);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionU, &vecU);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dmu, X, INSERT_VALUES, vecU);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmu, X, INSERT_VALUES, vecU);CHKERRQ(ierr);
    ierr = VecDestroy(&vecU);CHKERRQ(ierr);

    ierr = DMMeshGetSectionReal(dmk, "default", &sectionK);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionK, &vecK);CHKERRQ(ierr);
    ierr = VecCopy(user->Kloc, vecK);CHKERRQ(ierr);
    ierr = VecDestroy(&vecK);CHKERRQ(ierr);

    ierr = FormJacobianLocal_U(dmu, dmk, sectionU, sectionK, *B, user);CHKERRQ(ierr);
    ierr = SectionRealDestroy(&sectionU);CHKERRQ(ierr);
    ierr = SectionRealDestroy(&sectionK);CHKERRQ(ierr);
    break;
  case 1:
    ierr = DMMeshGetSectionReal(dmk, "default", &sectionK);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionK, &vecK);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dmk, X, INSERT_VALUES, vecK);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmk, X, INSERT_VALUES, vecK);CHKERRQ(ierr);
    ierr = VecDestroy(&vecK);CHKERRQ(ierr);

    ierr = DMMeshGetSectionReal(dmu, "default", &sectionU);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionU, &vecU);CHKERRQ(ierr);
    ierr = VecCopy(user->Uloc, vecU);CHKERRQ(ierr);
    ierr = VecDestroy(&vecU);CHKERRQ(ierr);

    ierr = FormJacobianLocal_K(dmu, dmk, sectionU, sectionK, *B, user);CHKERRQ(ierr);
    ierr = SectionRealDestroy(&sectionU);CHKERRQ(ierr);
    ierr = SectionRealDestroy(&sectionK);CHKERRQ(ierr);
    break;
  case 2: {
    Mat Buu,Buk,Bku,Bkk;
    IS *is;

    ierr = DMCompositeGetLocalVectors(user->pack, &Uloc, &Kloc);CHKERRQ(ierr);
    ierr = DMCompositeScatter(user->pack, X, Uloc, Kloc);CHKERRQ(ierr);

    ierr = DMMeshGetSectionReal(dmu, "default", &sectionU);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionU, &vecU);CHKERRQ(ierr);
    ierr = VecCopy(Uloc, vecU);CHKERRQ(ierr);
    ierr = VecDestroy(&vecU);CHKERRQ(ierr);

    ierr = DMMeshGetSectionReal(dmk, "default", &sectionK);CHKERRQ(ierr);
    ierr = SectionRealCreateLocalVector(sectionK, &vecK);CHKERRQ(ierr);
    ierr = VecCopy(Kloc, vecK);CHKERRQ(ierr);
    ierr = VecDestroy(&vecK);CHKERRQ(ierr);

    ierr = DMCompositeRestoreLocalVectors(user->pack, &Uloc, &Kloc);CHKERRQ(ierr);

    ierr = DMCompositeGetLocalISs(user->pack, &is);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(*B, is[0], is[0], &Buu);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(*B, is[0], is[1], &Buk);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(*B, is[1], is[0], &Bku);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(*B, is[1], is[1], &Bkk);CHKERRQ(ierr);
    ierr = FormJacobianLocal_U (dmu, dmk, sectionU, sectionK, Buu, user);CHKERRQ(ierr);
    ierr = FormJacobianLocal_UK(dmu, dmk, sectionU, sectionK, Buk, user);CHKERRQ(ierr);
    ierr = FormJacobianLocal_KU(dmu, dmk, sectionU, sectionK, Bku, user);CHKERRQ(ierr);
    ierr = FormJacobianLocal_K (dmu, dmk, sectionU, sectionK, Bkk, user);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(*B, is[0], is[0], &Buu);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(*B, is[0], is[1], &Buk);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(*B, is[1], is[0], &Bku);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(*B, is[1], is[1], &Bkk);CHKERRQ(ierr);

    ierr = SectionRealDestroy(&sectionU);CHKERRQ(ierr);
    ierr = SectionRealDestroy(&sectionK);CHKERRQ(ierr);
    ierr = ISDestroy(&is[0]);CHKERRQ(ierr);
    ierr = ISDestroy(&is[1]);CHKERRQ(ierr);
    ierr = PetscFree(is);CHKERRQ(ierr);
  } break;
  }
  ierr = MatAssemblyBegin(*B, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (*B, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != *B) {
    ierr = MatAssemblyBegin(*J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (*J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  //ierr = MatView(*B, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
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
  ierr = VecDestroy(&vecU);CHKERRQ(ierr);
  ierr = VecDestroy(&vecK);CHKERRQ(ierr);
  ierr = SectionRealDestroy(&coordinatesU);CHKERRQ(ierr);
  ierr = SectionRealDestroy(&coordinatesK);CHKERRQ(ierr);
  ierr = SectionRealDestroy(&sectionU);CHKERRQ(ierr);
  ierr = SectionRealDestroy(&sectionK);CHKERRQ(ierr);
  ierr = DMCompositeGather(user->pack, X, INSERT_VALUES, user->Uloc, user->Kloc);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  /* Create meshes */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,-10,1,1,PETSC_NULL,&dau);CHKERRQ(ierr);
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
  ierr = DMSetOptionsPrefix(dmu, "u_");CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(dmk, "k_");CHKERRQ(ierr);
  ierr = DMDestroy(&dau);CHKERRQ(ierr);
  ierr = DMDestroy(&dak);CHKERRQ(ierr);

#if 1
  {
    ALE::Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dmu, m);CHKERRQ(ierr);
    m->view("Mesh");
    m->getLabel("marker")->view("Marker");
  }
#endif

  ierr = PetscNew(struct _UserCtx, &user);CHKERRQ(ierr);
  user->hxu = 1.0/m;
  user->hxk = 1.0/(m-1);
  /* Setup dof layout.
   For a DMDA, this is automatic given the number of dof at each vertex. However, for a DMMesh, we need to specify this.
  */
  {
    /* There is perhaps a better way to do this that does not rely on the Discretization/BoundaryCondition objects in Mesh.hh */
    ierr = CreateProblem_gen_0(dmu, "u");CHKERRQ(ierr);
    ierr = CreateProblem_gen_0(dmk, "k");CHKERRQ(ierr);
  }
  SectionReal defaultSection;

  ierr = DMMeshGetSectionReal(dmu, "default", &defaultSection);CHKERRQ(ierr);
  ierr = DMMeshSetupSection(dmu, defaultSection);CHKERRQ(ierr);
  ierr = SectionRealDestroy(&defaultSection);CHKERRQ(ierr);
  ierr = DMMeshGetSectionReal(dmk, "default", &defaultSection);CHKERRQ(ierr);
  ierr = DMMeshSetupSection(dmk, defaultSection);CHKERRQ(ierr);
  ierr = SectionRealDestroy(&defaultSection);CHKERRQ(ierr);

  ierr = DMCompositeCreate(PETSC_COMM_WORLD, &pack);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(pack, dmu);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(pack, dmk);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(pack, "pack_");CHKERRQ(ierr);

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
  ierr = VecView(X, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  switch (user->ptype) {
  case 0:
    ierr = DMCompositeGetAccess(pack,X,&Xu,0);CHKERRQ(ierr);
    ierr = DMCompositeGetAccess(pack,F,&Fu,0);CHKERRQ(ierr);
    ierr = DMCreateMatrix(dmu,PETSC_NULL,&B);CHKERRQ(ierr);
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
    ierr = DMCreateMatrix(dmk,PETSC_NULL,&B);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,Fk,FormFunction_All,user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,B,B,FormJacobian_All,user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    ierr = SNESSolve(snes,PETSC_NULL,Xk);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(pack,X,0,&Xk);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccess(pack,F,0,&Fk);CHKERRQ(ierr);
    break;
  case 2:
    ierr = DMCreateMatrix(pack,PETSC_NULL,&B);CHKERRQ(ierr);
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
  ierr = DMDestroy(&dmu);CHKERRQ(ierr);
  ierr = DMDestroy(&dmk);CHKERRQ(ierr);
  ierr = DMDestroy(&pack);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFree(user);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
