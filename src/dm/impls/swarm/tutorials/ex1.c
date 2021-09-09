static char help[] = "Example program demonstrating projection between particle and finite element spaces\n\n";

#include "petscdmplex.h"
#include "petscds.h"
#include "petscdmswarm.h"
#include "petscksp.h"

static PetscErrorCode crd_func(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf_dummy, PetscScalar *u, void *actx)
{
  PetscInt i;
  PetscFunctionBeginUser;
  for (i = 0; i < dim; ++i) u[i] = x[i];
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM              dm, crddm, sw;
  PetscFE         fe;
  KSP             ksp;
  Mat             M_p, M;
  Vec             f, rho, rhs, crd_vec;
  PetscInt        dim, Nc = 1, timestep = 0, N, i, idx[3], faces[3];
  PetscInt        Np = 10, p, field = 0, zero = 0, bs;
  PetscReal       time = 0.0,  norm;
  PetscReal       lo[3], hi[3], h[3];
  PetscBool       removePoints = PETSC_TRUE;
  const PetscReal *xx, *vv;
  PetscReal       *wq, *coords;
  PetscDataType   dtype;
  PetscErrorCode  (*initu[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar [], void *);
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  /* Create a mesh */
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  i    = dim;
  ierr = PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &i, NULL);CHKERRQ(ierr);
  ierr = DMGetBoundingBox(dm, lo, hi);CHKERRQ(ierr);
  for (i=0;i<dim;i++) {
    h[i] = (hi[i] - lo[i])/faces[i];
    ierr = PetscPrintf(PETSC_COMM_SELF," lo = %g hi = %g n = %D h = %g\n",lo[i],hi[i],faces[i],h[i]);CHKERRQ(ierr);
  }

  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, Nc, PETSC_FALSE, "", PETSC_DECIDE, &fe);CHKERRQ(ierr);
  ierr = PetscFESetFromOptions(fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fe, "fe");CHKERRQ(ierr);
  ierr = DMSetField(dm, field, NULL, (PetscObject)fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  /* Create particle swarm */
  ierr = DMCreate(PETSC_COMM_SELF, &sw);CHKERRQ(ierr);
  ierr = DMSetType(sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(sw, dim);CHKERRQ(ierr);
  ierr = DMSwarmSetType(sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(sw, "w_q", Nc, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(sw);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(sw, Np, zero);CHKERRQ(ierr);
  ierr = DMSetFromOptions(sw);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, "w_q", &bs, &dtype, (void**)&wq);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords);CHKERRQ(ierr);
  for (p=0;p<Np;p++) {
    coords[p*2+0]  = -PetscCosReal((PetscReal)(p+1)/(PetscReal)(Np+1) * PETSC_PI);
    coords[p*2+1] =   PetscSinReal((PetscReal)(p+1)/(PetscReal)(Np+1) * PETSC_PI);
    wq[p]          = 1.0;
  }
  ierr = DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void**)&wq);CHKERRQ(ierr);
  ierr = DMSwarmMigrate(sw, removePoints);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sw, "Particle Grid");CHKERRQ(ierr);
  ierr = DMViewFromOptions(sw, NULL, "-swarm_view");CHKERRQ(ierr);
  /* Project particles to field */
  /* This gives M f = \int_\Omega \phi f, which looks like a rhs for a PDE */
  ierr = DMCreateMassMatrix(sw, dm, &M_p);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &rho);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)rho, "rho");CHKERRQ(ierr);
  ierr = DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)f, "weights");CHKERRQ(ierr);
  ierr = MatMultTranspose(M_p, f, rho);CHKERRQ(ierr);

  /* Visualize mesh field */
  ierr = DMSetOutputSequenceNumber(dm, timestep, time);CHKERRQ(ierr);
  ierr = VecViewFromOptions(rho, NULL, "-rho_view");CHKERRQ(ierr);

  /* create coordinate DM */
  ierr = DMClone(dm, &crddm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, dim, PETSC_FALSE, "", PETSC_DECIDE, &fe);CHKERRQ(ierr);
  ierr = PetscFESetFromOptions(fe);CHKERRQ(ierr);
  ierr = DMSetField(crddm, field, NULL, (PetscObject)fe);CHKERRQ(ierr);
  ierr = DMCreateDS(crddm);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  /* project coordinates to vertices */
  ierr = DMCreateGlobalVector(crddm, &crd_vec);CHKERRQ(ierr);
  initu[0] = crd_func;
  ierr = DMProjectFunction(crddm, 0.0, initu, NULL, INSERT_ALL_VALUES, crd_vec);CHKERRQ(ierr);
  ierr = VecViewFromOptions(crd_vec, NULL, "-coord_view");CHKERRQ(ierr);
  /* iterate over mesh data and get indices */
  ierr = VecGetArrayRead(crd_vec,&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(rho,&vv);CHKERRQ(ierr);
  ierr = VecGetLocalSize(rho,&N);CHKERRQ(ierr);
  for (p=0;p<N;p++) {
    for (i=0;i<dim;i++) idx[i] = (PetscInt)((xx[p*dim+i] - lo[i])/h[i] + 1.e-8);
    ierr = PetscPrintf(PETSC_COMM_SELF,"(%D,%D) = %g\n",idx[0],idx[1],vv[p]);CHKERRQ(ierr);
    /* access grid data here */
  }
  ierr = VecRestoreArrayRead(crd_vec,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(rho,&vv);CHKERRQ(ierr);
  ierr = VecDestroy(&crd_vec);CHKERRQ(ierr);
  /* Project field to particles */
  /*   This gives f_p = M_p^+ M f */
  ierr = DMCreateMassMatrix(dm, dm, &M);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &rhs);CHKERRQ(ierr);
  if (0) {
    ierr = MatMult(M, rho, rhs);CHKERRQ(ierr);  /* this is what you would do for an FE solve */
  } else {
    ierr = VecCopy(rho, rhs);CHKERRQ(ierr); /* Identity: M^1 M rho */
  }
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp, "ftop_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, M_p, M_p);CHKERRQ(ierr);
  ierr = KSPSolveTranspose(ksp, rhs, f);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr);

  /* Visualize particle field */
  ierr = DMSetOutputSequenceNumber(sw, timestep, time);CHKERRQ(ierr);
  ierr = VecViewFromOptions(f, NULL, "-weights_view");CHKERRQ(ierr);
  ierr = VecNorm(f,NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Total number density = %g\n", norm);CHKERRQ(ierr);
  /* Cleanup */
  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&M_p);CHKERRQ(ierr);
  ierr = VecDestroy(&rho);CHKERRQ(ierr);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&crddm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: !complex

  test:
    suffix: 0
    requires: double
    args: -dm_plex_simplex 0 -dm_plex_box_faces 4,2 -dm_plex_box_lower -2.0,0.0 -dm_plex_box_upper 2.0,2.0 -petscspace_degree 2 -ftop_ksp_type lsqr -ftop_pc_type none -dm_view
    filter: grep -v DM_ | grep -v atomic

TEST*/
