static char help[] = "Example program demonstrating projection between particle and finite element spaces\n\n";

#include "petscdmplex.h"
#include "petscds.h"
#include "petscdmswarm.h"
#include "petscksp.h"

int main(int argc, char **argv)
{
  DM              dm, sw;
  PetscFE         fe;
  KSP             ksp;
  PC              pc;
  Mat             M_p, PM_p=NULL;
  Vec             f, rho, rhs;
  PetscInt        dim, Nc = 1, timestep = 0, i, faces[3];
  PetscInt        Np = 10, p, field = 0, zero = 0, bs;
  PetscReal       time = 0.0,  norm, energy_0, energy_1;
  PetscReal       lo[3], hi[3], h[3];
  PetscBool       removePoints = PETSC_TRUE;
  PetscReal       *wq, *coords;
  PetscDataType   dtype;
  PetscErrorCode  ierr;
  PetscBool       is_bjac;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  /* Create a mesh */
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  i    = dim;
  ierr = PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &i, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-np", &Np, NULL);CHKERRQ(ierr);
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
  for (p=0,energy_0=0;p<Np;p++) {
    coords[p*2+0]  = -PetscCosReal((PetscReal)(p+1)/(PetscReal)(Np+1) * PETSC_PI);
    coords[p*2+1] =   PetscSinReal((PetscReal)(p+1)/(PetscReal)(Np+1) * PETSC_PI);
    wq[p]          = 1.0;
    energy_0 += wq[p]*(PetscSqr(coords[p*2+0])+PetscSqr(coords[p*2+1]));
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

  /* Project field to particles */
  /*   This gives f_p = M_p^+ M f */
  ierr = DMCreateGlobalVector(dm, &rhs);CHKERRQ(ierr);
  ierr = VecCopy(rho, rhs);CHKERRQ(ierr); /* Identity: M^1 M rho */

  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp, "ftop_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&is_bjac);
  if (is_bjac) {
    ierr = DMSwarmCreateMassMatrixSquare(sw, dm, &PM_p);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, M_p, PM_p);CHKERRQ(ierr);
  } else {
    ierr = KSPSetOperators(ksp, M_p, M_p);CHKERRQ(ierr);
  }
  ierr = KSPSolveTranspose(ksp, rhs, f);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr);

  /* Visualize particle field */
  ierr = DMSetOutputSequenceNumber(sw, timestep, time);CHKERRQ(ierr);
  ierr = VecViewFromOptions(f, NULL, "-weights_view");CHKERRQ(ierr);
  ierr = VecNorm(f,NORM_1,&norm);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f);CHKERRQ(ierr);

  /* compute energy */
  ierr = DMSwarmGetField(sw, "w_q", &bs, &dtype, (void**)&wq);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords);CHKERRQ(ierr);
  for (p=0,energy_1=0;p<Np;p++) {
    energy_1 += wq[p]*(PetscSqr(coords[p*2+0])+PetscSqr(coords[p*2+1]));
  }
  ierr = DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void**)&wq);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Total number = %20.12e. energy = %20.12e error = %20.12e\n", norm, energy_0, (energy_1-energy_0)/energy_0);CHKERRQ(ierr);
  /* Cleanup */
  ierr = MatDestroy(&M_p);CHKERRQ(ierr);
  ierr = MatDestroy(&PM_p);CHKERRQ(ierr);
  ierr = VecDestroy(&rho);CHKERRQ(ierr);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: !complex

  test:
    suffix: 0
    requires: double triangle
    args: -dm_plex_simplex 0 -dm_plex_box_faces 4,2 -np 50 -dm_plex_box_lower -2.0,0.0 -dm_plex_box_upper 2.0,2.0 -petscspace_degree 2 -ftop_ksp_type lsqr -ftop_pc_type none -dm_view -swarm_view -ftop_ksp_rtol 1.e-14
    filter: grep -v DM_ | grep -v atomic

  test:
    suffix: bjacobi
    requires: double triangle
    args: -dm_plex_simplex 0 -dm_plex_box_faces 4,2 -np 50 -dm_plex_box_lower -2.0,0.0 -dm_plex_box_upper 2.0,2.0 -petscspace_degree 2 -dm_plex_hash_location -ftop_ksp_type lsqr -ftop_pc_type bjacobi -ftop_sub_pc_type lu -ftop_sub_pc_factor_shift_type nonzero -dm_view -swarm_view -ftop_ksp_rtol 1.e-14
    filter: grep -v DM_ | grep -v atomic

TEST*/
