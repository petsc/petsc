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
  PetscBool       is_bjac;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL,help));
  /* Create a mesh */
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));

  CHKERRQ(DMGetDimension(dm, &dim));
  i    = dim;
  CHKERRQ(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &i, NULL));
  CHKERRQ(PetscOptionsGetInt(NULL, NULL, "-np", &Np, NULL));
  CHKERRQ(DMGetBoundingBox(dm, lo, hi));
  for (i=0;i<dim;i++) {
    h[i] = (hi[i] - lo[i])/faces[i];
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF," lo = %g hi = %g n = %D h = %g\n",lo[i],hi[i],faces[i],h[i]));
  }

  CHKERRQ(PetscFECreateDefault(PETSC_COMM_SELF, dim, Nc, PETSC_FALSE, "", PETSC_DECIDE, &fe));
  CHKERRQ(PetscFESetFromOptions(fe));
  CHKERRQ(PetscObjectSetName((PetscObject)fe, "fe"));
  CHKERRQ(DMSetField(dm, field, NULL, (PetscObject)fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(PetscFEDestroy(&fe));
  /* Create particle swarm */
  CHKERRQ(DMCreate(PETSC_COMM_SELF, &sw));
  CHKERRQ(DMSetType(sw, DMSWARM));
  CHKERRQ(DMSetDimension(sw, dim));
  CHKERRQ(DMSwarmSetType(sw, DMSWARM_PIC));
  CHKERRQ(DMSwarmSetCellDM(sw, dm));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(sw, "w_q", Nc, PETSC_SCALAR));
  CHKERRQ(DMSwarmFinalizeFieldRegister(sw));
  CHKERRQ(DMSwarmSetLocalSizes(sw, Np, zero));
  CHKERRQ(DMSetFromOptions(sw));
  CHKERRQ(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void**)&wq));
  CHKERRQ(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
  for (p=0,energy_0=0;p<Np;p++) {
    coords[p*2+0]  = -PetscCosReal((PetscReal)(p+1)/(PetscReal)(Np+1) * PETSC_PI);
    coords[p*2+1] =   PetscSinReal((PetscReal)(p+1)/(PetscReal)(Np+1) * PETSC_PI);
    wq[p]          = 1.0;
    energy_0 += wq[p]*(PetscSqr(coords[p*2+0])+PetscSqr(coords[p*2+1]));
  }
  CHKERRQ(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
  CHKERRQ(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void**)&wq));
  CHKERRQ(DMSwarmMigrate(sw, removePoints));
  CHKERRQ(PetscObjectSetName((PetscObject)sw, "Particle Grid"));
  CHKERRQ(DMViewFromOptions(sw, NULL, "-swarm_view"));

  /* Project particles to field */
  /* This gives M f = \int_\Omega \phi f, which looks like a rhs for a PDE */
  CHKERRQ(DMCreateMassMatrix(sw, dm, &M_p));
  CHKERRQ(DMCreateGlobalVector(dm, &rho));
  CHKERRQ(PetscObjectSetName((PetscObject)rho, "rho"));

  CHKERRQ(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));
  CHKERRQ(PetscObjectSetName((PetscObject)f, "weights"));
  CHKERRQ(MatMultTranspose(M_p, f, rho));

  /* Visualize mesh field */
  CHKERRQ(DMSetOutputSequenceNumber(dm, timestep, time));
  CHKERRQ(VecViewFromOptions(rho, NULL, "-rho_view"));

  /* Project field to particles */
  /*   This gives f_p = M_p^+ M f */
  CHKERRQ(DMCreateGlobalVector(dm, &rhs));
  CHKERRQ(VecCopy(rho, rhs)); /* Identity: M^1 M rho */

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &ksp));
  CHKERRQ(KSPSetOptionsPrefix(ksp, "ftop_"));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&is_bjac));
  if (is_bjac) {
    CHKERRQ(DMSwarmCreateMassMatrixSquare(sw, dm, &PM_p));
    CHKERRQ(KSPSetOperators(ksp, M_p, PM_p));
  } else {
    CHKERRQ(KSPSetOperators(ksp, M_p, M_p));
  }
  CHKERRQ(KSPSolveTranspose(ksp, rhs, f));
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&rhs));

  /* Visualize particle field */
  CHKERRQ(DMSetOutputSequenceNumber(sw, timestep, time));
  CHKERRQ(VecViewFromOptions(f, NULL, "-weights_view"));
  CHKERRQ(VecNorm(f,NORM_1,&norm));
  CHKERRQ(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));

  /* compute energy */
  CHKERRQ(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void**)&wq));
  CHKERRQ(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
  for (p=0,energy_1=0;p<Np;p++) {
    energy_1 += wq[p]*(PetscSqr(coords[p*2+0])+PetscSqr(coords[p*2+1]));
  }
  CHKERRQ(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
  CHKERRQ(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void**)&wq));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Total number = %20.12e. energy = %20.12e error = %20.12e\n", norm, energy_0, (energy_1-energy_0)/energy_0));
  /* Cleanup */
  CHKERRQ(MatDestroy(&M_p));
  CHKERRQ(MatDestroy(&PM_p));
  CHKERRQ(VecDestroy(&rho));
  CHKERRQ(DMDestroy(&sw));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
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
