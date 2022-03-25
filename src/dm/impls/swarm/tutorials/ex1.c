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

  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  /* Create a mesh */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(DMGetDimension(dm, &dim));
  i    = dim;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &i, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-np", &Np, NULL));
  PetscCall(DMGetBoundingBox(dm, lo, hi));
  for (i=0;i<dim;i++) {
    h[i] = (hi[i] - lo[i])/faces[i];
    PetscCall(PetscPrintf(PETSC_COMM_SELF," lo = %g hi = %g n = %D h = %g\n",lo[i],hi[i],faces[i],h[i]));
  }

  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, Nc, PETSC_FALSE, "", PETSC_DECIDE, &fe));
  PetscCall(PetscFESetFromOptions(fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "fe"));
  PetscCall(DMSetField(dm, field, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(PetscFEDestroy(&fe));
  /* Create particle swarm */
  PetscCall(DMCreate(PETSC_COMM_SELF, &sw));
  PetscCall(DMSetType(sw, DMSWARM));
  PetscCall(DMSetDimension(sw, dim));
  PetscCall(DMSwarmSetType(sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(sw, "w_q", Nc, PETSC_SCALAR));
  PetscCall(DMSwarmFinalizeFieldRegister(sw));
  PetscCall(DMSwarmSetLocalSizes(sw, Np, zero));
  PetscCall(DMSetFromOptions(sw));
  PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void**)&wq));
  PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
  for (p=0,energy_0=0;p<Np;p++) {
    coords[p*2+0]  = -PetscCosReal((PetscReal)(p+1)/(PetscReal)(Np+1) * PETSC_PI);
    coords[p*2+1] =   PetscSinReal((PetscReal)(p+1)/(PetscReal)(Np+1) * PETSC_PI);
    wq[p]          = 1.0;
    energy_0 += wq[p]*(PetscSqr(coords[p*2+0])+PetscSqr(coords[p*2+1]));
  }
  PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
  PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void**)&wq));
  PetscCall(DMSwarmMigrate(sw, removePoints));
  PetscCall(PetscObjectSetName((PetscObject)sw, "Particle Grid"));
  PetscCall(DMViewFromOptions(sw, NULL, "-swarm_view"));

  /* Project particles to field */
  /* This gives M f = \int_\Omega \phi f, which looks like a rhs for a PDE */
  PetscCall(DMCreateMassMatrix(sw, dm, &M_p));
  PetscCall(DMCreateGlobalVector(dm, &rho));
  PetscCall(PetscObjectSetName((PetscObject)rho, "rho"));

  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(PetscObjectSetName((PetscObject)f, "weights"));
  PetscCall(MatMultTranspose(M_p, f, rho));

  /* Visualize mesh field */
  PetscCall(DMSetOutputSequenceNumber(dm, timestep, time));
  PetscCall(VecViewFromOptions(rho, NULL, "-rho_view"));

  /* Project field to particles */
  /*   This gives f_p = M_p^+ M f */
  PetscCall(DMCreateGlobalVector(dm, &rhs));
  PetscCall(VecCopy(rho, rhs)); /* Identity: M^1 M rho */

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "ftop_"));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&is_bjac));
  if (is_bjac) {
    PetscCall(DMSwarmCreateMassMatrixSquare(sw, dm, &PM_p));
    PetscCall(KSPSetOperators(ksp, M_p, PM_p));
  } else {
    PetscCall(KSPSetOperators(ksp, M_p, M_p));
  }
  PetscCall(KSPSolveTranspose(ksp, rhs, f));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&rhs));

  /* Visualize particle field */
  PetscCall(DMSetOutputSequenceNumber(sw, timestep, time));
  PetscCall(VecViewFromOptions(f, NULL, "-weights_view"));
  PetscCall(VecNorm(f,NORM_1,&norm));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));

  /* compute energy */
  PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void**)&wq));
  PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
  for (p=0,energy_1=0;p<Np;p++) {
    energy_1 += wq[p]*(PetscSqr(coords[p*2+0])+PetscSqr(coords[p*2+1]));
  }
  PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
  PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void**)&wq));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Total number = %20.12e. energy = %20.12e error = %20.12e\n", norm, energy_0, (energy_1-energy_0)/energy_0));
  /* Cleanup */
  PetscCall(MatDestroy(&M_p));
  PetscCall(MatDestroy(&PM_p));
  PetscCall(VecDestroy(&rho));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
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
