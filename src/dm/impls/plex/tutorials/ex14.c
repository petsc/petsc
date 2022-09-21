const char help[] = "Set up a PetscSF for halo exchange between local vectors";

#include <petscdmplex.h>
#include <petscsf.h>

int main(int argc, char **argv)
{
  DM           dm;
  Vec          u;
  PetscSection section;
  PetscInt     dim, numFields, numBC, i;
  PetscMPIInt  rank;
  PetscInt     numComp[2];
  PetscInt     numDof[12];
  PetscInt    *remoteOffsets;
  PetscSF      pointSF;
  PetscSF      sectionSF;
  PetscScalar *array;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /* Create a mesh */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMGetDimension(dm, &dim));

  /** Describe the solution variables that are discretized on the mesh */
  /* Create scalar field u and a vector field v */
  numFields  = 2;
  numComp[0] = 1;
  numComp[1] = dim;
  for (i = 0; i < numFields * (dim + 1); ++i) numDof[i] = 0;
  /* Let u be defined on cells */
  numDof[0 * (dim + 1) + dim] = 1;
  /* Let v be defined on vertices */
  numDof[1 * (dim + 1) + 0] = dim;
  /* No boundary conditions */
  numBC = 0;

  /** Create a PetscSection to handle the layout of the discretized variables */
  PetscCall(DMSetNumFields(dm, numFields));
  PetscCall(DMPlexCreateSection(dm, NULL, numComp, numDof, numBC, NULL, NULL, NULL, NULL, &section));
  /* Name the Field variables */
  PetscCall(PetscSectionSetFieldName(section, 0, "u"));
  PetscCall(PetscSectionSetFieldName(section, 1, "v"));
  /* Tell the DM to use this data layout */
  PetscCall(DMSetLocalSection(dm, section));

  /** Construct the communication pattern for halo exchange between local vectors */
  /* Get the point SF: an object that says which copies of mesh points (cells,
   * vertices, faces, edges) are copies of points on other processes */
  PetscCall(DMGetPointSF(dm, &pointSF));
  /* Relate the locations of ghost degrees of freedom on this process
   * to their locations of the non-ghost copies on a different process */
  PetscCall(PetscSFCreateRemoteOffsets(pointSF, section, section, &remoteOffsets));
  /* Use that information to construct a star forest for halo exchange
   * for data described by the local section */
  PetscCall(PetscSFCreateSectionSF(pointSF, section, remoteOffsets, section, &sectionSF));
  PetscCall(PetscFree(remoteOffsets));

  /** Demo of halo exchange */
  /* Create a Vec with this layout */
  PetscCall(DMCreateLocalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "Local vector"));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  /* Set all mesh values to the MPI rank */
  PetscCall(VecSet(u, (PetscScalar)rank));
  /* Get the raw array of values */
  PetscCall(VecGetArrayWrite(u, &array));
  /*** HALO EXCHANGE ***/
  PetscCall(PetscSFBcastBegin(sectionSF, MPIU_SCALAR, array, array, MPI_REPLACE));
  /* local work can be done between Begin() and End() */
  PetscCall(PetscSFBcastEnd(sectionSF, MPIU_SCALAR, array, array, MPI_REPLACE));
  /* Restore the raw array of values */
  PetscCall(VecRestoreArrayWrite(u, &array));
  /* View the results: should show which process has the non-ghost copy of each degree of freedom */
  PetscCall(PetscSectionVecView(section, u, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&u));

  /** Cleanup */
  PetscCall(PetscSFDestroy(&sectionSF));
  PetscCall(PetscSectionDestroy(&section));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Test on a 1D mesh with overlap
  test:
    nsize: 3
    requires: !complex
    args: -dm_plex_dim 1 -dm_plex_box_faces 3 -dm_refine_pre 1 -petscpartitioner_type simple -dm_distribute_overlap 1

TEST*/
