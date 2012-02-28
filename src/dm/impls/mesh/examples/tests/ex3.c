static char help[] = "Mesh Distribution with SF.\n\n";
#include <petscdmmesh.h>
#include <petscsf.h>

typedef struct {
  DM            dm;                /* REQUIRED in order to use SNES evaluation functions */
  PetscInt      debug;             /* The debugging level */
  PetscMPIInt   rank;              /* The process rank */
  PetscMPIInt   numProcs;          /* The number of processes */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscBool     interpolate;       /* Generate intermediate mesh elements */
  PetscReal     refinementLimit;   /* The largest allowable cell volume */
  char          filename[2048];    /* Optional filename to read mesh from */
  char          partitioner[2048]; /* The graph partitioner */
  PetscLogEvent createMeshEvent;
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;
  options->interpolate     = PETSC_FALSE;
  options->refinementLimit = 0.0;

  ierr = MPI_Comm_size(comm, &options->numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Mesh Distribution Options", "DMMESH");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex1.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex1.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex1.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex1.c", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscStrcpy(options->filename, "");CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The input filename", "ex1.c", options->filename, options->filename, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscStrcpy(options->partitioner, "chaco");CHKERRQ(ierr);
  ierr = PetscOptionsString("-partitioner", "The graph partitioner", "ex1.c", options->partitioner, options->partitioner, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh",    DM_CLASSID,   &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim             = user->dim;
  PetscBool      interpolate     = user->interpolate;
  PetscReal      refinementLimit = user->refinementLimit;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = DMMeshCreateBoxMesh(comm, dim, interpolate, dm);CHKERRQ(ierr);
  {
    DM refinedMesh = PETSC_NULL;

    /* Refine mesh using a volume constraint */
    ierr = DMMeshRefine(*dm, refinementLimit, interpolate, &refinedMesh);CHKERRQ(ierr);
    if (refinedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = refinedMesh;
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Serial Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DistributeMesh"
/* Distribute cones
   - Partitioning:         input partition point map and naive sf, output sf with inverse of map, distribute points
   - Distribute section:   input current sf, communicate sizes and offsets, output local section and offsets (only use for new sf)
   - Create SF for values: input current sf and offsets, output new sf
   - Distribute values:    input new sf, communicate values
 */
PetscErrorCode DistributeMesh(DM dm, AppCtx *user, PetscSF *pointSF, DM *parallelDM)
{
  MPI_Comm       comm   = ((PetscObject) dm)->comm;
  const PetscInt height = 0;
  PetscInt       dim, numRemoteRanks;
  IS             cellPart,        part;
  PetscSection   cellPartSection, partSection;
  PetscSFNode   *remoteRanks;
  PetscSF        partSF;
  ISLocalToGlobalMapping renumbering;
  PetscSF        coneSF;
  PetscSection   originalConeSection, newConeSection;
  PetscInt      *remoteOffsets, newConesSize;
  PetscInt      *cones, *newCones;
  PetscMPIInt    numProcs, rank, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMMeshGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Create cell partition - We need to rewrite to use IS, use the MatPartition stuff */
  ierr = DMMeshCreatePartition(dm, &cellPartSection, &cellPart, height);CHKERRQ(ierr);
  /* Create SF assuming a serial partition for all processes: Could check for IS length here */
  if (!rank) {
    numRemoteRanks = numProcs;
  } else {
    numRemoteRanks = 0;
  }
  ierr = PetscMalloc(numRemoteRanks * sizeof(PetscSFNode), &remoteRanks);CHKERRQ(ierr);
  for(p = 0; p < numRemoteRanks; ++p) {
    remoteRanks[p].rank  = p;
    remoteRanks[p].index = 0;
  }
  ierr = PetscSFCreate(comm, &partSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(partSF, 1, numRemoteRanks, PETSC_NULL, PETSC_OWN_POINTER, remoteRanks, PETSC_OWN_POINTER);CHKERRQ(ierr);
  /* Debugging */
  ierr = PetscPrintf(comm, "Cell Partition:\n");CHKERRQ(ierr);
  ierr = PetscSectionView(cellPartSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISView(cellPart, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscSFView(partSF, PETSC_NULL);CHKERRQ(ierr);
  /* Close the partition over the mesh */
  ierr = DMMeshCreatePartitionClosure(dm, cellPartSection, cellPart, &partSection, &part);CHKERRQ(ierr);
  ierr = ISDestroy(&cellPart);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cellPartSection);CHKERRQ(ierr);
  /* Create new mesh */
  ierr = DMMeshCreate(comm, parallelDM);CHKERRQ(ierr);
  ierr = DMMeshSetDimension(*parallelDM, dim);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *parallelDM, "Parallel Mesh");CHKERRQ(ierr);
  /* Distribute sieve points and the global point numbering (replaces creating remote bases) */
  ierr = PetscSFConvertPartition(partSF, partSection, part, &renumbering, pointSF);CHKERRQ(ierr);
  /* Debugging */
  ierr = PetscPrintf(comm, "Point Partition:\n");CHKERRQ(ierr);
  ierr = PetscSectionView(partSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISView(part, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscSFView(*pointSF, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Point Renumbering after partition:\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(renumbering, PETSC_NULL);CHKERRQ(ierr);
  /* Cleanup */
  ierr = PetscSFDestroy(&partSF);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&partSection);CHKERRQ(ierr);
  ierr = ISDestroy(&part);CHKERRQ(ierr);
  /* Distribute cone section */
  ierr = DMMeshGetConeSection(dm, &originalConeSection);CHKERRQ(ierr);
  ierr = DMMeshGetConeSection(*parallelDM, &newConeSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(*pointSF, originalConeSection, &remoteOffsets, newConeSection);CHKERRQ(ierr);
  ierr = DMMeshSetUp(*parallelDM);CHKERRQ(ierr);
  /* Communicate and renumber cones */
  ierr = PetscSFCreateSectionSF(*pointSF, originalConeSection, remoteOffsets, newConeSection, &coneSF);CHKERRQ(ierr);
  ierr = DMMeshGetCones(dm, &cones);CHKERRQ(ierr);
  ierr = DMMeshGetCones(*parallelDM, &newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(newConeSection, &newConesSize);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(renumbering, IS_GTOLM_MASK, newConesSize, newCones, PETSC_NULL, newCones);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&renumbering);CHKERRQ(ierr);
  /* Debugging */
  ierr = PetscPrintf(comm, "Serial Cone Section:\n");CHKERRQ(ierr);
  ierr = PetscSectionView(originalConeSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Parallel Cone Section:\n");CHKERRQ(ierr);
  ierr = PetscSectionView(newConeSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscSFView(coneSF, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&coneSF);CHKERRQ(ierr);
  /* Create supports and stratify sieve */
  ierr = DMMeshSymmetrize(*parallelDM);CHKERRQ(ierr);
  ierr = DMMeshStratify(*parallelDM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DistributeCoordinates"
PetscErrorCode DistributeCoordinates(DM dm, PetscSF pointSF, DM parallelDM)
{
  PetscSF        coordSF;
  PetscSection   originalCoordSection, newCoordSection;
  Vec            coordinates, newCoordinates;
  PetscScalar   *coords,     *newCoords;
  PetscInt      *remoteOffsets, coordSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetCoordinateSection(dm, &originalCoordSection);CHKERRQ(ierr);
  ierr = DMMeshGetCoordinateSection(parallelDM, &newCoordSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(pointSF, originalCoordSection, &remoteOffsets, newCoordSection);CHKERRQ(ierr);

  ierr = DMMeshGetCoordinateVec(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMMeshGetCoordinateVec(parallelDM, &newCoordinates);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(newCoordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecSetSizes(newCoordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(newCoordinates);CHKERRQ(ierr);

  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(newCoordinates, &newCoords);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(pointSF, originalCoordSection, remoteOffsets, newCoordSection, &coordSF);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(coordSF, MPIU_SCALAR, coords, newCoords);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(coordSF, MPIU_SCALAR, coords, newCoords);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&coordSF);CHKERRQ(ierr);
  ierr = VecRestoreArray(newCoordinates, &newCoords);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  DM             dm, parallelDM;
  PetscSF        pointSF;
  AppCtx         user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &dm);CHKERRQ(ierr);
  ierr = DistributeMesh(dm, &user, &pointSF, &parallelDM);CHKERRQ(ierr);
  ierr = DistributeCoordinates(dm, pointSF, parallelDM);CHKERRQ(ierr);
  ierr = DMSetFromOptions(parallelDM);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&pointSF);CHKERRQ(ierr);
  ierr = DMDestroy(&parallelDM);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
