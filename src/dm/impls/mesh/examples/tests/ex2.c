static char help[] = "Test of Sieve vs. New Mesh and Field Distribution.\n\n";
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
    DM refinedMesh     = PETSC_NULL;
    DM distributedMesh = PETSC_NULL;
    const char *partitioner = user->partitioner;

    /* Refine mesh using a volume constraint */
    ierr = DMMeshRefine(*dm, refinementLimit, interpolate, &refinedMesh);CHKERRQ(ierr);
    if (refinedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = refinedMesh;
    }
    /* Distribute mesh over processes */
    ierr = DMMeshDistribute(*dm, partitioner, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshConvertOverlapToSF"
PetscErrorCode DMMeshConvertOverlapToSF(DM dm, PetscSF *sf)
{
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  PetscInt      *local;
  PetscSFNode   *remote;
  PetscInt       numPoints;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFCreate(((PetscObject) dm)->comm, sf);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  {
    /* The local points have degree 1
         We use the recv overlap
    */
    ALE::Obj<PETSC_MESH_TYPE::recv_overlap_type> overlap = mesh->getRecvOverlap();

    numPoints = overlap->getNumPoints();
    ierr = PetscMalloc(numPoints * sizeof(PetscInt), &local);CHKERRQ(ierr);
    ierr = PetscMalloc(numPoints * sizeof(PetscSFNode), &remote);CHKERRQ(ierr);
    for(PetscInt r = 0, i = 0; r < overlap->getNumRanks(); ++r) {
      const PetscInt                                                      rank   = overlap->getRank(r);
      const PETSC_MESH_TYPE::recv_overlap_type::supportSequence::iterator cBegin = overlap->supportBegin(rank);
      const PETSC_MESH_TYPE::recv_overlap_type::supportSequence::iterator cEnd   = overlap->supportEnd(rank);

      for(PETSC_MESH_TYPE::recv_overlap_type::supportSequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter, ++i) {
        local[i]        = *c_iter;
        remote[i].rank  = rank;
        remote[i].index = c_iter.color();
      }
    }
    ierr = PetscSFSetGraph(*sf, numPoints, numPoints, local, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFView(*sf, PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscSFCreateSectionSF"
PetscErrorCode PetscSFCreateSectionSF(PetscSF sf, PetscSection section, PetscSF *sectionSF)
{
  PetscInt           numRanks;
  const PetscInt    *ranks, *rankOffsets;
  const PetscMPIInt *localPoints, *remotePoints;
  PetscInt           numPoints, numIndices = 0;
  PetscInt          *remoteOffsets;
  PetscInt          *localIndices;
  PetscSFNode       *remoteIndices;
  PetscInt           i, r, ind;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscSFGetRanks(sf, &numRanks, &ranks, &rankOffsets, &localPoints, &remotePoints);CHKERRQ(ierr);
  numPoints = rankOffsets[numRanks];
  for(i = 0; i < numPoints; ++i) {
    PetscInt dof;

    ierr = PetscSectionGetDof(section, localPoints[i], &dof);CHKERRQ(ierr);
    numIndices += dof;
  }
  /* Communicate offsets for ghosted points */
#if 0
  PetscInt *localOffsets;
  ierr = PetscMalloc2(numPoints,PetscInt,&localOffsets,numPoints,PetscInt,&remoteOffsets);CHKERRQ(ierr);
  for(i = 0; i < numPoints; ++i) {
    ierr = PetscSectionGetOffset(section, localPoints[i], &localOffsets[i]);CHKERRQ(ierr);
  }
  ierr = PetscSFBcastBegin(sf, MPIU_INT, localOffsets, remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf, MPIU_INT, localOffsets, remoteOffsets);CHKERRQ(ierr);
  for(i = 0; i < numPoints; ++i) {
    ierr = PetscSynchronizedPrintf(((PetscObject) sf)->comm, "remoteOffsets[%d]: %d\n", i, remoteOffsets[i]);CHKERRQ(ierr);
  }
#else
  ierr = PetscMalloc((section->atlasLayout.pEnd - section->atlasLayout.pStart) * sizeof(PetscInt), &remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf, MPIU_INT, &section->atlasOff[-section->atlasLayout.pStart], &remoteOffsets[-section->atlasLayout.pStart]);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf, MPIU_INT, &section->atlasOff[-section->atlasLayout.pStart], &remoteOffsets[-section->atlasLayout.pStart]);CHKERRQ(ierr);
  for(i = section->atlasLayout.pStart; i < section->atlasLayout.pEnd; ++i) {
    ierr = PetscSynchronizedPrintf(((PetscObject) sf)->comm, "remoteOffsets[%d]: %d\n", i, remoteOffsets[i-section->atlasLayout.pStart]);CHKERRQ(ierr);
  }
#endif
  ierr = PetscSynchronizedFlush(((PetscObject) sf)->comm);CHKERRQ(ierr);
  ierr = PetscMalloc(numIndices * sizeof(PetscInt), &localIndices);CHKERRQ(ierr);
  ierr = PetscMalloc(numIndices * sizeof(PetscSFNode), &remoteIndices);CHKERRQ(ierr);
  /* Create new index graph */
  for(r = 0, ind = 0; r < numRanks; ++r) {
    PetscInt rank = ranks[r];

    for(i = rankOffsets[r]; i < rankOffsets[r+1]; ++i) {
      PetscInt localPoint   = localPoints[i];
      PetscInt remoteOffset = remoteOffsets[localPoint-section->atlasLayout.pStart];
      PetscInt localOffset, dof, d;

      ierr = PetscSectionGetOffset(section, localPoint, &localOffset);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(section, localPoint, &dof);CHKERRQ(ierr);
      for(d = 0; d < dof; ++d, ++ind) {
        localIndices[ind]        = localOffset+d;
        remoteIndices[ind].rank  = rank;
        remoteIndices[ind].index = remoteOffset+d;
      }
    }
  }
  ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  if (numIndices != ind) {SETERRQ2(((PetscObject) sf)->comm, PETSC_ERR_PLIB, "Inconsistency in indices, %d should be %d", ind, numIndices);}
  ierr = PetscSFCreate(((PetscObject) sf)->comm, sectionSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sectionSF, numIndices, numIndices, localIndices, PETSC_OWN_POINTER, remoteIndices, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFView(*sectionSF, PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  DM             dm;
  PetscSF        sf;
  AppCtx         user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &dm);CHKERRQ(ierr);
  ierr = DMMeshConvertOverlapToSF(dm, &sf);CHKERRQ(ierr);
  {
    PetscSection section;
    PetscSF      sectionSF;

    ierr = DMMeshGetCoordinateSection(dm, &section);CHKERRQ(ierr);
    ierr = PetscSectionView(section, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = PetscSFCreateSectionSF(sf, section, &sectionSF);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sectionSF);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
