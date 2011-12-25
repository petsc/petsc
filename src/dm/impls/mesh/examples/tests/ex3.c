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
#define __FUNCT__ "PetscSFConvertPartition"
PetscErrorCode PetscSFConvertPartition(DM dm, PetscSection partSection, IS partition, PetscSF *sf)
{
  MPI_Comm        comm = ((PetscObject) dm)->comm;
  PetscSF         sfCount;
  PetscSFNode    *remoteRanks, *remotePoints;
  IS              renumbering;
  PetscInt       *renumArray;
  PetscInt        numRemoteRanks = 0, p, i;
  PetscInt        localSize[2], partSize, *partSizes = PETSC_NULL, *partOffsets = PETSC_NULL;
  const PetscInt *partArray;
  PetscMPIInt     numProcs, rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  /*
   1. Count the number of ranks that your points should be sent to (nranks)
   2. Create a PetscSF that maps from nranks local points to (rank, 0). That is, each process has only one "owned" point. Call this sfcount.
   3. Put the number of points destined to each rank in outgoing, create another array outoffset of same size (nranks)
   4. incoming[0] = 0
   5. PetscSFFetchAndOpBegin/End(sf,MPIU_INT,incoming,outgoing,outoffset,MPIU_SUM);

   Now, incoming[0] holds the number of points you will receive and outoffset[i] holds the offset on rank[i] at which to place outgoing[i] points.

   6. Call PetscSFSetGraph() to build this new graph that communicates all your currently owned points to the remote processes at the offsets above.
   7. PetscSFReduceBegin/End(newsf,MPIU_POINT,outgoing_points,incoming_points,MPI_REPLACE);

   I would typically make MPIU_POINT just represent a PetscSFNode() indicating where the point data is in my local space. That would have successfully inverted the communication graph: now I have two-sided knowledge and local-to-global now maps from newowner-to-oldowner.
   */
  /* Assume all processes participate in the partition */
  if (!rank) { /* Could check for IS length here */
    numRemoteRanks = numProcs;
  }
  ierr = PetscMalloc(numRemoteRanks * sizeof(PetscSFNode), &remoteRanks);CHKERRQ(ierr);
  ierr = PetscMalloc2(2*numRemoteRanks,PetscInt,&partSizes,2*numRemoteRanks,PetscInt,&partOffsets);CHKERRQ(ierr);
  for(p = 0; p < numRemoteRanks; ++p) {
    remoteRanks[p].rank  = p;
    remoteRanks[p].index = 0;
    ierr = PetscSectionGetDof(partSection, p, &partSizes[2*p+0]);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(partSection, p, &partSizes[2*p+1]);CHKERRQ(ierr);
  }
  ierr = PetscSFCreate(comm, &sfCount);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sfCount, 1, numRemoteRanks, PETSC_NULL, PETSC_OWN_POINTER, remoteRanks, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFView(sfCount, PETSC_NULL);CHKERRQ(ierr);
  localSize[0] = 0; localSize[1] = 0;
  ierr = PetscSFFetchAndOpBegin(sfCount, MPIU_2INT, &localSize, partSizes, partOffsets, MPIU_SUM);CHKERRQ(ierr);
  ierr = PetscSFFetchAndOpEnd(sfCount, MPIU_2INT, &localSize, partSizes, partOffsets, MPIU_SUM);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(comm, "localSize %d %d\n", localSize[0], localSize[1]);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);
  for(p = 0; p < numRemoteRanks; ++p) {
    ierr = PetscPrintf(comm, "offset for rank %d: %d\n", p, partOffsets[p]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(partSizes,partOffsets);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfCount);CHKERRQ(ierr);
  /* Create the inverse graph for the partition */
  ierr = PetscMalloc(localSize[0] * sizeof(PetscSFNode), &remotePoints);CHKERRQ(ierr);
  for(i = 0; i < localSize[0]; ++i) {
    remotePoints[i].rank  = 0;
    remotePoints[i].index = localSize[1] + i;
  }
  ierr = ISGetLocalSize(partition, &partSize);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm, sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sf, partSize, localSize[0], PETSC_NULL, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
  /* Send global point numbers
     - owned values are sent
     - local values are received
  */
  ierr = PetscMalloc(localSize[0] * sizeof(PetscInt), &renumArray);CHKERRQ(ierr);
  ierr = ISGetIndices(partition, &partArray);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(*sf, MPIU_INT, partArray, renumArray);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(*sf, MPIU_INT, partArray, renumArray);CHKERRQ(ierr);
  ierr = ISRestoreIndices(partition, &partArray);CHKERRQ(ierr);

  ierr = ISCreateGeneral(comm, localSize[0], renumArray, PETSC_OWN_POINTER, &renumbering);CHKERRQ(ierr);
  ierr = ISView(renumbering, PETSC_NULL);CHKERRQ(ierr);
  ierr = ISDestroy(&renumbering);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DistributeMesh"
PetscErrorCode DistributeMesh(DM dm, AppCtx *user, DM *parallelDM)
{
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  MPI_Comm       comm = ((PetscObject) dm)->comm;
  PetscSF        partSF;
  IS             cellPart,        part;
  PetscSection   cellPartSection, partSection;
  PetscMPIInt    numProcs, rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  /* Create cell partition - We need to rewrite to use IS, use the MatPartition stuff */
  {
    typedef PETSC_MESH_TYPE::point_type          point_type;
    typedef ALE::Partitioner<>::part_type        rank_type;
    typedef ALE::ISection<rank_type, point_type> partition_type;
    const ALE::Obj<partition_type> cellPartition = new partition_type(comm, 0, numProcs, user->debug);
    const PetscInt                 height        = 0;

    ALE::Partitioner<>::createPartitionV(mesh, cellPartition, height);
    if (user->debug) {
      PetscViewer    viewer;
      PetscErrorCode ierr;

      cellPartition->view("Cell Partition");
      ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRXX(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRXX(ierr);
      ierr = PetscViewerFileSetName(viewer, "mesh.vtk");CHKERRXX(ierr);
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = DMView(dm, viewer);CHKERRQ(ierr);
      //ierr = ISView(mesh, cellPartition);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    /* Convert to PetscSection+IS */
    ierr = PetscSectionCreate(comm, &cellPartSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(cellPartSection, 0, numProcs);CHKERRQ(ierr);
    for(PetscInt p = 0; p < numProcs; ++p) {
      ierr = PetscSectionSetDof(cellPartSection, p, cellPartition->getFiberDimension(p));CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(cellPartSection);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, cellPartition->size(), cellPartition->restrictSpace(), PETSC_COPY_VALUES, &cellPart);CHKERRQ(ierr);
    ierr = PetscSectionView(cellPartSection, PETSC_NULL);CHKERRQ(ierr);
    ierr = ISView(cellPart, PETSC_NULL);CHKERRQ(ierr);
  }
  /* Close the partition over the mesh */
  ierr = ALE::Partitioner<>::createPartitionClosureV(mesh, cellPartSection, cellPart, &partSection, &part, 0);CHKERRQ(ierr);
  ierr = PetscSectionView(partSection, PETSC_NULL);CHKERRQ(ierr);
  ierr = ISView(part, PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscSFConvertPartition(dm, partSection, part, &partSF);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&partSF);CHKERRQ(ierr);
#if 0
  /* Create the remote bases -- We probably do not need this SF, only ones buikt on this. We do need to know how many points we are getting */
  {
    PetscSF         sf;
    PetscSFNode    *remotePoints;
    const PetscInt *partArray;
    PetscInt        numOwned, numLocal, r, i;

    ierr = PetscSectionGetDof(partSection, rank, &numOwned);CHKERRQ(ierr);
    ierr = ISGetLocalSize(part, &numLocal);CHKERRQ(ierr);
    ierr = ISGetIndices(part, &partArray);CHKERRQ(ierr);
    ierr = PetscMalloc(numLocal * sizeof(PetscSFNode), &remotePoints);CHKERRQ(ierr);
    for(r = 0, i = 0; r < numProcs; ++r) {
      PetscInt dof, offset, p;

      ierr = PetscSectionGetOffset(partSection, r, &offset);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(partSection, r, &dof);CHKERRQ(ierr);
      for(p = 0; p < dof; ++p, ++i) {
        remotePoints[i].rank  = r;
        remotePoints[i].index = p;
      }
    }
    ierr = PetscSFCreate(comm, &sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sf, numOwned, numLocal, partArray, PETSC_COPY_VALUES, remotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFView(sf, PETSC_NULL);CHKERRQ(ierr);

    // Test sending the points
    //   How do I know how many points are coming to me?
    {
      PetscInt *newPoints;
      PetscInt  numPoints = 10;

      ierr = PetscMalloc(numPoints * sizeof(PetscInt), &newPoints);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(sf, MPIU_INT, partArray, newPoints);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf, MPIU_INT, partArray, newPoints);CHKERRQ(ierr);
      ierr = PetscFree(newPoints);CHKERRQ(ierr);
    }

    ierr = ISRestoreIndices(part, &partArray);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  }
#endif
#if 0
      const Obj<partition_type> cellPartition = new partition_type(mesh->comm(), 0, mesh->commSize(), mesh->debug());
      const Obj<partition_type> partition     = new partition_type(mesh->comm(), 0, mesh->commSize(), mesh->debug());

      // Create the cell partition
      Partitioner::createPartitionV(mesh, cellPartition, height);
      if (mesh->debug()) {
        PetscViewer    viewer;
        PetscErrorCode ierr;

        cellPartition->view("Cell Partition");
        ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRXX(ierr);
        ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRXX(ierr);
        ierr = PetscViewerFileSetName(viewer, "mesh.vtk");CHKERRXX(ierr);
        ///TODO ierr = MeshView_Sieve_Ascii(mesh, cellPartition, viewer);CHKERRXX(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRXX(ierr);
      }
      // Close the partition over sieve points
      Partitioner::createPartitionClosureV(mesh, cellPartition, partition, height);
      if (mesh->debug()) {partition->view("Partition");}
      // Create the remote bases
      completeBaseV(mesh, partition, renumbering, newMesh, sendMeshOverlap, recvMeshOverlap);
      // Size the local mesh
      Partitioner::sizeLocalMeshV(mesh, partition, renumbering, newMesh, height);
      // Create the remote meshes
      completeConesV(mesh->getSieve(), newMesh->getSieve(), renumbering, sendMeshOverlap, recvMeshOverlap);
      // Create the local mesh
      Partitioner::createLocalMeshV(mesh, partition, renumbering, newMesh, height);
      newMesh->getSieve()->symmetrize();
      newMesh->stratify();
      return partition;
#endif
  ierr = ISDestroy(&cellPart);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cellPartSection);CHKERRQ(ierr);
  ierr = ISDestroy(&part);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&partSection);CHKERRQ(ierr);
  *parallelDM = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  DM             dm, parallelDM;
  AppCtx         user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &dm);CHKERRQ(ierr);
  ierr = DistributeMesh(dm, &user, &parallelDM);CHKERRQ(ierr);
  ierr = DMDestroy(&parallelDM);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
