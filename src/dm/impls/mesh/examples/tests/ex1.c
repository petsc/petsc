static char help[] = "Test of Mesh and Field Distribution.\n\n";
/*
The idea of this test is to compare different interfaces and implementations at the juncture
of the four main pieces of a computational simulation:

  - Domain description
  - Discretization (of both topology and functions)
  - Equations (for us this means weak forms)
  - Solver (PETSc)

Our interest here is the intersection of discrete domains (mesh), discrete functions (vector),
and solvers. The prototypical problem we have chosen in order to evaluate our API is mesh
redistribution (with the special case of starting with a serial mesh).

Problem Definition:

We must take a distributed mesh and a partition of said mesh, and move the mesh, its coordinates,
and a field defined over the mesh to respect the partition.

Mesh requirements:

We must be able to distribute meshes with these characteristics
  - 2D or 3D
  - simplicial, tensor product cells, or arbitrary cells (first two can be optimized)
  - only store necessary parts, e.g. cells and vertices

Partition requirements:

We must be able to partition any dimensional element, meaning that we specify a partition of cells, or vertices, or
edge, or faces, and then we must figure out how the rest of the mesh should be distributed. We must be able to pull
along the \textit{star} of the partition element, and also ghost to any specified overlap.

Solver requirements:

We want all field data, solution and coordinates, stored in PETSc Vec objects. DMGetLocalVector() must return the
restriction of a given field to the submesh prescribed by the partition, and DMGlobalVector() must return the entire
field. DMLocalToGlobal() must map between the two representations.

Proposed Mesh API:

Proposed Partition API:

Proposed Solver API:

I think we need a way to connect parts of the mesh to parts of the field defined over it. PetscSection is a map from
mesh pieces (sieve points) to Vec pieces (size and offset). In addition, it handles multiple fields and constraints.
The interface is here, http://petsc.cs.iit.edu/petsc/petsc-dev/annotate/eb9e8c4b5c78/include/petsc-private/vecimpl.h#l125.

Meshes Used:

The initial tests mesh the unit cube in the appropriate dimension. In 2D, we begin with 8 elements, so that the area of
a cell is $2^{-(k+3)}$ where $k$ is the number of refinement levels, and there are $2^{k+3}$ cells. So, if we want at
least N cells, then we need $k = \ceil{\lg N - 3}$ levels of refinement. In 3D, the refinement is less regular, but we
can still ask that the area of a cell be about $N^{-1}$.
*/
#include <petscdmmesh.h>

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
#define __FUNCT__ "ReadFEAPMesh"
PetscErrorCode ReadFEAPMesh(MPI_Comm comm, const char *filename, AppCtx *user, Vec *coordinates, Vec *elements)
{
  FILE          *fp;
  int            fd;
  char          *ret, line[1024];
  PetscScalar   *coords;
  PetscScalar   *elem;
  PetscViewer    viewer;
  PetscInt       numNodes, numLocalNodes, firstNode, numElems, numLocalElems, firstElem, nmat, ndm, numDof, numCorners, tmp;
  size_t         coordLineSize = 0, elemLineSize = 0;
  off_t          offset;
  PetscMPIInt    rank;
  const PetscInt dim   = 3;
  PetscBool      match = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  ierr = PetscViewerASCIIGetPointer(viewer, &fp);CHKERRQ(ierr);
  fd   = fileno(fp);
  // Skip everything until the line which begins with "FEAP"
  do {
    ret = fgets(line, 1023, fp);
    if (!ret) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Input file %s is not in FEAP format", filename);
    ierr = PetscStrncmp(line, "FEAP", 4, &match);CHKERRQ(ierr);
  } while(!match);
  // Read sizes
  ret = fgets(line, 256, fp);
  if (6 != sscanf(line, "%d %d %d %d %d %d", &numNodes, &numElems, &nmat, &ndm, &numDof, &numCorners)) {
    SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Malformed header in FEAP file");
  }
  ierr = VecCreate(comm, coordinates);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*coordinates, dim);CHKERRQ(ierr);
  ierr = VecSetSizes(*coordinates, PETSC_DETERMINE, numNodes*dim);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*coordinates);CHKERRQ(ierr);
  ierr = VecCreate(comm, elements);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*elements, numCorners);CHKERRQ(ierr);
  ierr = VecSetSizes(*elements, PETSC_DETERMINE, numElems*numCorners);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*elements);CHKERRQ(ierr);
  // Skip everything until "coordinates"
  match = PETSC_FALSE;
  do {
    ret = fgets(line, 1023, fp);
    if (!ret) {SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Input file %s is not in FEAP format", filename);}
    ierr = PetscStrncmp(line, "coor", 4, &match);CHKERRQ(ierr);
  } while(!match);
  // Rank 0 determines the length of a coordinate line and broadcasts it
  if (!rank) {
    ret = fgets(line, 1023, fp);
    if (!ret) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Missing coordinate line in FEAP file");}
    ierr = PetscStrlen(line, &coordLineSize);CHKERRQ(ierr);
    //ierr = PetscBinarySeek(fd, -coordLineSize, PETSC_BINARY_SEEK_CUR, &offset);CHKERRQ(ierr);
    fseek(fp, -coordLineSize, SEEK_CUR);
  }
  tmp = coordLineSize;
  ierr = MPI_Bcast(&coordLineSize, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  coordLineSize = tmp;
  // Read coordinates
  ierr = VecGetLocalSize(*coordinates, &numLocalNodes);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(*coordinates, &firstNode, PETSC_NULL);CHKERRQ(ierr);
  numLocalNodes /= dim;
  firstNode     /= dim;
  ierr = VecGetArray(*coordinates, &coords);CHKERRQ(ierr);
  //ierr = PetscBinarySeek(fd, firstNode*coordLineSize, PETSC_BINARY_SEEK_CUR, &offset);CHKERRQ(ierr);
  fseek(fp, firstNode*coordLineSize, SEEK_CUR);
  for(PetscInt n = 0; n < numLocalNodes; ++n) {
    PetscInt num, id;

    ret = fgets(line, 1023, fp);
    if (!ret) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Bad coordinate line");}
    if (5 != sscanf(line, "%d %d %le %le %le", &num, &id, &coords[n*dim+0], &coords[n*dim+1], &coords[n*dim+2])) {
      SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Malformed coordinate line in FEAP file <%s>", line);
    }
  }
  ierr = VecRestoreArray(*coordinates, &coords);CHKERRQ(ierr);
  // Skip everything until "elements"
  //ierr = PetscBinarySeek(fd, (numNodes - (firstNode+numLocalNodes))*coordLineSize, PETSC_BINARY_SEEK_CUR, &offset);CHKERRQ(ierr);
  fseek(fp, (numNodes - (firstNode+numLocalNodes))*coordLineSize, SEEK_CUR);
  match = PETSC_FALSE;
  do {
    ret = fgets(line, 1023, fp);
    if (!ret) {SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Input file %s is not in FEAP format", filename);}
    ierr = PetscStrncmp(line, "elem", 4, &match);CHKERRQ(ierr);
  } while(!match);
  // Rank 0 determines the length of a coordinate line and broadcasts it
  if (!rank) {
    ret = fgets(line, 1023, fp);
    if (!ret) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Missing element line in FEAP file");}
    ierr = PetscStrlen(line, &elemLineSize);CHKERRQ(ierr);
    //ierr = PetscBinarySeek(fd, -elemLineSize, PETSC_BINARY_SEEK_CUR, &offset);CHKERRQ(ierr);
    fseek(fp, -elemLineSize, SEEK_CUR);
  }
  tmp = elemLineSize;
  ierr = MPI_Bcast(&elemLineSize, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  elemLineSize = tmp;
  // Read in elements
  ierr = VecGetLocalSize(*elements, &numLocalElems);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(*elements, &firstElem, PETSC_NULL);CHKERRQ(ierr);
  numLocalElems /= numCorners;
  firstElem     /= numCorners;
  ierr = VecGetArray(*elements, &elem);CHKERRQ(ierr);
  //ierr = PetscBinarySeek(fd, firstElem*elemLineSize, PETSC_BINARY_SEEK_CUR, &offset);CHKERRQ(ierr);
  fseek(fp, firstElem*elemLineSize, SEEK_CUR);
  for(PetscInt n = 0; n < numLocalElems; ++n) {
    //         1         0     1         1         2         6         5       145       146       150       149
    PetscInt num, id, matid;
    PetscInt e[8];

    ret = fgets(line, 1023, fp);
    if (!ret) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Bad element line in FEAP file");}
    switch(numCorners) {
    case 8:
      if (3+8 != sscanf(line, "%d %d %d %d %d %d %d %d %d %d %d", &num, &id, &matid, &e[0], &e[1], &e[2], &e[3], &e[4], &e[5], &e[6], &e[7])) {
        SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Malformed element line in FEAP file");
      }
      break;
    default:
      SETERRQ1(comm, PETSC_ERR_SUP, "We do not support %d nodes per element", numCorners);
    }
    for(PetscInt c = 0; c < numCorners; ++c) {
      elem[n*numCorners+c] = e[c]-1;
    }
  }
  ierr = VecRestoreArray(*elements, &elem);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadMesh"
PetscErrorCode ReadMesh(MPI_Comm comm, const char *filename, AppCtx *user, DM *dm)
{
  Vec            coordinates, elements;
  PetscScalar   *coords, *elems;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = ReadFEAPMesh(comm, filename, user, &coordinates, &elements);CHKERRQ(ierr);
  try {
    typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
    PetscInt *cells, *cone, *coneO, *idx;
    PetscInt  dim, numCells, numTotalCells, numCorners, newV = 0;

    ierr = DMMeshCreate(comm, dm);CHKERRQ(ierr);
    ierr = VecGetBlockSize(coordinates, &dim);CHKERRQ(ierr);
    ierr = VecGetSize(elements,         &numTotalCells);CHKERRQ(ierr);
    ierr = VecGetLocalSize(elements,    &numCells);CHKERRQ(ierr);
    ierr = VecGetBlockSize(elements,    &numCorners);CHKERRQ(ierr);
    numCells      /= numCorners;
    numTotalCells /= numCorners;
    ierr = PetscMalloc(numCells*numCorners * sizeof(PetscInt), &cells);CHKERRQ(ierr);
    ierr = VecGetArray(elements,    &elems);CHKERRQ(ierr);
    for(PetscInt c = 0; c < numCells*numCorners; ++c) {
      cells[c] = elems[c];
    }
    ierr = VecRestoreArray(elements,    &elems);CHKERRQ(ierr);
    ALE::Obj<PETSC_MESH_TYPE>             mesh  = new PETSC_MESH_TYPE(comm, dim, 0);
    ALE::Obj<PETSC_MESH_TYPE::sieve_type> sieve = new PETSC_MESH_TYPE::sieve_type(comm, 0);
    PETSC_MESH_TYPE::renumbering_type     renumbering;

    // Renumber vertices
    for (PetscInt c = 0; c < numCells; ++c) {
      for(PetscInt v = 0; v < numCorners; ++v) {
        PetscInt vertex = cells[c*numCorners+v]+numTotalCells;

        if (renumbering.find(vertex) == renumbering.end()) {
          renumbering[vertex] = numCells + newV++;
        }
      }
    }
    ierr = PetscMalloc(newV*dim * sizeof(PetscInt), &idx);CHKERRQ(ierr);
    for (PetscInt c = 0; c < numCells; ++c) {
      for(PetscInt v = 0; v < numCorners; ++v) {
        PetscInt vertex = cells[c*numCorners+v]+numTotalCells;

        idx[renumbering[vertex] - numCells] = vertex;
      }
    }
    // Set chart
    sieve->setChart(PETSC_MESH_TYPE::sieve_type::chart_type(0, numCells+newV));
    // Set cone and support sizes
    for (PetscInt c = 0; c < numCells; ++c) {
      sieve->setConeSize(c, numCorners);
    }
    sieve->symmetrizeSizes(numCells, numCorners, cells, numCells);
    // Allocate point storage
    sieve->allocate();
    // Fill up cones
    ierr = PetscMalloc2(numCorners,PetscInt,&cone,numCorners,PetscInt,&coneO);CHKERRQ(ierr);
    for(PetscInt v = 0; v < numCorners; ++v) {
      coneO[v] = 1;
    }
    for(PetscInt c = 0; c < numCells; ++c) {
      for(PetscInt v = 0; v < numCorners; ++v) {
        cone[v] = renumbering[cells[c*numCorners+v]+numTotalCells];
      }
      sieve->setCone(cone, c);
      sieve->setConeOrientation(coneO, c);
    }
    ierr = PetscFree2(cone, coneO);CHKERRQ(ierr);
    ierr = PetscFree(cells);CHKERRQ(ierr);
    // Symmetrize to fill up supports
    sieve->symmetrize();
    mesh->setSieve(sieve);
    mesh->stratify();
    // Get ghosted coordinates
    Vec        ghostedCoordinates;
    VecScatter scatter;
    IS         is;

    ierr = VecCreate(PETSC_COMM_SELF, &ghostedCoordinates);CHKERRQ(ierr);
    ierr = VecSetBlockSize(ghostedCoordinates, dim);CHKERRQ(ierr);
    ierr = VecSetSizes(ghostedCoordinates, newV*dim, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(ghostedCoordinates);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, newV*dim, idx, PETSC_OWN_POINTER, &is);CHKERRQ(ierr);
    ierr = VecScatterCreate(coordinates, is, ghostedCoordinates, PETSC_NULL, &scatter);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter, coordinates, ghostedCoordinates, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter, coordinates, ghostedCoordinates, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
    // Put coordinates in the mesh
    ierr = VecGetArray(ghostedCoordinates, &coords);CHKERRQ(ierr);
    ALE::SieveBuilder<PETSC_MESH_TYPE>::buildCoordinates(mesh, dim, coords, numTotalCells);
    ierr = VecRestoreArray(ghostedCoordinates, &coords);CHKERRQ(ierr);
    ierr = VecDestroy(&ghostedCoordinates);CHKERRQ(ierr);
    {
      typedef PETSC_MESH_TYPE::point_type point_type;
      PETSc::Log::Event("CreateOverlap").begin();
      ALE::Obj<PETSC_MESH_TYPE::send_overlap_type> sendParallelMeshOverlap = mesh->getSendOverlap();
      ALE::Obj<PETSC_MESH_TYPE::recv_overlap_type> recvParallelMeshOverlap = mesh->getRecvOverlap();
      ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(renumbering);

      ALE::OverlapBuilder<>::constructOverlap(globalPoints, renumbering, sendParallelMeshOverlap, recvParallelMeshOverlap);
      //sendParallelMeshOverlap->view("Send Overlap");
      //recvParallelMeshOverlap->view("Recieve Overlap");
      mesh->setCalculatedOverlap(true);
      PETSc::Log::Event("CreateOverlap").end();
    }
    ierr = DMMeshSetMesh(*dm, mesh);CHKERRQ(ierr);
    ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid mesh: %s", e.message());
  }
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&elements);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  if (user.filename[0]) {
    ierr = ReadMesh(comm, user.filename, &user, &dm);CHKERRQ(ierr);
  } else {
    ierr = CreateMesh(comm, &user, &dm);CHKERRQ(ierr);
    {
      DM          distributedMesh = PETSC_NULL;
      const char *partitioner     = user.partitioner;

      /* Distribute mesh over processes */
      ierr = DMMeshDistribute(dm, partitioner, &distributedMesh);CHKERRQ(ierr);
      if (distributedMesh) {
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm  = distributedMesh;
        ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
