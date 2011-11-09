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

We must be able to partition any dimensional element. Thus, we partition cells for finite elements, but faces for
finite volumes, and vertices for finite difference. We must be able to pull along the \textit{start} of the partition
element, and also ghost to any specified overlap.

Solver requirements:

We want all field data, solution and coordinates, stored in PETSc Vec objects. DMGetLocalVector() must return the
restriction of a given field to the submesh prescribed by the partition, and DMGlobalVector() must return the entire
field. DMLocalToGlobal() must map between the two representations.

Proposed Mesh API:

Proposed Partition API:

Proposed Solver API:

I think we need a way to connect parts of the mesh to parts of the field defined over it. PetscSection is a map from
mesh pieces (sieve points) to Vec pieces (size and offset). In addition, it handles multiple fields and constraints.
The interface is here, http://petsc.cs.iit.edu/petsc/petsc-dev/annotate/eb9e8c4b5c78/include/private/vecimpl.h#l125.

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
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
