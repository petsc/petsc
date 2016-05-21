static char help[] = "Tests mesh adaptation with DMPlex and pragmatic.\n";

#include <petsc/private/dmpleximpl.h>

typedef struct {
  DM        dm;
  /* Definition of the test case (mesh and metric field) */
  PetscInt  dim;                          /* The topological mesh dimension */
  char      mshNam[PETSC_MAX_PATH_LEN];   /* Name of the mesh filename if any */
  PetscInt  nbrVerEdge;                   /* Number of vertices per edge if unit square/cube generated */
  char      bdyLabel[PETSC_MAX_PATH_LEN]; /* Name of the label marking boundary facets */
  PetscInt  metOpt;                       /* Different choices of metric */
  PetscReal hmax, hmin;                   /* Max and min sizes prescribed by the metric */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim        = 2;
  ierr = PetscStrcpy(options->mshNam, "");CHKERRQ(ierr);
  options->nbrVerEdge = 5;
  ierr = PetscStrcpy(options->bdyLabel, "");CHKERRQ(ierr);
  options->metOpt     = 1;
  options->hmin       = 0.05;
  options->hmax       = 0.5;

  ierr = PetscOptionsBegin(comm, "", "Meshing Adaptation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex19.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-msh", "Name of the mesh filename if any", "ex19.c", options->mshNam, options->mshNam, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nbrVerEdge", "Number of vertices per edge if unit square/cube generated", "ex19.c", options->nbrVerEdge, &options->nbrVerEdge, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-bdyLabel", "Name of the label marking boundary facets", "ex19.c", options->bdyLabel, options->bdyLabel, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-met", "Different choices of metric", "ex19.c", options->metOpt, &options->metOpt, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hmax", "Max size prescribed by the metric", "ex19.c", options->hmax, &options->hmax, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hmin", "Min size prescribed by the metric", "ex19.c", options->hmin, &options->hmin, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user)
{
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcmp(user->mshNam, "", &flag);CHKERRQ(ierr);
  if (flag) {
    ierr = DMPlexCreateBoxMesh(comm, user->dim, user->nbrVerEdge-1, PETSC_TRUE, (void *) user);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->mshNam, PETSC_TRUE, &user->dm);CHKERRQ(ierr);
    ierr = DMGetDimension(user->dm, &user->dim);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeMetric"
static PetscErrorCode ComputeMetric(DM dm, AppCtx *user, Vec *metric)
{
  Vec                coordinates, met;
  const PetscInt     dim = user->dim;
  const PetscScalar *coords;
  PetscReal          h, lambda1, lambda2, lambda3, lbd, lmax;
  PetscInt           vStart, vEnd, numVertices, i;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  numVertices = vEnd - vStart;

  ierr = VecCreate(PetscObjectComm((PetscObject) dm), metric);CHKERRQ(ierr);
  ierr = VecSetSizes(*metric, PETSC_DECIDE, numVertices*dim*dim);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*metric);CHKERRQ(ierr);

  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  for (i = 0; i < numVertices; ++i) {
    switch (user->metOpt) {
    case 0:
      lbd = 1/(user->hmax*user->hmax);
      lambda1 = lambda2 = lambda3 = lbd;
      break;
    case 1:
      h = user->hmax - (user->hmax-user->hmin)*coords[dim*i];
      h = h*h;
      lmax = 1/(user->hmax*user->hmax);
      lambda1 = 1/h;
      lambda2 = lmax;
      lambda3 = lmax;
      break;
    case 2:
      h = user->hmax*fabs(1-exp(-fabs(coords[dim*i]-0.5))) + user->hmin;
      lbd = 1/(h*h);
      lmax = 1/(user->hmax*user->hmax);
      lambda1 = lbd;
      lambda2 = lmax;
      lambda3 = lmax;
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "metOpt = 0, 1 or 2, cannot be %d", user->metOpt);
    }
    if (dim == 2) {
      ierr = VecSetValue(*metric, 4*i  , lambda1, INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(*metric, 4*i+1, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(*metric, 4*i+2, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(*metric, 4*i+3, lambda2, INSERT_VALUES);CHKERRQ(ierr);
    } else {
      ierr = VecSetValue(*metric, 9*i  , lambda1, INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(*metric, 9*i+1, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(*metric, 9*i+2, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(*metric, 9*i+3, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(*metric, 9*i+4, lambda2, INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(*metric, 9*i+5, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(*metric, 9*i+6, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(*metric, 9*i+7, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(*metric, 9*i+8, lambda3, INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(*metric);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*metric);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc, char * argv[]) {
  AppCtx         user;                 /* user-defined work context */
  MPI_Comm       comm;
  DM             dma;
  Vec            metric;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);

  ierr = CreateMesh(comm, &user);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) user.dm, "DMinit");CHKERRQ(ierr);
  ierr = DMViewFromOptions(user.dm, NULL, "-init_dm_view");CHKERRQ(ierr);

  ierr = ComputeMetric(user.dm, &user, &metric);CHKERRQ(ierr);
  ierr = DMPlexAdapt(user.dm, metric, user.bdyLabel, &dma);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dma, "DMadapt");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dma, NULL, "-adapt_dm_view");CHKERRQ(ierr);

  ierr = VecDestroy(&metric);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dma);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
