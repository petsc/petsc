static char help[] = "Tests mesh adaptation with DMPlex and pragmatic.\n";

#include <petsc/private/dmpleximpl.h>

typedef struct {
  DM        dm;
  /* Definition of the test case (mesh and metric field) */
  PetscInt  dim;                         /* The topological mesh dimension */
  char      mshNam[PETSC_MAX_PATH_LEN];  /* Name of the mesh filename if any */
  PetscInt  nbrVerEdge;                  /* Number of vertices per edge if unit square/cube generated */
  char      bdLabel[PETSC_MAX_PATH_LEN]; /* Name of the label marking boundary facets */
  PetscInt  metOpt;                      /* Different choices of metric */
  PetscReal hmax, hmin;                  /* Max and min sizes prescribed by the metric */
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
  ierr = PetscStrcpy(options->bdLabel, "");CHKERRQ(ierr);
  options->metOpt     = 1;
  options->hmin       = 0.05;
  options->hmax       = 0.5;

  ierr = PetscOptionsBegin(comm, "", "Meshing Adaptation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex19.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-msh", "Name of the mesh filename if any", "ex19.c", options->mshNam, options->mshNam, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nbrVerEdge", "Number of vertices per edge if unit square/cube generated", "ex19.c", options->nbrVerEdge, &options->nbrVerEdge, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-bdLabel", "Name of the label marking boundary facets", "ex19.c", options->bdLabel, options->bdLabel, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
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
    ierr = DMPlexCreateBoxMesh(comm, user->dim, user->nbrVerEdge-1, PETSC_TRUE, &user->dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->mshNam, PETSC_TRUE, &user->dm);CHKERRQ(ierr);
    ierr = DMGetDimension(user->dm, &user->dim);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(user->dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeMetric"
static PetscErrorCode ComputeMetric(DM dm, AppCtx *user, Vec *metric)
{
  DM                 cdm, mdm;
  PetscSection       csec, msec;
  Vec                coordinates;
  const PetscScalar *coords;
  PetscScalar       *met;
  PetscReal          h, lambda[3], lbd, lmax;
  PetscInt           pStart, pEnd, p, d;
  const PetscInt     dim = user->dim, Nd = dim*dim;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMClone(cdm, &mdm);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(cdm, &csec);CHKERRQ(ierr);

  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &msec);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(msec, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(msec, 0, Nd);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(csec, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(msec, pStart, pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    ierr = PetscSectionSetDof(msec, p, Nd);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(msec, p, 0, Nd);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(msec);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(mdm, msec);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&msec);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(mdm, metric);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(*metric, &met);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    const PetscScalar *pcoords;
    PetscScalar       *pmet;

    ierr = DMPlexPointLocalRead(cdm, p, coords, &pcoords);CHKERRQ(ierr);
    switch (user->metOpt) {
    case 0:
      lbd = 1/(user->hmax*user->hmax);
      lambda[0] = lambda[1] = lambda[2] = lbd;
      break;
    case 1:
      h = user->hmax - (user->hmax-user->hmin)*pcoords[0];
      h = h*h;
      lmax = 1/(user->hmax*user->hmax);
      lambda[0] = 1/h;
      lambda[1] = lmax;
      lambda[2] = lmax;
      break;
    case 2:
      h = user->hmax*fabs(1-exp(-fabs(pcoords[0]-0.5))) + user->hmin;
      lbd = 1/(h*h);
      lmax = 1/(user->hmax*user->hmax);
      lambda[0] = lbd;
      lambda[1] = lmax;
      lambda[2] = lmax;
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "metOpt = 0, 1 or 2, cannot be %d", user->metOpt);
    }
    /* Only set the diagonal */
    ierr = DMPlexPointGlobalRef(mdm, p, met, &pmet);CHKERRQ(ierr);
    for (d = 0; d < dim; ++d) pmet[d*(dim+1)] = lambda[d];
  }
  ierr = VecRestoreArray(*metric, &met);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMDestroy(&mdm);CHKERRQ(ierr);
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
  ierr = DMPlexAdapt(user.dm, metric, user.bdLabel, &dma);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dma, "DMadapt");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dma, NULL, "-adapt_dm_view");CHKERRQ(ierr);

  ierr = VecDestroy(&metric);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dma);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
