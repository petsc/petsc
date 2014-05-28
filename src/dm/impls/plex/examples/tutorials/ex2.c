static char help[] = "Read in a mesh and test whether it is valid\n\n";

#include <petscdmplex.h>
#if defined(PETSC_HAVE_CGNS)
#include <cgnslib.h>
#endif
#if defined(PETSC_HAVE_EXODUSII)
#include <exodusII.h>
#endif

typedef struct {
  PetscBool interpolate;                  /* Generate intermediate mesh elements */
  char      filename[PETSC_MAX_PATH_LEN]; /* Mesh filename */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->interpolate = PETSC_FALSE;
  options->filename[0] = '\0';

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex1.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex7.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscBool      interpolate = user->interpolate;
  const char    *filename    = user->filename;
  const char    *extGmsh     = ".msh";
  const char    *extCGNS     = ".cgns";
  const char    *extExodus   = ".exo";
  size_t         len;
  PetscBool      isGmsh, isCGNS, isExodus;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (!len) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Must provide input file using -filename");
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)], extGmsh,   4, &isGmsh);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-5)], extCGNS,   5, &isCGNS);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)], extExodus, 4, &isExodus);CHKERRQ(ierr);
  if (isGmsh) {
    PetscViewer viewer;

    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
    ierr = DMPlexCreateGmsh(comm, viewer, interpolate, dm);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  } else if (isCGNS) {
#if defined(PETSC_HAVE_CGNS)
    int cgid = -1;

    if (!rank) {
      ierr = cg_open(filename, CG_MODE_READ, &cgid);CHKERRQ(ierr);
      if (cgid <= 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "cg_open(\"%s\",...) did not return a valid file ID", filename);
    }
    ierr = DMPlexCreateCGNS(comm, cgid, interpolate, dm);CHKERRQ(ierr);
    if (!rank) {ierr = cg_close(cgid);CHKERRQ(ierr);}
#else
    SETERRQ(comm, PETSC_ERR_SUP, "Loading meshes requires CGNS support. Reconfigure using --with-cgns-dir");
#endif
  } else if (isExodus) {
#if defined(PETSC_HAVE_EXODUSII)
    int   CPU_word_size = 0, IO_word_size = 0, exoid;
    float version;

    if (!rank) {
      exoid = ex_open(filename, EX_READ, &CPU_word_size, &IO_word_size, &version);
      if (exoid <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ex_open(\"%s\",...) did not return a valid file ID",filename);
    }
    ierr = DMPlexCreateExodus(comm, exoid, interpolate, dm);CHKERRQ(ierr);
    if (!rank) {ierr = ex_close(exoid);CHKERRQ(ierr);}
#else
    SETERRQ(comm, PETSC_ERR_SUP, "Loading meshes requires Exodus support. Reconfigure using --download-exodusii");
#endif
  } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot load file %s: unrecognized extension", filename);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckMeshTopology"
static PetscErrorCode CheckMeshTopology(DM dm)
{
  PetscInt       dim, coneSize, cStart;
  PetscBool      isSimplex;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
  isSimplex = coneSize == dim+1 ? PETSC_TRUE : PETSC_FALSE;
  ierr = DMPlexCheckSymmetry(dm);CHKERRQ(ierr);
  ierr = DMPlexCheckSkeleton(dm, isSimplex, 0);CHKERRQ(ierr);
  ierr = DMPlexCheckFaces(dm, isSimplex, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckMeshGeometry"
static PetscErrorCode CheckMeshGeometry(DM dm)
{
  PetscInt       dim, coneSize, cStart, cEnd, c;
  PetscReal     *v0, *J, *invJ, detJ;
  PetscBool      isSimplex;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
  isSimplex = coneSize == dim+1 ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscMalloc3(dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = DMPlexComputeCellGeometry(dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for cell %d", detJ, c);
  }
  ierr = PetscFree3(v0,J,invJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = CheckMeshTopology(dm);CHKERRQ(ierr);
  ierr = CheckMeshGeometry(dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
