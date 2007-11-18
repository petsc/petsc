// This example will solve the Bratu problem eventually
static char help[] = "This example solves the Bratu problem.\n\n";

#include <petscmesh.hh>
#include <petscmesh_viewers.hh>
#include <petscmesh_formats.hh>
#include "Generator.hh"

#include "GMVFileAscii.hh" // USES GMVFileAscii
#include "GMVFileBinary.hh" // USES GMVFileBinary

using ALE::Obj;

typedef struct {
  PetscInt      debug;                       // The debugging level
  PetscInt      dim;                         // The topological mesh dimension
  PetscTruth    generateMesh;                // Generate the unstructure mesh
  PetscTruth    interpolate;                 // Generate intermediate mesh elements
  PetscTruth    refineLocal;                 // Locally refine the mesh
  PetscReal    *refinementLimit;             // The largest allowable cell volume on each process
  char          baseFilename[2048];          // The base filename for mesh files
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscMPIInt    size;
  PetscInt       numLimits;
  ostringstream  filename;
  PetscTruth     flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug        = 0;
  options->dim          = 2;
  options->generateMesh = PETSC_TRUE;
  options->interpolate  = PETSC_TRUE;
  options->refineLocal  = PETSC_FALSE;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = PetscMalloc(size * sizeof(PetscReal), &options->refinementLimit);CHKERRQ(ierr);
  for(int p = 0; p < size; ++p) options->refinementLimit[p] = 0.0;

  ierr = PetscOptionsBegin(comm, "", "Bratu Problem Options", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "refinement.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "refinement.cxx", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-generate", "Generate the unstructured mesh", "refinement.cxx", options->generateMesh, &options->generateMesh, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Generate intermediate mesh elements", "refinement.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-refine_local", "Locally refine the mesh", "refinement.cxx", options->refineLocal, &options->refineLocal, PETSC_NULL);CHKERRQ(ierr);
    numLimits = size;
    ierr = PetscOptionsRealArray("-refinement_limit", "The largest allowable cell volume per process", "refinement.cxx", options->refinementLimit, &numLimits, &flag);CHKERRQ(ierr);
    if (flag) {
      if (numLimits == 1) {
	for(int p = 1; p < size; ++p) options->refinementLimit[p] = options->refinementLimit[0];
      } else if (numLimits != size) {
        SETERRQ1(PETSC_ERR_ARG_WRONG, "Cannot specify refinement limits on a subset (%d) of processes", numLimits);
      }
    }
    filename << "data/refinement_" << options->dim <<"d";
    ierr = PetscStrcpy(options->baseFilename, filename.str().c_str());CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_filename", "The base filename for mesh files", "refinement.cxx", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
// Creates a field whose value is the processor rank on each element
PetscErrorCode CreatePartition(Mesh mesh, SectionInt *partition)
{
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetCellSectionInt(mesh, 1, partition);CHKERRQ(ierr);
  const Obj<ALE::Mesh::label_sequence>&     cells = m->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator end   = cells->end();
  const int                                 rank  = m->commRank();

  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
    ierr = SectionIntUpdate(*partition, *c_iter, &rank);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ViewSection"
PetscErrorCode ViewSection(Mesh mesh, SectionReal section, const char filename[], bool vertexwise = true)
{
  MPI_Comm       comm;
  SectionInt     partition;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  ierr = MeshView(mesh, viewer);CHKERRQ(ierr);
  if (!vertexwise) {ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);}
  ierr = SectionRealView(section, viewer);CHKERRQ(ierr);
  ierr = CreatePartition(mesh, &partition);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
  ierr = SectionIntView(partition, viewer);CHKERRQ(ierr);
  ierr = SectionIntDestroy(partition);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ViewMesh"
PetscErrorCode ViewMesh(Mesh mesh, const char filename[])
{
  MPI_Comm       comm;
  SectionInt     partition;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  ierr = MeshView(mesh, viewer);CHKERRQ(ierr);
  ierr = CreatePartition(mesh, &partition);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
  ierr = SectionIntView(partition, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = SectionIntDestroy(partition);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MeshIDBoundary(Mesh);

void FlipCellOrientation(pylith::int_array * const cells, const int numCells, const int numCorners, const int meshDim) {
  if (3 == meshDim && 4 == numCorners) {
    for(int iCell = 0; iCell < numCells; ++iCell) {
      const int i1 = iCell*numCorners+1;
      const int i2 = iCell*numCorners+2;
      const int tmp = (*cells)[i1];
      (*cells)[i1] = (*cells)[i2];
      (*cells)[i2] = tmp;
    }
  }
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  Mesh        mesh;
  PetscTruth  view;
  PetscMPIInt size, rank;

  if (options->generateMesh) {
    Mesh boundary;

    ierr = MeshCreate(comm, &boundary);CHKERRQ(ierr);
    if (options->dim == 2) {
      double lower[2]  = {0.0, 0.0};
      double upper[2]  = {1.0, 1.0};
      int    edges[2]  = {2, 2};
      Obj<ALE::Mesh> mB;

      mB = ALE::MeshBuilder::createSquareBoundary(comm, lower, upper, edges, options->debug);
      ierr = MeshSetMesh(boundary, mB);CHKERRQ(ierr);
    } else if (options->dim == 3) {
      double lower[3] = {0.0, 0.0, 0.0};
      double upper[3] = {1.0, 1.0, 1.0};
      int    faces[3] = {3, 3, 3};

      Obj<ALE::Mesh> mB = ALE::MeshBuilder::createCubeBoundary(comm, lower, upper, faces, options->debug);
      ierr = MeshSetMesh(boundary, mB);CHKERRQ(ierr);
    } else {
      SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", options->dim);
    }
    ierr = MeshGenerate(boundary, options->interpolate, &mesh);CHKERRQ(ierr);
    ierr = MeshDestroy(boundary);CHKERRQ(ierr);
  } else {
    Obj<ALE::Mesh>             m     = new ALE::Mesh(comm, options->dim, options->debug);
    Obj<ALE::Mesh::sieve_type> sieve = new ALE::Mesh::sieve_type(comm, options->debug);
    bool                 flipEndian = false;
    int                  dim;
    pylith::int_array    cells;
    pylith::double_array coordinates;
    pylith::int_array    materialIds;
    int                  numCells = 0, numVertices = 0, numCorners = 0;

    if (!m->commRank()) {
      if (pylith::meshio::GMVFile::isAscii(options->baseFilename)) {
	pylith::meshio::GMVFileAscii filein(options->baseFilename);
	filein.read(&coordinates, &cells, &materialIds, &dim, &dim, &numVertices, &numCells, &numCorners);
	if (options->interpolate) {
	  FlipCellOrientation(&cells, numCells, numCorners, dim);
	}
      } else {
	pylith::meshio::GMVFileBinary filein(options->baseFilename, flipEndian);
	filein.read(&coordinates, &cells, &materialIds, &dim, &dim, &numVertices, &numCells, &numCorners);
	if (!options->interpolate) {
	  FlipCellOrientation(&cells, numCells, numCorners, dim);
	}
      } // if/else
    }
    ALE::SieveBuilder<ALE::Mesh>::buildTopology(sieve, dim, numCells, const_cast<int*>(&cells[0]), numVertices, options->interpolate, numCorners, -1, m->getArrowSection("orientation"));
    m->setSieve(sieve);
    m->stratify();
    ALE::SieveBuilder<ALE::Mesh>::buildCoordinates(m, dim, const_cast<double*>(&coordinates[0]));

    ierr = MeshCreate(comm, &mesh);CHKERRQ(ierr);
    ierr = MeshSetMesh(mesh, m);CHKERRQ(ierr);
    ierr = MeshIDBoundary(mesh);CHKERRQ(ierr);
  }
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (size > 1) {
    Mesh parallelMesh;

    ierr = MeshDistribute(mesh, PETSC_NULL, &parallelMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(mesh);CHKERRQ(ierr);
    mesh = parallelMesh;
  }
  PetscTruth refine = PETSC_FALSE;

  for(int p = 0; p < size; ++p) {
    if (options->refinementLimit[p] > 0.0) {
      refine = PETSC_TRUE;
      break;
    }
  }
  if (refine) {
    Mesh refinedMesh;
    Obj<ALE::Mesh> m;

    if (options->refineLocal) {
      ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
      ierr = MeshCreate(m->comm(), &refinedMesh);CHKERRQ(ierr);
      Obj<ALE::Mesh> refM = ALE::Generator::refineMeshLocal(m, options->refinementLimit[rank], options->interpolate);
      ierr = MeshSetMesh(refinedMesh, refM);CHKERRQ(ierr);
    } else {
      ierr = MeshRefine(mesh, options->refinementLimit[rank], options->interpolate, &refinedMesh);CHKERRQ(ierr);
    }
    ierr = MeshDestroy(mesh);CHKERRQ(ierr);
    mesh = refinedMesh;
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_vtk", &view);CHKERRQ(ierr);
  if (view) {ierr = ViewMesh(mesh, "refinement.vtk");CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
  if (view) {
    Obj<ALE::Mesh> m;
    ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
    m->view("Mesh");
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_simple", &view);CHKERRQ(ierr);
  if (view) {ierr = MeshView(mesh, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  *dm = (DM) mesh;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DestroyMesh"

PetscErrorCode DestroyMesh(DM dm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshDestroy((Mesh) dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Options        options;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
  try {
    ierr = CreateMesh(comm, &dm, &options);CHKERRQ(ierr);
    ierr = DestroyMesh(dm, &options);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    std::cerr << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
