#include <petscmesh.hh>
#include <petscmesh_viewers.hh>
#include <petscmesh_formats.hh>
#include "MeshSurgery.hh"
#include "Hierarchy.hh"

typedef struct {
  int dim;
  int debug;
  PetscTruth useZeroBase;
  char baseFilename[2048];
  PetscInt flips;
  PetscTruth dolfin;
  PetscTruth generate;
} Options;

#undef  __FUNCT__ 
#define __FUNCT__ "ProcessOptions"

PetscErrorCode ProcessOptions(MPI_Comm comm, Options * options) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  options->dolfin = PETSC_FALSE;
  options->dim = 2;
  options->generate = PETSC_FALSE;
  options->debug = false;
  options->useZeroBase = PETSC_FALSE;
  options->flips = 100000000;
  ierr = PetscStrcpy(options->baseFilename, "data/flip_test");CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-generate", "generates a basic mesh for testing.", "mesh_surgery.cxx", options->generate, &options->generate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Options:", "mesh_surgery.cxx");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The Mesh Dimension", "mesh_surgery.cxx", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-use_zero_base", "Use zero-base feature indexing", "mesh_surgery.cxx", options->useZeroBase, &options->useZeroBase, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-flips", "the number of flips to perform", "mesh_surgery.cxx", options->flips, &options->flips, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-dolfin", "The mesh is a dolfin mesh", "mesh_surgery.cxx", options->dolfin, &options->dolfin, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-base_filename", "the base filename of the mesh used", "mesh_surgery.cxx", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"

int main (int argc, char * argv[]) {

  MPI_Comm comm;
  Options options;
  PetscErrorCode ierr;
  Mesh mesh;
  PetscFunctionBegin;
  try {
    ierr = PetscInitialize(&argc, &argv, (char *) 0, NULL);CHKERRQ(ierr);
    comm = PETSC_COMM_WORLD;
    double upper_coords[3] = {1.0, 1.0, 1.0};
    double lower_coords[3] = {0.0, 0.0, 0.0};
    int faces[3] = {1, 1, 1};
    ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
    ALE::Obj<ALE::Mesh> m;
    ALE::Obj<ALE::Mesh> bound_m;
    double refinementLimit;
    if (!options.generate) {
      m = ALE::PCICE::Builder::readMesh(comm, options.dim, options.baseFilename, options.useZeroBase, false, options.debug);
    } else {
      refinementLimit = 0.125;
      if (options.dim == 2) {
        bound_m = ALE::MeshBuilder::createSquareBoundary(PETSC_COMM_WORLD, upper_coords, lower_coords, faces);
      } else if (options.dim == 3) {
        refinementLimit = 0.01;
        bound_m = ALE::MeshBuilder::createCubeBoundary(PETSC_COMM_WORLD, upper_coords, lower_coords, faces);
      }
      
      m = ALE::Generator::refineMesh(ALE::Generator::generateMesh(bound_m, PETSC_FALSE), refinementLimit, PETSC_FALSE);
      
    }

    MeshCreate(comm, &mesh);CHKERRQ(ierr);
    MeshSetMesh(mesh, m);
    ierr = MeshIDBoundary(mesh);
    m->markBoundaryCells("marker");
    ALE::Obj<ALE::Mesh::label_sequence> vertices = m->depthStratum(0);
    ALE::Obj<ALE::Mesh::label_type> boundary = m->getLabel("marker");
    ALE::Mesh::label_sequence::iterator v_iter = vertices->begin();
    ALE::Mesh::label_sequence::iterator v_iter_end = vertices->end(); 
    ALE::Mesh::sieve_type::supportSet int_vertices = ALE::Mesh::sieve_type::supportSet();
    //calculate maxIndex
    ALE::Mesh::point_type maxIndex = -1;
    while (v_iter != v_iter_end) {
      if (*v_iter > maxIndex) maxIndex = *v_iter;
      if (m->getValue(boundary, *v_iter) != 1) {
        int_vertices.insert(*v_iter);
      } else {
        double curvature = Curvature(m, *v_iter);
        if (curvature < 0.5 && options.dim == 2) {
          //PetscPrintf(m->comm(), "Curvature: %f\n", curvature);
          int_vertices.insert(*v_iter);
        }
      }
      v_iter++;
    }
 
    ALE::Mesh::sieve_type::supportSet::iterator iv_iter = int_vertices.begin();
    ALE::Mesh::sieve_type::supportSet::iterator iv_iter_end = int_vertices.end();
    PetscPrintf(m->comm(), "Size of the mesh: %d vertices, %d faces\n", m->depthStratum(0)->size(), m->heightStratum(0)->size());
    ALE::Obj<ALE::Mesh::label_sequence> cells = m->heightStratum(0);
    ALE::Mesh::label_sequence::iterator c_iter = cells->begin();
    ALE::Mesh::label_sequence::iterator c_iter_end = cells->end();
    while (c_iter != c_iter_end) {
      if (*c_iter > maxIndex) maxIndex = *c_iter;
      c_iter++;
    }
    int nRemove = options.flips;
    int cur_Remove = 0;
    while (iv_iter != iv_iter_end && cur_Remove < nRemove) {
      cur_Remove++;
      PetscPrintf(m->comm(), "Deleting Vertex %d\n", *iv_iter);
      if (m->getDimension() == 2) {
        maxIndex = Surgery_2D_Remove_Vertex(m, *iv_iter, maxIndex);
      } else if (m->getDimension() == 3) {
        maxIndex = Surgery_3D_Remove_Vertex(m, *iv_iter, maxIndex);
      }
      //m->stratify();
      //PetscPrintf(m->comm(), "Size of the mesh: %d vertices, %d faces\n", m->depthStratum(0)->size(), m->heightStratum(0)->size());
      iv_iter++;
    }
    //next: the boundary! check the curvature and go from there
    m->stratify();
    PetscPrintf(m->comm(), "Size of the mesh: %d vertices, %d faces\n", m->depthStratum(0)->size(), m->heightStratum(0)->size());
    //check consistency:
    cells = m->heightStratum(0);
    c_iter = cells->begin();
    c_iter_end = cells->end();
    PetscPrintf(m->comm(), "Checking Consistency\n");
    while (c_iter != c_iter_end) {
      if ((int)m->getSieve()->cone(*c_iter)->size() != m->getDimension() + 1) {
        PetscPrintf(m->comm(), "Bad Cell: %d\n", *c_iter);
      }
      c_iter++;
    }
    PetscPrintf(m->comm(), "Writing VTK\n");
    PetscViewer viewer;
    PetscViewerCreate(m->comm(), &viewer);
    PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
    PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);
    PetscViewerFileSetName(viewer, "surgery_mesh.vtk");
    VTKViewer::writeHeader(viewer);
    VTKViewer::writeVertices(m, viewer);
    VTKViewer::writeElements(m, viewer);
    PetscViewerDestroy(viewer);
    PetscFunctionReturn(0);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
}
