static char help[] = "Testing the interface with Dolfin";

#include <petscmesh.hh>
#include <petscmesh_formats.hh>
#include <Selection.hh>

using ALE::Obj;

PetscErrorCode ConstructBoundary(const Obj<ALE::Mesh>& mesh, Obj<ALE::Mesh> boundary) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(mesh->comm(), "Build mesh boundary\n");
  boundary = ALE::Selection<ALE::Mesh>::submesh(mesh, mesh->getIntSection("traction"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  char           filename[2048];
  PetscInt       debug = 0;
  PetscTruth     doBoundary  = PETSC_FALSE;
  PetscTruth     interpolate = PETSC_FALSE;
  PetscTruth     view        = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Dolfin Options", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "dolfin.cxx", debug, &debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscStrcpy(filename, "triangle.xml");CHKERRQ(ierr);
    ierr = PetscOptionsString("-filename", "The mesh filename", "dolfin.cxx", filename, filename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-boundary", "Construct the boundary mesh", "dolfin.cxx", doBoundary, &doBoundary, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Interpolate the mesh", "dolfin.cxx", interpolate, &interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-view", "Interpolate the mesh", "dolfin.cxx", view, &view, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  Obj<ALE::Mesh>             mesh  = new ALE::Mesh(PETSC_COMM_WORLD, 2, debug);
  Obj<ALE::Mesh::sieve_type> sieve = new ALE::Mesh::sieve_type(PETSC_COMM_WORLD, debug);

  try {
    std::cout << "Reading mesh from file " << filename << std::endl;
    mesh->setSieve(sieve);
    ALE::Dolfin::Builder::readMesh(mesh, filename);
    if (view) {mesh->view("");}
    if (interpolate) {
      mesh = ALE::Selection<ALE::Mesh>::interpolateMesh(mesh);
      if (view) {mesh->view("");}
    }
    if (doBoundary) {
      Obj<ALE::Mesh> boundary;

      ierr = ConstructBoundary(mesh, boundary);CHKERRQ(ierr);
      if (view) {boundary->view("");}
    }
  } catch(ALE::Exception e) {
    std::cout << "ERROR: " << e.msg() << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
