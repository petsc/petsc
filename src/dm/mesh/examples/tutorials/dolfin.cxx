static char help[] = "Testing the interface with Dolfin";

#include <petscmesh.hh>
#include <petscmesh_formats.hh>

using ALE::Obj;

int main(int argc, char **argv)
{
  char           filename[2048];
  PetscInt       debug = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Dolfin Options", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "dolfin.cxx", debug, &debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscStrcpy(filename, "triangle.xml");CHKERRQ(ierr);
    ierr = PetscOptionsString("-filename", "The mesh filename", "dolfin.cxx", filename, filename, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  Obj<ALE::Mesh>             mesh  = new ALE::Mesh(PETSC_COMM_WORLD, 2, debug);
  Obj<ALE::Mesh::sieve_type> sieve = new ALE::Mesh::sieve_type(PETSC_COMM_WORLD, debug);

  try {
    std::cout << "Reading mesh from file " << filename << std::endl;
    mesh->setSieve(sieve);
    ALE::Dolfin::Builder::readMesh(mesh, filename);
    mesh->view("");
  } catch(ALE::Exception e) {
    std::cout << "ERROR: " << e.msg() << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
