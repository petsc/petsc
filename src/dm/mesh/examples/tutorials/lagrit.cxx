static char help[] = "Testing the interface with LaGriT";

#include <petscmesh.hh>
#include <petscmesh_formats.hh>
#include <Selection.hh>

#include "CohesiveTopology.hh"

using ALE::Obj;

PetscErrorCode ConstructBoundary(const Obj<ALE::Mesh>& mesh, Obj<ALE::Mesh> boundary) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(mesh->comm(), "Build mesh boundary\n");
  boundary = ALE::Selection<ALE::Mesh>::submesh(mesh, mesh->getIntSection("boundary"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  char           filename[2048];
  char           pset[2048];
  std::string    psetFilename;
  PetscInt       debug = 0;
  PetscTruth     doBoundary  = PETSC_FALSE;
  PetscTruth     interpolate = PETSC_FALSE;
  PetscTruth     split       = PETSC_FALSE;
  PetscTruth     view        = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "LaGriT Options", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "lagrit.cxx", debug, &debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscStrcpy(filename, "triangle.xml");CHKERRQ(ierr);
    ierr = PetscOptionsString("-filename", "The mesh filename", "lagrit.cxx", filename, filename, 2048, PETSC_NULL);CHKERRQ(ierr);
    pset[0] = '\0';
    ierr = PetscOptionsString("-pset", "The fault pset filename", "lagrit.cxx", pset, pset, 2048, PETSC_NULL);CHKERRQ(ierr);
    psetFilename = pset;
    ierr = PetscOptionsTruth("-boundary", "Construct the boundary mesh", "lagrit.cxx", doBoundary, &doBoundary, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Interpolate the mesh", "lagrit.cxx", interpolate, &interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-split", "Split the mesh along a fault", "lagrit.cxx", split, &split, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-view", "View the mesh", "lagrit.cxx", view, &view, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  try {
    std::cout << "Reading mesh from file " << filename << std::endl;
    Obj<ALE::Mesh> mesh = ALE::LaGriT::Builder::readMesh(PETSC_COMM_WORLD, 3, filename, interpolate, debug);
    if (psetFilename != "") {
      if (view) {mesh->view("");}
      ALE::LaGriT::Builder::readFault(mesh, psetFilename);
      if (split) {
        ALE::Obj<ALE::Mesh> fault;

        pylith::faults::CohesiveTopology::create(&fault, mesh, mesh->getIntSection("pfault_no4"), 42, true);
        if (view) {
          fault->view("Fault Mesh");
          mesh->view("Cohesive Mesh");
        }
      }
    } else if (view) {
      mesh->view("");
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
