// Test conversion of flexible sieve into optimized sieve
#include<Mesh.hh>
#include<SieveBuilder.hh>
#include<petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "TestInterpolatedConversion"
PetscErrorCode TestInterpolatedConversion() {
  PetscInt debug = 0;
  ALE::Obj<PETSC_MESH_TYPE::sieve_type> sieve = new PETSC_MESH_TYPE::sieve_type(PETSC_COMM_WORLD, debug);
  PetscInt dim = 2;
  PetscInt numCells = 2;
  PetscInt numCorners = 3;
  PetscInt cells[2*3] = {0, 1, 2,  1, 3, 2};
  PetscInt numVertices = 4;
  bool interpolate = true;

  try {
    typedef ALE::Mesh<PetscInt, PetscScalar> FlexMesh;
    ALE::Obj<FlexMesh::sieve_type> s = new FlexMesh::sieve_type(sieve->comm(), sieve->debug());
    ALE::Obj<FlexMesh::arrow_section_type> orientation = new FlexMesh::arrow_section_type(sieve->comm(), sieve->debug());
    std::map<FlexMesh::point_type,FlexMesh::point_type> renumbering;

    s->setDebug(2);
    ALE::SieveBuilder<FlexMesh>::buildTopology(s, dim, numCells, cells, numVertices, interpolate, numCorners, -1, orientation);
    ALE::ISieveConverter::convertSieve(*s, *sieve, renumbering);
    ALE::ISieveConverter::convertOrientation(*s, *sieve, renumbering, orientation.ptr());
    sieve->view("Optimized Mesh");
  } catch(ALE::Exception e) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "ERROR: %s", e.message());
  }
}

int main(int argc, char **argv) {
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, 0, 0);CHKERRQ(ierr);
  ierr = TestInterpolatedConversion();CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return(0);
}
