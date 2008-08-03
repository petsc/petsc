static char help[] = "This example solves the problem specified by a UFC form.\n\n";

#define ALE_HAVE_CXX_ABI

#include <problem/Ex_UFC.hh>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  try {
    MPI_Comm                      comm  = PETSC_COMM_WORLD;
    ALE::Obj<ALE::Problem::Ex_UFC> ex_ufc = new ALE::Problem::Ex_UFC(comm);

    ierr = ex_ufc->createMesh();CHKERRQ(ierr);
    ierr = ex_ufc->createProblem();CHKERRQ(ierr);
    //throw ALE::Exception("Problem creation stop.");
    ierr = ex_ufc->createExactSolution();CHKERRQ(ierr);
    ierr = ex_ufc->checkError(ex_ufc->exactSolution());CHKERRQ(ierr);
    ierr = ex_ufc->checkResidual(ex_ufc->exactSolution());CHKERRQ(ierr);
    ierr = ex_ufc->createSolver();CHKERRQ(ierr);
    ierr = ex_ufc->solve();CHKERRQ(ierr);

  } catch(ALE::Exception e) {
    std::cerr << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
