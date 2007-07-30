#include <petscmesh.hh>
#include <Selection.hh>

typedef ALE::Selection<ALE::Mesh> selection;
using ALE::Obj;

class Processor {
public:
  void operator()(const Obj<selection::PointArray>& subset) {
    std::cout << "  subset:";
    for(selection::PointArray::const_iterator p_iter = subset->begin(); p_iter != subset->end(); ++p_iter) {
      std::cout << " " << *p_iter;
    }
    std::cout << std::endl;
  };
};

int main(int argc, char *argv[]) {
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, 0);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  try {
    Obj<selection::PointArray> set = new selection::PointArray();
    Processor                  processor;

    for(int i = 0; i < 15; ++i) if (i%3) set->push_back(i);
    selection::subsets(set, 3, processor);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
