static char help[] = "Section Boundary Condition Tests.\n\n";

#include <petscmesh.h>

using ALE::Obj;

typedef struct {
  int debug; // The debugging level
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug = 0;

  ierr = PetscOptionsBegin(comm, "", "Options for section boundary condition test", "Section");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "section2.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GeneralSectionTest"
PetscErrorCode GeneralSectionTest(MPI_Comm comm, Options *options)
{
  typedef ALE::GeneralSection<int, PetscReal> GeneralSection;
  const Obj<GeneralSection> section = new GeneralSection(comm, options->debug);

  PetscFunctionBegin;
  for(int p = 0; p < 9; ++p) {
    section->setFiberDimension(p, 3);
    if (p%3 == 2) {
      section->setConstraintDimension(p, 1);
    }
  }
  section->allocatePoint();
  for(int p = 0; p < 9; ++p) {
    if (p%3 == 2) {
      int dof = p/3;

      section->setConstraintDof(p, &dof);
    }
  }
  for(int p = 0; p < 9; ++p) {
    if (p%3 == 2) {
      PetscReal values[2];

      for(int v = 0, i = -1; v < 3; ++v) {
        if (v == p/3) continue;
        values[++i] = p*3+v;
      }
      section->updateFreePoint(p, values);
      values[0] = -(p*3+p/3);
      section->updatePointBC(p, values);
    } else {
      PetscReal values[3];

      for(int v = 0; v < 3; ++v) {
        values[v] = p*3+v;
      }
      section->updatePoint(p, values);
    }
  }
  section->view("Constrained section");
  // Map this to a vector and back
  //   Create a mesh
  Vec        localVec, vecIn, vecOut;
  VecScatter scatter;
  Obj<ALE::Mesh::real_section_type> newSection = new ALE::Mesh::real_section_type(section->comm(), section->debug());
  Obj<ALE::Mesh>                    m          = new ALE::Mesh(section->comm(), 1, section->debug());
  const Obj<ALE::Mesh::sieve_type>  sieve      = new ALE::Mesh::sieve_type(m->comm(), m->debug());
  PetscErrorCode ierr;

  newSection->setAtlas(section->getAtlas());
  newSection->allocateStorage();
  newSection->copyBC(section);
  newSection->view("Before");

  m->setSieve(sieve);
  for(int p = 0; p < 9; ++p) sieve->addArrow(p, 9, p);
  m->stratify();

  const Obj<ALE::Mesh::order_type>& order      = m->getFactory()->getGlobalOrder(m, "default", section);
  ierr = VecCreate(section->comm(), &vecIn);CHKERRQ(ierr);
  ierr = VecSetSizes(vecIn, order->getLocalSize(), order->getGlobalSize());CHKERRQ(ierr);
  ierr = VecSetFromOptions(vecIn);CHKERRQ(ierr);
  ierr = VecDuplicate(vecIn, &vecOut);CHKERRQ(ierr);

  ierr = MeshCreateGlobalScatter(m, section, &scatter);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1,section->sizeWithBC(), section->restrictSpace(), &localVec);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter, localVec, vecIn, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter, localVec, vecIn, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecDestroy(&localVec);CHKERRQ(ierr);
  ierr = VecView(vecIn, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecCopy(vecIn, vecOut);CHKERRQ(ierr);
  ierr = VecView(vecOut, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1,newSection->sizeWithBC(),  newSection->restrictSpace(), &localVec);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter, vecOut, localVec, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter, vecOut, localVec, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecDestroy(&localVec);CHKERRQ(ierr);
  ierr = VecDestroy(&vecIn);CHKERRQ(ierr);
  ierr = VecDestroy(&vecOut);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
  newSection->view("After");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Options        options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
  try {
    ierr = GeneralSectionTest(comm, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
