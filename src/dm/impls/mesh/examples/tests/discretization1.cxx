static char help[] = "Discretization Tests.\n\n";

#include <petscmesh.h>
#include <Generator.hh>

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
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, Obj<ALE::Mesh>& mesh, Options *options)
{
  double lower[2] = {0.0, 0.0};
  double upper[2] = {1.0, 1.0};
  int    edges[2] = {2, 2};

  PetscFunctionBegin;
  Obj<ALE::Mesh> mB = ALE::MeshBuilder::createSquareBoundary(comm, lower, upper, edges, options->debug);
  mesh = ALE::Generator::generateMesh(mB, true);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MixedTest"
PetscErrorCode MixedTest(const Obj<ALE::Mesh>& mesh, Options *options)
{
  const Obj<ALE::Discretization> velocity = new ALE::Discretization(mesh->comm(), options->debug);
  const Obj<ALE::Discretization> pressure = new ALE::Discretization(mesh->comm(), options->debug);

  PetscFunctionBegin;
  pressure->setName("Pressure");
  pressure->setNumDof(0, 1);
  velocity->setName("Velocity");
  velocity->setNumDof(0, 2);
  velocity->setNumDof(1, 2);
  velocity->addChild(pressure);
  velocity->calculateIndices(mesh);
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
    Obj<ALE::Mesh> mesh;

    ierr = CreateMesh(comm, mesh, &options);CHKERRQ(ierr);
    ierr = MixedTest(mesh, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
