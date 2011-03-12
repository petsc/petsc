static char help[] = "Meshing Tests. \n\n";

#include "Meshing.hh"

#include <petscmesh_viewers.hh>

using ALE::Obj;

typedef struct {
  int debug;
  int dim;
  PetscBool  interpolate;
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"

PetscErrorCode ProcessOptions(MPI_Comm comm, Options * options) {


  PetscErrorCode ierr;
  PetscFunctionBegin;
  options->debug = 0;
  options->dim = 2;
  options->interpolate = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Options for meshing testing", "Meshing");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "meshing.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The embedded dimension", "meshing.cxx", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "construct (don't eliminate) intermediate elements", "meshing.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSimplex"

PetscErrorCode CreateSimplex(MPI_Comm comm, ALE::Obj<ALE::Mesh> &m, Options * options) {
  m = ALE::Mesh(comm, options->dim, options->debug);
  ALE::Obj<ALE::Mesh::sieve_type> s = new ALE::Mesh::sieve_type();
  m->setSieve(s);
  ALE::Mesh::point_type current_max_point = options->dim+2;
  double * coord_template = new double[options->dim*options->dim+1];
  s->addCapPoint(0);
  m->stratify();
  for (int curdim = 0; curdim < options->dim; curdim++) {
    s->addCapPoint(curdim+1);
    m->stratify();
    ALE::Mesh::point_type current_cone_base = 0;
    if (curdim >= 1) {
      current_cone_base = *m->heightStratum(0)->begin();
    }
    Meshing_ConeConstruct(m, curdim+1, current_cone_base, current_max_point);
    //test join
    PetscPrintf(m->comm(), "JOIN TEST: join(%d, %d) size: %d\n", curdim, curdim+1, s->nJoin(curdim, curdim+1, m->getDimension())->size());
    //current_max_point = Meshing_ConeConstruct(m, curdim+1, current_cone_base, current_max_point);
    m->stratify();
    for (int i = 0; i <= curdim+1; i++) {
      PetscPrintf(comm, "%d %d-cells in the mesh\n", m->depthStratum(i)->size(), i);
    }
    m->view("simplex mesh", comm);
  }
  
  delete coord_template;
}

#undef __FUNCT__
#define __FUNCT__ "SimpleTest"

void SimpleTest(MPI_Comm comm, Options * options) {
  ALE::Obj<ALE::Mesh>m = ALE::Mesh(comm, options->dim, options->debug);
  ALE::Obj<ALE::Mesh::sieve_type> s = new ALE::Mesh::sieve_type();
  m->setSieve(s);
  //s->addCapPoint(0);
  //s->addCapPoint(1);
  s->addArrow(0, 4);
  s->addArrow(1, 4);
  s->view("simple sieve", comm);
  PetscPrintf(m->comm(), "JOIN TEST: join(%d, %d) size: %d\n", 0, 1, s->nJoin(0, 1, options->dim)->size());
}


#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc, char * argv[]) {
  MPI_Comm comm;
  Options options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
  try {
    SimpleTest(comm, &options);
    ALE::Obj<ALE::Mesh> m;
    CreateSimplex(comm, m, &options);

  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
}
