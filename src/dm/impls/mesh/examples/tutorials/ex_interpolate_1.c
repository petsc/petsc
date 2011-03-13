//create a very simple 2D domain to test getInterpolation()

#include <Mesh.hh>
#include <Distribution.hh>
#include "petscmesh.h"
#include "petscviewer.h"
#include "../src/dm/mesh/meshpcice.h"
#include "../src/dm/mesh/meshpylith.h"
#include <stdlib.h>
#include <string.h>
#include <list>

using namespace ALE;

#undef __FUNCT__
#define __FUNCT__ "getPointInterpolation"
void getPointInterpolation(Mat A, ALE::Obj<ALE::Mesh> mesh, ALE::Mesh::point_type e, double * point) {
  const Obj<ALE::Mesh::real_section_type>& coordinates = mesh->getRealSection("coordinates");
  int dim = coordinates->getFiberDimension(0, *mesh->getTopology()->depthStratum(0, 0)->begin());
  double v0[dim], J[dim*dim], invJ[dim*dim], detJ;
  double w[dim];
  mesh->computeElementGeometry(coordinates, e, v0, J, invJ, detJ);
  for (int i = 0; i < dim; i++) {
    w[i] = 0;
    for (int j = 0; j < dim; j++) {
      w[i] += invJ[i*dim+j]*(point[j]-v0[j]);
    }
    w[i] = w[i] - 1;
  printf("%f, ", w[i]);
  }
  printf("\n");
 // const ALE::Obj<ALE::Mesh::order_type>& order = mesh->getFactory()->getGlobalOrder(mesh->getTopology(), 0, "default", coordinates->getAtlas());
}


#undef __FUNCT__
#define __FUNCT__ "getInterpolation"

PetscErrorCode getInterpolation(Obj<ALE::Mesh> mesh_fine, Obj<ALE::Mesh> mesh_coarse, Mat A) {
/*
 2) For each fine vertex:
    a) Find the coarse triangle it is in (traversal)
    b) Transform to get the reference coordinates of that vertex (geometry)
    c) Get vals = [\phi_0(v_ref), \phi_1(v_ref), \phi_2(v_ref)] (
    d) updateOperator(v_f, e_c, vals)
 For now it's hardcoded to use the standard P1 phi function.
*/

//a)
  //create a traversal label on both the meshes; allowing for this step to be done quickly.
  Obj<ALE::Mesh::topology_type> topology_fine = mesh_fine->getTopology();
  Obj<ALE::Mesh::topology_type> topology_coarse = mesh_coarse->getTopology();

  const Obj<ALE::Mesh::real_section_type>& coordinates_fine = mesh_fine->getRealSection("coordinates");
  int dim = coordinates_fine->getFiberDimension(0, *mesh_fine->getTopology()->depthStratum(0, 0)->begin());

  double * tmpCoords;
  double v0[dim], J[dim*dim], invJ[dim*dim], detJ;

  const Obj<ALE::Mesh::real_section_type>& coordinates_coarse = mesh_coarse->getRealSection("coordinates");

  Obj<ALE::Mesh::topology_type::label_sequence> vertices = topology_fine->depthStratum(0, 0);
  ALE::Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin();
  ALE::Mesh::topology_type::label_sequence::iterator v_iter_end = vertices->end();
  Obj<ALE::Mesh::topology_type::label_sequence> cells = topology_coarse->heightStratum(0, 0);
  ALE::Mesh::topology_type::label_sequence::iterator c_iter = cells->begin();
  ALE::Mesh::topology_type::label_sequence::iterator c_iter_end = cells->end();
  //dumbest thing right now; adapt tandem traversals to new and weird situation later.
  double v_coords[dim];
  double v_ref_coords[dim];
  while (v_iter != v_iter_end) {
    tmpCoords = (double *)coordinates_fine->restrict(0, *v_iter);
    for (int i = 0; i < dim; i++) {
      v_coords[i] = tmpCoords[i];
    }
    bool v_is_found = false;
    c_iter = cells->begin();
    while (c_iter != c_iter_end && !v_is_found) {
      mesh_coarse->computeElementGeometry(coordinates_coarse, *c_iter, v0, J, invJ, detJ);
      double coordsum = 0;
      for (int i = 0; i < dim; i++) {
        v_ref_coords[i] = 0;
        for (int j = 0; j < dim; j++) {
          v_ref_coords[i] += invJ[i*dim+j]*(v_coords[j]-v0[j]);
        }
        v_ref_coords[i] = v_ref_coords[i] - 1;
        coordsum += v_ref_coords[i];
      }
      bool isInElement = true;
      if (coordsum > 1/2) isInElement = false;
      for (int i = 0; i < dim; i++) {
        if (v_ref_coords[i] < -1) isInElement = false;
      }
      if (isInElement) {
        v_is_found = true;
        //update the operator! (TODO)
        printf("point %d is in %d at ( ", *v_iter, *c_iter);
        for (int i = 0; i < dim; i++) printf("%f ", v_ref_coords[i]);
        printf(")\n");
      }
      c_iter++;
    }
    if (!v_is_found) {
      printf("Operator will not include point %d\n", *v_iter);
    }
  v_iter++;
  }  
}

void forcedP1_phi(double * w, int dim, double * v) { //assume that v is within the reference element and we want to get the coefficients for it.
  if (dim == 2) {
    w[1] = 0.5*(v[0] + 1);
    w[2] = 0.5*(v[1] + 1);
    w[0] =  1 - w[1] - w[2];
  } else if (dim == 3) {
    //ask about the 3D reference element.
  }
}

#undef __FUNCT__
#define __FUNCT__ "main"


static char help[] = "tests getInterpolation\n\n";

int main(int argc, char * argv[]) {
  int dim = 2;
  double c[4][2];
  c[0][0] = -2.0; c[0][1] = 0.0;
  c[1][0] = 0.0; c[1][1] = 7.0;
  c[2][0] = 2.0; c[2][1] = 0.0;
  for (int i = 0; i <= dim; i++) {
    for (int j = 0; j < dim; j++) {
      c[3][j] += c[i][j];
    }
  }
  for (int i = 0; i < dim; i++) {
    c[3][i] = c[3][i]/((double)(dim + 1));
  }
  MPI_Comm comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *)0, help);
  comm = PETSC_COMM_WORLD;
  Obj<ALE::Mesh> mesh;
  mesh = new ALE::Mesh(comm, 2, 0);
  Obj<ALE::Mesh::sieve_type> sieve = new ALE::Mesh::sieve_type(mesh->comm(), 0);
  Obj<ALE::Mesh::topology_type> topology = new ALE::Mesh::topology_type(mesh->comm(), 0);
/*mesh
             o203
            /|
           / |
          /  |
         /   |
        /    |
       /     |
     102    103
     /       |
    /        |
   /    0    |
  /          |
 /           |
o202____101__o201

*/
  //set up the edges
  sieve->addArrow(101, 0);
  sieve->addArrow(102, 0);
  sieve->addArrow(103, 0);
  //set up the vertices
  sieve->addArrow(201, 101);
  sieve->addArrow(201, 102);
  sieve->addArrow(202, 102);
  sieve->addArrow(202, 103);
  sieve->addArrow(203, 103);
  sieve->addArrow(203, 101);
  sieve->stratify();
  topology->setPatch(0, sieve);
  topology->stratify();
  mesh->setTopology(topology);

/*mesh2
             o203
            /|
           /||
          /| |
         /106|
        / | 103
       /  |  |
     102 | 3 |
     / 2 o204|
    /   /\   |
   / 104 105 |
  / /   1  \ |
 //         \|
o202____101__o201

*/

  Obj<ALE::Mesh> mesh2;
  mesh2 = new ALE::Mesh(comm, 2, 0);
  Obj<ALE::Mesh::sieve_type> sieve2 = new ALE::Mesh::sieve_type(mesh->comm(), 0);
  Obj<ALE::Mesh::topology_type> topology2 = new ALE::Mesh::topology_type(mesh->comm(), 0);
  //set up the edges
  sieve2->addArrow(101, 1);
  sieve2->addArrow(102, 2);
  sieve2->addArrow(103, 3);
  sieve2->addArrow(104, 1);
  sieve2->addArrow(104, 2);
  sieve2->addArrow(105, 2);
  sieve2->addArrow(105, 3);
  sieve2->addArrow(106, 3);
  sieve2->addArrow(106, 1);
  //set up the vertices
  sieve2->addArrow(201, 101);
  sieve2->addArrow(201, 102);
  sieve2->addArrow(202, 102);
  sieve2->addArrow(202, 103);
  sieve2->addArrow(203, 103);
  sieve2->addArrow(203, 101);
  sieve2->addArrow(204, 104);
  sieve2->addArrow(202, 104);
  sieve2->addArrow(204, 105);
  sieve2->addArrow(201, 105);
  sieve2->addArrow(204, 106);
  sieve2->addArrow(203, 106);
  
  sieve2->stratify();
  topology2->setPatch(0, sieve2);
  topology2->stratify();
  mesh2->setTopology(topology2);


  Obj<ALE::Mesh::real_section_type> coordinates2 = mesh2->getRealSection("coordinates");
  coordinates2->setFiberDimensionByDepth(0, 0, 2);
  coordinates2->allocate();
  Obj<ALE::Mesh::real_section_type> coordinates = mesh->getRealSection("coordinates");
  coordinates->setFiberDimensionByDepth(0, 0, 2);
  coordinates->allocate();
  coordinates->update(0, 201, c[0]);
  coordinates2->update(0, 201, c[0]);
  coordinates->update(0, 201, c[0]);
  coordinates2->update(0, 202, c[1]);
  coordinates->update(0, 202, c[1]);
  coordinates2->update(0, 203, c[2]);
  coordinates->update(0, 203, c[2]);
  coordinates2->update(0, 204, c[3]);
  //double testpoint[2];
  //double interp[3];
  Mat A;
  getInterpolation(mesh2, mesh, A);

}
