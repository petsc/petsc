//Just a file full of functions for ex_coarsen
 
#include <list>
#include <Distribution.hh>
#include "petscmesh.h"
#include "petscviewer.h"
#include "src/dm/mesh/meshpcice.h"
#include "src/dm/mesh/meshpylith.h"
#include <stdlib.h>
#include <string.h>
#include <string>
#include <triangle.h>
#include <tetgen.h>

using ALE::Obj;

char baseFile[2048]; //stores the base file name.
double c_factor, r_factor;


EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshView_Sieve(const Obj<ALE::Mesh>&, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT FieldView_Sieve(const Obj<ALE::Mesh>&, const std::string&, PetscViewer);
PetscErrorCode CreateMesh(MPI_Comm, Obj<ALE::Mesh>&);
PetscErrorCode OutputVTK(const Obj<ALE::Mesh>&, std::string, std::string, bool cell);
PetscErrorCode OutputMesh(const Obj<ALE::Mesh>&);
PetscErrorCode GenerateMesh (MPI_Comm, Obj<ALE::Mesh>& mesh);

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm)
{
  PetscErrorCode ierr;

  ierr = PetscStrcpy(baseFile, "data/ex1_2d");CHKERRQ(ierr);
  c_factor = 2; //default
  r_factor = 0;
  ierr = PetscOptionsBegin(comm, "", "Options for mesh loading", "DMMG");CHKERRQ(ierr);
//    ierr = PetscOptionsInt("-debug", "The debugging level", "ex1.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_file", "The base filename for mesh files", "ex_coarsen", "ex_coarsen", baseFile, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-coarsen", "The coarsening factor", "ex_coarsen.c", c_factor, &c_factor, PETSC_NULL);    
    ierr = PetscOptionsReal("-generate", "Generate the mesh with refinement limit placed after this.", "ex_coarsen.c", r_factor, &r_factor, PETSC_NULL);
  ierr = PetscOptionsEnd();
 PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, Obj<ALE::Mesh>& mesh)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
  ALE::LogStagePush(stage);
  ierr = PetscPrintf(comm, "Creating mesh\n");CHKERRQ(ierr);
if (r_factor <= 0.0) {
    mesh = ALE::PCICE::Builder::readMesh(comm, 2, baseFile, true, false, false);
  } else {
    //mesh = new ALE::Mesh(comm, 2, 0);
    GenerateMesh(comm, mesh);
  }
  ALE::LogStagePop(stage);
  Obj<ALE::Mesh::topology_type> topology = mesh->getTopologyNew();
  ierr = PetscPrintf(comm, "  Read %d elements\n", topology->heightStratum(0, 0)->size());CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  Read %d vertices\n", topology->depthStratum(0, 0)->size());CHKERRQ(ierr);
  if (0) {
    topology->view("Serial topology");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputVTK"
PetscErrorCode OutputVTK(const Obj<ALE::Mesh>& mesh, std::string filename, std::string field, bool cell)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
    ALE::LogStage stage = ALE::LogStageRegister("VTKOutput");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(mesh->comm(), "Creating VTK mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename.c_str());CHKERRQ(ierr);
    ierr = MeshView_Sieve(mesh, viewer);CHKERRQ(ierr);
    if (cell) {ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);}
    else {ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);}
    ierr = FieldView_Sieve(mesh, field.c_str(), viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    ALE::LogStagePop(stage);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputMesh"
PetscErrorCode OutputMesh(const Obj<ALE::Mesh>& mesh)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
    ALE::LogStage stage = ALE::LogStageRegister("MeshOutput");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(mesh->comm(), "Creating original format mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_PCICE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "testMesh.lcon");CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscExceptionTry1(PetscViewerFileSetName(viewer, "testMesh"), PETSC_ERR_FILE_OPEN);
        if (PetscExceptionValue(ierr)) {
          /* this means that a caller above me has also tryed this exception so I don't handle it here, pass it up */
        } else if (PetscExceptionCaught(ierr, PETSC_ERR_FILE_OPEN)) {
          ierr = 0;
        } 
        CHKERRQ(ierr);
    ierr = MeshView_Sieve(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    ALE::LogStagePop(stage);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TriangleToMesh"

PetscErrorCode TriangleToMesh(Obj<ALE::Mesh> mesh, triangulateio * src, ALE::Mesh::section_type::patch_type patch) {
   //try to keep in the same labeled order we're given
   //First create arrows from the faces to the edges, then edges to vertices (easy part given triangle)
  //ALE::Mesh::section_type::value_type rank = mesh->commRank();

  PetscErrorCode ierr;
  //PetscInt order = 0;
  Obj<ALE::Mesh::sieve_type> sieve = new ALE::Mesh::sieve_type(mesh->comm(), 0);
  Obj<ALE::Mesh::topology_type> topology = new ALE::Mesh::topology_type(mesh->comm(), 0);;

  //create arrows between the edges and their covering vertices.

/*  for (int i = 0; i < src->numberofedges; i++) {
        //ierr = PetscPrintf(PETSC_COMM_WORLD, "Adding Arrow: Point %d to Edge %d\n", src->edgelist[2*i], src->numberofpoints + i);CHKERRQ(ierr);
        //ierr = PetscPrintf(PETSC_COMM_WORLD, "Adding Arrow: Point %d to Edge %d\n", src->edgelist[2*i+1], src->numberofpoints + i);CHKERRQ(ierr);
  }

    int j = 0;
    while(j < src->numberofedges) {
	sieve->addArrow(src->edgelist[2*j]+src->numberoftriangles, src->numberofpoints + src->numberoftriangles + j,  src->numberofpoints + src->numberoftriangles + j);
	sieve->addArrow(src->edgelist[2*j+1]+src->numberoftriangles, src->numberofpoints + src->numberoftriangles + j,  src->numberofpoints + src->numberoftriangles + j);
      for (int i = 0; i < src->numberoftriangles; i++) {
      	for (int k = 0; k < 3; k++) {  //triangle edge index, 0 is corner 0 to 1, 1 is corner 1 to 2, 2 is corner 2 to 0; also look for reverse.
		if 	(((src->trianglelist[3*i + (k%3)] == src->edgelist[2*j]) && (src->trianglelist[3*i + ((k + 1)%3)] == src->edgelist[2*j + 1])) ||
			 ((src->trianglelist[3*i + (k%3)] == src->edgelist[2*j + 1]) && (src->trianglelist[3*i + ((k + 1)%3)] == src->edgelist[2*j]))) {
			sieve->addArrow(src->numberoftriangles + src->numberofpoints + j, i, k);
                        //ierr = PetscPrintf(PETSC_COMM_WORLD, "Adding Arrow: Edge %d to Triangle %d\n", src->numberofpoints + j, src->numberofpoints + src->numberoPetscErrorCode GenerateMesh (Obj<ALE::Mesh>& mesh) {fedges + i);CHKERRQ(ierr);
		}
	}
    }
    j++;  //check the next edge!
  }

*/

//REINVENTING THE WHEEL ABOVE: NOW JUST BUILD THE SUCKER.
//make the sieve and the topology actually count for something
ALE::New::SieveBuilder<ALE::Mesh::sieve_type>::buildTopology(sieve, 2, src->numberoftriangles, src->trianglelist, src->numberofpoints, false, 3);
  sieve->stratify();
  topology->setPatch(patch, sieve);
  topology->stratify();
  mesh->setTopologyNew(topology);
    int nvertices = topology->depthStratum(patch, 0)->size();
    int nedges = topology->heightStratum(patch, 1)->size();
    int ncells = topology->heightStratum(patch, 0)->size();
    ierr = PetscPrintf(mesh->comm(), "NEW MESH: %d vertices, %d edges, %d cells\n", nvertices, nedges, ncells);
  //create the coordinate section and dump in the coordinates.
  Obj<ALE::Mesh::section_type> coordinates = mesh->getSection("coordinates");
  coordinates->getAtlas()->setFiberDimensionByDepth(patch, 0, 2);  //puts two doubles on each node.
  coordinates->getAtlas()->orderPatches();
  coordinates->allocate();
  const Obj<ALE::Mesh::topology_type::label_sequence>& vertices = coordinates->getAtlas()->getTopology()->depthStratum(patch, 0);
  
	ALE::Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin();
	ALE::Mesh::topology_type::label_sequence::iterator v_iter_end = vertices->end();
	while(v_iter != v_iter_end) {
		coordinates->update(patch, *v_iter, src->pointlist+2*(*v_iter-src->numberoftriangles));
		v_iter++;
	}

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GenerateMesh"

//just create a frickin' square boundary, then interpolate on it.
/*
 1-(5)-2
 \     \
(4)   (6)
 \     \
 0-(7)-3
*/
PetscErrorCode GenerateMesh (MPI_Comm comm, Obj<ALE::Mesh>& mesh) {

  PetscFunctionBegin
  Obj<ALE::Mesh> boundary = new ALE::Mesh(comm, 2, 0);
  PetscScalar coords[8] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0};
  const ALE::Mesh::topology_type::patch_type patch = 0;
  const Obj<ALE::Mesh::sieve_type>    sieve    = new ALE::Mesh::sieve_type(comm, 0);
  const Obj<ALE::Mesh::topology_type> topology = new ALE::Mesh::topology_type(comm, 0);
  
  sieve->addArrow(0, 4, 0);
  sieve->addArrow(1, 4, 1);

  sieve->addArrow(1, 5, 2);
  sieve->addArrow(2, 5, 3);

  sieve->addArrow(2, 6, 4);
  sieve->addArrow(3, 6, 5);

  sieve->addArrow(3, 7, 6);
  sieve->addArrow(0, 7, 7);

  sieve->stratify();
  topology->setPatch(patch, sieve);
  topology->stratify();
  boundary->setTopologyNew(topology);
  ALE::PyLith::Builder::buildCoordinates(boundary->getSection("coordinates"), 2, coords);
  const Obj<ALE::Mesh::topology_type::patch_label_type>& markers = topology->createLabel(patch, "marker");

  for(int v = 0; v < 4; v++) {
      topology->setValue(markers, v, 1);
  }
  for(int e = 4; e < 8; e++) {
      topology->setValue(markers, e, 1);
  }
  mesh = ALE::Mesh(comm, 2, 0);
  //mesh = ALE::Generator::generateMesh(boundary, 0);
  
  //mesh = ALE::Generator::refineMesh(mesh, r_factor, 0);
  mesh->getTopologyNew()->view("Serial topology");
  PetscFunctionReturn(0);
}


