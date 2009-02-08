//Just a file full of functions for ex_coarsen
 
#include <list>
#include <Distribution.hh>
#include "petscmesh.h"
#include "petscviewer.h"
#include "../src/dm/mesh/meshpcice.h"
#include "../src/dm/mesh/meshpylith.h"
#include <stdlib.h>
#include <string.h>
#include <string>
#include <triangle.h>
#include <tetgen.h>

using ALE::Obj;

char baseFile[2048]; //stores the base file name.
double c_factor, r_factor;
int debug;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshView_Sieve(const Obj<ALE::Mesh>&, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT FieldView_Sieve(const Obj<ALE::Mesh>&, const std::string&, PetscViewer);
PetscErrorCode CreateMesh(MPI_Comm, Obj<ALE::Mesh>&);
PetscErrorCode OutputVTK(const Obj<ALE::Mesh>&, std::string, std::string, bool cell);
PetscErrorCode OutputMesh(const Obj<ALE::Mesh>&);
PetscErrorCode GenerateMesh (MPI_Comm, Obj<ALE::Mesh>&, double);
PetscErrorCode IdentifyBoundary(Obj<ALE::Mesh>&, int);

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm)
{
  PetscErrorCode ierr;

  ierr = PetscStrcpy(baseFile, "data/ex1_2d");CHKERRQ(ierr);
  c_factor = 2; //default
  r_factor = 0;
  debug = 0;
  ierr = PetscOptionsBegin(comm, "", "Options for mesh loading", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "ex1.c", debug, &debug, PETSC_NULL);CHKERRQ(ierr);
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
    mesh = ALE::PCICE::Builder::readMesh(comm, 2, baseFile, true, true, debug);
    IdentifyBoundary(mesh, 2);
  } else {
    mesh = new ALE::Mesh(comm, 2, 0);
    GenerateMesh(comm, mesh, r_factor);
  }
  ALE::LogStagePop(stage);
  Obj<ALE::Mesh::topology_type> topology = mesh->getTopology();
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
  mesh->setTopology(topology);
  int nvertices = topology->depthStratum(patch, 0)->size();
  int nedges = topology->depthStratum(patch, 1)->size();
  int ncells = topology->heightStratum(patch, 0)->size();
  ierr = PetscPrintf(mesh->comm(), "NEW MESH: %d vertices, %d edges, %d cells\n", nvertices, nedges, ncells);
  //create the coordinate section and dump in the coordinates.  At the same time set the boundary markers.
  Obj<ALE::Mesh::section_type> coordinates = mesh->getSection("coordinates");
  coordinates->setFiberDimensionByDepth(patch, 0, 2);  //puts two doubles on each node.
  coordinates->allocate();
  const Obj<ALE::Mesh::topology_type::label_sequence>& vertices = coordinates->getTopology()->depthStratum(patch, 0);
  
  ALE::Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin();
  ALE::Mesh::topology_type::label_sequence::iterator v_iter_end = vertices->end();
  const Obj<ALE::Mesh::topology_type::patch_label_type>& markers = topology->createLabel(patch, "marker");
  while(v_iter != v_iter_end) {
    topology->setValue(markers, *v_iter, src->pointmarkerlist[*v_iter - src->numberoftriangles]);
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
PetscErrorCode GenerateMesh (MPI_Comm comm, Obj<ALE::Mesh>& mesh, double ref_lim) {


   PetscFunctionBegin;
//create the previous boundary set, feed into triangle with the boundaries marked.
   triangulateio * input = new triangulateio;
 //  triangulateio * ioutput = new triangulateio;
   triangulateio * output = new triangulateio;

//set up input

   input->numberofpoints = 4;
   input->numberofpointattributes = 0;
   double coords[8] = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
   input->pointlist = coords;
   input->numberofpointattributes = 0;
   input->pointattributelist = NULL;
   int pointmarks[4] = {1,1,1,1};
   input->pointmarkerlist = pointmarks; //mark as boundaries

   input->numberoftriangles = 0;
   input->numberofcorners = 0;
   input->numberoftriangleattributes = 0;
   input->trianglelist = NULL;
   input->triangleattributelist = NULL;
   input->trianglearealist = NULL;

   int segments[8] = {0, 1, 1, 2, 2, 3, 3, 0};
   input->segmentlist = segments;
   int segmentmarks[4] = {1, 1, 1, 1};
   input->segmentmarkerlist = segmentmarks;  //mark as boundaries.
   input->numberofsegments = 4;

   input->holelist = NULL;
   input->numberofholes = 0;
   
   input->regionlist = NULL;
   input->numberofregions = 0;

//set up output

   output->pointlist = NULL;
   output->pointattributelist = NULL;
   output->pointmarkerlist = NULL;
   output->trianglelist = NULL;
   output->triangleattributelist = NULL;
   output->trianglearealist = NULL;
   output->neighborlist = NULL;
   output->segmentlist = NULL;
   output->segmentmarkerlist = NULL;
   output->holelist = NULL;
   output->regionlist = NULL;
   output->edgelist = NULL;
   output->edgemarkerlist = NULL;
   output->normlist = NULL;

   char triangleOptions[256]; 
   sprintf(triangleOptions, "-zeQa%f",ref_lim);
   triangulate(triangleOptions, input, output, NULL); //refine

   //for(int i = 0; i < output->numberofpoints; i++) {
   //  printf("%d", output->pointmarkerlist[i]);
   //}
   //printf("\n");

   TriangleToMesh(mesh, output, 0);

   PetscFunctionReturn(0);

}

//identify the boundary points and edges on an interpolated mesh by looking for the number of elements covered by edges (or faces in 3D).

PetscErrorCode IdentifyBoundary(Obj<ALE::Mesh>& mesh, int dim) {

   PetscFunctionBegin;

if (dim == 2) {
     ALE::Mesh::section_type::patch_type patch = 0;
     Obj<ALE::Mesh::topology_type> topology = mesh->getTopology();
     const Obj<ALE::Mesh::topology_type::label_sequence>& edges = topology->heightStratum(patch, 1);
     const Obj<ALE::Mesh::topology_type::label_sequence>& vertices = topology->depthStratum(patch, 0);


     ALE::Mesh::topology_type::label_sequence::iterator e_iter = edges->begin();
     ALE::Mesh::topology_type::label_sequence::iterator e_iter_end = edges->end();

     ALE::Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin();
     ALE::Mesh::topology_type::label_sequence::iterator v_iter_end = vertices->end();

     const Obj<ALE::Mesh::topology_type::patch_label_type>& markers = topology->createLabel(patch, "marker");

//initialize all the vertices

     while (v_iter != v_iter_end) {
        topology->setValue(markers, *v_iter, 0);
        v_iter++;
     }

//trace through the edges, initializing them to be non-boundary, then setting them as boundary.

int boundEdges = 0, boundVerts = 0;

    // int nBoundaryVertices = 0;
     while (e_iter != e_iter_end) {
       topology->setValue(markers, *e_iter, 0);
//find out if the edge is not supported on both sides, if so, this is a boundary node
       //printf("Edge %d supported by %d faces", *e_iter, topology->getPatch(patch)->support(*e_iter)->size());
       if (topology->getPatch(patch)->support(*e_iter)->size() < 2) {
        topology->setValue(markers, *e_iter, 1);
        boundEdges++;
        ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence> endpoints = topology->getPatch(patch)->cone(*e_iter); //the adjacent elements
        ALE::Mesh::sieve_type::traits::coneSequence::iterator p_iter = endpoints->begin();
        ALE::Mesh::sieve_type::traits::coneSequence::iterator p_iter_end = endpoints->end();
        while (p_iter != p_iter_end) {
           topology->setValue(markers, *p_iter, 1);
           boundVerts++;
           p_iter++;
        }
       }
      e_iter++;
   }
  printf("Boundary Edges: %d, Boundary Vertices: %d\n", boundEdges, boundVerts);
  } 
  PetscFunctionReturn(0);
}
