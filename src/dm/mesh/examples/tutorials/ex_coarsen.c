/*T
   Concepts: Creating a series of coarsened meshes using function-based coarsening and Triangle (or TetGen).
   Processors: n
T*/

/*

*/

static char help[] = "Reads, partitions, and outputs an unstructured mesh.\n\n";

#include "ex_coarsen.h"

PetscErrorCode CreateSpacingFunction(Obj<ALE::Mesh>);
PetscErrorCode CreateCoarsenedMesh(Obj<ALE::Mesh>, Obj<ALE::Mesh>, double);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  try {
    Obj<ALE::Mesh> mesh;
    ierr = ProcessOptions(comm);CHKERRQ(ierr);
    ierr = CreateMesh(comm, mesh);CHKERRQ(ierr);
    Obj<ALE::Mesh> coarseMesh = ALE::Mesh(comm, 2, 0);	
    ierr = CreateSpacingFunction(mesh);CHKERRQ(ierr);
    ierr = CreateCoarsenedMesh(mesh, coarseMesh, c_factor);CHKERRQ(ierr);
    ierr = CreateSpacingFunction(coarseMesh);CHKERRQ(ierr);

//        mesh->getTopologyNew()->view("Serial topology");
//	coarseMesh->getTopologyNew()->view("Serial topology");
//    ierr = OutputVTK(mesh, "notCoarseMesh.vtk", "spacing", 0); CHKERRQ(ierr);
    ierr = OutputVTK(coarseMesh, "coarseMesh.vtk", "spacing", 0);CHKERRQ(ierr);
    
//    ierr = OutputMesh(coarseMesh);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSpacingFunction"

PetscErrorCode CreateSpacingFunction(Obj<ALE::Mesh> mesh) {


//grab a new section for the spacing function
  Obj<ALE::Mesh::topology_type> topology = mesh->getTopologyNew();

  Obj<ALE::Mesh::section_type>        spacing = mesh->getSection("spacing");
  ALE::Mesh::section_type::patch_type patch = 0;
  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN;
  spacing->getAtlas()->setFiberDimensionByDepth(patch, 0, 1);
  spacing->getAtlas()->orderPatches();
  spacing->allocate();
  const Obj<ALE::Mesh::topology_type::label_sequence>& vertices = spacing->getAtlas()->getTopology()->depthStratum(patch, 0);


//grab the coordinate section

  Obj<ALE::Mesh::section_type> coords = mesh->getSection("coordinates");
	//ierr = PetscPrintf(mesh->comm(), "Spacing Values: ");CHKERRQ(ierr);
   ALE::Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin();
   ALE::Mesh::topology_type::label_sequence::iterator v_iter_end = vertices->end();
	while(v_iter != v_iter_end) {

	    ALE::Obj<ALE::Mesh::sieve_type::traits::supportSequence> support = topology->getPatch(patch)->support(*v_iter);

	    //ierr = PetscPrintf(mesh->comm(), "Edges: %d\n", support->size());CHKERRQ(ierr);
            //ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence> tCone = topology->getPatch(patch)->cone(support);
    		const double *vCoords = coords->restrict(patch, *v_iter);
		double v_x = vCoords[0], v_y = vCoords[1];
    		double        minDist = 10000.;

    ALE::Mesh::topology_type::label_sequence::iterator s_iter = support->begin();
    ALE::Mesh::topology_type::label_sequence::iterator s_iter_end = support->end();
    while(s_iter != s_iter_end) {
      ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence> neighbors = topology->getPatch(patch)->cone(*s_iter);
	    //ierr = PetscPrintf(mesh->comm(), "Edge Cardinality (This is stupid): %d, %d\n", neighbors->size(), *neighbors->end());CHKERRQ(ierr);
    ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter = neighbors->begin();
    ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter_end = neighbors->end();
    while(n_iter != n_iter_end) {
        if (*v_iter != *n_iter) {
          const double *nCoords = coords->restrict(patch, *n_iter);
	  double n_x = nCoords[0], n_y = nCoords[1];
          double        dist    = 0.0;

	  dist = (n_x - v_x)*(n_x - v_x) + (v_y - n_y)*(v_y - n_y);
//          for(int d = 0; d < dim; d++) {
//            dist += (vCoords[d] - nCoords[d])*(vCoords[d] - nCoords[d]);
	  //ierr = PetscPrintf(mesh->comm(), "%f\n", (vCoords[d] - nCoords[d])*(vCoords[d] - nCoords[d]));CHKERRQ(ierr);
          //}
	 // ierr = PetscPrintf(mesh->comm(), "%d: %f %f (%x) -> %d: %f %f (%x) = %f\n", *v_iter, vCoords[0], vCoords[1], vCoords, *n_iter, nCoords[0], nCoords[1], nCoords, dist);CHKERRQ(ierr);
          if (dist < minDist) minDist = dist;
        }
      n_iter++;
      }
    s_iter++;
    }
    //ierr = PetscPrintf(mesh->comm(), "%d: %f ", *v_iter, minDist);CHKERRQ(ierr);
    minDist = sqrt(minDist);
    spacing->update(patch, *v_iter, &minDist);
    v_iter++;
  }  
  PetscFunctionReturn(0);
}

struct coarsen_point {
float * coordinates;

};

PetscErrorCode CreateCoarsenedMesh(Obj<ALE::Mesh> srcMesh, Obj<ALE::Mesh> dstMesh, double beta) {
   PetscErrorCode ierr;
   PetscFunctionBegin;
    ierr = PetscPrintf(srcMesh->comm(), "Coarsening mesh by beta = %f\n", beta);CHKERRQ(ierr);
   std::list<double *> points;
  ALE::Mesh::section_type::patch_type patch = 0;  
   //come up with a list containing an independent set of bubbles.
   //arbitrarily add the first vertex to this list.
   double * add_point;
   const double * tmp_point;
  Obj<ALE::Mesh::section_type> coords = srcMesh->getSection("coordinates");
  Obj<ALE::Mesh::section_type> space = srcMesh->getSection("spacing");
  const Obj<ALE::Mesh::topology_type::label_sequence>& vertices = coords->getAtlas()->getTopology()->depthStratum(patch, 0);

   ALE::Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin();
   ALE::Mesh::topology_type::label_sequence::iterator v_iter_end = vertices->end();
   while(v_iter != v_iter_end) {
		double v_space;
		v_space = *space->restrict(patch, *v_iter); 
		tmp_point = coords->restrict(patch, *v_iter);
		std::list<double *>::iterator c_iter = points.begin(), c_iter_end = points.end();
		bool point_is_ok = true; //point doesn't intersect anything already in the list.
		while (c_iter != c_iter_end && point_is_ok) {
			double dist = sqrt(((*c_iter)[0] - tmp_point[0])*((*c_iter)[0] - tmp_point[0]) + ((*c_iter)[1] - tmp_point[1])*((*c_iter)[1] - tmp_point[1]));
			if(dist < beta*((*c_iter)[2] + v_space)/2.) {
				point_is_ok = false;
			//ierr = PetscPrintf(srcMesh->comm(), "(%f, %f) (%f) <> (%f, %f) (%f)\n", tmp_point[0], tmp_point[1], v_space, (*c_iter)[0], (*c_iter)[1], (*c_iter)[2]);

			}
		c_iter++;
		}
		if(point_is_ok) { //add to the set.
			add_point = new double[3];
			add_point[0] = tmp_point[0];
			add_point[1] = tmp_point[1];
			add_point[2] = v_space;
			points.push_front(add_point);
			//ierr = PetscPrintf(srcMesh->comm(), "added %d to the set\n", *v_iter);
		}
	v_iter++;
   }
   
   ierr = PetscPrintf(srcMesh->comm(), "Now have a set of vertices of size %d, triangulating\n", points.size());
   
  //get the set ready for triangle!
  //
   triangulateio * input = new triangulateio;
   triangulateio * output = new triangulateio;
   input->numberofpoints = points.size();
   input->numberofpointattributes = 0;
   input->pointlist = new double[2*input->numberofpoints];

   //copy the points over
   std::list<double *>::iterator c_iter = points.begin(), c_iter_end = points.end();

   int index = 0;
   while (c_iter != c_iter_end) {
        input->pointlist[2*index] = (*c_iter)[0];
	input->pointlist[2*index+1] = (*c_iter)[1];
	c_iter++;
        index++;
   }

   //ierr = PetscPrintf(srcMesh->comm(), "copy is ok\n");
   input->numberofpointattributes = 0;
   input->pointattributelist = NULL;
   input->pointmarkerlist = NULL;

   input->numberoftriangles = 0;
   input->numberofcorners = 0;
   input->numberoftriangleattributes = 0;
   input->trianglelist = NULL;
   input->triangleattributelist = NULL;
   input->trianglearealist = NULL;

   input->segmentlist = NULL;
   input->segmentmarkerlist = NULL;
   input->numberofsegments = 0;

   input->holelist = NULL;
   input->numberofholes = 0;
   
   input->regionlist = NULL;
   input->numberofregions = 0;

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

   string triangleOptions = "-zeQ"; //(z)ero indexing, output (e)dges, Quiet

   triangulate((char *)triangleOptions.c_str(), input, output, NULL);
   //initialize the dstmesh
   ierr = PetscPrintf(srcMesh->comm(), "triangles: %d edges: %d points: %d\n", output->numberoftriangles, output->numberofedges, output->numberofpoints);
   TriangleToMesh(dstMesh, output, 0);
   //rest are out only.

//cleanup

   //c_iter = points.begin(); c_iter_end = points.end();
   //while (c_iter != c_iter_end) {
//	delete *c_iter;
 //  }

   delete input->pointlist;
   delete output->pointlist;
   delete output->trianglelist;
   delete output->edgelist;
   delete input;
   delete output;
   

   
   //ALE::New::SieveBuilder<ALE::Mesh::sieve_type>::buildTopology(sieve, dim, numCells, cells, numVertices, interpolate, numCorners);
   PetscFunctionReturn(0);
}
