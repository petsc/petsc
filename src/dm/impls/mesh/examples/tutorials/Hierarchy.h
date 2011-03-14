#include <list>
#include <Mesh.hh>
#include <Distribution.hh>
#include <petscmesh.h>
#include <stdlib.h>
#include <string>
#include <triangle.h>

namespace ALE {
namespace Hierarchy {
  //Typedefs I'm SO SICK of typing
  typedef ALE::Mesh::topology_type                        topology_type;
  typedef ALE::Mesh::real_section_type                    real_section_type;
  typedef ALE::Mesh::topology_type::point_type            point_type;
  typedef ALE::Mesh::int_section_type                     int_section_type;
  typedef ALE::Mesh::sieve_type                           sieve_type;
  typedef ALE::Mesh::topology_type::patch_type            patch_type;
  typedef ALE::Mesh::topology_type::label_sequence        label_sequence;
  typedef ALE::Mesh::sieve_type::traits::coneSequence     coneSequence;
  typedef ALE::Mesh::sieve_type::traits::supportSequence  supportSequence;
  typedef ALE::Mesh::sieve_type::supportSet               supportSet;
  typedef ALE::Mesh::sieve_type::coneSet                  coneSet;

  class HierarchyBuilder {
    private:
      //I am SO SICK of typing these.
      
      Obj<ALE::Mesh> mesh;
      Obj<topology_type> topology;
      Obj<real_section_type> coordinates;
      Obj<real_section_type> spacing;
      Obj<topology_type::patch_label_type> boundary;
      Obj<topology_type::patch_label_type> location;  //The triangle where the point (currently orphaned) lives
      Obj<topology_type::patch_label_type> included_location; //The triangle where a currently ADDED but unmeshed point.
      Obj<topology_type::patch_label_type> prolongation;  //The name of the triangle in the next level up where the given meshed-over point lives (build prolongation operators out of these)
      int dim;
      int C_levels;
      double C_factor;
      patch_type patch;
      
      //private member functions
      PetscErrorCode CreateSpacingFunction();
      PetscErrorCode IdentifyBoundary();
      PetscErrorCode IdentifyCoarsestBoundary();
      PetscErrorCode BuildLevels();
      PetscErrorCode TriangleToMesh(triangulateio *, patch_type);
      PetscErrorCode BoundaryNodeDimension_2D(point_type vertex); //puts the boundary label on
      PetscErrorCode BuildTopLevel(); //does a dumbest-approach build on the top level.
      PetscErrorCode PointListToMesh(std::list<point_type> *, patch_type);
      PetscErrorCode GetLocalInterpolation(double * coeffs, point_type v, point_type e); //test case for this.
      
      //monotony relief
      void ElementCorners(double *, patch_type, point_type);

      //geometric member functions
      bool TrianglesOverlap(patch_type patchA, point_type TriA, patch_type patchB, point_type TriB);  //sees if the two triangles overlap.
      bool EdgesIntersect(double *, double *);  //sees if the two 2D edges overlap (meaningless in 3D, but still should work.)
      bool HierarchyBuilder::PointMidpointCollide(double *, double *, double);
      bool PointsCollide(double *, double *, double, double); //passing stuff in in these as doubles lets us limit the number of costly restricts we do.
      bool PointInEdgeRegion(double *, double *, double);  //sees if the point is in the box around the edge of half-width region.
      bool PointInEdgeRegionSub(double *, double *, double); //ditto, only the region is edgelength - dif
      bool PointIsInTriangle(double * t_Coords, double * p_Coords);  //sees if the point is in the triangle defined by t_coords

    public:
      HierarchyBuilder(Obj<ALE::Mesh> m, int dimensions, int levels, double C);
      ~HierarchyBuilder();
      Obj<ALE::Mesh> getMesh() {return mesh;}
      int getDimension() {return dim;}
      int getNLevels() {return C_levels;}
  };

/////////////////////////////////////////////////////////////////////////////////////////////////////

  PetscErrorCode HierarchyBuilder::GetLocalInterpolation(double * coeffs, point_type v, point_type e) {
   // TEST FOR BUILDING THE RESTRICT (INTERPOLATE) OPERATOR.
   double * v0, * J, * invJ, detJ;
   mesh->computeElementGeometry(coordinates, e, v0, J, invJ, detJ);
   
  }

  HierarchyBuilder::HierarchyBuilder(Obj<ALE::Mesh> m, int dimensions, int levels, double C) {

    //initialize the various variables that will cut our code size in half.
    patch = 0;
    mesh = m;
    topology = mesh->getTopology();
    coordinates = mesh->getRealSection("coordinates");
    
    //set up the spacing section
    
    spacing = mesh->getRealSection("spacing");
    spacing->setFiberDimensionByDepth(patch, 0, 1);
    spacing->allocate();
    boundary = topology->createLabel(patch, "boundary");
    location = topology->createLabel(patch, "location");
    included_location = topology->createLabel(patch, "includedlocation");
    prolongation = topology->createLabel(patch, "prolongation");
    C_levels = levels;
    C_factor = C;
    dim = dimensions;

    //mark the boundary, create the spacing function, build the hierarchy.
    IdentifyBoundary();
    IdentifyCoarsestBoundary();
    CreateSpacingFunction();
    BuildTopLevel();
    BuildLevels();
    return;
  }
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  PetscErrorCode HierarchyBuilder::TriangleToMesh(triangulateio * src, patch_type patch) {
    PetscFunctionBegin;
  // We store the global vertex numbers as markers to preserve them in the coarse mesh
  //   Here we convert from the new Triangle numbering to the original fine mesh numbering (same sieve points we started from)
  //   We also offset them by the number of coarse triangles, to preserve the numbers after buildTopology()
    for (int i = 0; i != src->numberoftriangles; i++) {
      src->trianglelist[i*3+0] = src->pointmarkerlist[src->trianglelist[i*3+0]] - src->numberoftriangles;
      src->trianglelist[i*3+1] = src->pointmarkerlist[src->trianglelist[i*3+1]] - src->numberoftriangles;
      src->trianglelist[i*3+2] = src->pointmarkerlist[src->trianglelist[i*3+2]] - src->numberoftriangles;
    }

    Obj<ALE::Mesh::sieve_type>           sieve    = new ALE::Mesh::sieve_type(mesh->comm(), 0);
    const Obj<ALE::Mesh::topology_type>& topology = mesh->getTopology();

  //make the sieve and the topology actually count for something
    ALE::New::SieveBuilder<ALE::Mesh::sieve_type>::buildTopology(sieve, 2, src->numberoftriangles, src->trianglelist, src->numberofpoints, true, 3);
    sieve->stratify();
    topology->setPatch(patch, sieve);
  // Actually we probably only want to stratify at the end, so that we do not recalculate a lot
    topology->stratify();
    PetscFunctionReturn(0);
  }

  
  PetscErrorCode HierarchyBuilder::IdentifyBoundary()
  {
    if (dim == 2) {
    //initialize all the vertices
      const Obj<label_sequence>& vertices = topology->depthStratum(patch, 0);
      label_sequence::iterator v_iter = vertices->begin();
      label_sequence::iterator v_iter_end = vertices->end();

      while (v_iter != v_iter_end) {
	topology->setValue(boundary, *v_iter, 0);
	v_iter++;
      }

    //trace through the edges, initializing them to be non-boundary, then setting them as boundary.
      const Obj<label_sequence>& edges = topology->depthStratum(patch, 1);
      label_sequence::iterator e_iter = edges->begin();
      label_sequence::iterator e_iter_end = edges->end();

    // int nBoundaryVertices = 0;
      while (e_iter != e_iter_end) {
      //topology->setValue(boundary, *e_iter, 0);
      //find out if the edge is not supported on both sides, if so, this is a boundary node
	if (mesh->debug()) {printf("Edge %d supported by %d faces\n", *e_iter, topology->getPatch(patch)->support(*e_iter)->size());}
	if (topology->getPatch(patch)->support(*e_iter)->size() < 2) {
        //topology->setValue(boundary, *e_iter, 1);
	  Obj<coneSequence> endpoints = topology->getPatch(patch)->cone(*e_iter); //the adjacent elements
	  coneSequence::iterator p_iter     = endpoints->begin();
	  coneSequence::iterator p_iter_end = endpoints->end();
	  while (p_iter != p_iter_end) {
	    if (topology->depth(patch, *p_iter) != 0) {
	      throw ALE::Exception("Bad point");
	    } 
	    if (topology->getValue(boundary, *p_iter) == 0) {
	      topology->setValue(boundary, *p_iter, BoundaryNodeDimension_2D(*p_iter));
	      if (mesh->debug()) {printf("set boundary dimension for %d as %d\n", *p_iter, topology->getValue(boundary, *p_iter));}
	    }
          //boundVerts++;
	    p_iter++;
	  }
	}
	e_iter++;
      }
    //boundary->view(std::cout, "Boundary label");
    } else if (dim == 3) {  //loop over the faces to determine the 
      
    } else {
      
    }
    PetscFunctionReturn(0);
  }
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  PetscErrorCode HierarchyBuilder::PointListToMesh(std::list<point_type> * points, patch_type newPatch) {
    triangulateio * input = new triangulateio;
    triangulateio * output = new triangulateio;
    input->numberofpoints = points->size();
    input->pointlist = new double[dim*input->numberofpoints];
    std::list<point_type>::iterator c_iter = points->begin(), c_iter_end = points->end();
    int index = 0;
    std::list<point_type> newBoundNodes;
    while (c_iter != c_iter_end) {
      PetscMemcpy(input->pointlist + dim*index, coordinates->restrict(patch, *c_iter), dim*sizeof(double));
      c_iter++;
      index++;
    }
    input->numberofpointattributes = 0;
    input->pointattributelist = NULL;

//set up the pointmarkerlist to hold the names of the points

    input->pointmarkerlist = new int[input->numberofpoints];
    c_iter = points->begin();
    c_iter_end = points->end();
    index = 0;
    while(c_iter != c_iter_end) {
      input->pointmarkerlist[index] = *c_iter;
      c_iter++;
      index++;
    }
    
    //input->numberofsegments = 0;
    //input->segmentlist = NULL;
    
    Obj<label_sequence> boundEdges = topology->heightStratum(C_levels+1, 0);
    label_sequence::iterator be_iter = boundEdges->begin();
    label_sequence::iterator be_iter_end = boundEdges->end();
//set up the boundary segments
      
    input->numberofsegments = boundEdges->size();
    input->segmentlist = new int[2*input->numberofsegments];
    for (int i = 0; i < 2*input->numberofsegments; i++) {
      input->segmentlist[i] = -1;  //initialize
    }
    
    index = 0;
    while (be_iter != be_iter_end) { //loop over the boundary segments
      Obj<coneSequence> neighbors = topology->getPatch(C_levels + 1)->cone(*be_iter);
      coneSequence::iterator n_iter = neighbors->begin();
      coneSequence::iterator n_iter_end = neighbors->end();
      while (n_iter != n_iter_end) {
	for (int i = 0; i < input->numberofpoints; i++) {
	  if(input->pointmarkerlist[i] == *n_iter) {
	    if (input->segmentlist[2*index] == -1) {
	      input->segmentlist[2*index] = i;
	    } else {
	      input->segmentlist[2*index + 1] = i;
	    }
	  }
	}
	n_iter++;
      }
      index++;
      be_iter++;
    }
    
    input->numberoftriangles = 0;
    input->numberofcorners = 0;
    input->numberoftriangleattributes = 0;
    input->trianglelist = NULL;
    input->triangleattributelist = NULL;
    input->trianglearealist = NULL;

      //input->segmentlist = NULL;
    input->segmentmarkerlist = NULL;
      //input->numberofsegments = 0;

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

    string triangleOptions = "-zpQ"; //(z)ero indexing, output (e)dges, Quiet, Delaunay
    triangulate((char *)triangleOptions.c_str(), input, output, NULL);
    TriangleToMesh(output, newPatch);
      //printf("computing the angles\n");
    delete input->pointlist;
    delete output->pointlist;
    delete output->trianglelist;
    delete output->edgelist;
    delete input;
    delete output;
  }
  
  int HierarchyBuilder::BoundaryNodeDimension_2D(point_type vertex) {

    const double *vCoords = coordinates->restrict(patch, vertex);
    double v_x = vCoords[0], v_y = vCoords[1];
    bool foundNeighbor = false;
    int isEssential = 1;
  
    double f_n_x, f_n_y;
  
    Obj<supportSequence> support = topology->getPatch(patch)->support(vertex);
    label_sequence::iterator s_iter = support->begin();
    label_sequence::iterator s_iter_end = support->end();
    while(s_iter != s_iter_end) {
      if (topology->getPatch(patch)->support(*s_iter)->size() < 2) {
	Obj<coneSequence> neighbors = topology->getPatch(patch)->cone(*s_iter);
	coneSequence::iterator n_iter = neighbors->begin();
	coneSequence::iterator n_iter_end = neighbors->end();
	while(n_iter != n_iter_end) {
	  if (vertex != *n_iter) {
	    if (!foundNeighbor) {
	      const double *nCoords = coordinates->restrict(patch, *n_iter);
	      f_n_x = nCoords[0]; f_n_y = nCoords[1];
	      foundNeighbor = true;
	    } else {
	      const double *nCoords = coordinates->restrict(patch, *n_iter);
	      double n_x = nCoords[0], n_y = nCoords[1];
	      double parArea = fabs((f_n_x - v_x) * (n_y - v_y) - (f_n_y - v_y) * (n_x - v_x));
	      double len = (f_n_x-n_x)*(f_n_x-n_x) + (f_n_y-n_y)*(f_n_y-n_y);
	      if (parArea > .001*len) isEssential = 2;
	      if(mesh->debug()) printf("Parallelogram area: %f\n", parArea);
	    }
	  }
	  n_iter++;
	}
      }
      s_iter++;
    }
    return isEssential;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  struct bound_trav {  //structure with all the info needed to do a DFS on the boundary to find the PSLG
    ALE::Mesh::point_type lastCorn;  //the last corner seen.  as we string along we will connect these two in the new topology definition
    ALE::Mesh::point_type lastNode;  //the last node seen, so we do not backtrace.
    ALE::Mesh::point_type thisNode;  //the currently enqueued node.
    ALE::Mesh::point_type firstEdge;  //a HACK to name the edge something that most definitely isn't a point name.
    ALE::Mesh::point_type lastEdge;   //a HACK to keep edges from getting duplicated.
    int length;
  };
  
  
  PetscErrorCode HierarchyBuilder::IdentifyCoarsestBoundary () {
      //creates a 2-level (PSLG) representation of the boundary for feeding into triangle or tetgen.
    PetscFunctionBegin;
    Obj<sieve_type> sieve = new sieve_type(mesh->comm(), 0);
    patch_type srcPatch = patch;
    int nEdges = 0; // just a counter for sanity checking.
      //const Obj<ALE::Mesh::topology_type::label_sequence>& vertices = topology->depthStratum(originalPatch, 0);
      //from here grab the corners
    Obj<label_sequence> corners = topology->getLabelStratum(srcPatch, "boundary", dim);
    label_sequence::iterator c_iter = corners->begin();
    label_sequence::iterator c_iter_end = corners->end();
    int nBoundaries = 0; //this loop should only run once if the space is a topological (dim-1)-sphere.
    std::list<bound_trav *> trav_queue;
    while (c_iter != c_iter_end) {
      if (!sieve->capContains(*c_iter)) {  //if it has not already been found and added to the new topology. (check to see we don't need to stratify)
	  // here we just create the first set of paths to travel on.
	nBoundaries++;
	ALE::Obj<supportSequence> support = topology->getPatch(srcPatch)->support(*c_iter);
	label_sequence::iterator s_iter = support->begin();
	label_sequence::iterator s_iter_end = support->end();
	//boundary = topology->getLabel(srcPatch, "boundary");
	  //bool foundNeighbor = false;
	while (s_iter != s_iter_end) {
	  Obj<coneSequence> neighbors = topology->getPatch(srcPatch)->cone(*s_iter);
	  coneSequence::iterator n_iter = neighbors->begin();
	  coneSequence::iterator n_iter_end = neighbors->end();
	  while (n_iter != n_iter_end) {
	    if (*n_iter != *c_iter) {
	      if (topology->getValue(boundary, *n_iter) >= dim-1) { //if it's a boundary/edge-of-boundary(3D), or essential node
		bound_trav * tmp_trav = new bound_trav;
		tmp_trav->lastCorn = *c_iter;
		tmp_trav->lastNode = *c_iter;
		tmp_trav->thisNode = *n_iter;
		tmp_trav->firstEdge = *s_iter;
		tmp_trav->lastEdge = *s_iter;
		tmp_trav->length = 0;
		trav_queue.push_front(tmp_trav);
		 // foundNeighbor = true;
	      }
	    }
	    n_iter++;
	  }
	  s_iter++;
	}
	  //we have set up the initial conditions for the traversal, now we must traverse!
	while (!trav_queue.empty()) {
	  bound_trav * cur_trav = *trav_queue.begin();
	  trav_queue.pop_front();
	    //essential boundary node case.
	  if ((topology->getValue(boundary, cur_trav->thisNode) == dim)) {
	      //PetscPrintf(mesh->comm(), "-%d\n", cur_trav->thisNode);
	    if (!sieve->capContains(cur_trav->thisNode)) { //if it has not yet been discovered.
	      ALE::Obj<supportSequence> support = topology->getPatch(srcPatch)->support(cur_trav->thisNode);
	      label_sequence::iterator s_iter = support->begin();
	      label_sequence::iterator s_iter_end = support->end();
	      while (s_iter != s_iter_end) {
		Obj<coneSequence> neighbors = topology->getPatch(srcPatch)->cone(*s_iter);
		coneSequence::iterator n_iter = neighbors->begin();
		coneSequence::iterator n_iter_end = neighbors->end();
		while (n_iter != n_iter_end) {
		  if (*n_iter != cur_trav->thisNode && *n_iter != cur_trav->lastNode && topology->getPatch(srcPatch)->support(*s_iter)->size() == 1) {
		    if (topology->getValue(boundary, *n_iter) >= dim-1) { //if it's a boundary/edge-of-boundary(3D), or essential node
		      bound_trav * tmp_trav = new bound_trav;
		      tmp_trav->lastCorn = cur_trav->thisNode;
		      tmp_trav->lastNode = cur_trav->thisNode;
		      tmp_trav->firstEdge = *s_iter;
		      tmp_trav->lastEdge = *s_iter;
		      tmp_trav->thisNode = *n_iter;
		      tmp_trav->length = 0;
		      trav_queue.push_front(tmp_trav);
		    }
		  }
		  n_iter++;
		}
		s_iter++;
	      }
	    }
	      // in either essential boundary node case (discovered or undiscovered) we must create the sieve elements.... hmm...
	      //this will involve: creating a new edge, and creating arrows to it from lastCorn, and thisNode.
	    if (!sieve->baseContains(cur_trav->lastEdge)) {  //makes sure we don't pick up the backwards edge as well.
	      sieve->addArrow(cur_trav->thisNode, cur_trav->firstEdge, 0);
	      sieve->addArrow(cur_trav->lastCorn, cur_trav->firstEdge, 0);
	      //PetscPrintf(mesh->comm(), "Added edge from %d to %d of length %d\n", cur_trav->thisNode, cur_trav->lastCorn, cur_trav->length);
	      nEdges++;
	    }
	    delete cur_trav; //we can get rid of this one.
	  } else {
	      //in this case we just continue travelling along the edge we already were at.  (assume that in 3D intersections DO NOT HAPPEN HERE or it would be essential.
	      //PetscPrintf(mesh->comm(), "|%d\n", cur_trav->thisNode);
	    Obj<supportSequence> support = topology->getPatch(srcPatch)->support(cur_trav->thisNode);
	    label_sequence::iterator s_iter = support->begin();
	    label_sequence::iterator s_iter_end = support->end();
	    bool foundPath = false;
	    while (s_iter != s_iter_end && !foundPath) {
	      Obj<coneSequence> neighbors = topology->getPatch(srcPatch)->cone(*s_iter);
	      coneSequence::iterator n_iter = neighbors->begin();
	      coneSequence::iterator n_iter_end = neighbors->end();
	      while (n_iter != n_iter_end && !foundPath) {
		if (*n_iter != cur_trav->thisNode && *n_iter != cur_trav->lastNode) {
		  if (topology->getValue(boundary, *n_iter) >= dim-1 && topology->getPatch(srcPatch)->support(*s_iter)->size() == 1 && !sieve->baseContains(*s_iter)) { //if it's a boundary or essential boundary node, AND we don't have the opposite direction already worked out
		    foundPath = true; //breaks out of the loops.
		    cur_trav->lastNode = cur_trav->thisNode;
		    cur_trav->thisNode = *n_iter;
		    cur_trav->lastEdge = *s_iter;
		    cur_trav->length++;
		    trav_queue.push_front(cur_trav);
		      //printf("travelling on");
		  }
		}
		n_iter++;
	      }
	      s_iter++;
	    }
	  }
	}  //end traversal while
      } //end not discovered if
      c_iter++;
    } //end while over boundary vertices
      //we have created the boundary.  This is good!
    sieve->stratify();
    topology->setPatch(C_levels+1, sieve);
    topology->stratify();
    if (mesh->debug()) ;
    PetscPrintf(mesh->comm(), "- Created %d segments in %d boundaries in the exterior PSLG\n", topology->heightStratum(C_levels+1, 0)->size(), nBoundaries);
    PetscFunctionReturn(0);
  }
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  
  PetscErrorCode HierarchyBuilder::CreateSpacingFunction() {
    PetscFunctionBegin;
    const Obj<label_sequence>& vertices = topology->depthStratum(patch, 0);
  
    label_sequence::iterator v_iter = vertices->begin();
    label_sequence::iterator v_iter_end = vertices->end();
  
    double vCoords[dim], nCoords[dim];
  
    while (v_iter != v_iter_end) {
	//printf("vertex: %d\n", *v_iter);
      const double * rBuf = coordinates->restrict(patch, *v_iter);
      PetscMemcpy(vCoords, rBuf, dim*sizeof(double));
	  
      double minDist = -1; //using the max is silly.
      Obj<supportSequence> support = topology->getPatch(patch)->support(*v_iter);
	Obj<coneSet> neighbors = topology->getPatch(patch)->cone(support);
	coneSet::iterator n_iter = neighbors->begin();
	coneSet::iterator n_iter_end = neighbors->end();
	while(n_iter != n_iter_end) {
	  if (*v_iter != *n_iter) {
	    rBuf = coordinates->restrict(patch, *n_iter);
	    PetscMemcpy(nCoords, rBuf, dim*sizeof(double));
	    double d_tmp, dist    = 0.0;

	    for (int d = 0; d < dim; d++) {
	      d_tmp = nCoords[d] - vCoords[d];
	      dist += d_tmp * d_tmp;
	    }

	    if (dist < minDist || minDist == -1) minDist = dist;
	  }
	  n_iter++;
	}
      minDist = sqrt(minDist);
      spacing->update(patch, *v_iter, &minDist);
      v_iter++;
    } 
    PetscFunctionReturn(0);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  //METHODOLOGY: Build the first level, and bucket the points in the triangles.  From then on we must 
  //compare as follows:
  // + +             +    +
  //+ 0 +___________+___0  +
  // +|+ .          .+ /  +
  //  | .\ .........  /
  //  | . .\ .    .  /
  //  | . x x\ . .  /
  // +|+.x o x.\+ +/
  //+ 0_+_x_x__+_0 +
  // + +        + +
  // The interior point (o) must be compared to the elements on the other side of the edge if the sphere
  // of radius space(o) + length(edge) - sum(length(edge(endpoints))) collides with the edge.  This is done recursively on the 
  // next step out as well, allowing us to keep a small number of comparisons to the total number of points.
  //
  // The reason this works is that the largest possible intersection involves a sphere on the edge of radius
  // such that it doesn't collide with the endpoints.  
  
  PetscErrorCode HierarchyBuilder::BuildLevels() { //ok we're going to try this: MESH-based coarsening.
    std::list<point_type> level_points;
    const Obj<label_sequence> & top_vertices = topology->depthStratum(C_levels, 0);
    label_sequence::iterator top_iter = top_vertices->begin();
    label_sequence::iterator top_iter_end = top_vertices->end();
    while (top_iter != top_iter_end) {
      level_points.push_front(*top_iter);
      top_iter++;
    }
    const double * tmpCoords;
    const Obj<topology_type::patch_label_type>& traversal = topology->getLabel(C_levels, "traversal");
    for(int curLevel = C_levels - 1; curLevel > 0; curLevel--) {
      double factor = pow(C_factor, curLevel);
      Obj<label_sequence> triangles = topology->heightStratum(curLevel+1, 0); //triangles in next level up
      printf("%d triangles\n", triangles->size());
      label_sequence::iterator t_iter = triangles->begin();
      label_sequence::iterator t_iter_end = triangles->end();
      while (t_iter != t_iter_end) {
        int numlookedat = 0;
        printf("doing comparisons for triangle %d", *t_iter);
        const Obj<label_sequence> & intpoints = topology->getLabelStratum(patch, "location", *t_iter);
        printf(" - %d points internal.", intpoints->size());
        label_sequence::iterator p_iter = intpoints->begin();
        label_sequence::iterator p_iter_end = intpoints->end();
        while (p_iter != p_iter_end) {
          numlookedat++;
          //printf("doing comparisons for point %d", *p_iter);
          tmpCoords = coordinates->restrict(patch, *p_iter);
          double p_coords[dim];
          for (int i = 0; i < dim; i++) {
            p_coords[i] = tmpCoords[i];
          }
          double p_space = *spacing->restrict(patch, *p_iter);
          bool p_is_ok = true;
          std::list<point_type> compare_list; //A comparison queue.
          topology->setValue(traversal, *t_iter, 1);
          compare_list.push_front(*t_iter);
          while (!compare_list.empty() && p_is_ok) { //start comparing the point to things in the queue
            point_type cur_object = *compare_list.begin();
            compare_list.pop_front();
            int cur_object_height = topology->height(curLevel+1, cur_object);
            if (cur_object_height == 0) { //triangle case
              //printf("- triangle %d ", cur_object);
              //compare to endpoints and already added internal nodes
              Obj<coneSet> parts = topology->getPatch(curLevel+1)->closure(cur_object);
              coneSet::iterator parts_iter = parts->begin();
              coneSet::iterator parts_iter_end = parts->end();
              while (parts_iter != parts_iter_end && p_is_ok) {
                int part_depth = topology->depth(curLevel+1, *parts_iter);
                if (part_depth == 0) { //point
                  double part_space = * spacing->restrict(patch, *parts_iter);
                  tmpCoords = coordinates->restrict(patch, *parts_iter);
                  if(PointsCollide(p_coords, (double *)tmpCoords, factor*p_space, factor*part_space)) p_is_ok = false;
                } else if (part_depth == 1) { //edge
                  //prepare for edge comparison!
                  double part_space = 0;
                  double part_coords[dim*dim];
                  Obj<coneSequence> part_endpoints = topology->getPatch(curLevel+1)->cone(*parts_iter);
                  coneSequence::iterator pend_iter = part_endpoints->begin();
                  coneSequence::iterator pend_iter_end = part_endpoints->end();
                  int ind = 0;
                  while (pend_iter != pend_iter_end) {
                    //part_space += *spacing->restrict(patch, *pend_iter);
                    tmpCoords = coordinates->restrict(patch, *pend_iter);
                    for (int i = 0; i < dim; i++) {
                      part_coords[dim*ind + i] = tmpCoords[i];
                    }
                    ind++;
                    pend_iter++;
                  }
                  if(topology->getPatch(curLevel+1)->support(*parts_iter)->size() == 1) {
                    if (PointInEdgeRegion(part_coords, p_coords, factor*p_space/2) && topology->getValue(boundary, *p_iter) == 0) p_is_ok = false;
                  }
                  else if(topology->getValue(traversal, *parts_iter) != 1 && PointMidpointCollide(part_coords, p_coords, factor*p_space)) {
                     compare_list.push_back(*parts_iter);
                     topology->setValue(traversal, *parts_iter, 1);
                  }
                  
                }
                parts_iter++;
              }
              if (p_is_ok) { //now we get the list of points already included in this triangle and compare it to it!
                Obj<label_sequence> incPoints = topology->getLabelStratum(patch, "includedlocation", cur_object);
                label_sequence::iterator inc_iter = incPoints->begin();
                label_sequence::iterator inc_iter_end = incPoints->end();
                while (inc_iter != inc_iter_end && p_is_ok) {
                  double inc_space = *spacing->restrict(patch, *inc_iter);
                  tmpCoords = coordinates->restrict(patch, *inc_iter);
                  if (PointsCollide((double *)tmpCoords, p_coords, factor*inc_space, factor*p_space)) p_is_ok = false;
                  inc_iter++;
                }
              }
              //topology->setValue(traversal, cur_object, 1);  //set the triangle to "traversed"
            } else { //edge case
                Obj<supportSequence> edgeSides = topology->getPatch(curLevel + 1)->support(cur_object);
                supportSequence::iterator es_iter = edgeSides->begin();
                supportSequence::iterator es_iter_end = edgeSides->end();
                while (es_iter != es_iter_end) {
                  int travvalue = topology->getValue(traversal, *es_iter);
                  if (travvalue != 1) {
                     topology->setValue(traversal, *es_iter, 1);
                     compare_list.push_back(*es_iter);
                }
                es_iter++;
              }
              //topology->setValue(traversal, cur_object, 1);
            }
          } //end of the compare_list while.
          //go erase the traversal path, we must do it in two stages due to the way the label iterators work.
          //printf(" - cleanup - ");
          std::list<point_type> incpoints;
          Obj<label_sequence> incpoints_label = topology->getLabelStratum(curLevel+1, "traversal", 1);
          label_sequence::iterator il_iter = incpoints_label->begin();
          label_sequence::iterator il_iter_end = incpoints_label->end();
          while (il_iter != il_iter_end) {
            incpoints.push_front(*il_iter);
            il_iter++;
          }
          while (!incpoints.empty()) {
            point_type curInc = *incpoints.begin();
            incpoints.pop_front();
            topology->setValue(traversal, curInc, -1);
          }
          if (p_is_ok) {
            topology->setValue(included_location, *p_iter, *t_iter);
            topology->setValue(prolongation, *p_iter, *t_iter);
            level_points.push_front(*p_iter);
           
          }
          //printf("\n");
          p_iter++;
        }
        //this triangle is finalized.
        Obj<label_sequence> included_points = topology->getLabelStratum(patch, "includedlocation", *t_iter);
        label_sequence::iterator ip_iter = included_points->begin();
        label_sequence::iterator ip_iter_end = included_points->end();
        int numincluded = included_points->size();
        while (ip_iter != ip_iter_end) {
          topology->setValue(location, *ip_iter, -1);
          ip_iter++;
        }
        printf(" %d points looked at, %d included.\n", numlookedat, numincluded);
        t_iter++;
      } //end while over triangles
      printf("Triangulating over %d points on level %d\n", level_points.size(), curLevel);
      PointListToMesh(&level_points, curLevel);
      //unset all the include_location labels on the previous level.
      const Obj<label_sequence> cur_verts = topology->depthStratum(curLevel, 0);
      label_sequence::iterator v_iter = cur_verts->begin();
      label_sequence::iterator v_iter_end = cur_verts->end();
      while (v_iter != v_iter_end) {
        topology->setValue(included_location, *v_iter, -1);
        v_iter++;
      }

      // Locate the points that need location in the new mesh
      const Obj<topology_type::patch_label_type>& traversal = topology->createLabel(curLevel, "traversal");
      triangles = topology->heightStratum(curLevel+1, 0);
      t_iter = triangles->begin();
      t_iter_end = triangles->end();
      while (t_iter != t_iter_end) {
      printf("Redistributing %d", *t_iter);
      int ntricomps = 0;
      //NEW STRATEGY: go through the points in this triangle doing the DUMBEST THING (tm) at each stage, aka growing the set of included triangles through repeated cone(support()) as I'm
      //obviously too retarded to write a good triangle collision routine.
        //Initialize the traversal list to contain the star of the endpoints of t_iter;
        std::list<point_type> trav_list;
        trav_list.clear();
        Obj<coneSet> tips = topology->getPatch(curLevel+1)->closure(*t_iter);
        coneSet::iterator tips_iter = tips->begin();
        coneSet::iterator tips_iter_end = tips->end();
        while (tips_iter != tips_iter_end) {
          //get the CURRENT level's star of this and the point coordinates.
          if (topology->depth(curLevel, *tips_iter) == 0) {
            Obj<supportSet> init_tris = topology->getPatch(curLevel)->star(*tips_iter);
            supportSet::iterator it_iter = init_tris->begin();
            supportSet::iterator it_iter_end = init_tris->end();
            while (it_iter != it_iter_end) {
              if (topology->height(curLevel, *it_iter) == 0 && topology->getValue(traversal, *it_iter) != 1) {
                  topology->setValue(traversal, *it_iter, 1);
                  trav_list.push_back(*it_iter);
              } 
              it_iter++;
            }
          }
          tips_iter++;
        }
        //grab the points in this triangle
        printf(" - %d forced comparisons.", trav_list.size());
        Obj<label_sequence> intPoints = topology->getLabelStratum(patch, "location", *t_iter);
        label_sequence::iterator ip_iter = intPoints->begin();
        label_sequence::iterator ip_iter_end = intPoints->end();
        std::list<point_type> pointList;
        while (ip_iter != ip_iter_end) {
          pointList.push_front(*ip_iter);
          ip_iter++;
        }
        //TRAVERSE:  AT THIS POINT EVERY THING THAT HAS A VALID LOCATION SHOULD HAVE BEEN LOCATED AT A HIGHER LEVEL!!!
        double t_coords[dim*(dim+1)];
        while ((!trav_list.empty()) && (!pointList.empty())) {
          ntricomps++;
          point_type curTri = *trav_list.begin();
          trav_list.pop_front();
          bool containsPoint = false;
          ElementCorners(t_coords, curLevel, curTri); 
          //CHECK EVERY POINT AGAINST THIS TRIANGLE; THE POINTS SHOULD DIE OFF FAST
          std::list<point_type>::iterator ipl_iter = pointList.begin();
          std::list<point_type>::iterator ipl_iter_end = pointList.end();
          while (ipl_iter != ipl_iter_end) {
            if(PointIsInTriangle(t_coords, (double *)coordinates->restrict(patch, *ipl_iter))) {
              containsPoint = true;
              topology->setValue(included_location, *ipl_iter, curTri);
              ipl_iter = pointList.erase(ipl_iter);
            } else ipl_iter++;
          }
          Obj<supportSet> neighbors = topology->getPatch(curLevel)->support(topology->getPatch(curLevel)->cone(curTri));
          supportSet::iterator n_iter = neighbors->begin();
          supportSet::iterator n_iter_end = neighbors->end();
          while (n_iter != n_iter_end) {
            if(topology->getValue(traversal, *n_iter) != 1) {
              topology->setValue(traversal, *n_iter, 1);
              trav_list.push_back(*n_iter);
            }
            n_iter++;
          }
        }
        if (pointList.size() > 0) {
          printf(" - ERROR - %d Points unaccounted for.", pointList.size());
          //GO THROUGH THE LIST AND -1 OUT THAT! (There will be no prolongation operator for this one :( In fact, this shouldn't happen at all once I take out the boundaries)
          std::list<point_type>::iterator bp_iter = pointList.begin();
          std::list<point_type>::iterator bp_iter_end = pointList.end();
          while (bp_iter != bp_iter_end) {
            topology->setValue(included_location, *bp_iter, -1);
            topology->setValue(location, *bp_iter, -1);
            bp_iter++;
          }
        }
        printf(" - %d triangles spanned\n", ntricomps);
        std::list<point_type> incpoints;
        incpoints.clear();
        Obj<label_sequence> incpoints_label = topology->getLabelStratum(curLevel, "traversal", 1);
        label_sequence::iterator il_iter = incpoints_label->begin();
        label_sequence::iterator il_iter_end = incpoints_label->end();
        while (il_iter != il_iter_end) {
          incpoints.push_front(*il_iter);
          il_iter++;
        }
        while (!incpoints.empty()) {
          point_type curInc = *incpoints.begin();
          incpoints.pop_front();
          topology->setValue(traversal, curInc, -1);
        }
        t_iter++;
      }
      //ONE MORE TRAVERSAL: THE NEW TRIANGLES.  GET ALL THINGS INCLUDED IN EACH ONE AS DENOTED BY INCLUDEDLOCATION, -1 OUT INCLUDEDLOCATION, AND MOVE IT TO LOCATION
      Obj<label_sequence> nTris = topology->heightStratum(curLevel, 0);
      label_sequence::iterator nT_iter = nTris->begin();
      label_sequence::iterator nT_iter_end = nTris->end();
      while (nT_iter != nT_iter_end) {
        std::list<point_type> ntpoints;
        Obj<label_sequence> ntpoints_label = topology->getLabelStratum(patch, "includedlocation", *nT_iter);
        label_sequence::iterator ntp_iter = ntpoints_label->begin();
        label_sequence::iterator ntp_iter_end = ntpoints_label->end();
        while (ntp_iter != ntp_iter_end) {
          ntpoints.push_front(*ntp_iter);
          ntp_iter++;
        }
        while (!ntpoints.empty()) {
          point_type curP = *ntpoints.begin();
          ntpoints.pop_front();
          topology->setValue(location, curP, *nT_iter);
          topology->setValue(included_location, curP, -1);
        }
        nT_iter++;
      }
    }  //end for over levels
  }
  PetscErrorCode HierarchyBuilder::BuildTopLevel() { //does a greedy build of the topmost level of the mesh.
    PetscFunctionBegin;
    double factor = pow(C_factor, C_levels);
    const Obj<label_sequence>& vertices = topology->depthStratum(patch, 0);
    label_sequence::iterator v_iter = vertices->begin();
    label_sequence::iterator v_iter_end = vertices->end();
    std::list<point_type> incPoints;
    //add the coarsest boundary corners to the list.
    const Obj<label_sequence>& corners = topology->depthStratum(C_levels+1, 0);//topology->getLabelStratum(0, "boundary", dim);
    printf("%d forced corners\n", corners->size());
    label_sequence::iterator c_iter = corners->begin();
    label_sequence::iterator c_iter_end = corners->end();
    while (c_iter != c_iter_end) {
      incPoints.push_front(*c_iter);
      c_iter++;
    }
    while (v_iter != v_iter_end) {
      double v_space = *spacing->restrict(patch, *v_iter);
      double v_coords[dim];
      PetscMemcpy(v_coords, coordinates->restrict(patch, *v_iter), dim*sizeof(double));
      std::list<point_type>::iterator p_iter = incPoints.begin();
      std::list<point_type>::iterator p_iter_end = incPoints.end();
      bool v_is_ok = true;
      while (p_iter != p_iter_end && v_is_ok) {
	double p_space = *spacing->restrict(patch, *p_iter);
	double p_coords[dim];
	PetscMemcpy(p_coords, coordinates->restrict(patch, *p_iter), dim*sizeof(double));
	if (PointsCollide(v_coords, p_coords, v_space*factor, p_space*factor)) {
	  v_is_ok = false; 
	} 
	p_iter++;
      }
      //enforce the condition that the sphere packing must be within the coarsest domain.
      const Obj<label_sequence>& roughestEdges = topology->heightStratum(C_levels+1, 0);
      label_sequence::iterator e_iter = roughestEdges->begin();
      label_sequence::iterator e_iter_end = roughestEdges->end();
      while (e_iter != e_iter_end && v_is_ok) {
        const Obj<coneSequence>& edgeEndPoints = topology->getPatch(C_levels+1)->cone(*e_iter);
        coneSequence::iterator ep_iter = edgeEndPoints->begin();
        coneSequence::iterator ep_iter_end =edgeEndPoints->end();
        double e_coords[2*dim];
        int index = 0;
        //int bound = 0;
        while (ep_iter != ep_iter_end) {
          if (topology->getValue(boundary, *ep_iter) != 0);
          const double * tmpCoords = coordinates->restrict(patch, *ep_iter);
          for (int i = 0; i < dim; i++) {
            e_coords[dim*index+i] = tmpCoords[i];
          }
          ep_iter++;
          index++;
        }
        if(PointInEdgeRegion(e_coords, v_coords, factor*v_space) && topology->getValue(boundary, *v_iter) == 0) v_is_ok = false;
      e_iter++;
      }
      if (v_is_ok) incPoints.push_front(*v_iter);
      v_iter++;
    }
    printf("Attempting to triangulate over %d points for the top level.\n", incPoints.size());
    PointListToMesh(&incPoints, C_levels);
    const Obj<topology_type::patch_label_type>& traversal = topology->createLabel(C_levels, "traversal");
    //POINT LOCATION - ASSOCIATE ALL POINTS NOT IN THIS MESH WITH A TRIANGLE IN THIS MESH -
    const Obj<label_sequence>& triangles = topology->heightStratum(C_levels, 0);
    //we have restratified the topology, regrab the vertices.
//    const Obj<label_sequence>& verts = topology->getLabelStratum(patch, "boundary", 0); //get the internal vertices
    const Obj<label_sequence>& verts = topology->depthStratum(patch, 0); //get the vertices
    v_iter = verts->begin();
    v_iter_end = verts->end();

    //DO THIS BY TANDEM TRAVERSALS
   while (v_iter != v_iter_end) {
     topology->setValue(location, *v_iter, -2);
     v_iter++;
   }
   v_iter = verts->begin();
   while (v_iter != v_iter_end) {
     std::list<point_type> pQueue;
     std::list<point_type> triQueue;
     if (topology->getPatch(C_levels)->capContains(*v_iter)) {
       topology->setValue(location, *v_iter, -1);
       //v_iter++;
     } else if (topology->getValue(location, *v_iter) == -2) {
       printf("Starting from point %d in element ", *v_iter);
       //naively find the first point, and traverse outward from its neighbors.
       label_sequence::iterator t_iter = triangles->begin();
       label_sequence::iterator t_iter_end = triangles->end();
       bool point_located = false;
       //point_type containingTri;
       double tri_points[dim*(dim+1)];
       while (t_iter != t_iter_end && !point_located) {
         ElementCorners(tri_points, C_levels, *t_iter);
         if (PointIsInTriangle(tri_points, (double *)coordinates->restrict(patch, *v_iter))) {
           point_located = true;
           topology->setValue(location, *v_iter, *t_iter);
           pQueue.push_front(*v_iter);
           triQueue.push_front(*t_iter);
           printf("%d.\n", *t_iter);
         }
         t_iter++;
       }
       //traverse through the neighbors of our first located point
       while (!pQueue.empty()) {
         point_type curPoint = *pQueue.begin();
         double pq_coords[dim];
         const double * tmpCoords = coordinates->restrict(patch, curPoint);
         for (int i = 0; i < dim; i++) {
           pq_coords[i] = tmpCoords[i];
         }
         point_type curTri = *triQueue.begin();
         pQueue.pop_front();
         triQueue.pop_front();
         //printf("%d: ", curPoint);
         Obj<coneSet> neighbors = topology->getPatch(patch)->cone(topology->getPatch(patch)->support(curPoint));
         coneSet::iterator n_iter = neighbors->begin();
         coneSet::iterator n_iter_end = neighbors->end();
         while (n_iter != n_iter_end) {
           if (topology->getValue(location, *n_iter) == -2) {  //it's an interior point that hasn't yet been updated.
             double n_coords[dim];
             tmpCoords = coordinates->restrict(patch, *n_iter);
             for (int i = 0; i < dim; i++) {
               n_coords[i] = tmpCoords[i];
             }
             //locate this point, trying the triangle its neighbor was located in first;
             printf("locating point %d - ", *n_iter);
             std::list<point_type> curTriQueue;
             curTriQueue.push_front(curTri);
             topology->setValue(traversal, curTri, 1);
             while (!curTriQueue.empty()) {
               point_type compTri = *curTriQueue.begin();
               curTriQueue.pop_front();
               ElementCorners(tri_points, C_levels, compTri);
               if (PointIsInTriangle(tri_points, n_coords)) {
                 printf("located in %d.\n", compTri);
                 topology->setValue(location, *n_iter, compTri);
                 pQueue.push_back(*n_iter);
                 triQueue.push_back(compTri);
                 curTriQueue.clear();
               } else { //add the neighbors of the triangle to the comparison queue.
                 Obj<supportSet> tri_neighbors = topology->getPatch(C_levels)->support(topology->getPatch(C_levels)->cone(compTri));
                 supportSet::iterator tn_iter = tri_neighbors->begin();
                 supportSet::iterator tn_iter_end = tri_neighbors->end();
                 while (tn_iter != tn_iter_end) {
                   if (topology->getValue(traversal, *tn_iter) != 1) {
                     curTriQueue.push_back(*tn_iter);
                     topology->setValue(traversal, *tn_iter, 1);
                   }
                   tn_iter++;
                 }
               } 
             } //end of triangle traversal
             //clean up the traversal label
             Obj<label_sequence> visited_triangles_label = topology->getLabelStratum(C_levels, "traversal", 1);
             label_sequence::iterator vtl_iter = visited_triangles_label->begin();
             label_sequence::iterator vtl_iter_end = visited_triangles_label->end();
             std::list<point_type> visited_triangles;
             while (vtl_iter != vtl_iter_end) {
              visited_triangles.push_front(*vtl_iter);
              vtl_iter++;
             }
             while (!visited_triangles.empty()) {
               point_type cur_vis_tri = *visited_triangles.begin();
               visited_triangles.pop_front();
               topology->setValue(traversal, cur_vis_tri, 0); //reset so we can retraverse
             }
           }  //end of if determining if the neighbors are already evaluated
           n_iter++;
         } //end of while over neighbors of last point reachable from v_iter
       } //end of while over points reachable from last v_iter
     } //end of if testing for overall unvisitedness
     v_iter++;
   }
   printf("done\n");
   /*
    printf("Starting loop\n");
    while (v_iter != v_iter_end) {
      if(topology->getPatch(C_levels)->capContains(*v_iter)) {
        topology->setValue(location, *v_iter, -1);
      } else {
        bool not_located = true;
        PetscMemcpy(v_coords, coordinates->restrict(patch, *v_iter), sizeof(double)*dim);
        label_sequence::iterator t_iter = triangles->begin();
        label_sequence::iterator t_iter_end = triangles->end();

        while (t_iter != t_iter_end && not_located) {
          Obj<coneSet> triPoints = topology->getPatch(C_levels)->closure(*t_iter); //make iterative for compatibility with 3D.
          //printf("%d\n", triPoints->size());
          coneSet::iterator po_iter = triPoints->begin();
          coneSet::iterator po_iter_end = triPoints->end();
          int index = 0;
          while (po_iter != po_iter_end) {
            if (topology->depth(C_levels, *po_iter) == 0) {
              const double * tmpCoords = coordinates->restrict(patch, *po_iter);
              for (int i = 0; i < dim; i++) {
                t_coords[dim*index + i] = tmpCoords[i];
              }
              index++;
            }
            po_iter++;
          }
          if (PointIsInTriangle(t_coords, v_coords)) {
            not_located = false;
            topology->setValue(location, *v_iter, *t_iter);
            //printf("Point %d is in triangle %d\n", *v_iter, *t_iter);
          }
          t_iter++;
        }
        if (not_located) printf("ERROR: Couldn't find triangle in which %d was located.\n", *v_iter);
      }
      v_iter++;
    }
   */
    PetscFunctionReturn(0);
  }

  ////////////////////////////////////////////////////////////////////////////////////
  //GEOMETRIC SUBFUNCTIONS - KEEP SIEVE/MESH INDEPENDENT BECAUSE WE CAN CONSERVE RESTRICTS FURTHER UP
  bool HierarchyBuilder::PointsCollide(double * a, double * b, double a_space, double b_space) {
    double dist = 0;
    for (int i = 0; i < dim; i++) {
      dist += (a[i] - b[i])*(a[i] - b[i]);
    }
    double mdist = a_space + b_space;
    if (dist < mdist*mdist/4) return true;
    else return false;
  }
  
  
  bool HierarchyBuilder::PointInEdgeRegion(double * e_coords, double * p_coords, double region) {
    double res_len = 0;
    double e_dot_e = 0;
    double p_dot_e = 0;
    for (int i = 0; i < dim; i++) { //put the first edge endpoint at (0, 0);
      e_dot_e += (e_coords[dim + i] - e_coords[i]) * (e_coords[dim + i] - e_coords[i]);
      p_dot_e += (e_coords[dim + i]- e_coords[i]) * (p_coords[i] - e_coords[i]);
    }
    double r = p_dot_e/e_dot_e;
    if (r > 1 || r < 0) return false; //it's outside of our juristiction.
    for (int i = 0; i < dim; i++) {
      double trm = (p_coords[i] - e_coords[i]) - r*(e_coords[dim + i] - e_coords[i]);
      res_len += trm*trm;
    }
    if (res_len < region*region) return true; //it's in the region surrounding the edge.
    return false;
  }
  bool HierarchyBuilder::PointMidpointCollide(double * e_coords, double * p_coords, double p_space) {
    double midpoint[dim];
    double edgelensq = 0;
    double disttomid = 0;
    for (int i = 0; i < dim; i++) {
      midpoint[i] = (e_coords[i] + e_coords[dim+i])/2;
      edgelensq += (e_coords[i] + e_coords[dim+i])*(e_coords[i] + e_coords[dim+i]);
      disttomid += (midpoint[i] - p_coords[i])*(midpoint[i] - p_coords[i]);
    }
    double edgerad = sqrt(edgelensq)/2;
    if (disttomid < (p_space/2 + edgerad)*(p_space/2 + edgerad)) return true;
    return false;
  }
  bool HierarchyBuilder::PointInEdgeRegionSub(double * e_coords, double * p_coords, double dif) { //version for triangle crossover calculation, diff being the added spacing functions of the endpoints
    double res_len = 0;
    double edg_len = 0;
    double e_dot_e = 0;
    double p_dot_e = 0;
    for (int i = 0; i < dim; i++) { //put the first edge endpoint at (0, 0);
      e_dot_e += (e_coords[dim + i] - e_coords[i]) * (e_coords[dim + i] - e_coords[i]);
      p_dot_e += (e_coords[dim + i]- e_coords[i]) * (p_coords[i] - e_coords[i]);
    }
    double r = p_dot_e/e_dot_e;
    if (r > 1 || r < 0) return false; //it's outside of our juristiction.
    for (int i = 0; i < dim; i++) {
      double trm = (p_coords[i] - e_coords[i]) - r*(e_coords[dim + i] - e_coords[i]);
      double lentrm = (1 - (dif*dif/e_dot_e))*(e_coords[dim +i] - e_coords[i]);
      res_len += trm*trm;
      edg_len += lentrm*lentrm;
    }
    if (res_len < edg_len) return true; //it's in the region surrounding the edge.
    return false;
  }

  bool HierarchyBuilder::TrianglesOverlap(patch_type patchA, point_type TriA, patch_type patchB, point_type TriB) {
    double pointsA[3*dim];
    double pointsB[3*dim];
    double edgesA[6*dim];
    double edgesB[6*dim];
    Obj<coneSet> closureA = topology->getPatch(patchA)->closure(TriA);
    Obj<coneSet> closureB = topology->getPatch(patchB)->closure(TriB);
    coneSet::iterator c_iter = closureA->begin();
    coneSet::iterator c_iter_end = closureA->end();
    int index = 0;
    while (c_iter != c_iter_end) {
      if (topology->depth(patchA, *c_iter) == 0) {
        const double * tmpCoords = coordinates->restrict(patch, *c_iter);
        for (int i = 0; i < dim; i++) {
          pointsA[index*dim + i] = tmpCoords[i];
          edgesA[((2*index+1)%6)*dim + i] = tmpCoords[i];
          edgesA[((2*index+2)%6)*dim + i] = tmpCoords[i];
        }
        index++;
      }
      c_iter++;
    }
    c_iter = closureB->begin();
    c_iter_end = closureB->end();
    index = 0;
    while (c_iter != c_iter_end) {
      if (topology->depth(patchB, *c_iter) == 0) {
        const double * tmpCoords = coordinates->restrict(patch, *c_iter);
        for (int i = 0; i < dim; i++) {
          pointsB[index*dim + i] = tmpCoords[i];
          edgesB[((2*index+1)%6)*dim + i] = tmpCoords[i];
          edgesB[((2*index+2)%6)*dim + i] = tmpCoords[i];
        }
        index++;
      }
      c_iter++;
    }
    //test for any point intersections.
    for (int i = 0; i < 3; i++) {
      if(PointIsInTriangle(pointsA, &pointsB[dim*i]) || PointIsInTriangle(pointsB, &pointsA[dim*i])) return true;
    }
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        if(EdgesIntersect(&edgesA[i*2*dim], &edgesB[j*2*dim])) return true;
      }
    }
    return false;
  }
  bool HierarchyBuilder::EdgesIntersect(double * aCoords, double * bCoords) {
    //methodology: if for both edges the projection of the other edges endpoints onto the segment involves a sign flip of the residual signed distance, then those edges collide.
    //project B onto A:
    double a_dot_a = 0;
    double a_dot_b1 = 0;
    double a_dot_b2 = 0;
    double b_dot_b = 0;
    double b_dot_a1 = 0;
    double b_dot_a2 = 0;
    for (int i = 0; i < dim; i++) {
      double intera = aCoords[dim + i] - aCoords[i];
      a_dot_a += intera * intera;
      double intera1 = bCoords[i] - aCoords[i];
      a_dot_b1 += intera1*intera;
      double intera2 = bCoords[dim + i] - aCoords[i];
      a_dot_b2 += intera2*intera;
      double interb = bCoords[dim + i] - bCoords[i];
      b_dot_b += interb * interb;
      double interb1 = aCoords[i] - bCoords[i];
      b_dot_a1 += interb * interb1;
      double interb2 = aCoords[i + dim] - bCoords[i];
      b_dot_a2 += interb * interb2;
    }
    double ra1 = a_dot_b1/a_dot_a;
    double ra2 = a_dot_b2/a_dot_a;
    double rb1 = b_dot_a1/b_dot_b;
    double rb2 = b_dot_a2/b_dot_b;
    double resDota = 0, resDotb = 0;
    //compute the dot product of the residuals; if it's negative, then continue
    for (int i = 0; i < dim; i++) {
      double trma = ((bCoords[i] - aCoords[i]) - ra1*(aCoords[dim + i] - aCoords[i]))*((bCoords[dim+i] - aCoords[i]) - ra2*(aCoords[dim + i] - aCoords[i]));
      double trmb = ((aCoords[i] - bCoords[i]) - rb1*(bCoords[dim + i] - bCoords[i]))*((aCoords[dim+i] - bCoords[i]) - rb2*(bCoords[dim + i] - bCoords[i]));
      resDota += trma;
      resDotb += trmb;
    }
    if (resDota < 0 && resDotb < 0) return true;
    return false;
  }

  void HierarchyBuilder::ElementCorners(double * buffer, patch_type aPatch, point_type aTriangle) {
    Obj<coneSet> tt_support = topology->getPatch(aPatch)->closure(aTriangle);
    coneSet::iterator tts_iter = tt_support->begin();
    coneSet::iterator tts_iter_end = tt_support->end();
    int index = 0;
    while (tts_iter != tts_iter_end) {
      if (topology->depth(aPatch, *tts_iter) == 0) {
        const double * tmpCoords = coordinates->restrict(patch, *tts_iter);
        for (int i = 0; i < dim; i++) {
          buffer[index*dim + i] = tmpCoords[i];
        }
      index++;
      }
     tts_iter++;
     }
    return;
  }

  bool HierarchyBuilder::PointIsInTriangle(double * p_coords, double * v_coords) {
    double area = 0; //compute the area of the triangle/volume of the tet
    //if (dim == 2) {
      area = fabs((p_coords[2] - p_coords[0])*(p_coords[5] - p_coords[1]) - (p_coords[4] - p_coords[0])*(p_coords[3] - p_coords[1]));
    //}
    //compute the area of the various subvolumes induced by the point.
    double t_area = 0;
    if (dim == 2) for (int i = 1; i <= dim; i++) { //loop choosing the first point.
      for (int j = 0; j < i; j++) {  //loop choosing the second point.
        t_area += fabs((p_coords[dim*i] - v_coords[0])*(p_coords[dim*j+1] - v_coords[1]) - (p_coords[dim*i+1] - v_coords[1])*(p_coords[dim*j] - v_coords[0]));
      }
    }
    //printf("Comparing triangle area %f with %f\n", area, t_area);
    if (t_area - area  > 0.0000001*area) return false;
    return true;
  }
}
}

