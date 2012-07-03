#include <list>
#include <Distribution.hh>
#include <petscmesh.h>
#include <petscviewer.h>
#include <../src/dm/mesh/meshpcice.h>
#include <../src/dm/mesh/meshpylith.h>
#include "tree_mis.h"
#include "id_bound.h"
#include <stdlib.h>
#include <string.h>
#include <string>
  
namespace ALE {
namespace Coarsener {
  
  PetscErrorCode IdentifyBoundary(Obj<ALE::Mesh>&, int);  //identify the boundary faces/edges/nodes.
  PetscErrorCode CreateSpacingFunction(Obj<ALE::Mesh>&, int);  //same 'ol, same 'ol.  (puts a nearest neighbor value on each vertex) (edges?)
  PetscErrorCode CreateCoarsenedHierarchy(Obj<ALE::Mesh>&, int, int, float); //returns the meshes!
#ifdef PETSC_HAVE_TRIANGLE
  PetscErrorCode TriangleToMesh(Obj<ALE::Mesh>, triangulateio *, ALE::Mesh::real_section_type::patch_type);
#endif
  PetscErrorCode LevelCoarsen(Obj<ALE::Mesh>&, int,  ALE::Mesh::real_section_type::patch_type, bool, float);
  int BoundaryNodeDimension_2D(Obj<ALE::Mesh>&, ALE::Mesh::point_type);
  int BoundaryNodeDimension_3D(Obj<ALE::Mesh>&, ALE::Mesh::point_type);
  
  
  ////////////////////////////////////////////////////////////////
  
  PetscErrorCode CreateSpacingFunction(Obj<ALE::Mesh> & mesh, int dim) {
    Obj<ALE::Mesh::topology_type> topology = mesh->getTopology();
    ALE::Mesh::real_section_type::patch_type patch = 0;
    const Obj<ALE::Mesh::topology_type::label_sequence>& vertices = topology->depthStratum(patch, 0);
  
    PetscFunctionBegin;
  
    //initialize the spacing function section
  
    Obj<ALE::Mesh::real_section_type> spacing = mesh->getRealSection("spacing");
    spacing->setFiberDimensionByDepth(patch, 0, 1);
    spacing->allocate();
    
    Obj<ALE::Mesh::real_section_type> coords = mesh->getRealSection("coordinates");
    
    ALE::Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin();
    ALE::Mesh::topology_type::label_sequence::iterator v_iter_end = vertices->end();
  
    double vCoords[dim], nCoords[dim];
  
    while (v_iter != v_iter_end) {
	//printf("vertex: %d\n", *v_iter);
      const double * rBuf = coords->restrict(patch, *v_iter);
      PetscMemcpy(vCoords, rBuf, dim*sizeof(double));
	  
      double minDist = -1; //using the max is silly.
    ALE::Obj<ALE::Mesh::sieve_type::traits::supportSequence> support = topology->getPatch(patch)->support(*v_iter);
    ALE::Mesh::topology_type::label_sequence::iterator s_iter     = support->begin();
    ALE::Mesh::topology_type::label_sequence::iterator s_iter_end = support->end();
    while(s_iter != s_iter_end) {
      ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence> neighbors = topology->getPatch(patch)->cone(*s_iter);
      ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter = neighbors->begin();
      ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter_end = neighbors->end();
      while(n_iter != n_iter_end) {
	if (*v_iter != *n_iter) {
	  rBuf = coords->restrict(patch, *n_iter);
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
      s_iter++;
     }
    minDist = sqrt(minDist);
    spacing->update(patch, *v_iter, &minDist);
    v_iter++;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IdentifyBoundary(Obj<ALE::Mesh>& mesh, int dim)
{
  ALE::Mesh::real_section_type::patch_type patch = 0;
  Obj<ALE::Mesh::topology_type> topology = mesh->getTopology();
  const Obj<ALE::Mesh::topology_type::patch_label_type>& boundary = topology->createLabel(patch, "boundary");

  if (dim == 2) {
    //initialize all the vertices
    const Obj<ALE::Mesh::topology_type::label_sequence>& vertices = topology->depthStratum(patch, 0);
    ALE::Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin();
    ALE::Mesh::topology_type::label_sequence::iterator v_iter_end = vertices->end();

    while (v_iter != v_iter_end) {
      topology->setValue(boundary, *v_iter, 0);
      v_iter++;
    }

    //trace through the edges, initializing them to be non-boundary, then setting them as boundary.
    const Obj<ALE::Mesh::topology_type::label_sequence>& edges = topology->depthStratum(patch, 1);
    ALE::Mesh::topology_type::label_sequence::iterator e_iter = edges->begin();
    ALE::Mesh::topology_type::label_sequence::iterator e_iter_end = edges->end();

    // int nBoundaryVertices = 0;
    while (e_iter != e_iter_end) {
      //topology->setValue(boundary, *e_iter, 0);
      //find out if the edge is not supported on both sides, if so, this is a boundary node
      if (mesh->debug()) {printf("Edge %d supported by %d faces\n", *e_iter, (int)topology->getPatch(patch)->support(*e_iter)->size());}
      if (topology->getPatch(patch)->support(*e_iter)->size() < 2) {
        //topology->setValue(boundary, *e_iter, 1);
        ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence> endpoints = topology->getPatch(patch)->cone(*e_iter); //the adjacent elements
        ALE::Mesh::sieve_type::traits::coneSequence::iterator p_iter     = endpoints->begin();
        ALE::Mesh::sieve_type::traits::coneSequence::iterator p_iter_end = endpoints->end();
        while (p_iter != p_iter_end) {
          if (topology->depth(patch, *p_iter) != 0) {
            throw ALE::Exception("Bad point");
          } 
          if (topology->getValue(boundary, *p_iter) == 0) {
	    topology->setValue(boundary, *p_iter, BoundaryNodeDimension_2D(mesh, *p_iter));
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

PetscErrorCode CreateCoarsenedHierarchy(Obj<ALE::Mesh>& mesh, int dim, int nMeshes, float beta = 1.41) {
   //in this function we will assume that the original mesh is given to us in patch 0, and that its boundary has been identified with IdentifyBoundary.  We will put nMesh - 1 coarsenings in patches 1 through nMeshes.
   
  for (int curLevel = nMeshes; curLevel > 0; curLevel--) {
    bool isTopLevel = (curLevel == nMeshes);
    double crsBeta = pow(beta, curLevel);
    printf("Creating coarsening level: %d with beta = %f\n", curLevel, crsBeta);
    //LevelCoarsen(mesh, dim, curLevel, !isTopLevel, crsBeta);
    tree_mis(mesh, dim, curLevel, nMeshes+1, !isTopLevel, crsBeta);
    if (mesh->debug()) {
      ostringstream txt;
      txt << "Sieve for coarsening level " << curLevel;
      mesh->getTopology()->getPatch(curLevel)->view(txt.str().c_str());
    }
  }
  mesh->getTopology()->stratify();
  PetscFunctionReturn(0);
}

PetscErrorCode LevelCoarsen(Obj<ALE::Mesh>& mesh, int dim, ALE::Mesh::real_section_type::patch_type newPatch, bool includePrevious, float beta) {
  PetscFunctionBegin;
  ALE::Mesh::real_section_type::patch_type originalPatch = 0;
  std::list<ALE::Mesh::point_type> incPoints;
  Obj<ALE::Mesh::topology_type> topology = mesh->getTopology();
  double v_coord[dim], c_coord[dim];
  Obj<ALE::Mesh::real_section_type> coords  = mesh->getRealSection("coordinates");
  Obj<ALE::Mesh::real_section_type> spacing = mesh->getRealSection("spacing");

  //const Obj<ALE::Mesh::topology_type::patch_label_type>& boundary = topology->getLabel(originalPatch, "boundary");
  if(includePrevious) {

    ALE::Mesh::real_section_type::patch_type coarserPatch = newPatch+1;
    const Obj<ALE::Mesh::topology_type::label_sequence>& previousVertices = topology->depthStratum(coarserPatch, 0);

    //Add the vertices from the next coarser patch to the list of included vertices.
    ALE::Mesh::topology_type::label_sequence::iterator v_iter = previousVertices->begin();
    ALE::Mesh::topology_type::label_sequence::iterator v_iter_end = previousVertices->end();
    while(v_iter != v_iter_end) {
      incPoints.push_front(*v_iter);
      v_iter++;
    }
  } else {
    //get the essential boundary nodes and add them to the set.
    const Obj<ALE::Mesh::topology_type::label_sequence>& essVertices = topology->getLabelStratum(originalPatch, "boundary", dim);
    printf("- adding %d boundary nodes.\n", (int)essVertices->size());
    ALE::Mesh::topology_type::label_sequence::iterator v_iter     = essVertices->begin();
    ALE::Mesh::topology_type::label_sequence::iterator v_iter_end = essVertices->end();
    while (v_iter != v_iter_end) {
      if (mesh->debug()) {std::cout << "--> added " << *v_iter << std::endl;}
      incPoints.push_front(*v_iter);
      v_iter++;
    }
  }
  //for now just loop over the points
  const Obj<ALE::Mesh::topology_type::label_sequence>& verts = topology->depthStratum(originalPatch, 0);
  ALE::Mesh::topology_type::label_sequence::iterator v_iter = verts->begin();
  ALE::Mesh::topology_type::label_sequence::iterator v_iter_end = verts->end();
  while (v_iter != v_iter_end) {
    PetscMemcpy(v_coord, coords->restrict(originalPatch, *v_iter), dim*sizeof(double));
    double v_space = *spacing->restrict(originalPatch, *v_iter);
    std::list<ALE::Mesh::point_type>::iterator c_iter = incPoints.begin(), c_iter_end = incPoints.end();
    bool isOk = true;
    while (c_iter != c_iter_end) {
      PetscMemcpy(c_coord, coords->restrict(originalPatch, *c_iter), dim*sizeof(double));
      double dist = 0;
      double c_space = *spacing->restrict(originalPatch, *c_iter);
      for (int d = 0; d < dim; d++) {
        dist += (v_coord[d] - c_coord[d])*(v_coord[d] - c_coord[d]);
      }
      double mdist = c_space + v_space;
      if (dist < beta*beta*mdist*mdist/4) {
        isOk = false;
        break;
      }
      c_iter++;
    }
    if (isOk) {
      incPoints.push_front(*v_iter);
      //printf("  - Adding point %d to the new mesh\n", *v_iter);
    }
    v_iter++;
  }

  printf("- creating input to triangle: %d points\n", (int)incPoints.size());
#ifdef PETSC_HAVE_TRIANGLE
  //At this point we will set up the triangle(tetgen) calls (with preservation of vertex order.  This is why I do not use the functions build in).
  triangulateio * input = new triangulateio;
  triangulateio * output = new triangulateio;
  
  input->numberofpoints = incPoints.size();
  input->numberofpointattributes = 0;
  input->pointlist = new double[dim*input->numberofpoints];

  //copy the points over
  std::list<ALE::Mesh::point_type>::iterator c_iter = incPoints.begin(), c_iter_end = incPoints.end();

  int index = 0;
  while (c_iter != c_iter_end) {
     PetscMemcpy(input->pointlist + dim*index, coords->restrict(originalPatch, *c_iter), dim*sizeof(double));
     c_iter++;
     index++;
  }

  //ierr = PetscPrintf(srcMesh->comm(), "copy is ok\n");
  input->numberofpointattributes = 0;
  input->pointattributelist = NULL;

//set up the pointmarkerlist to hold the names of the points
  input->segmentlist = NULL;
  input->numberofsegments = 0;
  input->segmentmarkerlist = NULL;

  input->pointmarkerlist = new int[input->numberofpoints];
  c_iter = incPoints.begin();
  c_iter_end = incPoints.end();
  index = 0;

  while(c_iter != c_iter_end) {
    input->pointmarkerlist[index] = *c_iter;
    c_iter++;
    index++;
    
  }


  input->numberoftriangles = 0;
  input->numberofcorners = 0;
  input->numberoftriangleattributes = 0;
  input->trianglelist = NULL;
  input->triangleattributelist = NULL;
  input->trianglearealist = NULL;

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
  TriangleToMesh(mesh, output, newPatch);
  delete input->pointlist;
  delete output->pointlist;
  delete output->trianglelist;
  delete output->edgelist;
  delete input;
  delete output;
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "No mesh generator available!");
#endif
  PetscFunctionReturn(0);
}

int BoundaryNodeDimension_2D(Obj<ALE::Mesh>& mesh, ALE::Mesh::point_type vertex) {

  ALE::Mesh::real_section_type::patch_type patch = 0; 
  Obj<ALE::Mesh::topology_type> topology = mesh->getTopology();
  Obj<ALE::Mesh::real_section_type> coords = mesh->getRealSection("coordinates");
  const double *vCoords = coords->restrict(patch, vertex);
  double v_x = vCoords[0], v_y = vCoords[1];
  bool foundNeighbor = false;
  int isEssential = 1;
  
  double f_n_x, f_n_y;
  
  ALE::Obj<ALE::Mesh::sieve_type::traits::supportSequence> support = topology->getPatch(patch)->support(vertex);
  ALE::Mesh::topology_type::label_sequence::iterator s_iter = support->begin();
  ALE::Mesh::topology_type::label_sequence::iterator s_iter_end = support->end();
  while(s_iter != s_iter_end) {
      if (topology->getPatch(patch)->support(*s_iter)->size() < 2) {
      ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence> neighbors = topology->getPatch(patch)->cone(*s_iter);
      ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter = neighbors->begin();
      ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter_end = neighbors->end();
      while(n_iter != n_iter_end) {
	if (vertex != *n_iter) {
	  if (!foundNeighbor) {
	    const double *nCoords = coords->restrict(patch, *n_iter);
	    f_n_x = nCoords[0]; f_n_y = nCoords[1];
	    foundNeighbor = true;
	  } else {
	    const double *nCoords = coords->restrict(patch, *n_iter);
	    double n_x = nCoords[0], n_y = nCoords[1];
	    double parArea = fabs((f_n_x - v_x) * (n_y - v_y) - (f_n_y - v_y) * (n_x - v_x));
            double len = (f_n_x-n_x)*(f_n_x-n_x) + (f_n_y-n_y)*(f_n_y-n_y);
	    if (parArea > .00001*len) isEssential = 2;
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

int BoundaryNodeDimension_3D(Obj<ALE::Mesh>& mesh, ALE::Mesh::point_type vertex) {
//determines if two triangles are coplanar
  //given the point,get the support of every element of the point's support and see if it is a "crease".  Count the creases
//if there are two crease support elements, it is a rank 2, if there are more it's 3, if there are 0 (there cannot be 1) it is rank 1
//here we must make sure that it is a boundary node as well.
  Obj<ALE::Mesh::topology_type> topology = mesh->getTopology();
  return 1; // stub
}

bool areCoPlanar(Obj<ALE::Mesh>& mesh, ALE::Mesh::point_type tri1, ALE::Mesh::point_type tri2) {
  Obj<ALE::Mesh::topology_type> topology = mesh->getTopology();
    
  return false; // stub
}

#ifdef PETSC_HAVE_TRIANGLE
PetscErrorCode TriangleToMesh(Obj<ALE::Mesh> mesh, triangulateio * src, ALE::Mesh::real_section_type::patch_type patch) {
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
  ALE::New::SieveBuilder<ALE::Mesh>::buildTopology(sieve, 2, src->numberoftriangles, src->trianglelist, src->numberofpoints, false, 3);
  sieve->stratify();
  topology->setPatch(patch, sieve);
  // Actually we probably only want to stratify at the end, so that we do not recalculate a lot
  topology->stratify();
  PetscFunctionReturn(0);
}
#endif
}  //end Coarsener
}  //end ALE
