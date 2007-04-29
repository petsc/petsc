//rewritten Hierarchy.h made explicitly to output directly to DMMG
#include <list>
#include <Mesh.hh>
#include <stdlib.h>
#include <string>
//#include <triangle.h>

//#include "petscmesh.h"
#include "petscdmmg.h"
//#include "petscmat.h"
#include "private/meshimpl.h"   /*I      "petscmesh.h"   I*/
#include <Distribution.hh>
#include <Generator.hh>
//helper functions:

void SetupTriangulateio(triangulateio *, triangulateio *);
void TeardownTriangulateio(triangulateio *, triangulateio *);
//void SetupTetgenio(tetgenio *, tetgenio *);
bool PointIsInElement(ALE::Obj<ALE::Mesh> m, ALE::Mesh::point_type element, double * point);
//double PointDist(double * pointA, double * pointB, int dim);

// Functions only used here

PetscErrorCode MeshSpacingFunction(Mesh mesh); //builds the spacing function for the mesh
PetscErrorCode MeshIDBoundary(Mesh mesh); //finds the boundary of the mesh.

// DMMG Top Function

PetscErrorCode DMMGFunctionBasedCoarsen(Mesh finemesh, int levels, double coarsen_factor, DMMG dmmg);  //sets up the DMMG object to do this

//SetupTriangulateio: set all the fields of the triangulateio structures to be good for initial input/output

void SetupTriangulateio(triangulateio * input, triangulateio * output) {
  input->numberofsegments = 0;
  input->segmentlist = NULL;
  input->numberoftriangles = 0;
  input->numberofcorners = 0;
  input->numberofpointattributes = 0;
  input->pointattributelist = NULL;
  input->numberoftriangleattributes = 0;
  input->trianglelist = NULL;
  input->triangleattributelist = NULL;
  input->trianglearealist = NULL;
  input->segmentmarkerlist = NULL;
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
  return;
}

double Curvature_2D(ALE::Obj<ALE::Mesh> m, ALE::Mesh::point_type p) {
  PetscErrorCode ierr;
  const ALE::Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  ALE::Obj<ALE::Mesh::label_type> boundary = m->getLabel("marker");
  int dim = m->getDimension();
  if (dim != 2) throw ALE::Exception("Called the 2D curvature routine on a non-2D mesh.");
  double curvature;
  double pCoords[dim], qCoords[dim], rCoords[dim];
  const double * tmpCoords;
  double normvec[dim];
  int levels = m->height(p); //allows for interpolated or noninterpolated cases
  
  const ALE::Obj<ALE::Mesh::sieve_type::supportSet> neighbors = m->getSieve()->cone(m->getSieve()->support(p)); //get the neighboring points
  ALE::Mesh::sieve_type::supportSet::iterator n_iter = neighbors->begin();
  ALE::Mesh::sieve_type::supportSet::iterator n_iter_end = neighbors->end();
  std::list<ALE::Mesh::point_type> edgnlist;
  while (n_iter != n_iter_end) {
    if (m->debug()) PetscPrintf(m->comm(), "size %d\n", m->getSieve()->nJoin(p, *n_iter, levels)->size());
    if ((*n_iter != p) && (m->getSieve()->nJoin(p, *n_iter, levels)->size() == levels)) {//we have to test that getting to this node from the original goes along a boundary edge.
      edgnlist.push_front(*n_iter);
    }
    n_iter++;
  }
  if (edgnlist.size() != 2) throw ALE::Exception("There is either a pathological boundary here, or this algorithm is wrong!");
  //ok, we have an arc. n1 -> p -> n2  we want to go through the arc in order, in order to get the normal.
  ALE::Mesh::point_type n1 = *edgnlist.begin();
  ALE::Mesh::point_type n2 = *(++edgnlist.begin());

  ierr = PetscMemcpy(pCoords, coordinates->restrictPoint(p), dim*sizeof(double));
  ierr = PetscMemcpy(qCoords, coordinates->restrictPoint(n1), dim*sizeof(double));
  ierr = PetscMemcpy(rCoords, coordinates->restrictPoint(n2), dim*sizeof(double));

  if (m->debug()) PetscPrintf(m->comm(), "Edges: %d--%d--%d : (%f, %f)--(%f, %f)--(%f, %f)\n", n1, p, n2, qCoords[0], qCoords[1], pCoords[0], pCoords[1], rCoords[0], rCoords[1]);

  normvec[0] = pCoords[1] - qCoords[1];
  normvec[1] = qCoords[0] - pCoords[0];

  normvec[0] += rCoords[1] - pCoords[1];
  normvec[1] += pCoords[0] - rCoords[0];
  //normalize the normal.
  double normlen = sqrt(normvec[0]*normvec[0] + normvec[1]*normvec[1]);
  if (normlen < 0.000000000001) return 0.; //give up
  normvec[0] = normvec[0]/normlen;
  normvec[1] = normvec[1]/normlen;
  if (m->debug()) PetscPrintf(m->comm(), "normal: (%f, %f)\n", normvec[0], normvec[1]);
  //ok, take the min dot product of this with the two edges used before.
  double qnorm = sqrt((pCoords[0] - qCoords[0])*(pCoords[0] - qCoords[0]) + (pCoords[1] - qCoords[1])*(pCoords[1] - qCoords[1]));
  double rnorm = sqrt((rCoords[0] - pCoords[0])*(rCoords[0] - pCoords[0]) + (rCoords[1] - pCoords[1])*(rCoords[1] - pCoords[1]));
  double c1 = ((qCoords[1] - pCoords[1])*normvec[1]+(qCoords[0] - pCoords[0])*normvec[0])/qnorm;
  double c2 = ((rCoords[1] - pCoords[1])*normvec[1]+(rCoords[0] - pCoords[0])*normvec[0])/rnorm;
  if (fabs(c1) > fabs(c2)) return fabs(c2);
  return fabs(c1);
}
//MeshSpacingFunction: Build the spacing function in the "spacing" section on the mesh.  

PetscErrorCode MeshSpacingFunction(Mesh mesh) {
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);
  int dim = m->getDimension();
  //setup the spacing section
  const ALE::Obj<ALE::Mesh::real_section_type>& spacing = m->getRealSection("spacing");
  spacing->setFiberDimension(m->depthStratum(0), 1);
  m->allocate(spacing);
  //vertices
  const ALE::Obj<ALE::Mesh::real_section_type>&  coordinates = m->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::label_sequence>& vertices = m->depthStratum(0);
  ALE::Mesh::label_sequence::iterator v_iter = vertices->begin();
  ALE::Mesh::label_sequence::iterator v_iter_end = vertices->end();
  double vCoords[3], nCoords[3];
  while (v_iter != v_iter_end) {
    const double * tmpCoords = coordinates->restrictPoint(*v_iter);
    for (int i = 0; i < dim; i++) {
      vCoords[i] = tmpCoords[i];
    }
    //get the neighbors
    ALE::Obj<ALE::Mesh::sieve_type::coneSet> neighbors = m->getSieve()->cone(m->getSieve()->support(*v_iter));
    ALE::Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
    ALE::Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
    //go through the neighbors
    double minDist = 0.;
    while (n_iter != n_iter_end) {
      double dist = 0.;
      const double * rBuf = coordinates->restrictPoint(*n_iter);
      PetscMemcpy(nCoords, rBuf, dim*sizeof(double));
      double d_tmp;
      for (int d = 0; d < dim; d++) {
	d_tmp = nCoords[d] - vCoords[d];
	dist += d_tmp * d_tmp;
      }
      if ((dist < minDist && dist > 0.) || minDist == 0.) minDist = dist;
      n_iter++;
    }
    minDist = sqrt(minDist);
    spacing->updatePoint(*v_iter, &minDist);
    v_iter++;
  }
  PetscFunctionReturn(0);
}

//MeshIDBoundary: create the "marker" label needed by many such things
PetscErrorCode MeshIDBoundary(Mesh mesh) {
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);
  int dim = m->getDimension();
  ALE::Obj<ALE::Mesh::label_type> boundary = m->createLabel("marker");
  const ALE::Obj<ALE::Mesh::label_sequence>& vertices = m->depthStratum(0);
  ALE::Mesh::label_sequence::iterator v_iter = vertices->begin();
  ALE::Mesh::label_sequence::iterator v_iter_end = vertices->end();
  //to make it work for interpolated and noninterpolated meshes we will only work with the top and bottom
  int interplevels = m->height(*v_iter);
  if (interplevels == 1) { //noninterpolated case
    while (v_iter != v_iter_end) {
      ALE::Obj<ALE::Mesh::sieve_type::supportArray> vsupport = m->getSieve()->nSupport(*v_iter, interplevels);
      ALE::Mesh::sieve_type::supportArray::iterator s_iter = vsupport->begin();
      ALE::Mesh::sieve_type::supportArray::iterator s_iter_end = vsupport->end();
      bool isBound = false;
      while (s_iter != s_iter_end) {
        //check that each of the neighbor vertices is represented as part of the closure of at least dim of the surrounding volumes
        //this means that join(*v_iter, *n_iter) should be greater than or equal to dim for interior nodes.
        ALE::Obj<ALE::Mesh::sieve_type::coneArray> neighbors = m->getSieve()->nCone(*s_iter, interplevels);
        ALE::Mesh::sieve_type::supportArray::iterator n_iter = neighbors->begin();
        ALE::Mesh::sieve_type::supportArray::iterator n_iter_end = neighbors->end();
        while (n_iter != n_iter_end) {
          if (m->getSieve()->join(*v_iter, *n_iter)->size() < (unsigned int)dim) {
            isBound = true;
          }
          n_iter++;
        }
        s_iter++;
      }
      if (isBound) m->setValue(boundary, *v_iter, 1);
      v_iter++;
    }
  } else { //interpolated case -- easier
    while (v_iter != v_iter_end) {
      ALE::Obj<ALE::Mesh::sieve_type::supportArray> vsupport = m->getSieve()->nSupport(*v_iter, interplevels-1); //3D faces or 2D edges
      ALE::Mesh::sieve_type::supportArray::iterator s_iter = vsupport->begin();
      ALE::Mesh::sieve_type::supportArray::iterator s_iter_end = vsupport->end();
      bool isBound = false;
      while (s_iter != s_iter_end) {
        //check the support of each dim-1 element; if it's supported on one side it is attached to boundary nodes
        ALE::Obj<ALE::Mesh::sieve_type::supportSequence> fsupport = m->getSieve()->support(*s_iter);
        if (fsupport->size() < 2) isBound = true;
        s_iter++;
      }
      if (isBound) m->setValue(boundary, *v_iter, 1);
      v_iter++;
    }
  }
  PetscFunctionReturn(0);
}

//MeshCoarsenMesh: Do a naive top-level coarsen based upon an assumed spacing function
//REQUIRES: both meshes initialized, finemesh has section "spacing" initialized with NN info
//finemesh has the boundary labeled in "marker"
//O(n_fine * n_coarsest) with n_coarsest assumed to be tuned (through coarsen_factor) to be constant.

PetscErrorCode MeshCoarsenMesh(Mesh finemesh, double coarsen_factor, Mesh * outmesh) {
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  //BOUNDARY Coarsen: pick a nonintersecting set of boundary balls.
  ierr = MeshGetMesh(finemesh, m);
  std::list<ALE::Mesh::point_type> incPoints;
  int dim = m->getDimension();
  const ALE::Obj<ALE::Mesh::real_section_type>&  coordinates = m->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::real_section_type>& spacing = m->getRealSection("spacing");
  ALE::Obj<ALE::Mesh::label_sequence> boundpoints = m->getLabelStratum("marker", 0);
  ALE::Mesh::label_sequence::iterator v_iter = boundpoints->begin();
  ALE::Mesh::label_sequence::iterator v_iter_end = boundpoints->end();
  double vCoords[dim];
  const double * tmpCoords;
  double vSpace;
  double dist;

  while (v_iter != v_iter_end) {
    bool can_include = true;
    tmpCoords = coordinates->restrictPoint(*v_iter);
    for (int i = 0; i < dim; i++) {
      vCoords[i] = tmpCoords[i];
    }
    vSpace = *spacing->restrictPoint(*v_iter);
    std::list<ALE::Mesh::point_type>::iterator inc_iter = incPoints.begin();
    std::list<ALE::Mesh::point_type>::iterator inc_iter_end = incPoints.end();
    while (inc_iter != inc_iter_end && can_include) {
      double iSpace = *spacing->restrictPoint(*inc_iter);
      tmpCoords = coordinates->restrictPoint(*inc_iter);
      dist = 0;
      for (int i = 0; i < dim; i++) {
        dist += (tmpCoords[i] - vCoords[i])*(tmpCoords[i] - vCoords[i]);
      }
      dist = sqrt(dist);
      if (dist < 0.5*coarsen_factor*(vSpace + iSpace)) can_include = false;
      inc_iter++;
    }
    if (can_include) {
      incPoints.push_front(*v_iter); //include it in the set.
    }
   v_iter++;
  }
  //now try to include the other points
  ALE::Obj<ALE::Mesh::label_sequence> vertices = m->depthStratum(0);
  v_iter = vertices->begin();
  v_iter_end = vertices->end();
  while (v_iter != v_iter_end) {
    bool can_include = true;
    tmpCoords = coordinates->restrictPoint(*v_iter);
    for (int i = 0; i < dim; i++) {
      vCoords[i] = tmpCoords[i];
    }
    vSpace = *spacing->restrictPoint(*v_iter);
    std::list<ALE::Mesh::point_type>::iterator inc_iter = incPoints.begin();
    std::list<ALE::Mesh::point_type>::iterator inc_iter_end = incPoints.end();
    while (inc_iter != inc_iter_end && can_include) {
      double iSpace = *spacing->restrictPoint(*inc_iter);
      tmpCoords = coordinates->restrictPoint(*inc_iter);
      dist = 0;
      for (int i = 0; i < dim; i++) {
        dist += (tmpCoords[i] - vCoords[i])*(tmpCoords[i] - vCoords[i]);
      }
      dist = sqrt(dist);
      if (dist < 0.5*coarsen_factor*(vSpace + iSpace)) can_include = false;
      inc_iter++;
    }
    if (can_include) {
      incPoints.push_front(*v_iter); //include it in the set.
    }
    v_iter++;
  }
  PetscPrintf(MPI_COMM_WORLD, "%d included at the toplevel\n", incPoints.size());
  //create a coordinate array from the list we have made
  double coords[dim * incPoints.size()];
  int indices[dim * incPoints.size()];
  std::list<ALE::Mesh::point_type>::iterator inc_iter = incPoints.begin();
  std::list<ALE::Mesh::point_type>::iterator inc_iter_end = incPoints.end();
  int index = 0;
  const double * tmpcoords;
  while (inc_iter != inc_iter_end) {
    tmpcoords = coordinates->restrictPoint(*inc_iter);
    for (int i = 0; i < dim; i++) {
      coords[index*dim+i] = tmpcoords[i];
    }
    index++;
    inc_iter++;
  }
  //call triangle or tetgen: turns out the options we want on are the same
  std::string triangleOptions = "-zQe"; //(z)ero indexing, output (e)dges, Quiet
  double * finalcoords;
  int * connectivity;
  int nelements;
  int nverts;
  if (dim == 2) {
    triangulateio tridata[2];
    SetupTriangulateio(&tridata[0], &tridata[1]);
    tridata[0].pointlist = coords;
    tridata[0].numberofpoints = incPoints.size();
    tridata[0].pointmarkerlist = indices;
    //triangulate
    triangulate((char *)triangleOptions.c_str(), &tridata[0], &tridata[1], NULL);
    finalcoords = tridata[1].pointlist;
    connectivity = tridata[1].trianglelist;
    nelements = tridata[1].numberoftriangles;
    nverts = tridata[1].numberofpoints;
  } else if (dim == 3) {
    tetgenio * tetdata = new tetgenio[2];
    //push the points into the thing
    tetdata[0].pointlist = coords;
    tetdata[0].pointmarkerlist = indices;
    tetdata[0].numberofpoints = incPoints.size();
    //tetrahedralize
    tetrahedralize((char *)triangleOptions.c_str(), &tetdata[0], &tetdata[1]);
    finalcoords = tetdata[1].pointlist;
    connectivity = tetdata[1].tetrahedronlist;
    nelements = tetdata[1].numberoftetrahedra;
    nverts = tetdata[1].numberofpoints;
  }
  //make it into a mesh;
  ALE::Obj<ALE::Mesh::sieve_type> sieve = new ALE::Mesh::sieve_type(m->comm(), m->debug());
  ALE::SieveBuilder<ALE::Mesh>::buildTopology(sieve, dim, nelements, connectivity, nverts, false, dim+1, nelements);
  ALE::Obj<ALE::Mesh> newmesh = new ALE::Mesh(m->comm(), m->debug());
  newmesh->setDimension(dim);
  newmesh->setSieve(sieve);
  newmesh->stratify();
  ALE::SieveBuilder<ALE::Mesh>::buildCoordinates(newmesh, dim, finalcoords);
  //remove trivial elements; ie ones where all the tri/tet corners are marked as boundary points.
  PetscPrintf(m->comm(), "%d points, %d elements in the new mesh.\n", newmesh->depthStratum(0)->size(), newmesh->heightStratum(0)->size());
  ierr = MeshSetMesh(*outmesh, newmesh);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshCreateHierarchLabel"

//MeshCreateHierarchyLabel: Create a label that tells what the highest level a given vertex appears in where 0 is fine and n is coarsest.
PetscErrorCode MeshCreateHierarchyLabel(Mesh finemesh, double beta, int nLevels, Mesh * outmeshes, Mat * outmats = PETSC_NULL) {
  PetscErrorCode ierr;
  ALE::Obj<ALE::Mesh> m;
  PetscFunctionBegin;
  ierr = MeshGetMesh(finemesh, m);CHKERRQ(ierr);
  int dim = m->getDimension();
  const ALE::Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::real_section_type>& spacing = m->getRealSection("spacing");
  const ALE::Obj<ALE::Mesh::label_type> hdepth = m->createLabel("hdepth");
  const ALE::Obj<ALE::Mesh::label_type> dompoint = m->createLabel("dompoint");
  const ALE::Obj<ALE::Mesh::label_type> traversal = m->createLabel("traversal");
  const ALE::Obj<ALE::Mesh::label_type>& boundary = m->getLabel("marker");
  const ALE::Obj<ALE::Mesh::label_sequence>& vertices = m->depthStratum(0);
  ALE::Mesh::label_sequence::iterator v_iter = vertices->begin();
  ALE::Mesh::label_sequence::iterator v_iter_end = vertices->end();
  double maxspace = -1., minspace = -1.;
  while(v_iter != v_iter_end) {
    //initialize the label to 0.
    m->setValue(hdepth, *v_iter, 0);
    //discover the maximum and minimum spacing functions in the mesh.
    double vspace = *spacing->restrictPoint(*v_iter);
    if ((vspace > maxspace) || (maxspace == -1.)) maxspace = vspace;
    if ((vspace < minspace) || (minspace == -1.)) minspace = vspace;
    v_iter++;
  }
  //PUT IN PART FOR AUTOMATICALLY ADDING HIGH-CURVATURE BOUNDARY NODES
  const ALE::Obj<ALE::Mesh::label_sequence>& boundaryvertices = m->getLabelStratum("marker", 1); //boundary
  ALE::Mesh::label_sequence::iterator bv_iter = boundaryvertices->begin();
  ALE::Mesh::label_sequence::iterator bv_iter_end = boundaryvertices->end();
  PetscPrintf(m->comm(), "NUMBER OF BOUNDARY POINTS: %d\n", boundaryvertices->size());
  while (bv_iter != bv_iter_end) {
    if (Curvature_2D(m, *bv_iter) > 0.01) {
      m->setValue(hdepth, *bv_iter, nLevels-1);
    }
    bv_iter++;
  }
  PetscPrintf(m->comm(), "Forced in %d especially curved boundary nodes.\n", m->getLabelStratum("hdepth", nLevels-1)->size());
  double bvCoords[dim];
  std::list<ALE::Mesh::point_type> complist;
  std::list<ALE::Mesh::point_type> domlist; //stores the points dominated by the current point.
  int curmeshsize = 0; //the size of the current mesh
  for (int curLevel = nLevels-1; curLevel > 0; curLevel--) {
    double curBeta = pow(beta, curLevel);
   //OUR MODUS OPERANDI:
    //1. do the boundary and the interior identically but separately
    //2. keep track of the point that eliminates each point on each level.  This should work sort of like an approximation to the voronoi partitions.  Compare against these first as they're more likely to collide than neighbors.  Also compare to the points that eliminate the neighbors in the same fashion.
    //3. If the point is not eliminated by its old eliminator we must traverse out to max(space(v)) + space(i).
    //GOAL: only eliminate each point once! if we add a point that eliminates other points get rid of them in the traversal! (and set their elimination marker appropriately.)
    ALE::Mesh::label_sequence::iterator bv_iter = boundaryvertices->begin();
    ALE::Mesh::label_sequence::iterator bv_iter_end = boundaryvertices->end();
    while (bv_iter != bv_iter_end) {
      ALE::Mesh::point_type bvdom = m->getValue(dompoint, *bv_iter);
      bool skip = false;
      if (bvdom != -1) {
        if (m->getValue(hdepth, bvdom) == curLevel) skip = true; 
      }
      bool canAdd = true;
      if (m->getValue(hdepth, *bv_iter) == 0 && !skip) { //if not yet included or invalidated
        m->setValue(traversal, *bv_iter, 1);
        double bvSpace = *spacing->restrictPoint(*bv_iter);
        ierr = PetscMemcpy(bvCoords, coordinates->restrictPoint(*bv_iter), dim*sizeof(double));
        //get its neighbors and add them to the comparison queue.
        ALE::Obj<ALE::Mesh::sieve_type::coneSet> neighbors = m->getSieve()->cone(m->getSieve()->support(*bv_iter));
        ALE::Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
        ALE::Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
        while (n_iter != n_iter_end) {
          if (m->getValue(boundary, *n_iter) == 1) {
            m->setValue(traversal, *n_iter, 1);
            complist.push_front(*n_iter);
          }
          n_iter++;
        }
        //push the last point to invalidate the current point to the front of the list of comparisons.
        if (bvdom != -1) {
           complist.push_front(bvdom);
        }

        while ((!complist.empty()) && canAdd) {
          ALE::Mesh::point_type curpt = *complist.begin();
          complist.pop_front();
          double dist = 0.;
          double curSpace = *spacing->restrictPoint(curpt);
          const double * curCoords = coordinates->restrictPoint(curpt); 
          for (int i = 0; i < dim; i++) {
            dist += (curCoords[i] - bvCoords[i])*(curCoords[i] - bvCoords[i]);
          }
          dist = sqrt(dist);
          ALE::Mesh::point_type curpt_dom = m->getValue(dompoint, curpt);
          int curpt_depth = m->getValue(hdepth, curpt);
          int curpt_bound = m->getValue(boundary, curpt);
          if ((dist < 0.5*curBeta*(bvSpace + curSpace))&&(curpt_depth > 0)) { //collision with an already added node
            canAdd = false;
            m->setValue(dompoint, *bv_iter, curpt);
          } else if (dist < 0.5*curBeta*(bvSpace + maxspace)) { 
            ALE::Obj<ALE::Mesh::sieve_type::coneSet> cneighbors = m->getSieve()->cone(m->getSieve()->support(curpt));
            ALE::Mesh::sieve_type::coneSet::iterator cn_iter = cneighbors->begin();
            ALE::Mesh::sieve_type::coneSet::iterator cn_iter_end = cneighbors->end();
            while (cn_iter != cn_iter_end) {
              if ((curpt_bound == 1) && (m->getValue(traversal, *cn_iter) == 0)) {
                m->setValue(traversal, *cn_iter, 1);
                complist.push_back(*cn_iter);
              }
              cn_iter++;
            }
          }
          if ((dist < 0.5*curBeta*(bvSpace + curSpace)) && (curpt_depth == 0)) { //add the point to the list of points dominated by this node; points eliminated in one step later
            domlist.push_front(curpt);
            if (curpt_dom != -1) {
              if (m->getValue(traversal, curpt_dom) == 0) {
                complist.push_front(curpt_dom);
                m->setValue(traversal, curpt_dom, 1);
              }
            }
          }
        }  //end of complist deal
        complist.clear();
        if (canAdd == true) { 
          m->setValue(hdepth, *bv_iter, curLevel);
          std::list<ALE::Mesh::point_type>::iterator dom_iter = domlist.begin();
          std::list<ALE::Mesh::point_type>::iterator dom_iter_end = domlist.end();
          while (dom_iter != dom_iter_end) {
            m->setValue(dompoint, *dom_iter, *bv_iter);
            dom_iter++;
          }
        }
        domlist.clear();
        //unset the traversal listing
        ALE::Obj<ALE::Mesh::label_sequence> travnodes = m->getLabelStratum("traversal", 1);
        ALE::Mesh::label_sequence::iterator tn_iter = travnodes->begin();
        ALE::Mesh::label_sequence::iterator tn_iter_end = travnodes->end();
        while (tn_iter != tn_iter_end) {
          complist.push_front(*tn_iter);
          tn_iter++;
        }
        while (!complist.empty()) {
          ALE::Mesh::point_type emptpt = *complist.begin();
          complist.pop_front();
          m->setValue(traversal, emptpt, 0);
        }
      }
      bv_iter++;
    }
    PetscPrintf(m->comm(), "Added %d new boundary vertices\n", m->getLabelStratum("hdepth", curLevel)->size());
    //INTERIOR NODES:
    ALE::Obj<ALE::Mesh::label_sequence> intverts = m->depthStratum(0);
    bv_iter = intverts->begin();
    bv_iter_end = intverts->end();
    while (bv_iter != bv_iter_end) {
      ALE::Mesh::point_type bvdom = m->getValue(dompoint, *bv_iter);
      bool skip = false;
      if (bvdom != -1) {
        if (m->getValue(hdepth, bvdom) == curLevel) skip = true; 
      }
      m->setValue(traversal, *bv_iter, 1);
      bool canAdd = true;
      if ((m->getValue(boundary, *bv_iter) != 1) && (m->getValue(hdepth, *bv_iter) == 0) && !skip) { //if not in the boundary and not included (or excluded)
        double bvSpace = *spacing->restrictPoint(*bv_iter);
        ALE::Mesh::point_type bvdom = m->getValue(dompoint, *bv_iter);
        ierr = PetscMemcpy(bvCoords, coordinates->restrictPoint(*bv_iter), dim*sizeof(double));
        //get its neighbors and add them to the comparison queue.
        ALE::Obj<ALE::Mesh::sieve_type::coneSet> neighbors = m->getSieve()->cone(m->getSieve()->support(*bv_iter));
        ALE::Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
        ALE::Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
        while (n_iter != n_iter_end) {
          if (m->getValue(boundary, *n_iter) != 1) {
            m->setValue(traversal, *n_iter, 1);
            complist.push_front(*n_iter);
          }
          n_iter++;
        }
        if (bvdom != -1) {
           complist.push_front(bvdom);
        }
        while ((!complist.empty()) && canAdd) {
          ALE::Mesh::point_type curpt = *complist.begin();
          complist.pop_front();
          double dist = 0.;
          double curSpace = *spacing->restrictPoint(curpt);
          const double * curCoords = coordinates->restrictPoint(curpt); 
          for (int i = 0; i < dim; i++) {
            dist += (curCoords[i] - bvCoords[i])*(curCoords[i] - bvCoords[i]);
          }
          dist = sqrt(dist);
          int curpt_depth = m->getValue(hdepth, curpt);
          int curpt_bound = m->getValue(boundary, curpt);
          ALE::Mesh::point_type curpt_dom = m->getValue(dompoint, curpt);
          if ((dist < 0.5*curBeta*(bvSpace + curSpace))&&(curpt_depth > 0)) {
            canAdd = false;
            m->setValue(dompoint, *bv_iter, curpt);
          } else if ((dist < 0.5*curBeta*(bvSpace)) && (curpt_bound == 1)) {
            canAdd = false;
            m->setValue(dompoint, *bv_iter, curpt);
          } else if (dist < 0.5*curBeta*(bvSpace + maxspace)) { 
            ALE::Obj<ALE::Mesh::sieve_type::coneSet> cneighbors = m->getSieve()->cone(m->getSieve()->support(curpt));
            ALE::Mesh::sieve_type::coneSet::iterator cn_iter = cneighbors->begin();
            ALE::Mesh::sieve_type::coneSet::iterator cn_iter_end = cneighbors->end();
            while (cn_iter != cn_iter_end) {
              if ((m->getValue(boundary, *cn_iter) != 1) && (m->getValue(traversal, *cn_iter) != 1)) {
                m->setValue(traversal, *cn_iter, 1);
                complist.push_back(*cn_iter);
              }
              cn_iter++;
            }
          }
          if ((dist < 0.5*curBeta*(bvSpace + curSpace)) && (curpt_depth == 0)) {
            domlist.push_front(curpt);
            if (curpt_dom != -1) {
              if (m->getValue(traversal, curpt_dom) == 0) {
                complist.push_front(curpt_dom);
                m->setValue(traversal, curpt_dom, 1);
              }
            }
          }
        }  //end of complist deal
        complist.clear();
        if (canAdd == true) { 
          m->setValue(hdepth, *bv_iter, curLevel);
          std::list<ALE::Mesh::point_type>::iterator dom_iter = domlist.begin();
          std::list<ALE::Mesh::point_type>::iterator dom_iter_end = domlist.end();
          while (dom_iter != dom_iter_end) {
            m->setValue(dompoint, *dom_iter, *bv_iter);
            dom_iter++;
          }
        } 
        domlist.clear();
        complist.clear();
        //unset the traversal listing
        ALE::Obj<ALE::Mesh::label_sequence> travnodes = m->getLabelStratum("traversal", 1);
        ALE::Mesh::label_sequence::iterator tn_iter = travnodes->begin();
        ALE::Mesh::label_sequence::iterator tn_iter_end = travnodes->end();
        while (tn_iter != tn_iter_end) {
          complist.push_front(*tn_iter);
          tn_iter++;
        }
        while (!complist.empty()) {
          ALE::Mesh::point_type emptpt = *complist.begin();
          complist.pop_front();
          m->setValue(traversal, emptpt, 0);
        }
      }
      bv_iter++;
    }
    PetscPrintf(m->comm(), "Included %d new points in level %d\n", m->getLabelStratum("hdepth", curLevel)->size(), curLevel);
    curmeshsize += m->getLabelStratum("hdepth", curLevel)->size();
    //MESHING AND CONTINUITY CHECKING: MAKE SURE:
    //1. ELIMINATE COMPLETELY CONSTRAINED ELEMENTS, BEING ONES ON WHICH ALL CORNERS ARE BOUNDARY PLACES.
    //2. MAKE SURE THAT NO INTERNAL NODES ARE IN THE BOUNDARY.  IF AN INTERNAL NODE IS IN THE BOUNDARY, PUT THEM BACK TO LEVEL '0' AND REMESH.  REPEAT UNTIL SANE (THIS REALLY SHOULDN'T HAPPEN GIVEN OUR POINT ADDITION CRITERIA).
    //load the points and their names in this mesh into a list
    //triangulate/tetrahedralize
    //make into a new sieve.  place coordinates and names on the sieve.



    double coords[dim * curmeshsize];
    int indices[dim * curmeshsize];
    const double * tmpcoords;
    int index = 0;
    for (int i = curLevel; i < nLevels; i++) {
      ALE::Obj<ALE::Mesh::label_sequence> curLevVerts = m->getLabelStratum("hdepth", i);
      ALE::Mesh::label_sequence::iterator clv_iter = curLevVerts->begin();
      ALE::Mesh::label_sequence::iterator clv_iter_end = curLevVerts->end();
      while (clv_iter != clv_iter_end) {
        tmpcoords = coordinates->restrictPoint(*clv_iter);
        for (int j = 0; j < dim; j++) {
          coords[index*dim+j] = tmpcoords[j];
        }
        indices[index] = *clv_iter;
        index++;
        clv_iter++;
      }
    }
    //call triangle or tetgen: turns out the options we want on are the same
    std::string triangleOptions = "-zQe"; //(z)ero indexing, output (e)dges, Quiet
    double * finalcoords;
    int * connectivity;
    int * oldpositions;
    int nelements;
    int nverts;
    if (dim == 2) {
      triangulateio tridata[2];
      SetupTriangulateio(&tridata[0], &tridata[1]);
      tridata[0].pointlist = coords;
      tridata[0].numberofpoints = curmeshsize;
      tridata[0].pointmarkerlist = indices;
      //triangulate
      triangulate((char *)triangleOptions.c_str(), &tridata[0], &tridata[1], NULL);
      finalcoords = tridata[1].pointlist;
      connectivity = tridata[1].trianglelist;
      oldpositions = tridata[1].pointmarkerlist;
      nelements = tridata[1].numberoftriangles;
      nverts = tridata[1].numberofpoints;
    } else if (dim == 3) {
      tetgenio * tetdata = new tetgenio[2];
      //push the points into the thing
      tetdata[0].pointlist = coords;
      tetdata[0].pointmarkerlist = indices;
      tetdata[0].numberofpoints = curmeshsize;
      //tetrahedralize
      tetrahedralize((char *)triangleOptions.c_str(), &tetdata[0], &tetdata[1]);
      finalcoords = tetdata[1].pointlist;
      connectivity = tetdata[1].tetrahedronlist;
      oldpositions = tetdata[1].pointmarkerlist;
      nelements = tetdata[1].numberoftetrahedra;
      nverts = tetdata[1].numberofpoints;
    }
    //make it into a mesh;
    ALE::Obj<ALE::Mesh::sieve_type> sieve = new ALE::Mesh::sieve_type(m->comm(), m->debug());
    ALE::SieveBuilder<ALE::Mesh>::buildTopology(sieve, dim, nelements, connectivity, nverts, false, dim+1, nelements);
    ALE::Obj<ALE::Mesh> newmesh = new ALE::Mesh(m->comm(), m->debug());
    newmesh->setDimension(dim);
    newmesh->setSieve(sieve);
    newmesh->stratify();
    ALE::SieveBuilder<ALE::Mesh>::buildCoordinates(newmesh, dim, finalcoords);
    //UPDATE THE MARKER AND FINEMESH VERTEX NUMBERING LABELS
    ALE::Obj<ALE::Mesh::label_type> boundary_new = newmesh->createLabel("marker");
    ALE::Obj<ALE::Mesh::label_type> fine_corresponds = newmesh->createLabel("fine");
    ALE::Obj<ALE::Mesh::label_sequence> newverts = newmesh->depthStratum(0);
    ALE::Mesh::label_sequence::iterator nv_iter = newverts->begin();
    ALE::Mesh::label_sequence::iterator nv_iter_end = newverts->end();
    while (nv_iter != nv_iter_end) {
      newmesh->setValue(fine_corresponds, *nv_iter, oldpositions[*nv_iter - nelements]);
      if(m->getValue(boundary, oldpositions[*nv_iter - nelements]) == 1) newmesh->setValue(boundary_new, *nv_iter, 1);
      nv_iter++;
    }
    PetscPrintf(m->comm(), "%d boundary vertices here\n", newmesh->getLabelStratum("marker", 1)->size());
    //eliminate the completely constrained triangles.
    ALE::Obj<ALE::Mesh::label_sequence> coarsele = newmesh->heightStratum(0);
    int nRemoved = 0;
    int intlevels = newmesh->depth(*coarsele->begin());
    ALE::Mesh::label_sequence::iterator ce_iter = coarsele->begin();
    ALE::Mesh::label_sequence::iterator ce_iter_end = coarsele->end();
    std::list<ALE::Mesh::point_type> rempoints;
    while (ce_iter != ce_iter_end) {
      ALE::Obj<ALE::Mesh::sieve_type::coneArray> children = newmesh->getSieve()->nCone(*ce_iter, intlevels); //get the vertices here.
      //printf("%d", children->size());
      ALE::Mesh::sieve_type::coneArray::iterator ch_iter = children->begin();
      ALE::Mesh::sieve_type::coneArray::iterator ch_iter_end = children->end();
      bool canRemove = true;
      while (ch_iter != ch_iter_end) {
        //if ANY of them are the interior vertex, OR this is the only simplex supporting the given point, we cannot do this
        if ((newmesh->getSieve()->nSupport(*ch_iter, intlevels)->size() < 2) || (newmesh->getValue(boundary_new, *ch_iter) != 1)) canRemove = false;
        //printf(" %d", newmesh->getValue(boundary_new, *ch_iter));
        ch_iter++;
      }
      //printf("\n");
      if (canRemove) {
        rempoints.push_front(*ce_iter);
        nRemoved++;
      }
      ce_iter++;
    }
    PetscPrintf(m->comm(), "Removed %d trivial cells\n", nRemoved);
    while (!rempoints.empty()) {
      ALE::Mesh::point_type currem = *rempoints.begin();
      rempoints.pop_front();
      newmesh->getSieve()->removeCapPoint(currem);
      newmesh->getSieve()->removeBasePoint(currem);
    }
    //if interpolated remove any lower-dimensional simplices that have been support-orphaned in this process.
    for (int i = 1; i < intlevels-1; i++) {
      //at each stage remove the orphaned simplices.
      ALE::Obj<ALE::Mesh::label_sequence> cursim = newmesh->heightStratum(i);
      ALE::Mesh::label_sequence::iterator cs_iter = cursim->begin();
      ALE::Mesh::label_sequence::iterator cs_iter_end = cursim->end();
      while (cs_iter != cs_iter_end) {
         if (newmesh->getSieve()->support(*cs_iter)->size() == 0)rempoints.push_front(*cs_iter);
      }
      while (!rempoints.empty()) {
        ALE::Mesh::point_type currem = *rempoints.begin();
        newmesh->getSieve()->removeCapPoint(currem);
        newmesh->getSieve()->removeBasePoint(currem);
      }
    }
    //check the border. LATER
    //repeat if broken.
    MeshSetMesh(outmeshes[curLevel-1], newmesh);
    //BUILD THE INTERPOLATION OPERATORS (admittedly for P1.)
    //MODUS OPERANDI:  TAKE THE DOMINATING POINT OF EVERY POINT IN THIS MESH THAT ISN'T IN THE NEXT MESH UP.  IF THE DOMINATING POINT ALSO ISN'T THERE IT IS A BOUNDARY, SO TAKE *ITS* DOMINATING POINT.
    //TAKE THE TRIANGLES ON THE PERIPHERY OF THE DOMINATING POINT IN THE NEXT MESH UP AND TEST AGAINST THEM.  IF NOT FOUND THEN TAKE THEIR NEIGHBORS AND DO THE SAME.  REPEAT UNTIL FOUND OR GIVE UP.
    //DON'T CORRECT THE BOUNDARY
    //loop over elements
    //locate each vertex
    //FUTURE: locate each lagrange basis point by the same magic.
    //update the operator
  } //end of level for
  PetscFunctionReturn(0);
}


//MeshLocateInMesh: Create a label between the meshes.

PetscErrorCode MeshLocateInMesh(Mesh finemesh, Mesh coarsemesh) {
  ALE::Obj<ALE::Mesh> fm, cm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MeshGetMesh(finemesh, fm);CHKERRQ(ierr);
  ierr = MeshGetMesh(coarsemesh, cm);CHKERRQ(ierr);

  //set up the prolongation section if it doesn't already exist
  //bool prolongexists = fm->hasLabel("prolongation");
  const ALE::Obj<ALE::Mesh::label_type>& prolongation = fm->createLabel("prolongation");

  //we have a prolongation label that does not correspond to our current mesh.  Reset it to -1s.
  const ALE::Obj<ALE::Mesh::label_sequence>& finevertices = fm->depthStratum(0);
  ALE::Mesh::label_sequence::iterator fv_iter = finevertices->begin();
  ALE::Mesh::label_sequence::iterator fv_iter_end = finevertices->end();

  while (fv_iter != fv_iter_end) {
    fm->setValue(prolongation, *fv_iter, -1);
    fv_iter++;
  }
  //traversal labels on both layers
  ALE::Obj<ALE::Mesh::label_type> coarsetraversal = cm->createLabel("traversal");
  const ALE::Obj<ALE::Mesh::real_section_type>&  finecoordinates = fm->getRealSection("coordinates");
  //const ALE::Obj<ALE::Mesh::real_section_type>&  coarsecoordinates = cm->getRealSection("coordinates");
  int dim = fm->getDimension();
  //PetscPrintf(cm->comm(), "Dimensions: %d and %d\n", dim, cm->getDimension());
  if (dim != cm->getDimension()) throw ALE::Exception("Dimensions of the fine and coarse meshes do not match"); 
  //do the tandem traversal thing.  it is possible that if the section already existed then the locations of some of the points are known if they exist in both the meshes.
  fv_iter = finevertices->begin();
  fv_iter_end = finevertices->end();
  const ALE::Obj<ALE::Mesh::label_sequence>& coarseelements = cm->heightStratum(0);
  ALE::Mesh::label_sequence::iterator ce_iter = coarseelements->begin();
  ALE::Mesh::label_sequence::iterator ce_iter_end = coarseelements->end();
  double fvCoords[dim], nvCoords[dim];
  std::list<ALE::Mesh::point_type> travlist;  //store point
//  std::list<ALE::Mesh::point_type> travElist; //store element location "guesses"
  std::list<ALE::Mesh::point_type> eguesslist; // store the next guesses for the location of the current point.
  while (fv_iter != fv_iter_end) {

    //locate an initial point.
    if (fm->getValue(prolongation, *fv_iter) == -1) {
      ce_iter = coarseelements->begin();
      ce_iter_end = coarseelements->end();
      bool isLocated = false;
      ierr = PetscMemcpy(fvCoords, finecoordinates->restrictPoint(*fv_iter), dim*sizeof(double));
      while ((ce_iter != ce_iter_end) && (!isLocated)) {
        if (PointIsInElement(cm, *ce_iter, fvCoords)) {
          isLocated = true;
          fm->setValue(prolongation, *fv_iter, *ce_iter);
          PetscPrintf(fm->comm(), "INITIAL: Point %d located in %d.\n",  *fv_iter, *ce_iter);
          //OK WE HAVE A STARTING POINT.  Go through its neighbors looking at the unfound ones and finding them homes.
          ALE::Obj<ALE::Mesh::sieve_type::coneSet> neighbors = fm->getSieve()->cone(fm->getSieve()->support(*fv_iter));
          ALE::Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
          ALE::Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
          while (n_iter != n_iter_end) {
            if (fm->getValue(prolongation, *n_iter) == -1) {
//              travElist.push_front(*ce_iter);
              travlist.push_back(*n_iter);
              fm->setValue(prolongation, *n_iter, *ce_iter); //guess the next prolongation
            }
            n_iter++;
          }

          //do a DFS across the finemesh with BFSes on the coarse mesh for each point using assumed regularity of edgelength as a justification for guessing neighboring point's locations.
          while (!travlist.empty()) {
            ALE::Mesh::point_type curVert = *travlist.begin();
            PetscMemcpy(nvCoords, finecoordinates->restrictPoint(curVert), dim*sizeof(double));
            ALE::Mesh::point_type curEle =  fm->getValue(prolongation, curVert);
            travlist.pop_front();
            //travElist.pop_front();
            eguesslist.push_front(curEle);
            cm->setValue(coarsetraversal, curEle, 1);
            bool locationDiscovered = false;
            while ((!eguesslist.empty()) && (!locationDiscovered)) {
              ALE::Mesh::point_type curguess = *eguesslist.begin();
              eguesslist.pop_front();
              if (PointIsInElement(cm, curguess, nvCoords)) {
                locationDiscovered = true;
                //set the label.
                fm->setValue(prolongation, curVert, curguess);
                PetscPrintf(fm->comm(), "Point %d located in %d.\n",  curVert, curguess);
                //stick its neighbors in the queue along with its location as a good guess of the location of its neighbors
                neighbors = fm->getSieve()->cone(fm->getSieve()->support(curVert));
                n_iter = neighbors->begin();
                n_iter_end = neighbors->end();
                while (n_iter != n_iter_end) {
                  if (fm->getValue(prolongation, *n_iter) == -1) { //unlocated neighbor
                    travlist.push_back(*n_iter);
                    //travElist.push_front(curguess);
                    fm->setValue(prolongation, *n_iter, curguess);
                  }
                  n_iter++;
                }
              } else {
              //add the current guesses neighbors to the comparison queue and start over.
                ALE::Obj<ALE::Mesh::sieve_type::supportSet> curguessneighbors = cm->getSieve()->support(cm->getSieve()->cone(curguess));
                ALE::Mesh::sieve_type::supportSet::iterator cgn_iter = curguessneighbors->begin();
                ALE::Mesh::sieve_type::supportSet::iterator cgn_iter_end = curguessneighbors->end();
                while (cgn_iter != cgn_iter_end) {
                  if (cm->getValue(coarsetraversal, *cgn_iter) != 1) {
                    eguesslist.push_back(*cgn_iter);
                    cm->setValue(coarsetraversal, *cgn_iter, 1);
                  }
                  cgn_iter++;
                }
              }
            }
            if (!locationDiscovered) { 
              fm->setValue(prolongation, curVert, -2); //put it back in the list of orphans.
              PetscPrintf(fm->comm(), "Point %d (%f, %f) not located.\n",  curVert, nvCoords[0], nvCoords[1]);
            }
            eguesslist.clear(); //we've discovered the location of the point or exhausted our possibilities on this contiguous block of elements.
            //unset the traversed element list
            ALE::Obj<ALE::Mesh::label_sequence> traved_elements = cm->getLabelStratum("traversal", 1);
            PetscPrintf(cm->comm(), "%d\n", traved_elements->size());
            ALE::Mesh::label_sequence::iterator tp_iter = traved_elements->begin();
            ALE::Mesh::label_sequence::iterator tp_iter_end = traved_elements->end();
            while (tp_iter != tp_iter_end) {
              eguesslist.push_back(*tp_iter);
              tp_iter++;
            }
            while (!eguesslist.empty()) {
              cm->setValue(coarsetraversal, *eguesslist.begin(), 0);
              eguesslist.pop_front();
            }
          }
        }
        ce_iter++;
      }
      if (!isLocated) {
       fm->setValue(prolongation, *fv_iter, -2);
      }
    }
   // printf("-");
    fv_iter++;
  }
  PetscFunctionReturn(0);
}

bool PointIsInElement(ALE::Obj<ALE::Mesh> mesh, ALE::Mesh::point_type e, double * point) {
      int dim = mesh->getDimension();
      double v0[dim], J[dim*dim], invJ[dim*dim], detJ;
      mesh->computeElementGeometry(mesh->getRealSection("coordinates"), e, v0, J, invJ, detJ);
/*      if (dim == 2) {
        double xi   = invJ[0*dim+0]*(point[0] - v0[0]) + invJ[0*dim+1]*(point[1] - v0[1]);
        double eta  = invJ[1*dim+0]*(point[0] - v0[0]) + invJ[1*dim+1]*(point[1] - v0[1]);
        if ((xi >= 0.0) && (eta >= 0.0) && (xi + eta <= 1.0)) { return true;
        } else return false;
      } else if (dim == 3) {
        double xi   = invJ[0*dim+0]*(point[0] - v0[0]) + invJ[0*dim+1]*(point[1] - v0[1]) + invJ[0*dim+2]*(point[2] - v0[2]);
        double eta  = invJ[1*dim+0]*(point[0] - v0[0]) + invJ[1*dim+1]*(point[1] - v0[1]) + invJ[1*dim+2]*(point[2] - v0[2]);
        double zeta = invJ[2*dim+0]*(point[0] - v0[0]) + invJ[2*dim+1]*(point[1] - v0[1]) + invJ[2*dim+2]*(point[2] - v0[2]);

        if ((xi >= 0.0) && (eta >= 0.0) && (zeta >= 0.0) && (xi + eta + zeta <= 1.0)) { return true;
        } else return false;
      } else throw ALE::Exception("Location only in 2D or 3D");*/
      double coeffs[dim];
      double sum = 0.;
      for (int i = 0; i < dim; i++) {
        coeffs[i] = 0.;
        for (int j = 0; j < dim; j++) {
          coeffs[i] += invJ[i*dim+j]*(point[j] - v0[j]);
        }
        sum += coeffs[i];
      }
      if (sum > 1.) return false;
      for (int i = 0; i < dim; i++) {
        if (coeffs[i] < 0.) return false;
      }
      return true;
}




