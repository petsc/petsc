//rewritten Hierarchy.h made explicitly to output directly to DMMG
#include <list>
#include <Mesh.hh>
#include <stdlib.h>
#include <string>

//#include "petscmesh.h"
#include "petscdmmg.h"
//#include "petscmat.h"
#include "private/meshimpl.h"   /*I      "petscmesh.h"   I*/
#include <Distribution.hh>
#include <Generator.hh>
//helper functions:

bool PointIsInElement(ALE::Obj<ALE::Mesh> m, ALE::Mesh::point_type element, double * point);
//double PointDist(double * pointA, double * pointB, int dim);

// Functions only used here

PetscErrorCode MeshSpacingFunction(Mesh mesh); //builds the spacing function for the mesh
PetscErrorCode MeshIDBoundary(Mesh mesh); //finds the boundary of the mesh.

// DMMG Top Function

PetscErrorCode DMMGFunctionBasedCoarsen(Mesh finemesh, int levels, double coarsen_factor, DMMG dmmg);  //sets up the DMMG object to do this

#ifdef PETSC_HAVE_TRIANGLE
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
#endif

double Curvature_2D(ALE::Obj<ALE::Mesh> m, ALE::Mesh::point_type p) {
  PetscErrorCode ierr;
  const ALE::Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  ALE::Obj<ALE::Mesh::label_type> boundary = m->getLabel("marker");
  int dim = m->getDimension();
  if (dim != 2) throw ALE::Exception("Called the 2D curvature routine on a non-2D mesh.");
  if (m->height(p) != 2) throw ALE::Exception("Curvatures available for interpolated meshes only.");
  double pCoords[dim], qCoords[dim], rCoords[dim];
  double normvec[dim];
  
  const ALE::Obj<ALE::Mesh::sieve_type::supportSequence> neighbors = m->getSieve()->support(p); //get the set of edges with p as an endpoint
  ALE::Mesh::sieve_type::supportSequence::iterator n_iter = neighbors->begin();
  ALE::Mesh::sieve_type::supportSequence::iterator n_iter_end = neighbors->end();
  std::list<ALE::Mesh::point_type> edgnlist;
  while (n_iter != n_iter_end) {
      if (m->getSieve()->support(*n_iter)->size() == 1) {
        ALE::Mesh::sieve_type::coneSequence::iterator npoint = m->getSieve()->cone(*n_iter)->begin();
        if (*npoint != p) {
          edgnlist.push_front(*npoint);
        } else {
          npoint++;
          edgnlist.push_front(*npoint);
        }
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

double Curvature_3D(ALE::Obj<ALE::Mesh> m, ALE::Mesh::point_type p) {
  PetscErrorCode ierr;
  const ALE::Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  ALE::Obj<ALE::Mesh::label_type> boundary = m->getLabel("marker");
  int dim = m->getDimension();
  if (dim != 3) throw ALE::Exception("Called the 3D curvature routine on a non-3D mesh.");
  if (m->height(p) != 3) throw ALE::Exception("Curvatures available for interpolated meshes only.");
  double pCoords[dim], qCoords[dim], rCoords[dim];
  ierr = PetscMemcpy(pCoords, coordinates->restrictPoint(p), dim*sizeof(double));
  double normvec[dim];
  normvec[0] = 0.;
  normvec[1] = 0.;
  normvec[2] = 0.;
  //ok, traverse the pointface in one direction.
  //find the first exterior face
  ALE::Obj<ALE::Mesh::sieve_type::supportArray> faces = m->getSieve()->nSupport(p, 2);
  ALE::Mesh::sieve_type::supportArray::iterator f_iter = faces->begin();
  ALE::Mesh::sieve_type::supportArray::iterator f_iter_end = faces->end();
  ALE::Mesh::point_type curface, curpt, lastpt, firstpt;
  bool found = false;
  while (f_iter != f_iter_end && !found) {
    if (m->getSieve()->support(*f_iter)->size() == 1) {
      curface = *f_iter;
      found = true;
    }
    f_iter++;
  }
  
  ALE::Obj<ALE::Mesh::sieve_type::coneArray> neighbors = m->getSieve()->nCone(curface, 2);
  ALE::Mesh::sieve_type::coneArray::iterator n_iter = neighbors->begin();
  ALE::Mesh::sieve_type::coneArray::iterator n_iter_end = neighbors->end();
  //set cur and lastpt as appropriate.
  while (n_iter != n_iter_end) {
    if (*n_iter != p) {
      lastpt = curpt;
      curpt = *n_iter;
    }
    n_iter++;
  }
  firstpt = curpt;
  //ok, proceed in the following fashion:  compute the normal of curface, add it to normvec, and move on to the next triangle.  give up when you're back at firstpt.
    //printf("%d ->", curpt);
  bool startup = true;
  while (curpt != firstpt || startup) {
    startup = false;
    //normal computation:
    ierr = PetscMemcpy(qCoords, coordinates->restrictPoint(curpt), dim*sizeof(double));
    ierr = PetscMemcpy(rCoords, coordinates->restrictPoint(lastpt), dim*sizeof(double));
    double tmp = (qCoords[1] - pCoords[1])*(rCoords[2] - pCoords[2]) - (qCoords[2] - pCoords[2])*(rCoords[1] - pCoords[1]);
    normvec[0] += tmp;
    //printf("%f,",tmp);
    tmp = (qCoords[2] - pCoords[2])*(rCoords[0] - pCoords[0]) - (qCoords[0] - pCoords[0])*(rCoords[2] - pCoords[2]);
    normvec[1] += tmp;
    //printf("%f,",tmp);
    tmp = (qCoords[0] - pCoords[0])*(rCoords[1] - pCoords[1]) - (qCoords[1] - pCoords[1])*(rCoords[0] - pCoords[0]);
    normvec[2] += tmp;
    //printf("%f\n",tmp);
    found = false;
    f_iter = faces->begin();
    f_iter_end = faces->end();
    while (f_iter != f_iter_end && !found) {
      //get the points in the cone of this face and see if it a) isn't the current face, and b) has p, curpt, and NOT lastpt in its cone
      if (m->getSieve()->support(*f_iter)->size() < 2) {
        neighbors = m->getSieve()->nCone(*f_iter, 2);
        n_iter = neighbors->begin();
        n_iter_end = neighbors->end();
        bool containsp = false;
        bool containscurpt = false;
        bool containslastpt = false;
        ALE::Mesh::point_type posspt;
        while (n_iter != n_iter_end) {
          if (*n_iter == p){ containsp = true;}
          else if (*n_iter == curpt) {containscurpt = true;}
          else if (*n_iter == lastpt) {containslastpt = true;}
          else posspt = *n_iter;
          n_iter++;
        }
        if (containscurpt && containsp && !containslastpt) {
          lastpt = curpt;
          curpt = posspt;
          found = true;
          curface = *f_iter;
        }
      }
      f_iter++;
    }
    //printf("%d ->", curpt);
  }
  //printf("\n");
  //normalize the normal.
  double normlen = sqrt(normvec[0]*normvec[0] + normvec[1]*normvec[1] + normvec[2]*normvec[2]); 
  if (normlen < 0.000000000001) return 0.; //give up
  normvec[0] = normvec[0]/normlen;
  normvec[1] = normvec[1]/normlen;
  normvec[2] = normvec[2]/normlen;
  //PetscPrintf(m->comm(), "%f, %f, %f\n", normvec[0], normvec[1], normvec[2]);
  //now go through the adjacent edges, seeing how off they are from orthogonal with the normal.
  double curvature = 1.0;
  ALE::Obj<ALE::Mesh::sieve_type::coneSet> pneighbors = m->getSieve()->cone(m->getSieve()->support(p));
  ALE::Mesh::sieve_type::coneSet::iterator p_iter = pneighbors->begin();
  ALE::Mesh::sieve_type::coneSet::iterator p_iter_end = pneighbors->end();
  while (p_iter != p_iter_end) {
    if (*p_iter != p) {
      ierr = PetscMemcpy(qCoords, coordinates->restrictPoint(*p_iter), dim*sizeof(double));
      double curlen = sqrt((qCoords[0] - pCoords[0])*(qCoords[0] - pCoords[0]) + (qCoords[1] - pCoords[1])*(qCoords[1] - pCoords[1]) + (qCoords[2] - pCoords[2])*(qCoords[2] - pCoords[2]));
      double curcurve = fabs((normvec[0]*(qCoords[0] - pCoords[0]) + normvec[1]*(qCoords[1] - pCoords[1]) + normvec[2]*(qCoords[2] - pCoords[2]))/curlen);
      if (curcurve < curvature)curvature = curcurve;
    }
    p_iter++;
  }
  //printf("Curvature: %f\n", curvature);
  return curvature;
}

double Curvature(ALE::Obj<ALE::Mesh> m, ALE::Mesh::point_type p) {
  int dim = m->getDimension();
  if (dim == 2) {return Curvature_2D(m, p);}
  else if (dim == 3) {return Curvature_3D(m, p);}
  else throw ALE::Exception("Cannot do Curvature in dimensions other than 2 and 3D");
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

#undef __FUNCT__
#define __FUNCT__ "MeshCreateHierarchyLabel"

//MeshCreateHierarchyLabel: Create a label that tells what the highest level a given vertex appears in where 0 is fine and n is coarsest.
PetscErrorCode MeshCreateHierarchyLabel(Mesh finemesh, double beta, int nLevels, Mesh * outmeshes, Mat * outmats = PETSC_NULL) {
  PetscErrorCode ierr;
  ALE::Obj<ALE::Mesh> m;
  PetscFunctionBegin;
  PetscTruth info;
  ierr = PetscOptionsHasName(PETSC_NULL, "-dmmg_coarsen_info", &info);CHKERRQ(ierr);
  ierr = MeshGetMesh(finemesh, m);CHKERRQ(ierr);
  int dim = m->getDimension();
  if (info)PetscPrintf(m->comm(), "Original Mesh: %d vertices, %d elements\n", m->depthStratum(0)->size(), m->heightStratum(0)->size());
  const ALE::Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::real_section_type>& spacing = m->getRealSection("spacing");
  const ALE::Obj<ALE::Mesh::label_type> hdepth = m->createLabel("hdepth");
  const ALE::Obj<ALE::Mesh::label_type> dompoint = m->createLabel("dompoint");
  const ALE::Obj<ALE::Mesh::label_type> traversal = m->createLabel("traversal");
  const ALE::Obj<ALE::Mesh::label_type>& boundary = m->getLabel("marker");
  const ALE::Obj<ALE::Mesh::label_sequence>& vertices = m->depthStratum(0);
  ALE::Mesh::label_sequence::iterator v_iter = vertices->begin();
  ALE::Mesh::label_sequence::iterator v_iter_end = vertices->end();
  double maxspace = -1., minspace = -1., dist;
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
 // PetscPrintf(m->comm(), "NUMBER OF BOUNDARY POINTS: %d\n", boundaryvertices->size());
  while (bv_iter != bv_iter_end) {
    if (m->depth(*bv_iter) == 0) if (Curvature(m, *bv_iter) > 0.01) {
      m->setValue(hdepth, *bv_iter, nLevels-1);
    }
    bv_iter++;
  }
 // PetscPrintf(m->comm(), "Forced in %d especially curved boundary nodes.\n", m->getLabelStratum("hdepth", nLevels-1)->size());
  double bvCoords[dim];
  ALE::Mesh::point_type bvdom;
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
    double comparison_const;
    ALE::Mesh::label_sequence::iterator bv_iter = boundaryvertices->begin();
    ALE::Mesh::label_sequence::iterator bv_iter_end = boundaryvertices->end();
    ALE::Obj<ALE::Mesh::sieve_type::coneSet> neighbors = m->getSieve()->cone(m->getSieve()->support(*bv_iter));
    ALE::Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
    ALE::Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
    while (bv_iter != bv_iter_end) {
      bvdom = m->getValue(dompoint, *bv_iter);
      bool skip = false;
      if (bvdom != -1) {
        if (m->getValue(hdepth, bvdom) == curLevel) skip = true; 
      }
      bool canAdd = true;
      if (m->getValue(hdepth, *bv_iter) == 0 && !skip && m->depth(*bv_iter) == 0) { //if not yet included or invalidated
        m->setValue(traversal, *bv_iter, 1);
        double bvSpace = *spacing->restrictPoint(*bv_iter);
        ierr = PetscMemcpy(bvCoords, coordinates->restrictPoint(*bv_iter), dim*sizeof(double));
        //get its neighbors and add them to the comparison queue.
        neighbors = m->getSieve()->cone(m->getSieve()->support(*bv_iter));
        n_iter = neighbors->begin();
        n_iter_end = neighbors->end();
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
          dist = 0.;
          double curSpace = *spacing->restrictPoint(curpt);
          const double * curCoords = coordinates->restrictPoint(curpt); 
          for (int i = 0; i < dim; i++) {
            dist += (curCoords[i] - bvCoords[i])*(curCoords[i] - bvCoords[i]);
          }
          dist = sqrt(dist);
          ALE::Mesh::point_type curpt_dom = m->getValue(dompoint, curpt);
          int curpt_depth = m->getValue(hdepth, curpt);
          int curpt_bound = m->getValue(boundary, curpt);
          comparison_const = 0.5*curBeta;
          if ((dist < comparison_const*(bvSpace + curSpace))&&(curpt_depth > 0)) { //collision with an already added node
            canAdd = false;
            m->setValue(dompoint, *bv_iter, curpt);
          } else if (dist < comparison_const*(bvSpace + maxspace)) { 
            neighbors = m->getSieve()->cone(m->getSieve()->support(curpt));
            n_iter = neighbors->begin();
            n_iter_end = neighbors->end();
            while (n_iter != n_iter_end) {
              if ((curpt_bound == 1) && (m->getValue(traversal, *n_iter) == 0)) {
                m->setValue(traversal, *n_iter, 1);
                complist.push_back(*n_iter);
              }
              n_iter++;
            }
          }
          if ((dist < comparison_const*(bvSpace + curSpace)) && (curpt_depth == 0)) { //add the point to the list of points dominated by this node; points eliminated in one step later
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
 //   PetscPrintf(m->comm(), "Added %d new boundary vertices\n", m->getLabelStratum("hdepth", curLevel)->size());
    //INTERIOR NODES:
    complist.clear();
    ALE::Obj<ALE::Mesh::label_sequence> intverts = m->depthStratum(0);
    bv_iter = intverts->begin();
    bv_iter_end = intverts->end();
    while (bv_iter != bv_iter_end) {
      bvdom = m->getValue(dompoint, *bv_iter);
      bool skip = false;
      if (bvdom != -1) {
        if (m->getValue(hdepth, bvdom) == curLevel) skip = true; 
      }
      bool canAdd = true;
      if ((m->getValue(boundary, *bv_iter) != 1) && (m->getValue(hdepth, *bv_iter) == 0) && !skip) { //if not in the boundary and not included (or excluded)
        double bvSpace = *spacing->restrictPoint(*bv_iter);
        ierr = PetscMemcpy(bvCoords, coordinates->restrictPoint(*bv_iter), dim*sizeof(double));
        //get its neighbors and add them to the comparison queue.
        neighbors = m->getSieve()->cone(m->getSieve()->support(*bv_iter));
        n_iter = neighbors->begin();
        n_iter_end = neighbors->end();
        m->setValue(traversal, *bv_iter, 1);
        while (n_iter != n_iter_end) {
          if (*n_iter != *bv_iter) {
            //PetscPrintf(m->comm(), "Added %d to the list\n", *n_iter);
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
          //PetscPrintf(m->comm(), "Comparing %d to %d\n", *bv_iter, curpt);
          dist = 0.;
          double curSpace = *spacing->restrictPoint(curpt);
          const double * curCoords = coordinates->restrictPoint(curpt); 
          for (int i = 0; i < dim; i++) {
            dist += (curCoords[i] - bvCoords[i])*(curCoords[i] - bvCoords[i]);
          }
          dist = sqrt(dist);
          int curpt_depth = m->getValue(hdepth, curpt);
          int curpt_bound = m->getValue(boundary, curpt);
          comparison_const = 0.5*curBeta;
          ALE::Mesh::point_type curpt_dom = m->getValue(dompoint, curpt);
          if ((dist < comparison_const*(bvSpace + curSpace))&&(curpt_depth > 0)) {
            canAdd = false;
            m->setValue(dompoint, *bv_iter, curpt);
          } else if ((dist < comparison_const*(bvSpace+curSpace)) && (curpt_bound == 1)) {
            canAdd = false;
            m->setValue(dompoint, *bv_iter, curpt);
          } else if (dist < comparison_const*(bvSpace + maxspace)) { 
            neighbors = m->getSieve()->cone(m->getSieve()->support(curpt));
            n_iter = neighbors->begin();
            n_iter_end = neighbors->end();
            while (n_iter != n_iter_end) {
              if ((m->getValue(boundary, *n_iter) != 1) && (m->getValue(traversal, *n_iter) != 1)) {
                m->setValue(traversal, *n_iter, 1);
                complist.push_back(*n_iter);
              }
              n_iter++;
            }
          }
          if ((dist < comparison_const*(bvSpace + curSpace)) && (curpt_depth == 0)) {
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
  //  PetscPrintf(m->comm(), "Included %d new points in level %d\n", m->getLabelStratum("hdepth", curLevel)->size(), curLevel);
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
    std::string triangleOptions = "zQe"; //(z)ero indexing, output (e)dges, Quiet
    double * finalcoords;
    int * connectivity;
    int * oldpositions;
    int nelements;
    int nverts;
    if (dim == 2) {
#ifdef PETSC_HAVE_TRIANGLE
      //create a segmentlist to keep triangle from doing dumb things.
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
#else
      SETERRQ(PETSC_ERR_SUP, "Must have Triangle installed to use this method. Reconfigure with --download-triangle");
#endif
    } else if (dim == 3) {
#ifdef PETSC_HAVE_TETGEN
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
#else
      SETERRQ(PETSC_ERR_SUP, "Must have TetGen installed to use this method. Reconfigure with --download-tetgen");
#endif
    }
    //make it into a mesh;
    ALE::Obj<ALE::Mesh> newmesh = new ALE::Mesh(m->comm(), m->debug());
    newmesh->setDimension(dim);
    ALE::Obj<ALE::Mesh::sieve_type> sieve = new ALE::Mesh::sieve_type(m->comm(), m->debug());
    ALE::SieveBuilder<ALE::Mesh>::buildTopology(sieve, dim, nelements, connectivity, nverts, true, dim+1, -1, newmesh->getArrowSection("orientation"));
    newmesh->setSieve(sieve);
    newmesh->stratify();
    //UPDATE THE MARKER AND FINEMESH VERTEX NUMBERING LABELS
    ALE::Obj<ALE::Mesh::label_type> boundary_new = newmesh->createLabel("marker");
    ALE::Obj<ALE::Mesh::label_type> fine_corresponds = newmesh->createLabel("fine");
    ALE::Obj<ALE::Mesh::label_sequence> newverts = newmesh->depthStratum(0);
    if (info)PetscPrintf(m->comm(), "%d: %d vertices, %d elements\n", curLevel, newmesh->depthStratum(0)->size(), newmesh->heightStratum(0)->size());
    ALE::Mesh::label_sequence::iterator nv_iter = newverts->begin();
    ALE::Mesh::label_sequence::iterator nv_iter_end = newverts->end();
    while (nv_iter != nv_iter_end) {
      newmesh->setValue(fine_corresponds, *nv_iter, oldpositions[*nv_iter - nelements]);
      if(m->getValue(boundary, oldpositions[*nv_iter - nelements]) == 1) newmesh->setValue(boundary_new, *nv_iter, 1);
      nv_iter++;
    }
    ALE::Obj<ALE::Mesh::real_section_type> s = newmesh->getRealSection("default");
    //set up the default section
    newmesh->setDiscretization(m->getDiscretization());
    newmesh->setBoundaryCondition(m->getBoundaryCondition());
    newmesh->setupField(s);

    MeshSetMesh(outmeshes[curLevel-1], newmesh);
    ALE::SieveBuilder<ALE::Mesh>::buildCoordinates(newmesh, dim, finalcoords);
/*    ierr = SectionRealCreate(newmesh->comm(), &section);CHKERRQ(ierr);
    ierr = SectionRealGetBundle(section, newmesh);CHKERRQ(ierr);
    ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) section, "default");CHKERRQ(ierr);
    ierr = MeshSetSectionReal(outmeshes[curLevel-1], section);CHKERRQ(ierr);
    ierr = SectionRealDestroy(section);*/
  } //end of level for
  std::list<ALE::Mesh::point_type> tricomplist;
  for (int curLevel = 0; curLevel < nLevels-1; curLevel++) {
    //PetscPrintf(m->comm(), "Building the prolongation section from level %d to level %d\n", curLevel, curLevel+1);
    ALE::Obj<ALE::Mesh> c_m;
    ALE::Obj<ALE::Mesh> f_m;
    if (curLevel == 0) {
      f_m = m;
    } else {
      ierr = MeshGetMesh(outmeshes[curLevel-1], f_m);CHKERRQ(ierr);
    }
    ierr = MeshGetMesh(outmeshes[curLevel], c_m);CHKERRQ(ierr);
    ALE::Obj<ALE::Mesh::label_type> prolongation = f_m->createLabel("prolongation");
    ALE::Obj<ALE::Mesh::label_type> coarse_traversal = c_m->createLabel("traversal");
    ALE::Obj<ALE::Mesh::label_sequence> levelVertices = m->getLabelStratum("hdepth", curLevel);
   // PetscPrintf(m->comm(), "%d points in level %d\n", levelVertices->size(), curLevel);
    ALE::Mesh::label_sequence::iterator lv_iter = levelVertices->begin();
    ALE::Mesh::label_sequence::iterator lv_iter_end = levelVertices->end();
    int interpolatelevels = c_m->height(*c_m->depthStratum(0)->begin()); //see if the mesh is interpolated or not
    double lvCoords[dim];
    while (lv_iter != lv_iter_end) {
      int ncomps = 0;
      ierr = PetscMemcpy(lvCoords, coordinates->restrictPoint(*lv_iter), dim*sizeof(double));
      if (m->getValue(boundary, *lv_iter) != 1) {
        //get the triangle/tet fan around the dominating point in the next level up
        ALE::Mesh::point_type dp = m->getValue(dompoint, *lv_iter);
        if (m->getValue(hdepth, dp) < curLevel+1) dp = m->getValue(dompoint, dp); //if it's a boundary node it can be a dominating point and NOT be in the topmesh
        ALE::Obj<ALE::Mesh::sieve_type::supportArray> trifan = c_m->getSieve()->nSupport(*c_m->getLabelStratum("fine", dp)->begin(), interpolatelevels);
        ALE::Mesh::sieve_type::supportArray::iterator tf_iter = trifan->begin();
        ALE::Mesh::sieve_type::supportArray::iterator tf_iter_end = trifan->end();
        while (tf_iter != tf_iter_end) {
          tricomplist.push_front(*tf_iter); //initialize the closest-guess comparison list.
          c_m->setValue(coarse_traversal, *tf_iter, 1);
          tf_iter++;
        }
        //PetscPrintf(m->comm(), "%d initial guesses\n", trifan->size());
        bool isFound = false;
        while (!tricomplist.empty() && !isFound) {
          ALE::Mesh::point_type curTri = *tricomplist.begin();
          tricomplist.pop_front();
          ncomps++;
          if (PointIsInElement(c_m, curTri, lvCoords)) {
            ALE::Mesh::point_type fmpoint;
            if (curLevel == 0) {
              fmpoint = *lv_iter;
            } else {
              fmpoint = *f_m->getLabelStratum("fine", *lv_iter)->begin();
            } 
            f_m->setValue(prolongation, fmpoint, curTri);
            isFound = true;
          //  PetscPrintf(m->comm(), "%d located in %d\n", *lv_iter, curTri);
          } else {
            ALE::Obj<ALE::Mesh::sieve_type::supportSet> trineighbors = c_m->getSieve()->support(c_m->getSieve()->cone(curTri));
            ALE::Mesh::sieve_type::supportSet::iterator tn_iter = trineighbors->begin();
            ALE::Mesh::sieve_type::supportSet::iterator tn_iter_end = trineighbors->end();
            while (tn_iter != tn_iter_end) {
              if (c_m->getValue(coarse_traversal, *tn_iter) != 1){
                tricomplist.push_back(*tn_iter);
                c_m->setValue(coarse_traversal, *tn_iter, 1);
              }
              tn_iter++;
            }
          }
        }
        ALE::Obj<ALE::Mesh::label_sequence> travtris = c_m->getLabelStratum("traversal", 1);
        ALE::Mesh::label_sequence::iterator tt_iter = travtris->begin();
        ALE::Mesh::label_sequence::iterator tt_iter_end = travtris->end();
        while (tt_iter != tt_iter_end) {
          tricomplist.push_front(*tt_iter);
          tt_iter++;
        }
        while (!tricomplist.empty()) {
          c_m->setValue(coarse_traversal, *tricomplist.begin(), 0);
          tricomplist.pop_front();
        }
        if (!isFound) {
          PetscPrintf(m->comm(), "ERROR: Could Not Locate Point %d (%f, %f) in %d comparisons\n", *lv_iter, lvCoords[0], lvCoords[1], ncomps);
        }
      } else { //DO NOT correct the boundary nodes only added on this level. (might break neumann)
        ALE::Mesh::point_type fmpoint;
        if (curLevel == 0) {
          fmpoint = *lv_iter;
        } else {
          fmpoint = *f_m->getLabelStratum("fine", *lv_iter)->begin();
        } 
        f_m->setValue(prolongation, fmpoint, -1);
      }
      lv_iter++;
    }
    //set the prolongation label for the guys in this mesh in such a way that there's an element associated for the points in higher levels also.
    for (int upLevel = curLevel + 1; upLevel < nLevels - 1; upLevel++) {
      levelVertices = m->getLabelStratum("hdepth", upLevel);
      lv_iter = levelVertices->begin();
      lv_iter_end = levelVertices->end();
      while (lv_iter != lv_iter_end) {
        //if (m->getValue(boundary, *lv_iter) != 1) { we CAN correct for the boundary, and should!
          //nothing complicated, just take the FIRST element of the nSupport of it in the next mesh up.
          ALE::Mesh::point_type cmpt = *c_m->getLabelStratum("fine", *lv_iter)->begin();
          ALE::Mesh::point_type fmpoint;
          if (curLevel == 0) {
            fmpoint = *lv_iter;
          } else {
            fmpoint = *f_m->getLabelStratum("fine", *lv_iter)->begin();
          } 
          f_m->setValue(prolongation, fmpoint, *c_m->getSieve()->nSupport(cmpt, interpolatelevels)->begin());
        //}
        lv_iter++;
      }
    }
  }
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
      if (dim == 2) {
        double xi   = invJ[0*dim+0]*(point[0] - v0[0]) + invJ[0*dim+1]*(point[1] - v0[1]);
        double eta  = invJ[1*dim+0]*(point[0] - v0[0]) + invJ[1*dim+1]*(point[1] - v0[1]);
        //PetscPrintf(mesh->comm(), "Location Try: (%d) (%f, %f, %f)\n", e, xi, eta, xi + eta);
        if ((xi >= -0.000001) && (eta >= -0.000001) && (xi + eta <= 2.000001)) {return true;
        } else return false;
      } else if (dim == 3) {
        double xi   = invJ[0*dim+0]*(point[0] - v0[0]) + invJ[0*dim+1]*(point[1] - v0[1]) + invJ[0*dim+2]*(point[2] - v0[2]);
        double eta  = invJ[1*dim+0]*(point[0] - v0[0]) + invJ[1*dim+1]*(point[1] - v0[1]) + invJ[1*dim+2]*(point[2] - v0[2]);
        double zeta = invJ[2*dim+0]*(point[0] - v0[0]) + invJ[2*dim+1]*(point[1] - v0[1]) + invJ[2*dim+2]*(point[2] - v0[2]);

        if ((xi >= -0.000001) && (eta >= -0.000001) && (zeta >= -0.000001) && (xi + eta + zeta <= 2.000001)) { return true;
        } else return false;
      } else throw ALE::Exception("Location only in 2D or 3D");
/*      double coeffs[dim];
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
      return true; */
}




