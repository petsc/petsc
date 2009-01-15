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
#include <SieveAlgorithms.hh>
#include <Selection.hh>
//helper functions:

bool PointIsInElement(ALE::Obj<PETSC_MESH_TYPE> m, PETSC_MESH_TYPE::point_type element, double * point);
PetscErrorCode MeshLocateInMesh(Mesh finemesh, Mesh coarsemesh);
//double PointDist(double * pointA, double * pointB, int dim);

// Functions only used here

PetscErrorCode MeshSpacingFunction(Mesh mesh); //builds the spacing function for the mesh
PetscErrorCode MeshIDBoundary(Mesh mesh); //finds the boundary of the mesh.

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

double vertex_angle (const ALE::Obj<PETSC_MESH_TYPE::real_section_type> coords, int dim, PETSC_MESH_TYPE::point_type p, PETSC_MESH_TYPE::point_type q, PETSC_MESH_TYPE::point_type r) {

// o is our origin point, p is one endpoint, q is the other; find their angle given the coordinates
  double *p_coords = new double[dim], *q_coords = new double[dim], *r_coords = new double[dim], mag_q, mag_r, dot_product, angle;
  PetscErrorCode ierr;
  ierr = PetscMemcpy(p_coords, coords->restrictPoint(p), dim*sizeof(double));CHKERRQ(ierr);
  ierr = PetscMemcpy(q_coords, coords->restrictPoint(q), dim*sizeof(double));CHKERRQ(ierr);
  ierr = PetscMemcpy(r_coords, coords->restrictPoint(r), dim*sizeof(double));CHKERRQ(ierr);
  for (int i = 0; i < dim; i++) {
    mag_q += (p_coords[i] - q_coords[i])*(p_coords[i] - q_coords[i]);
    mag_r += (p_coords[i] - r_coords[i])*(p_coords[i] - r_coords[i]);
    dot_product += (p_coords[i] -q_coords[i])*(p_coords[i] - r_coords[i]);
  }
  if (mag_q == 0 || mag_r == 0) return 0.;
  angle = acos(dot_product/sqrt(mag_q*mag_r));
  delete [] p_coords; delete [] q_coords; delete [] r_coords;
  return angle;
} 

double Curvature(ALE::Obj<PETSC_MESH_TYPE> m, PETSC_MESH_TYPE::point_type v) {
  //Make it work for 2D and 3D in an abstract fashion.  we know that the 
  //integral of the curvature over a triangle fan in 3D is merely the sum
  //of the angles radiating from its centerpoint minus 2pi, and that
  //the point on the triangle fan will be a singularity of curvature

  double pi = M_PI;
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
  double curvature;
//  PetscErrorCode ierr;
  int dim = m->getDimension();  
  if (m->depth(v) != 0) {
    throw ALE::Exception("Curvature only defined on vertices");
  }

  double triangleSum = 0.;

  ALE::Obj<PETSC_MESH_TYPE::sieve_type> s = m->getSieve();

  if (dim == 3) {
    if (m->depth() == 1) { //uninterpolated case, we have to ID the faces
      ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSequence> cells = s->support(v);
      PETSC_MESH_TYPE::sieve_type::supportSequence::iterator c_iter = cells->begin();
      PETSC_MESH_TYPE::sieve_type::supportSequence::iterator c_iter_end = cells->end();
      while (c_iter != c_iter_end) {
        //why must cells and corners both start with c?
        ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneSequence> tips = s->cone(*c_iter);
        PETSC_MESH_TYPE::sieve_type::coneSequence::iterator t_iter = tips->begin();
        PETSC_MESH_TYPE::sieve_type::coneSequence::iterator t_iter_exclude = tips->begin();
        PETSC_MESH_TYPE::sieve_type::coneSequence::iterator t_iter_end = tips->end();
        ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> face = new PETSC_MESH_TYPE::sieve_type::supportSet();
        while (t_iter_exclude != t_iter_end) {
          face->clear();
          if (*t_iter_exclude != v) {
            t_iter = tips->begin();
            while (t_iter != t_iter_end) {
              //make the face into a supportSet
              if (t_iter != t_iter_exclude) {
                face->insert(*t_iter);
              }
              t_iter++;
            }
            if (s->nJoin1(face)->size() == 1) { //face on boundary
              //compute the angle
              if (face->size() != 3) throw ALE::Exception("Curvature: Bad face size");
              PETSC_MESH_TYPE::sieve_type::supportSet::iterator tf_iter = face->begin();
              PETSC_MESH_TYPE::sieve_type::supportSet::iterator tf_iter_end = face->end();
              
              PETSC_MESH_TYPE::point_type v_op[2]; //opposing vertices
              int index = 0;
              while (tf_iter != tf_iter_end) {
                if (*tf_iter != v) {
                  v_op[index] = *tf_iter;
                  index++;
                }
                tf_iter++;
              }
              triangleSum += vertex_angle(coordinates, dim, v, v_op[0], v_op[1]);
            } //end face-on-boundary if
          } //end t_iter_exclude != v if
          t_iter_exclude++;
        }
        c_iter++;
      }
    } else {
    //the triangles are there for us!
      ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportArray> faces = s->nSupport(v, 2);
      PETSC_MESH_TYPE::sieve_type::supportArray::iterator f_iter = faces->begin();
      PETSC_MESH_TYPE::sieve_type::supportArray::iterator f_iter_end = faces->end();
      while (f_iter != f_iter_end) {
        if (s->support(*f_iter)->size() == 1) { //boundary edge
          ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneArray> tips = s->nCone(*f_iter, 2);
          if (tips->size() != 3) throw ALE::Exception("Curvature: wrong size face");
          PETSC_MESH_TYPE::sieve_type::coneArray::iterator tf_iter = tips->begin();
          PETSC_MESH_TYPE::sieve_type::coneArray::iterator tf_iter_end = tips->end();
          int index = 0;
          PETSC_MESH_TYPE::point_type v_op[2];
          while (tf_iter != tf_iter_end) {
                if (*tf_iter != v) {
                  v_op[index] = *tf_iter;
                  index++;
                }
                tf_iter++;
          }
          triangleSum += vertex_angle(coordinates, dim, v, v_op[0], v_op[1]);
        }
        f_iter++;
      }
    }
    curvature = 2*pi - triangleSum;
  } else if (dim == 2) {
    ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportArray> surrounding_triangles = s->nSupport(v, m->depth());
    PETSC_MESH_TYPE::sieve_type::supportArray::iterator st_iter = surrounding_triangles->begin();
    PETSC_MESH_TYPE::sieve_type::supportArray::iterator st_iter_end = surrounding_triangles->end();
    while (st_iter != st_iter_end) {
      ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneArray> tips = s->nCone(*st_iter, m->depth());
      if (tips->size() != 3) throw ALE::Exception("Curvature: not a triangle!");
      PETSC_MESH_TYPE::sieve_type::coneArray::iterator t_iter = tips->begin();
      PETSC_MESH_TYPE::sieve_type::coneArray::iterator t_iter_end = tips->end();
      PETSC_MESH_TYPE::point_type v_op[2];
      int index = 0;
      while (t_iter != t_iter_end) {
        if (*t_iter != v) {
          v_op[index] = *t_iter;
          index++;
        }
        t_iter++;
      }
      triangleSum += vertex_angle(coordinates, dim, v, v_op[0], v_op[1]);
      st_iter++;
    }
    curvature = fabs(triangleSum - pi);    
  } //end of dimension 2 case
  return curvature;
}


#if 0

double Curvature_2D(ALE::Obj<PETSC_MESH_TYPE> m, PETSC_MESH_TYPE::point_type p) {
  PetscErrorCode ierr;
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
  ALE::Obj<PETSC_MESH_TYPE::label_type> boundary = m->getLabel("marker");
  int dim = m->getDimension();
  if (dim != 2) throw ALE::Exception("Called the 2D curvature routine on a non-2D mesh.");
  double pCoords[dim], qCoords[dim], rCoords[dim];
  double normvec[dim];
  
  const ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSequence> neighbors = m->getSieve()->support(p); //get the set of edges with p as an endpoint
  PETSC_MESH_TYPE::sieve_type::supportSequence::iterator n_iter = neighbors->begin();
  PETSC_MESH_TYPE::sieve_type::supportSequence::iterator n_iter_end = neighbors->end();
  std::list<PETSC_MESH_TYPE::point_type> edgnlist;
  while (n_iter != n_iter_end) {
      if (m->getSieve()->support(*n_iter)->size() == 1) {
        PETSC_MESH_TYPE::sieve_type::coneSequence::iterator npoint = m->getSieve()->cone(*n_iter)->begin();
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
  PETSC_MESH_TYPE::point_type n1 = *edgnlist.begin();
  PETSC_MESH_TYPE::point_type n2 = *(++edgnlist.begin());

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

double Curvature_3D(ALE::Obj<PETSC_MESH_TYPE> m, PETSC_MESH_TYPE::point_type p) {
  PetscErrorCode ierr;
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
  ALE::Obj<PETSC_MESH_TYPE::label_type> boundary = m->getLabel("marker");
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
  ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportArray> faces = m->getSieve()->nSupport(p, 2);
  PETSC_MESH_TYPE::sieve_type::supportArray::iterator f_iter = faces->begin();
  PETSC_MESH_TYPE::sieve_type::supportArray::iterator f_iter_end = faces->end();
  PETSC_MESH_TYPE::point_type curface, curpt, lastpt, firstpt;
  bool found = false;
  while (f_iter != f_iter_end && !found) {
    if (m->getSieve()->support(*f_iter)->size() == 1) {
      curface = *f_iter;
      found = true;
    }
    f_iter++;
  }
  
  ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneArray> neighbors = m->getSieve()->nCone(curface, 2);
  PETSC_MESH_TYPE::sieve_type::coneArray::iterator n_iter = neighbors->begin();
  PETSC_MESH_TYPE::sieve_type::coneArray::iterator n_iter_end = neighbors->end();
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
        PETSC_MESH_TYPE::point_type posspt;
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
  ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneSet> pneighbors = m->getSieve()->cone(m->getSieve()->support(p));
  PETSC_MESH_TYPE::sieve_type::coneSet::iterator p_iter = pneighbors->begin();
  PETSC_MESH_TYPE::sieve_type::coneSet::iterator p_iter_end = pneighbors->end();
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

double Curvature(ALE::Obj<PETSC_MESH_TYPE> m, PETSC_MESH_TYPE::point_type p) {
  int dim = m->getDimension();
  if (dim == 2) {return Curvature_2D(m, p);}
  else if (dim == 3) {return Curvature_3D(m, p);}
  else throw ALE::Exception("Cannot do Curvature in dimensions other than 2 and 3D.");
}

#endif

//MeshSpacingFunction: Build the spacing function in the "spacing" section on the mesh.  

PetscErrorCode MeshSpacingFunction(Mesh mesh) {
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);
  int dim = m->getDimension();
  //setup the spacing section
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& spacing = m->getRealSection("spacing");
  spacing->setFiberDimension(m->depthStratum(0), 1);
  m->allocate(spacing);
  //vertices
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&  coordinates = m->getRealSection("coordinates");
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& vertices = m->depthStratum(0);
  PETSC_MESH_TYPE::label_sequence::iterator v_iter = vertices->begin();
  PETSC_MESH_TYPE::label_sequence::iterator v_iter_end = vertices->end();
  double vCoords[3], nCoords[3];
  while (v_iter != v_iter_end) {
    const double * tmpCoords = coordinates->restrictPoint(*v_iter);
    for (int i = 0; i < dim; i++) {
      vCoords[i] = tmpCoords[i];
    }
    //get the neighbors
    ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneSet> neighbors = m->getSieve()->cone(m->getSieve()->support(*v_iter));
    PETSC_MESH_TYPE::sieve_type::coneSet::iterator n_iter = neighbors->begin();
    PETSC_MESH_TYPE::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
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
      dist = 0.5*sqrt(dist);
      if ((dist < minDist && dist > 0.) || minDist == 0.) minDist = dist;
      n_iter++;
    }
    spacing->updatePoint(*v_iter, &minDist);
    v_iter++;
  }
  PetscFunctionReturn(0);
}

//MeshIDBoundary: create the "marker" label needed by many such things

PetscErrorCode MeshIDBoundary(Mesh mesh) {
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);
  //int dim = m->getDimension();
  ALE::Obj<PETSC_MESH_TYPE::label_type> boundary = m->createLabel("marker");

  int interplevels = m->height(*m->depthStratum(0)->begin());
  if (interplevels == 1) { //noninterpolated case
    ALE::Obj<PETSC_MESH_TYPE::label_sequence> cells = m->heightStratum(0);
    PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin();
    PETSC_MESH_TYPE::label_sequence::iterator c_iter_end = cells->end();
    while (c_iter != c_iter_end) {
      //for now just do for simplicial meshes
      ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneSequence> corners = m->getSieve()->cone(*c_iter);
      PETSC_MESH_TYPE::sieve_type::coneSequence::iterator v_iter = corners->begin();
      PETSC_MESH_TYPE::sieve_type::coneSequence::iterator v_remove = corners->begin();
      PETSC_MESH_TYPE::sieve_type::coneSequence::iterator v_iter_end = corners->end();
      while (v_remove != v_iter_end) {
        ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> face = PETSC_MESH_TYPE::sieve_type::supportSet();
        v_iter = corners->begin();
        while (v_iter != v_iter_end) {
          if (v_iter != v_remove) {
          face->insert(*v_iter);
          }
          v_iter++;
        }
        if (m->getSieve()->nJoin1(face)->size() == 1) {
          PETSC_MESH_TYPE::sieve_type::supportSet::iterator f_iter = face->begin();
          PETSC_MESH_TYPE::sieve_type::supportSet::iterator f_iter_end = face->end();
          while (f_iter != f_iter_end){
            m->setValue(boundary, *f_iter, 1);
            ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSequence> f_support = m->getSieve()->support(*f_iter);
            PETSC_MESH_TYPE::sieve_type::supportSequence::iterator fs_iter = f_support->begin();
            PETSC_MESH_TYPE::sieve_type::supportSequence::iterator fs_iter_end = f_support->end();
            while (fs_iter != fs_iter_end) {
              m->setValue(boundary,*fs_iter, 2);
              fs_iter++;
            }
            f_iter++;
          }
       }
        v_remove++;
      }
      c_iter++;
    }
  } else {
    const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& faces = m->heightStratum(1);
    PETSC_MESH_TYPE::label_sequence::iterator f_iter = faces->begin();
    PETSC_MESH_TYPE::label_sequence::iterator f_iter_end = faces->end();
    while (f_iter != f_iter_end) {
      //mark the boundary faces and every element of their closure as "1".  Mark their support tri/tet as "2" (Matt - Is this right?)
      ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSequence> support = m->getSieve()->support(*f_iter);
      if (support->size() == 1) {
        m->setValue(boundary, *f_iter, 1);
        m->setValue(boundary, *support->begin(), 2);
        ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneArray> boundclose = ALE::SieveAlg<PETSC_MESH_TYPE>::closure(m, *f_iter);
        PETSC_MESH_TYPE::sieve_type::coneArray::iterator bc_iter = boundclose->begin();
        PETSC_MESH_TYPE::sieve_type::coneArray::iterator bc_iter_end = boundclose->end();
        while (bc_iter != bc_iter_end) {
          m->setValue(boundary, *bc_iter, 1);
          bc_iter++;
        }
      }
      f_iter++;
    }

/*    while (v_iter != v_iter_end) {
      ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportArray> vsupport = m->getSieve()->nSupport(*v_iter, interplevels-1); //3D faces or 2D edges
      PETSC_MESH_TYPE::sieve_type::supportArray::iterator s_iter = vsupport->begin();
      PETSC_MESH_TYPE::sieve_type::supportArray::iterator s_iter_end = vsupport->end();
      bool isBound = false;
      while (s_iter != s_iter_end) {
        //check the support of each dim-1 element; if it's supported on one side it is attached to boundary nodes
        ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSequence> fsupport = m->getSieve()->support(*s_iter);
        if (fsupport->size() < 2) isBound = true;
        s_iter++;
      }
      if (isBound) m->setValue(boundary, *v_iter, 1);
      v_iter++;
    }
*/
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshCreateHierarchyMesh"

#if defined PETSC_HAVE_TETGEN || defined PETSC_HAVE_TRIANGLE

ALE::Obj<PETSC_MESH_TYPE> MeshCreateHierarchyMesh(ALE::Obj<PETSC_MESH_TYPE> m, ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> includedVertices) {
  int curmeshsize = 0;
  int dim = m->getDimension();
  //const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& boundary = m->getLabel("marker");
  //const ALE::Obj<PETSC_MESH_TYPE::label_type> hdepth = m->getLabel("hdepth");
  //for (int i = curLevel; i < nLevels; i++) {
  //  curmeshsize += m->getLabelStratum("hdepth", i)->size();
  //}
  curmeshsize = includedVertices->size();
  //PetscPrintf(m->comm(), "new mesh size should be %d vertices.\n", curmeshsize);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& vertices = m->depthStratum(0);
  PETSC_MESH_TYPE::label_sequence::iterator v_iter = vertices->begin();
  PETSC_MESH_TYPE::label_sequence::iterator v_iter_end = vertices->end();
  //double coords[dim * curmeshsize];
  //int indices[dim * curmeshsize];
  //const double * tmpcoords;
  //int index = 0;
    //PetscPrintf(m->comm(), "Mesh Size: %d\n", curmeshsize);
    //load the points and their names in this mesh into a list
    //triangulate/tetrahedralize
    //make into a new sieve.  place coordinates and names on the sieve.

    ALE::Obj<PETSC_MESH_TYPE::sieve_type> boundary_sieve = new PETSC_MESH_TYPE::sieve_type(m->comm(), m->debug());
    ALE::Obj<PETSC_MESH_TYPE> boundary_mesh = new PETSC_MESH_TYPE(m->comm(), m->debug());
    boundary_mesh->setSieve(boundary_sieve);
    ALE::Obj<PETSC_MESH_TYPE::label_type> boundary_marker = boundary_mesh->createLabel("marker");
    //rebuild the boundary, then coarsen it.
    if (dim == 2) {
      boundary_mesh->setDimension(1);
      ALE::Obj<PETSC_MESH_TYPE::label_sequence> bndPoints = m->getLabelStratum("marker", 1);
      PETSC_MESH_TYPE::label_sequence::iterator b_iter = bndPoints->begin();
      PETSC_MESH_TYPE::label_sequence::iterator b_iter_end = bndPoints->end();

      //PetscPrintf(m->comm(),"Copying the Boundary Mesh: %d items\n", bndPoints->size());
      while (b_iter != b_iter_end) {
        if (m->height(*b_iter) > 1) {
          ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSequence> bnd_support = m->getSieve()->support(*b_iter);
          PETSC_MESH_TYPE::sieve_type::supportSequence::iterator bs_iter = bnd_support->begin();
          PETSC_MESH_TYPE::sieve_type::supportSequence::iterator bs_iter_end = bnd_support->end();
          while (bs_iter != bs_iter_end) {
            if (m->getValue(boundary, *bs_iter) == 1) {
              boundary_sieve->addArrow(*b_iter, *bs_iter);
              boundary_mesh->setValue(boundary_marker, *bs_iter, 1);
            }
            bs_iter++;
          }
        }
        b_iter++;
      }
      //coarsen the 2D boundary mesh
      v_iter = vertices->begin();
      v_iter_end = vertices->end();
      while (v_iter != v_iter_end) {
        if (m->getValue(boundary, *v_iter) == 1 && includedVertices->find(*v_iter) == includedVertices->end()) {
          //remove this one and reconnect its sides
          ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSequence> rem_support = boundary_sieve->support(*v_iter);
          PETSC_MESH_TYPE::sieve_type::supportSequence::iterator rs_iter = rem_support->begin();
          PETSC_MESH_TYPE::sieve_type::supportSequence::iterator rs_iter_end = rem_support->end();
          PETSC_MESH_TYPE::point_type bnd_segments [2];
          PETSC_MESH_TYPE::point_type endpts [2];
          int bnd_index = 0;
          while (rs_iter != rs_iter_end) {
            bnd_segments[bnd_index] = *rs_iter;
            ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneSequence> rs_cone = boundary_sieve->cone(*rs_iter);
            PETSC_MESH_TYPE::sieve_type::coneSequence::iterator rsc_iter = rs_cone->begin();
            PETSC_MESH_TYPE::sieve_type::coneSequence::iterator rsc_iter_end = rs_cone->end();
            while (rsc_iter != rsc_iter_end) {
              if (*rsc_iter != *v_iter) {
                endpts[bnd_index] = *rsc_iter;
              }
              rsc_iter++;
            }
            bnd_index++;
            rs_iter++;
          }
          boundary_sieve->removeBasePoint(bnd_segments[0]);
          boundary_sieve->removeCapPoint(*v_iter);
          boundary_sieve->addArrow(endpts[0], bnd_segments[1]);
          //PetscPrintf(m->comm(), "taking %d -%d- %d -%d- %d to %d -%d- %d\n", endpts[0], bnd_segments[0], *v_iter, bnd_segments[1], endpts[1], endpts[0], bnd_segments[1], endpts[1]);
          //boundary_sieve->view();
        }
        v_iter++;
      }
    } else if (dim == 3) {
      boundary_mesh->setDimension(2);
    }
    //insert the interior vertices
    v_iter = vertices->begin();
    v_iter_end = vertices->end();
    while (v_iter != v_iter_end) {
      if ((!boundary_sieve->hasPoint(*v_iter)) && (includedVertices->find(*v_iter) != includedVertices->end())) {
        boundary_sieve->addCapPoint(*v_iter);
        if (m->getValue(boundary, *v_iter) == 1) boundary_mesh->setValue(boundary_marker, *v_iter, 1);
      }
      v_iter++;
    }
    boundary_mesh->stratify();
    //set the boundary meshes coordinate section
    boundary_mesh->setRealSection("coordinates", m->getRealSection("coordinates"));

#if 0    
    //boundary_sieve->view();
    //PetscPrintf(m->comm(),"Copied the Boundary Mesh: %d vertices, %d edges\n", boundary_mesh->depthStratum(0)->size(), boundary_mesh->depthStratum(1)->size());
    //call triangle or tetgen: turns out the options we want on are the same
    //std::string triangleOptions = "zQp"; //(z)ero indexing, output (e)dges, Quiet
    double * finalcoords;
    int * connectivity;
    int * oldpositions;
    int nelements;
    int nverts;
    if (dim == 2) {
#ifdef PETSC_HAVE_TRIANGLE
     boundary_mesh->stratify();
     ALE::Obj<PETSC_MESH_TYPE::label_type> numbering = boundary_mesh->createLabel("numbering");
       //now, take the edges of this sieve and use them to construct a segmentlist.
       //ALE::Obj<PETSC_MESH_TYPE::numbering_type> bnd_vertices_numbering  = boundary_mesh->getFactory()->getNumbering(boundary_mesh, 0);
       //ALE::Obj<PETSC_MESH_TYPE::numbering_type> bnd_edges_numbering = boundary_mesh->getFactory()->getNumbering(boundary_mesh, 1);
       
       //copy over the boundary vertex coordinates:
       ALE::Obj<PETSC_MESH_TYPE::label_sequence> bnd_vertices = boundary_mesh->depthStratum(0);
       PETSC_MESH_TYPE::label_sequence::iterator bndv_iter = bnd_vertices->begin();
       PETSC_MESH_TYPE::label_sequence::iterator bndv_iter_end = bnd_vertices->end();
       while (bndv_iter != bndv_iter_end) {
         tmpcoords = coordinates->restrictPoint(*bndv_iter);
         for (int j = 0; j < dim; j++) {
            coords[dim*index+j] = tmpcoords[j];
            boundary_mesh->setValue(numbering, *bndv_iter, index);
            //PetscPrintf(m->comm(), "%d\n", index);
         }
         indices[index] = *bndv_iter;
         bndv_iter++;
         index++;
      }
      ALE::Obj<PETSC_MESH_TYPE::label_sequence> bnd_edges = boundary_mesh->depthStratum(1);
      int segments[2*bnd_edges->size()];
      int nSegments = bnd_edges->size();
      PETSC_MESH_TYPE::label_sequence::iterator bnde_iter = bnd_edges->begin();
      PETSC_MESH_TYPE::label_sequence::iterator bnde_iter_end = bnd_edges->end();
      int bnd_index = 0;
      while (bnde_iter != bnde_iter_end) {
        ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneSequence> bnde_cone = boundary_sieve->cone(*bnde_iter);
        PETSC_MESH_TYPE::sieve_type::coneSequence::iterator bndec_iter = bnde_cone->begin();
        segments[2*bnd_index] = boundary_mesh->getValue(numbering, *bndec_iter);
        bndec_iter++;
        segments[2*bnd_index + 1] = boundary_mesh->getValue(numbering, *bndec_iter);
        bnd_index++;
        bnde_iter++;
      }

     //PetscPrintf(m->comm(),"Created the segmentlist\n");
      for (int i = curLevel; i < nLevels; i++) {
        //PetscPrintf(m->comm(), "level %d\n", i);
        ALE::Obj<PETSC_MESH_TYPE::label_sequence> curLevVerts = m->getLabelStratum("hdepth", i);
        PETSC_MESH_TYPE::label_sequence::iterator clv_iter = curLevVerts->begin();
        PETSC_MESH_TYPE::label_sequence::iterator clv_iter_end = curLevVerts->end();
          while (clv_iter != clv_iter_end) {
            if (m->getValue(boundary, *clv_iter) != 1) {
            tmpcoords = coordinates->restrictPoint(*clv_iter);
            for (int j = 0; j < dim; j++) {
              coords[index*dim+j] = tmpcoords[j];
            }
            indices[index] = *clv_iter;
            //PetscPrintf(m->comm(), "%d\n", index);
            index++;
          }
          clv_iter++;
        }
      }
     //PetscPrintf(m->comm(),"Triangulate\n");
      //create a segmentlist to keep triangle from doing dumb things.
      triangulateio tridata[2];
      SetupTriangulateio(&tridata[0], &tridata[1]);
      tridata[0].segmentlist = segments;
      tridata[0].numberofsegments = nSegments;
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
      throw ALE::Exception("Must have Triangle installed to use this method. Reconfigure with --download-triangle");
#endif
    } else if (dim == 3) {
#ifdef PETSC_HAVE_TETGEN
      std::string tetgenOptions = "zQe"; //(z)ero indexing, output (e)dges, Quiet
      for (int i = curLevel; i < nLevels; i++) {
        ALE::Obj<PETSC_MESH_TYPE::label_sequence> curLevVerts = m->getLabelStratum("hdepth", i);
        PETSC_MESH_TYPE::label_sequence::iterator clv_iter = curLevVerts->begin();
        PETSC_MESH_TYPE::label_sequence::iterator clv_iter_end = curLevVerts->end();
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
      tetgenio * tetdata = new tetgenio[2];
      //push the points into the thing
      tetdata[0].pointlist = coords;
      tetdata[0].pointmarkerlist = indices;
      tetdata[0].numberofpoints = curmeshsize;
      //tetrahedralize
      tetrahedralize((char *)tetgenOptions.c_str(), &tetdata[0], &tetdata[1]);
      finalcoords = tetdata[1].pointlist;
      connectivity = tetdata[1].tetrahedronlist;
      oldpositions = tetdata[1].pointmarkerlist;
      nelements = tetdata[1].numberoftetrahedra;
      nverts = tetdata[1].numberofpoints;
#else
      throw ALE::Exception("Must have TetGen installed to use this method. Reconfigure with --download-tetgen");
#endif
    }
    //make it into a mesh;
    ALE::Obj<PETSC_MESH_TYPE> newmesh = new PETSC_MESH_TYPE(m->comm(), m->debug());
    newmesh->setDimension(dim);
    ALE::Obj<PETSC_MESH_TYPE::sieve_type> sieve = new PETSC_MESH_TYPE::sieve_type(m->comm(), m->debug());
    ALE::SieveBuilder<PETSC_MESH_TYPE>::buildTopology(sieve, dim, nelements, connectivity, nverts, true, dim+1, -1, newmesh->getArrowSection("orientation"));
    newmesh->setSieve(sieve);
    newmesh->stratify();
    //UPDATE THE MARKER AND FINEMESH VERTEX NUMBERING LABELS
    ALE::Obj<PETSC_MESH_TYPE::label_type> boundary_new = newmesh->createLabel("marker");
    ALE::Obj<PETSC_MESH_TYPE::label_type> fine_corresponds = newmesh->createLabel("fine");
    ALE::Obj<PETSC_MESH_TYPE::label_sequence> newverts = newmesh->depthStratum(0);
    PETSC_MESH_TYPE::label_sequence::iterator nv_iter = newverts->begin();
    PETSC_MESH_TYPE::label_sequence::iterator nv_iter_end = newverts->end();
    while (nv_iter != nv_iter_end) {
      newmesh->setValue(fine_corresponds, *nv_iter, oldpositions[*nv_iter - nelements]);
      if(m->getValue(boundary, oldpositions[*nv_iter - nelements]) == 1) newmesh->setValue(boundary_new, *nv_iter, 1);
      nv_iter++;
    }
#endif //end of LONG if 0
    if (dim == 3) {// HACKED UP: set the height of the vertices to be "1"
      ALE::Obj<PETSC_MESH_TYPE::label_sequence> bound_verts = boundary_mesh->depthStratum(0);
      PETSC_MESH_TYPE::label_sequence::iterator bv_iter = bound_verts->begin();
      PETSC_MESH_TYPE::label_sequence::iterator bv_iter_end = bound_verts->end();
      ALE::Obj<PETSC_MESH_TYPE::label_type> bound_height = boundary_mesh->getLabel("height");
      while (bv_iter != bv_iter_end) {
        boundary_mesh->setValue(bound_height, *bv_iter, 1);
	bv_iter++;
      }
    }
    //PetscPrintf(m->comm(), "%d, %d\n", boundary_mesh->depthStratum(0)->size(), boundary_mesh->heightStratum(0)->size());
    ALE::Obj<PETSC_MESH_TYPE> newmesh = ALE::Generator<PETSC_MESH_TYPE>::generateMesh(boundary_mesh, (m->depth() != 1), true);
    //PetscPrintf(m->comm(), "%d, %d\n", newmesh->depthStratum(0)->size(), newmesh->heightStratum(0)->size());
    ALE::Obj<PETSC_MESH_TYPE::real_section_type> s = newmesh->getRealSection("default");
    const Obj<std::set<std::string> >& discs = m->getDiscretizations();
    //set up the default section
    for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
      newmesh->setDiscretization(*f_iter, m->getDiscretization(*f_iter));
     }
    newmesh->setupField(s);
    newmesh->markBoundaryCells("marker", 1, 2, true);
    return newmesh;
}

#else


ALE::Obj<PETSC_MESH_TYPE> MeshCreateHierarchyMesh(ALE::Obj<PETSC_MESH_TYPE> m, ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> includedVertices) {
  throw ALE::Exception("reconfigure PETSc with --download-triangle and --download-tetgen to use this method.");
  return m;
}

#endif


#undef __FUNCT__
#define __FUNCT__ "MeshCreateHierarchyLabel_Link"
PetscErrorCode MeshCreateHierarchyLabel_Link(Mesh finemesh, double beta, int nLevels, Mesh * outmeshes, Mat * outmats = PETSC_NULL, double curvatureCutoff = 1.0) {
    PetscErrorCode ierr;
  
  PetscFunctionBegin;

  //initialization
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> includedVertices = new PETSC_MESH_TYPE::sieve_type::supportSet();
  includedVertices->clear();
  int nComparisons;
  PetscTruth info;
  ierr = PetscOptionsHasName(PETSC_NULL, "-dmmg_coarsen_info", &info);CHKERRQ(ierr);
  double nComparisons_perPoint_Total = 0.;
  double maxspace = -1., minspace = -1., dist, current_beta;
  double n_space, c_space, v_space;
  double n_coords[3], c_coords[3];
  PETSC_MESH_TYPE::point_type v_point;
  PETSC_MESH_TYPE::point_type c_point, n_point;
  int v_bound;
  
  std::list<PETSC_MESH_TYPE::point_type> comparison_list;
  std::list<PETSC_MESH_TYPE::point_type> coarsen_candidates;
  std::list<PETSC_MESH_TYPE::point_type> support_list;
  
  ierr = MeshGetMesh(finemesh, m);CHKERRQ(ierr);
  int interplevels = m->height(*m->depthStratum(0)->begin());
  //if (interplevels == 1) { //noninterpolated case -- fix later as join is broken
  //  throw ALE::Exception("Cannot Coarsen a Non-Interpolated Mesh (for now, fix is easy)");
  //}
  int dim = m->getDimension();

  if (!m->hasLabel("marker"))ierr = MeshIDBoundary(finemesh);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& boundary = m->getLabel("marker");

  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& spacing = m->getRealSection("spacing");
  const ALE::Obj<PETSC_MESH_TYPE::int_section_type>& seen = m->getIntSection("seen");
  seen->setFiberDimension(m->depthStratum(0), 1);
  m->allocate(seen);
  //PetscPrintf(m->comm(), "allocated seen");
  //const ALE::Obj<PETSC_MESH_TYPE::label_type> hdepth = m->createLabel("hdepth");
  if (info)PetscPrintf(m->comm(), "Original Mesh: %d vertices, %d elements\n", m->depthStratum(0)->size(), m->heightStratum(0)->size());
  const ALE::Obj<PETSC_MESH_TYPE::sieve_type> sieve = m->getSieve();
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& vertices = m->depthStratum(0);
  PETSC_MESH_TYPE::label_sequence::iterator v_iter = vertices->begin();
  PETSC_MESH_TYPE::label_sequence::iterator v_iter_end = vertices->end();
  PETSC_MESH_TYPE::point_type max_vertex_index = -1;

  //Setup, find the minimum and maximum spacing values and force in high-curvature nodes.

  while(v_iter != v_iter_end) {

    v_point = *v_iter;
    v_bound = m->getValue(boundary, v_point);
    //m->setValue(hdepth, v_point, 0);
    v_space = *spacing->restrictPoint(v_point);
    if (*v_iter > max_vertex_index) max_vertex_index = *v_iter;
    if ((v_space > maxspace) || (maxspace == -1.)) maxspace = v_space;
    if ((v_space < minspace) || (minspace == -1.)) minspace = v_space;
    if (m->depth(v_point) == 0 && v_bound == 1) {
        double cur_curvature = Curvature(m, v_point);
        if (info && fabs(cur_curvature) > curvatureCutoff) PetscPrintf(m->comm(), "Curvy point: %f\n", cur_curvature);
        if (fabs(cur_curvature) > curvatureCutoff) includedVertices->insert(*v_iter);
    }
    int pt_val = 0;
    seen->updatePoint(*v_iter, &pt_val);
    v_iter++;
  }
  for (int curLevel = nLevels-1; curLevel > 0; curLevel--) {
    coarsen_candidates.clear();
    nComparisons = 0;
    current_beta = pow(beta, curLevel);
    if (info) PetscPrintf(m->comm(), "coarsening by %f\n", current_beta);
    //recopy the first two levels of the sieve into a new sieve.
    ALE::Obj<PETSC_MESH_TYPE::sieve_type> coarsen_sieve = new PETSC_MESH_TYPE::sieve_type(m->comm(), m->debug());
    ALE::Obj<PETSC_MESH_TYPE> coarsen_mesh = new PETSC_MESH_TYPE(m->comm(), m->debug());
    coarsen_mesh->setSieve(coarsen_sieve);
    //INEFFICIENT!
    coarsen_candidates.clear();
    //ALE::Obj<PETSC_MESH_TYPE::label_type> coarsen_candidates = m->createLabel("candidates");
    v_iter = vertices->begin();
    v_iter_end = vertices->end();
    if (interplevels > 1) { //interpolated case; merely copy the edges out
      while (v_iter != v_iter_end) {
        ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSequence> adjacent_edges = sieve->support(*v_iter);
        PETSC_MESH_TYPE::sieve_type::supportSequence::iterator ae_iter = adjacent_edges->begin();
        PETSC_MESH_TYPE::sieve_type::supportSequence::iterator ae_iter_end = adjacent_edges->end();
        while (ae_iter != ae_iter_end) {
          coarsen_sieve->addArrow(*v_iter, *ae_iter);
          ae_iter++;
        }
        v_iter++;
      }
    } else { //noninterpolated case, build the edges
      //PETSC_MESH_TYPE::sieve_type::supportSet seen; sets have logarithmic complexity
      PETSC_MESH_TYPE::point_type min_coarsen_edge_index = max_vertex_index + 1;
      while (v_iter != v_iter_end) {
        //get the neighbors of this point
        ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneSet> mockup_edge_endpoints = sieve->cone(sieve->support(*v_iter));
        PETSC_MESH_TYPE::sieve_type::coneSet::iterator m_iter = mockup_edge_endpoints->begin();
        PETSC_MESH_TYPE::sieve_type::coneSet::iterator m_iter_end = mockup_edge_endpoints->end();
        while (m_iter != m_iter_end) {
          if (*m_iter != *v_iter && *seen->restrictPoint(*m_iter) == 0) {
            coarsen_sieve->addArrow(*v_iter, min_coarsen_edge_index);
            coarsen_sieve->addArrow(*m_iter, min_coarsen_edge_index);
            min_coarsen_edge_index++;
          }
          m_iter++;
        }
        int pt_val = 1;
        seen->updatePoint(*v_iter, &pt_val);
        v_iter++;
      }
    }
    coarsen_mesh->stratify();
    //PetscPrintf(coarsen_mesh->comm(), "%d vertices, %d edges in the coarsen structure\n", coarsen_mesh->depthStratum(0)->size(), coarsen_mesh->heightStratum(0)->size());
    //Interior Pass: Eliminate Nodes that collide with already included nodes or the boundary.
    //now, take the points that neighbor the already included points in the main sieve AND THE BOUNDARY:
    //for (int compare_marker = 1; compare_marker <= 2; compare_marker++) { //two passes, interior and boundary
    v_iter = vertices->begin();
    v_iter_end = vertices->end();
    while (v_iter != v_iter_end) {
      if ((includedVertices->find(*v_iter) != includedVertices->end()) || (m->getValue(boundary, *v_iter) == 1)) {
        ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSequence> tangent_edges = coarsen_sieve->support(*v_iter);
        PETSC_MESH_TYPE::sieve_type::supportSequence::iterator te_iter = tangent_edges->begin();
        PETSC_MESH_TYPE::sieve_type::supportSequence::iterator te_iter_end = tangent_edges->end();
        while (te_iter != te_iter_end) {
          if ((m->getValue(boundary, *te_iter) == 0)) {
            comparison_list.push_back(*te_iter);
          }
          te_iter++;
        }
      }
      if ((includedVertices->find(*v_iter) == includedVertices->end()) && (m->getValue(boundary, *v_iter) == 0)) {
        //m->setValue(coarsen_candidates, *v_iter, 1);
        coarsen_candidates.push_front(*v_iter);
      } 
      v_iter++;
    }
    //Interior Run: Treat the boundaries as barriers
    bool notDone = true;
    while (notDone) {
    while (!comparison_list.empty()){
      PETSC_MESH_TYPE::point_type c_edge = *comparison_list.begin();
      comparison_list.pop_front();
      //PetscPrintf(m->comm(), "comparing across edge %d\n", c_edge);
      ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneSequence> end_points = coarsen_sieve->cone(c_edge);
      PETSC_MESH_TYPE::sieve_type::coneSequence::iterator ep_iter = end_points->begin();
      PETSC_MESH_TYPE::point_type ep_1 = *ep_iter;
      ep_iter++;
      PETSC_MESH_TYPE::point_type  ep_2 = *ep_iter;
      bool ep_1_hdepth = (includedVertices->find(ep_1) != includedVertices->end()); //m->getValue(hdepth, ep_1);
      bool ep_2_hdepth = (includedVertices->find(ep_2) != includedVertices->end());//m->getValue(hdepth, ep_2);
      int ep_1_bound = m->getValue(boundary, ep_1);
      int ep_2_bound = m->getValue(boundary, ep_2);
      bool ep_1_dominates = ((ep_1_hdepth == true)  || ((ep_1_bound == 1)));
      bool ep_2_dominates = ((ep_2_hdepth == true)  || ((ep_2_bound == 1)));
      //if ((!ep_1_dominates) && (!ep_2_dominates)) {
      //  PetscPrintf(m->comm(),"%d, %d\n", ep_1_bound, ep_2_bound);
      //  throw ALE::Exception("Coarse To Coarse Comparison: This Should Not Have Happened");
      //}
      if ((!ep_1_dominates && ep_2_dominates) || (ep_1_dominates && !ep_2_dominates)) {
        if (ep_1_dominates) { 
        c_point = ep_2;
        n_point = ep_1;
        } else {
         c_point = ep_1;
         n_point = ep_2;
        }
        nComparisons++;
        PetscMemcpy(c_coords, coordinates->restrictPoint(c_point), dim*sizeof(double));
        c_space = *spacing->restrictPoint(c_point);
        PetscMemcpy(n_coords, coordinates->restrictPoint(n_point), dim*sizeof(double));
        n_space = *spacing->restrictPoint(n_point);
        dist = 0.;
        for (int i = 0; i < dim; i++) {
            dist += (n_coords[i] - c_coords[i])*(n_coords[i] - c_coords[i]);
        }
        dist = sqrt(dist);
        if (dist < current_beta*(n_space + c_space)) {
          //remove the point and this edge from existence and link its neighbors up with the eliminating point, adding the neighboring edges to THE LIST.
          //m->setValue(coarsen_candidates, c_point, 0);

          //Replace this with a more intelligent topology reconnection eventually.

          ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSequence> move_these_edges = coarsen_sieve->support(c_point);
          PETSC_MESH_TYPE::sieve_type::supportSequence::iterator mte_iter = move_these_edges->begin();
          PETSC_MESH_TYPE::sieve_type::supportSequence::iterator mte_iter_end = move_these_edges->end();

          while (mte_iter != mte_iter_end) {
            if (*mte_iter != c_edge) {
              coarsen_sieve->addArrow(n_point, *mte_iter);
              if (coarsen_sieve->cone(*mte_iter)->size() == 2) {  //it's one of the edges that should be eliminated here, don't compare across it.
                support_list.push_front(*mte_iter); //save for elimination because we don't want to screw up the iterator
              } else { //compare across the just-changed edge
                comparison_list.push_back(*mte_iter);
              }
            }
            mte_iter++;
          }
          while (!support_list.empty()) {
            PETSC_MESH_TYPE::point_type delete_this_point = *support_list.begin();
            support_list.pop_front();
            coarsen_sieve->removeBasePoint(delete_this_point);
          }
          coarsen_sieve->removeBasePoint(c_edge);
          coarsen_sieve->removeCapPoint(c_point);

        }
      } else {
        //do nothing .. this edge will never be bothered again save for the difference between the boundary and non-boundary passes.
      }
    }


    //KILL THIS IT'S INEFFICENT  
    //ALE::Obj<PETSC_MESH_TYPE::label_sequence> new_candidate_options = m->getLabelStratum("candidates", 1);
    PETSC_MESH_TYPE::point_type new_candidate_point = *coarsen_candidates.begin();
    coarsen_candidates.pop_front();
    while (!coarsen_sieve->hasPoint(new_candidate_point) && coarsen_candidates.size() > 0) {
      new_candidate_point = *coarsen_candidates.begin();
      coarsen_candidates.pop_front();
    }
    if (coarsen_candidates.size() == 0) {
      notDone = false;
    } else { //add this point to the mesh, adding its neighbors to the comparison queue.
      //m->setValue(coarsen_candidates, new_candidate_point, 0);
      //      m->setValue(hdepth, new_candidate_point, curLevel);
      includedVertices->insert(new_candidate_point);
      ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSequence> neighbor_edges = coarsen_sieve->support(new_candidate_point);
      PETSC_MESH_TYPE::sieve_type::supportSequence::iterator ne_iter = neighbor_edges->begin();
      PETSC_MESH_TYPE::sieve_type::supportSequence::iterator ne_iter_end = neighbor_edges->end();
      //PetscPrintf(coarsen_mesh->comm(), "%d edges added to the comparison queue for vertex %d.\n", neighbor_edges->size(), new_candidate_point);
      while (ne_iter != ne_iter_end) {
        if ((m->getValue(boundary, *ne_iter) == 0)) {
          comparison_list.push_back(*ne_iter);
        }
        ne_iter++;
      }
    }
    }
    //Boundary Run: ONLY treat the boundary.
    ALE::Obj<PETSC_MESH_TYPE::label_sequence> bound_points = m->getLabelStratum("marker", 1);
    v_iter = bound_points->begin();
    v_iter_end = bound_points->end();
    while (v_iter != v_iter_end) {
      if ((m->depth(*v_iter) == 0)&&(includedVertices->find(*v_iter) != includedVertices->end())) {
        ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSequence> tangent_edges = coarsen_sieve->support(*v_iter);
        PETSC_MESH_TYPE::sieve_type::supportSequence::iterator te_iter = tangent_edges->begin();
        PETSC_MESH_TYPE::sieve_type::supportSequence::iterator te_iter_end = tangent_edges->end();
        while (te_iter != te_iter_end) {
          if ((m->getValue(boundary, *te_iter) == 1)) {
            comparison_list.push_back(*te_iter);
	  }
          te_iter++;
        }
      }
      if ((includedVertices->find(*v_iter) == includedVertices->end()) && (m->getValue(boundary, *v_iter) == 1) && (m->depth(*v_iter) == 0)) {
        //m->setValue(coarsen_candidates, *v_iter, 1);
        coarsen_candidates.push_front(*v_iter);
        //PetscPrintf(coarsen_mesh->comm(), "Size of the support of %d is %d\n", *v_iter, coarsen_sieve->support(*v_iter)->size());
      }  
      v_iter++;
    }
    //PetscPrintf(m->comm(), "Comparison_queue size: %d, Candidate_list size %d\n", comparison_list.size(), coarsen_candidates.size());
    notDone = true;
    while (notDone) {
    while (!comparison_list.empty()){
      PETSC_MESH_TYPE::point_type c_edge = *comparison_list.begin();
      comparison_list.pop_front();
      //PetscPrintf(m->comm(), "comparing across edge %d\n", c_edge);
      ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneSequence> end_points = coarsen_sieve->cone(c_edge);
      PETSC_MESH_TYPE::sieve_type::coneSequence::iterator ep_iter = end_points->begin();
      PETSC_MESH_TYPE::point_type ep_1 = *ep_iter;
      ep_iter++;
      PETSC_MESH_TYPE::point_type  ep_2 = *ep_iter;
      //int ep_1_hdepth = m->getValue(hdepth, ep_1);
      //int ep_2_hdepth = m->getValue(hdepth, ep_2);
      bool ep_1_dominates = (includedVertices->find(ep_1) != includedVertices->end()); //((ep_1_hdepth != 0));
      bool ep_2_dominates = (includedVertices->find(ep_2) != includedVertices->end()); //((ep_2_hdepth != 0));
      //if ((!ep_1_dominates) && (!ep_2_dominates)) {
      //  PetscPrintf(m->comm(),"%d, %d\n", ep_1_bound, ep_2_bound);
      //  throw ALE::Exception("Coarse To Coarse Comparison: This Should Not Have Happened");
      //}
      if ((!ep_1_dominates && ep_2_dominates) || (ep_1_dominates && !ep_2_dominates)) {
        if (ep_1_dominates) { 
        c_point = ep_2;
        n_point = ep_1;
        } else {
         c_point = ep_1;
         n_point = ep_2;
        }
        nComparisons++;
        PetscMemcpy(c_coords, coordinates->restrictPoint(c_point), dim*sizeof(double));
        c_space = *spacing->restrictPoint(c_point);
        PetscMemcpy(n_coords, coordinates->restrictPoint(n_point), dim*sizeof(double));
        n_space = *spacing->restrictPoint(n_point);
        dist = 0.;
        for (int i = 0; i < dim; i++) {
            dist += (n_coords[i] - c_coords[i])*(n_coords[i] - c_coords[i]);
        }
        dist = sqrt(dist);
        if (dist < current_beta*(n_space + c_space)) {
          //remove the point and this edge from existence and link its neighbors up with the eliminating point, adding the neighboring edges to THE LIST.
          //m->setValue(coarsen_candidates, c_point, 0);
          ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSequence> move_these_edges = coarsen_sieve->support(c_point);
          PETSC_MESH_TYPE::sieve_type::supportSequence::iterator mte_iter = move_these_edges->begin();
          PETSC_MESH_TYPE::sieve_type::supportSequence::iterator mte_iter_end = move_these_edges->end();
          while (mte_iter != mte_iter_end) {
            if (*mte_iter != c_edge) {
              coarsen_sieve->addArrow(n_point, *mte_iter); 
              if (coarsen_sieve->cone(*mte_iter)->size() == 2) {  //it's one of the edges that should be eliminated here, don't compare across it.
                support_list.push_front(*mte_iter); //save for elimination because we don't want to screw up the iterator
              } else { //compare across the just-changed edge
                if (m->getValue(boundary, *mte_iter) == 1) 
                comparison_list.push_back(*mte_iter);
              }
            }
            mte_iter++;
          }
          while (!support_list.empty()) {
            PETSC_MESH_TYPE::point_type delete_this_point = *support_list.begin();
            support_list.pop_front();
            coarsen_sieve->removeBasePoint(delete_this_point);
          }
          coarsen_sieve->removeBasePoint(c_edge);
          coarsen_sieve->removeCapPoint(c_point);
          //PetscPrintf(m->comm(), "Removed %d\n", c_point);
        }
      } else {
        //do nothing .. this edge will never be bothered again save for the difference between the boundary and non-boundary passes.
      }
    }

    //ALE::Obj<PETSC_MESH_TYPE::label_sequence> new_candidate_options = m->getLabelStratum("candidates", 1);

    PETSC_MESH_TYPE::point_type new_candidate_point = *coarsen_candidates.begin();
    coarsen_candidates.pop_front();
    while (!coarsen_sieve->hasPoint(new_candidate_point) && coarsen_candidates.size() > 0) {
      new_candidate_point = *coarsen_candidates.begin();
      coarsen_candidates.pop_front();
      //PetscPrintf(coarsen_mesh->comm(), "%d edges in the support of the current coarsen candidate; %d somethings in the cone\n", coarsen_sieve->support(new_candidate_point)->size(), coarsen_sieve->cone(new_candidate_point)->size());
    }
    if (coarsen_candidates.size() == 0) {
      notDone = false;
    } else { //add this point to the mesh, adding its neighbors to the comparison queue.
      //m->setValue(coarsen_candidates, *new_candidate, 0);
      //m->setValue(hdepth, new_candidate_point, curLevel);
      includedVertices->insert(new_candidate_point);
      ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSequence> neighbor_edges = coarsen_sieve->support(new_candidate_point);
      PETSC_MESH_TYPE::sieve_type::supportSequence::iterator ne_iter = neighbor_edges->begin();
      PETSC_MESH_TYPE::sieve_type::supportSequence::iterator ne_iter_end = neighbor_edges->end();
      //PetscPrintf(coarsen_mesh->comm(), "%d edges added to the comparison queue for vertex %d.\n", neighbor_edges->size(), new_candidate_point);
      if (neighbor_edges->size() == 0) throw ALE::Exception("bad vertex.");
      while (ne_iter != ne_iter_end) {
        if (m->getValue(boundary, *ne_iter) == 1) {
          comparison_list.push_back(*ne_iter);
	}
        ne_iter++;
      }
    }
    }


    //}
    coarsen_mesh->stratify();
    ALE::Obj<PETSC_MESH_TYPE::label_sequence> coarsen_vertices = coarsen_mesh->depthStratum(0);
    PETSC_MESH_TYPE::label_sequence::iterator cv_iter = coarsen_vertices->begin();
    PETSC_MESH_TYPE::label_sequence::iterator cv_iter_end = coarsen_vertices->end();
    //what does this part do again?
    includedVertices->clear();
    while (cv_iter != cv_iter_end) {
      // m->setValue(hdepth, *cv_iter, curLevel);
      includedVertices->insert(*cv_iter);
      cv_iter++;
    }
  if(info) PetscPrintf(m->comm(), "%d points in %d comparisons\n", coarsen_mesh->depthStratum(0)->size(), nComparisons);
  nComparisons_perPoint_Total += nComparisons;
  ALE::Obj<PETSC_MESH_TYPE> newmesh = MeshCreateHierarchyMesh(m, includedVertices);
  if(info) PetscPrintf(m->comm(), "%d: %d vertices, %d elements\n", curLevel, newmesh->depthStratum(0)->size(), newmesh->heightStratum(0)->size());
  MeshSetMesh(outmeshes[curLevel-1], newmesh);
  }
  nComparisons_perPoint_Total = nComparisons_perPoint_Total/((double)m->depthStratum(0)->size());
  if(info) PetscPrintf(m->comm(), "Comparisons Per Point: %f\n", nComparisons_perPoint_Total);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshCreateHierarchyLabel"

//MeshCreateHierarchyLabel: Create a label that tells what the highest level a given vertex appears in where 0 is fine and n is coarsest.
PetscErrorCode MeshCreateHierarchyLabel(Mesh finemesh, double beta, int nLevels, Mesh * outmeshes, Mat * outmats = PETSC_NULL) {
  PetscErrorCode ierr;
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscFunctionBegin;
  PetscTruth info;
  int overallComparisons = 0;
  ierr = PetscOptionsHasName(PETSC_NULL, "-dmmg_coarsen_info", &info);CHKERRQ(ierr);
  ierr = MeshGetMesh(finemesh, m);CHKERRQ(ierr);
  int dim = m->getDimension();
  if (info)PetscPrintf(m->comm(), "Original Mesh: %d vertices, %d elements\n", m->depthStratum(0)->size(), m->heightStratum(0)->size());
  if (!m->hasLabel("marker"))ierr = MeshIDBoundary(finemesh);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& spacing = m->getRealSection("spacing");
  const ALE::Obj<PETSC_MESH_TYPE::label_type> hdepth = m->createLabel("hdepth");
  const ALE::Obj<PETSC_MESH_TYPE::label_type> dompoint = m->createLabel("dompoint");
  const ALE::Obj<PETSC_MESH_TYPE::label_type> traversal = m->createLabel("traversal");
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& boundary = m->getLabel("marker");
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& vertices = m->depthStratum(0);
  PETSC_MESH_TYPE::label_sequence::iterator v_iter = vertices->begin();
  PETSC_MESH_TYPE::label_sequence::iterator v_iter_end = vertices->end();
  double maxspace = -1., minspace = -1., dist;
  PETSC_MESH_TYPE::point_type max_vertex_index = -1;
  while(v_iter != v_iter_end) {
    //initialize the label to 0.
    m->setValue(hdepth, *v_iter, 0);
    //discover the maximum and minimum spacing functions in the mesh.
    double vspace = *spacing->restrictPoint(*v_iter);
    if ((vspace > maxspace) || (maxspace == -1.)) maxspace = vspace;
    if ((vspace < minspace) || (minspace == -1.)) minspace = vspace;
    if (*v_iter > max_vertex_index) max_vertex_index = *v_iter;
    v_iter++;
  }
  //PUT IN PART FOR AUTOMATICALLY ADDING HIGH-CURVATURE BOUNDARY NODES
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& boundaryvertices = m->getLabelStratum("marker", 1); //boundary
  PETSC_MESH_TYPE::label_sequence::iterator bv_iter = boundaryvertices->begin();
  PETSC_MESH_TYPE::label_sequence::iterator bv_iter_end = boundaryvertices->end();
 // PetscPrintf(m->comm(), "NUMBER OF BOUNDARY POINTS: %d\n", boundaryvertices->size());
  while (bv_iter != bv_iter_end) {
    if (m->depth(*bv_iter) == 0) if (Curvature(m, *bv_iter) > 0.1) {
      m->setValue(hdepth, *bv_iter, nLevels-1);
    }
    bv_iter++;
  }
 // PetscPrintf(m->comm(), "Forced in %d especially curved boundary nodes.\n", m->getLabelStratum("hdepth", nLevels-1)->size());
  double *bvCoords = new double[dim];
  PETSC_MESH_TYPE::point_type bvdom;
  std::list<PETSC_MESH_TYPE::point_type> complist;
  std::list<PETSC_MESH_TYPE::point_type> domlist; //stores the points dominated by the current point.
  for (int curLevel = nLevels-1; curLevel > 0; curLevel--) {
    double curBeta = pow(beta, curLevel);
   //OUR MODUS OPERANDI:
    //1. do the boundary and the interior identically but separately
    //2. keep track of the point that eliminates each point on each level.  This should work sort of like an approximation to the voronoi partitions.  Compare against these first as they're more likely to collide than neighbors.  Also compare to the points that eliminate the neighbors in the same fashion.
    //3. If the point is not eliminated by its old eliminator we must traverse out to max(space(v)) + space(i).
    //GOAL: only eliminate each point once! if we add a point that eliminates other points get rid of them in the traversal! (and set their elimination marker appropriately.)
    double comparison_const;
    PETSC_MESH_TYPE::label_sequence::iterator bv_iter = boundaryvertices->begin();
    PETSC_MESH_TYPE::label_sequence::iterator bv_iter_end = boundaryvertices->end();
    ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneSet> neighbors = m->getSieve()->cone(m->getSieve()->support(*bv_iter));
    PETSC_MESH_TYPE::sieve_type::coneSet::iterator n_iter = neighbors->begin();
    PETSC_MESH_TYPE::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
    int nComparisonsTotal = 0;
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
        bool setDist = false;
        if (bvdom != -1) {
           setDist = true;
           complist.push_front(bvdom);
        }

        while ((!complist.empty()) && canAdd) {
          nComparisonsTotal++;
          PETSC_MESH_TYPE::point_type curpt = *complist.begin();
          complist.pop_front();
          dist = 0.;
          double curSpace = *spacing->restrictPoint(curpt);
          const double * curCoords = coordinates->restrictPoint(curpt); 
          for (int i = 0; i < dim; i++) {
            dist += (curCoords[i] - bvCoords[i])*(curCoords[i] - bvCoords[i]);
          }
          //gets the distance to bvdom as the maximum radius to compare out
          dist = sqrt(dist);
          comparison_const = 0.5*curBeta;
          if (setDist == true) {
            maxspace = dist;
            setDist = false;
          } else {
            maxspace = 3.*curSpace*comparison_const;
          } 
          PETSC_MESH_TYPE::point_type curpt_dom = m->getValue(dompoint, curpt);
          int curpt_depth = m->getValue(hdepth, curpt);
          int curpt_bound = m->getValue(boundary, curpt);
          if ((dist < comparison_const*(bvSpace + curSpace))&&(curpt_depth > 0)) { //collision with an already added node
            canAdd = false;
            m->setValue(dompoint, *bv_iter, curpt);
          } else if (dist < maxspace) { //go out to the dompoint
            neighbors = m->getSieve()->cone(m->getSieve()->support(curpt));
            n_iter = neighbors->begin();
            n_iter_end = neighbors->end();
            while (n_iter != n_iter_end) {
              if ((m->getValue(traversal, *n_iter) == 0)) {
                m->setValue(traversal, *n_iter, 1);
                complist.push_back(*n_iter);
              }
              n_iter++;
            }
          }
          if ((dist < comparison_const*(bvSpace + curSpace)) && (curpt_depth == 0)) { //add the point to the list of points dominated by this node; points eliminated in one step later
            domlist.push_front(curpt);
            if (curpt_bound == 0) {
              m->setValue(dompoint, curpt, *bv_iter);
            }
          }
          if ((dist < maxspace) && (curpt_depth == 0)) {
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
          std::list<PETSC_MESH_TYPE::point_type>::iterator dom_iter = domlist.begin();
          std::list<PETSC_MESH_TYPE::point_type>::iterator dom_iter_end = domlist.end();
          while (dom_iter != dom_iter_end) {
            m->setValue(dompoint, *dom_iter, *bv_iter);
            dom_iter++;
          }
        }
        domlist.clear();
        //unset the traversal listing
        ALE::Obj<PETSC_MESH_TYPE::label_sequence> travnodes = m->getLabelStratum("traversal", 1);
        PETSC_MESH_TYPE::label_sequence::iterator tn_iter = travnodes->begin();
        PETSC_MESH_TYPE::label_sequence::iterator tn_iter_end = travnodes->end();
        while (tn_iter != tn_iter_end) {
          complist.push_front(*tn_iter);
          tn_iter++;
        }
        while (!complist.empty()) {
          PETSC_MESH_TYPE::point_type emptpt = *complist.begin();
          complist.pop_front();
          m->setValue(traversal, emptpt, 0);
        }
      }
      bv_iter++;
    }
 //   PetscPrintf(m->comm(), "Added %d new boundary vertices\n", m->getLabelStratum("hdepth", curLevel)->size());
    //INTERIOR NODES:
    complist.clear();
    ALE::Obj<PETSC_MESH_TYPE::label_sequence> intverts = m->depthStratum(0);
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
        maxspace = 0.5*bvSpace*curBeta;
        bool setDist = false;
        if (bvdom != -1) {
           setDist = true;
           complist.push_front(bvdom);
        } 
        while ((!complist.empty()) && canAdd) {
          nComparisonsTotal++;
          PETSC_MESH_TYPE::point_type curpt = *complist.begin();
          complist.pop_front();
          //PetscPrintf(m->comm(), "Comparing %d to %d\n", *bv_iter, curpt);
          dist = 0.;
          double curSpace = *spacing->restrictPoint(curpt);
          const double * curCoords = coordinates->restrictPoint(curpt); 
          for (int i = 0; i < dim; i++) {
            dist += (curCoords[i] - bvCoords[i])*(curCoords[i] - bvCoords[i]);
          }
          comparison_const = 0.5*curBeta;
          dist = sqrt(dist);
          if (setDist == true) {
            maxspace = dist;
            setDist = false;
          } else {
            maxspace = 3.*curSpace*comparison_const;
          }
          int curpt_depth = m->getValue(hdepth, curpt);
          int curpt_bound = m->getValue(boundary, curpt);
          PETSC_MESH_TYPE::point_type curpt_dom = m->getValue(dompoint, curpt);
          if ((dist < comparison_const*(bvSpace + curSpace))&&(curpt_depth > 0)) {
            canAdd = false;
            m->setValue(dompoint, *bv_iter, curpt);
          } else if ((dist < comparison_const*(bvSpace+curSpace)) && (curpt_bound == 1)) {
            canAdd = false;
            m->setValue(dompoint, *bv_iter, curpt);
          } else if (dist < maxspace) {
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
          }
          if ((dist < maxspace) && (curpt_depth == 0)) {
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
          std::list<PETSC_MESH_TYPE::point_type>::iterator dom_iter = domlist.begin();
          std::list<PETSC_MESH_TYPE::point_type>::iterator dom_iter_end = domlist.end();
          while (dom_iter != dom_iter_end) {
            m->setValue(dompoint, *dom_iter, *bv_iter);
            dom_iter++;
          }
        } 
        domlist.clear();
        complist.clear();
        //unset the traversal listing
        ALE::Obj<PETSC_MESH_TYPE::label_sequence> travnodes = m->getLabelStratum("traversal", 1);
        PETSC_MESH_TYPE::label_sequence::iterator tn_iter = travnodes->begin();
        PETSC_MESH_TYPE::label_sequence::iterator tn_iter_end = travnodes->end();
        while (tn_iter != tn_iter_end) {
          complist.push_front(*tn_iter);
          tn_iter++;
        }
        while (!complist.empty()) {
          PETSC_MESH_TYPE::point_type emptpt = *complist.begin();
          complist.pop_front();
          m->setValue(traversal, emptpt, 0);
        }
      }
      bv_iter++;
    }
  //  PetscPrintf(m->comm(), "Included %d new points in level %d\n", m->getLabelStratum("hdepth", curLevel)->size(), curLevel);
    ALE::Obj<PETSC_MESH_TYPE> newmesh;
    // = MeshCreateHierarchyMesh(m, nLevels, curLevel);
    if (info)PetscPrintf(m->comm(), "%d: %d vertices, %d elements in %d comparisons\n", curLevel, newmesh->depthStratum(0)->size(), newmesh->heightStratum(0)->size(), nComparisonsTotal);
    overallComparisons += nComparisonsTotal; //yeah yeah horrible names

    MeshSetMesh(outmeshes[curLevel-1], newmesh);
/*    ierr = SectionRealCreate(newmesh->comm(), &section);CHKERRQ(ierr);
    ierr = SectionRealGetBundle(section, newmesh);CHKERRQ(ierr);
    ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) section, "default");CHKERRQ(ierr);
    ierr = MeshSetSectionReal(outmeshes[curLevel-1], section);CHKERRQ(ierr);
    ierr = SectionRealDestroy(section);*/
  } //end of level for
  delete [] bvCoords;
  if (info)PetscPrintf(m->comm(), "Comparisons per Point: %f\n", (float)overallComparisons/vertices->size());
  //std::list<PETSC_MESH_TYPE::point_type> tricomplist;
/*  for (int curLevel = 0; curLevel < nLevels-1; curLevel++) {
    if (info) PetscPrintf(m->comm(), "Building the prolongation section from level %d to level %d\n", curLevel, curLevel+1);
    ALE::Obj<PETSC_MESH_TYPE> c_m;
    ALE::Obj<PETSC_MESH_TYPE> f_m;
    if (curLevel == 0) {
      f_m = m;
    } else {
      ierr = MeshGetMesh(outmeshes[curLevel-1], f_m);CHKERRQ(ierr);
    }
    ierr = MeshGetMesh(outmeshes[curLevel], c_m);CHKERRQ(ierr);
    ALE::Obj<PETSC_MESH_TYPE::label_type> prolongation = f_m->createLabel("prolongation");
    ALE::Obj<PETSC_MESH_TYPE::label_type> coarse_traversal = c_m->createLabel("traversal");
    ALE::Obj<PETSC_MESH_TYPE::label_sequence> levelVertices = m->getLabelStratum("hdepth", curLevel);
   // PetscPrintf(m->comm(), "%d points in level %d\n", levelVertices->size(), curLevel);
    PETSC_MESH_TYPE::label_sequence::iterator lv_iter = levelVertices->begin();
    PETSC_MESH_TYPE::label_sequence::iterator lv_iter_end = levelVertices->end();
    int interpolatelevels = c_m->height(*c_m->depthStratum(0)->begin()); //see if the mesh is interpolated or not
    double lvCoords[dim];
    while (lv_iter != lv_iter_end) {
      int ncomps = 0;
      ierr = PetscMemcpy(lvCoords, coordinates->restrictPoint(*lv_iter), dim*sizeof(double));
      if (m->getValue(boundary, *lv_iter) != 1) {
        //get the triangle/tet fan around the dominating point in the next level up
        PETSC_MESH_TYPE::point_type dp = m->getValue(dompoint, *lv_iter);
        if (m->getValue(hdepth, dp) < curLevel+1) dp = m->getValue(dompoint, dp); //if it's a boundary node it can be a dominating point and NOT be in the topmesh
        ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportArray> trifan = c_m->getSieve()->nSupport(*c_m->getLabelStratum("fine", dp)->begin(), interpolatelevels);
        PETSC_MESH_TYPE::sieve_type::supportArray::iterator tf_iter = trifan->begin();
        PETSC_MESH_TYPE::sieve_type::supportArray::iterator tf_iter_end = trifan->end();
        while (tf_iter != tf_iter_end) {
          tricomplist.push_front(*tf_iter); //initialize the closest-guess comparison list.
          c_m->setValue(coarse_traversal, *tf_iter, 1);
          tf_iter++;
        }
        //PetscPrintf(m->comm(), "%d initial guesses\n", trifan->size());
        bool isFound = false;
        while (!tricomplist.empty() && !isFound) {
          PETSC_MESH_TYPE::point_type curTri = *tricomplist.begin();
          tricomplist.pop_front();
          ncomps++;
          if (PointIsInElement(c_m, curTri, lvCoords)) {
            PETSC_MESH_TYPE::point_type fmpoint;
            if (curLevel == 0) {
              fmpoint = *lv_iter;
            } else {
              fmpoint = *f_m->getLabelStratum("fine", *lv_iter)->begin();
            } 
            f_m->setValue(prolongation, fmpoint, curTri);
            isFound = true;
          //  PetscPrintf(m->comm(), "%d located in %d\n", *lv_iter, curTri);
          } else {
            ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> trineighbors = c_m->getSieve()->support(c_m->getSieve()->cone(curTri));
            PETSC_MESH_TYPE::sieve_type::supportSet::iterator tn_iter = trineighbors->begin();
            PETSC_MESH_TYPE::sieve_type::supportSet::iterator tn_iter_end = trineighbors->end();
            while (tn_iter != tn_iter_end) {
              if (c_m->getValue(coarse_traversal, *tn_iter) != 1){
                tricomplist.push_back(*tn_iter);
                c_m->setValue(coarse_traversal, *tn_iter, 1);
              }
              tn_iter++;
            }
          }
        }
        ALE::Obj<PETSC_MESH_TYPE::label_sequence> travtris = c_m->getLabelStratum("traversal", 1);
        PETSC_MESH_TYPE::label_sequence::iterator tt_iter = travtris->begin();
        PETSC_MESH_TYPE::label_sequence::iterator tt_iter_end = travtris->end();
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
        PETSC_MESH_TYPE::point_type fmpoint;
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
          PETSC_MESH_TYPE::point_type cmpt = *c_m->getLabelStratum("fine", *lv_iter)->begin();
          PETSC_MESH_TYPE::point_type fmpoint;
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
  }*/
/*  if (info) PetscPrintf(m->comm(), "Building the prolongation section from level %d to level %d\n", 0, 1);
  ierr = MeshLocateInMesh(finemesh, outmeshes[0]);
  for (int curLevel = 1; curLevel < nLevels-1; curLevel++) {
    if (info) PetscPrintf(m->comm(), "Building the prolongation section from level %d to level %d\n", curLevel, curLevel+1);
    ierr = MeshLocateInMesh(outmeshes[curLevel-1], outmeshes[curLevel]);CHKERRQ(ierr);
  }
*/ 
 PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "MeshCreateHierarchyLabel_NEW"

//MeshCreateHierarchyLabel: Create a label that tells what the highest level a given vertex appears in where 0 is fine and n is coarsest.
PetscErrorCode MeshCreateHierarchyLabel_NEW(Mesh finemesh, double beta, int nLevels, Mesh * outmeshes) {
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  //initialization
  ALE::Obj<PETSC_MESH_TYPE> m;
  double maxspace = -1., minspace = -1., dist, current_beta;
  double v_space, c_space, total_space, max_total_space;
  double v_coords[3], c_coords[3];

  int mesh_vertices_size = 0;

  int v_point, v_bound, v_hdepth, v_dominating_point;
  int c_point, c_bound, c_hdepth, c_dominating_point;
  int n_point, n_bound, n_traversal;
  
  std::list<PETSC_MESH_TYPE::point_type> comparison_list;

  std::list<PETSC_MESH_TYPE::point_type> traversed_list;

  std::list<PETSC_MESH_TYPE::point_type> dominated_list;
  std::list<PETSC_MESH_TYPE::point_type>::iterator dominated_list_iter = dominated_list.begin();
  std::list<PETSC_MESH_TYPE::point_type>::iterator dominated_list_iter_end = dominated_list.end();

  
  bool canAdd, skip;
  
  ierr = MeshGetMesh(finemesh, m);CHKERRQ(ierr);

  int dim = m->getDimension();

  if (!m->hasLabel("marker"))ierr = MeshIDBoundary(finemesh);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& boundary = m->getLabel("marker");

  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& spacing = m->getRealSection("spacing");
  const ALE::Obj<PETSC_MESH_TYPE::label_type> hdepth = m->createLabel("hierarchy_depth");
  const ALE::Obj<PETSC_MESH_TYPE::label_type> dompoint = m->createLabel("dominating_point");
  const ALE::Obj<PETSC_MESH_TYPE::label_type> traversal = m->createLabel("traversal");

  const ALE::Obj<PETSC_MESH_TYPE::sieve_type> sieve = m->getSieve();
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& vertices = m->depthStratum(0);
  PETSC_MESH_TYPE::label_sequence::iterator v_iter = vertices->begin();
  PETSC_MESH_TYPE::label_sequence::iterator v_iter_end = vertices->end();
  
  //Setup, find the minimum and maximum spacing values and force in high-curvature nodes.
  while(v_iter != v_iter_end) {

    v_point = *v_iter;
    v_bound = m->getValue(boundary, v_point);
    m->setValue(dompoint, v_point, -1);
    m->setValue(hdepth, v_point, 0);
    v_space = *spacing->restrictPoint(v_point);

    if ((v_space > maxspace) || (maxspace == -1.)) maxspace = v_space;
    if ((v_space < minspace) || (minspace == -1.)) minspace = v_space;

    if (m->depth(v_point) == 0 && v_bound == 1) if (Curvature(m, v_point) > 0.01) {
      m->setValue(hdepth, v_point, nLevels-1);
    }

    v_iter++;
  }

  //main loop over levels
  for (int current_level = nLevels - 1; current_level > 0; current_level--) {

    current_beta = 0.5*pow(beta, current_level);
    v_iter = vertices->begin();
    v_iter_end = vertices->end();
    while (v_iter != v_iter_end) {

      v_point = *v_iter;
      v_hdepth = m->getValue(hdepth, v_point);
      v_dominating_point = m->getValue(dompoint, v_point);

      //determine if this point has already been eliminated this round
      skip = false;
      if (v_dominating_point != -1) {
        if (m->getValue(hdepth, v_dominating_point) == current_level) skip = true;
      }

      if (v_hdepth == 0 && !skip) {
        v_bound = m->getValue(boundary, v_point);
        //PetscPrintf(m->comm(), "%d\n", v_point);
        //get the coordinates and spacing function values for this point.
        v_space = *spacing->restrictPoint(v_point);
        ierr = PetscMemcpy(v_coords, coordinates->restrictPoint(v_point), dim*sizeof(double));
        max_total_space = current_beta*(v_space*3);  //keep EVERYTHING local, max-space blows up on highly-graded meshes

        //seed the comparison list and set the current point as traversed
        m->setValue(traversal, v_point, 1);
        traversed_list.push_front(v_point);
        if (v_dominating_point != -1) {
          comparison_list.push_front(v_dominating_point);
          traversed_list.push_front(v_dominating_point);
          m->setValue(traversal, v_dominating_point, 1);
        }
        ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneSet> neighbors = sieve->cone(sieve->support(v_point));
        PETSC_MESH_TYPE::sieve_type::coneSet::iterator n_iter = neighbors->begin();
        PETSC_MESH_TYPE::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
        

        while (n_iter != n_iter_end) {
          n_point = *n_iter;
          if ((v_point != n_point)&&((m->getValue(boundary, n_point) == 1 && v_bound == 1) || v_bound == 0)) {
            m->setValue(traversal, n_point, 1);
            traversed_list.push_front(n_point);
            comparison_list.push_back(n_point);
          }
          n_iter++;
        }
        
        //traverse the mesh out to the point where nothing can collide with this point
        canAdd = true;
        while (!comparison_list.empty() && canAdd) {
          
          c_point = comparison_list.front();
          comparison_list.pop_front();
          //PetscPrintf(m->comm(), "%d\n", c_point);
          c_bound = m->getValue(boundary, c_point);
          c_hdepth = m->getValue(hdepth, c_point);
          c_dominating_point = m->getValue(dompoint, c_point);
          ierr = PetscMemcpy(c_coords, coordinates->restrictPoint(c_point), dim*sizeof(double));CHKERRQ(ierr);
          
          //compute the distance
          dist = 0.;
          for (int i = 0; i < dim; i++) {
            dist += (c_coords[i] - v_coords[i])*(c_coords[i] - v_coords[i]);
          }
          dist = sqrt(dist);
          
          if (dist < max_total_space) { //if it could possibly collide in the function-ball sense:
            c_space = *spacing->restrictPoint(c_point);
            // if it collides in the function-ball sense:
            total_space = current_beta*(v_space + c_space);
            if (dist < total_space) {
              if (v_bound == 1) { //boundary case
                if (c_hdepth != 0) {
                  canAdd = false;
                  m->setValue(dompoint, v_point, c_point);
                } else {
                  dominated_list.push_front(c_point);
                }
              } else { //interior case
                if (c_hdepth != 0 || c_bound == 1) {
                   canAdd = false; 
                   m->setValue(dompoint, v_point, c_point);
                } else {  //add the current inspected point to the list of points potentially dominated
                  dominated_list.push_front(c_point);
                  //if the dominating point is defined, test against it next.
                  if (c_dominating_point != -1) {
                    if (m->getValue(traversal, c_dominating_point) != 1) {
                      comparison_list.push_front(c_dominating_point);
                      traversed_list.push_front(c_dominating_point);
                      m->setValue(traversal, c_dominating_point, 1);
                    }
                  }
                }
              }
            } 
            //add neighbors to the comparison queue
            neighbors = sieve->cone(sieve->support(c_point));
            n_iter = neighbors->begin();
            n_iter_end = neighbors->end();
            while (n_iter != n_iter_end) {
              
              n_point = *n_iter;
              //PetscPrintf(m->comm(), "-%d\n", n_point);
              n_traversal = m->getValue(traversal, n_point);

              if (n_traversal != 1) {

                n_bound = m->getValue(boundary, n_point);

                if (c_bound == 1) { //boundary c_point case: only add boundary points
                  if (n_bound == 1) {
                    comparison_list.push_back(n_point);
                    m->setValue(traversal, n_point, 1);
                    traversed_list.push_front(n_point);
                  }
                } else { //nonboundary c_point case: add anything
                    comparison_list.push_back(n_point);
                    m->setValue(traversal, n_point, 1);
                    traversed_list.push_front(n_point);
                }
              }

              n_iter++;
            }

          }  // collision endif

        }  // comparison traversal loop end

        comparison_list.clear();

        //clean up the traversal label
        while (!traversed_list.empty()) {
          n_point = traversed_list.front();
          traversed_list.pop_front();
          m->setValue(traversal, n_point, 0);
        }

        if (canAdd == true) { 
          //we can add the point, so put it at this level and set the points it dominates
          m->setValue (hdepth, v_point, current_level);
          dominated_list_iter = dominated_list.begin();
          dominated_list_iter_end = dominated_list.end();
          while (dominated_list_iter != dominated_list_iter_end) {
            m->setValue(dompoint, *dominated_list_iter, v_point); 
            dominated_list_iter++;
          }
        }
        dominated_list.clear();
      }  // eligible for comparison endif
      v_iter++;
    } // vertices loop end
    mesh_vertices_size += m->getLabelStratum("hierarchy_depth", current_level)->size();
    PetscPrintf(m->comm(), "Size of mesh: %d\n", mesh_vertices_size);
  } // levels loop end

  PetscFunctionReturn(0);
}

//MeshLocateInMesh: Create a label between the meshes.

PetscErrorCode MeshLocateInMesh(Mesh finemesh, Mesh coarsemesh) {
  ALE::Obj<PETSC_MESH_TYPE> fm, cm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  //int maxSearches = 30;
  ierr = MeshGetMesh(finemesh, fm);CHKERRQ(ierr);
  ierr = MeshGetMesh(coarsemesh, cm);CHKERRQ(ierr);

  //set up the prolongation section if it doesn't already exist
  //bool prolongexists = fm->hasLabel("prolongation");
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& prolongation = fm->createLabel("prolongation");

  //we have a prolongation label that does not correspond to our current mesh.  Reset it to -1s.
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& finevertices = fm->depthStratum(0);
  PETSC_MESH_TYPE::label_sequence::iterator fv_iter = finevertices->begin();
  PETSC_MESH_TYPE::label_sequence::iterator fv_iter_end = finevertices->end();

  while (fv_iter != fv_iter_end) {
    fm->setValue(prolongation, *fv_iter, -1);
    fv_iter++;
  }
  //traversal labels on both layers
  ALE::Obj<PETSC_MESH_TYPE::label_type> coarsetraversal = cm->createLabel("traversal");
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&  finecoordinates = fm->getRealSection("coordinates");
  //const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&  coarsecoordinates = cm->getRealSection("coordinates");
  int dim = fm->getDimension();
  //PetscPrintf(cm->comm(), "Dimensions: %d and %d\n", dim, cm->getDimension());
  if (dim != cm->getDimension()) throw ALE::Exception("Dimensions of the fine and coarse meshes do not match"); 
  //do the tandem traversal thing.  it is possible that if the section already existed then the locations of some of the points are known if they exist in both the meshes.
  fv_iter = finevertices->begin();
  fv_iter_end = finevertices->end();
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& coarseelements = cm->heightStratum(0);
  PETSC_MESH_TYPE::label_sequence::iterator ce_iter = coarseelements->begin();
  PETSC_MESH_TYPE::label_sequence::iterator ce_iter_end = coarseelements->end();
  double *fvCoords = new double[dim], *nvCoords = new double[dim];
  std::list<PETSC_MESH_TYPE::point_type> travlist;  //store point
//  std::list<PETSC_MESH_TYPE::point_type> travElist; //store element location "guesses"
  std::list<PETSC_MESH_TYPE::point_type> eguesslist; // store the next guesses for the location of the current point.
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
          //PetscPrintf(fm->comm(), "INITIAL: Point %d located in %d.\n",  *fv_iter, *ce_iter);
          //OK WE HAVE A STARTING POINT.  Go through its neighbors looking at the unfound ones and finding them homes.
          ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneSet> neighbors = fm->getSieve()->cone(fm->getSieve()->support(*fv_iter));
          PETSC_MESH_TYPE::sieve_type::coneSet::iterator n_iter = neighbors->begin();
          PETSC_MESH_TYPE::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
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
            PETSC_MESH_TYPE::point_type curVert = *travlist.begin();
            PetscMemcpy(nvCoords, finecoordinates->restrictPoint(curVert), dim*sizeof(double));
            PETSC_MESH_TYPE::point_type curEle =  fm->getValue(prolongation, curVert);
            travlist.pop_front();
            //travElist.pop_front();
            eguesslist.push_front(curEle);
            cm->setValue(coarsetraversal, curEle, 1);
            bool locationDiscovered = false;
            while ((!eguesslist.empty()) && (!locationDiscovered)) {
              PETSC_MESH_TYPE::point_type curguess = *eguesslist.begin();
              eguesslist.pop_front();
              if (PointIsInElement(cm, curguess, nvCoords)) {
                locationDiscovered = true;
                //set the label.
                fm->setValue(prolongation, curVert, curguess);
                //PetscPrintf(fm->comm(), "Point %d located in %d.\n",  curVert, curguess);
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
                ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> curguessneighbors = cm->getSieve()->support(cm->getSieve()->cone(curguess));
                PETSC_MESH_TYPE::sieve_type::supportSet::iterator cgn_iter = curguessneighbors->begin();
                PETSC_MESH_TYPE::sieve_type::supportSet::iterator cgn_iter_end = curguessneighbors->end();
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
              if(fm->debug())PetscPrintf(fm->comm(), "Point %d (%f, %f) not located.\n",  curVert, nvCoords[0], nvCoords[1]);
            }
            eguesslist.clear(); //we've discovered the location of the point or exhausted our possibilities on this contiguous block of elements.
            //unset the traversed element list
            ALE::Obj<PETSC_MESH_TYPE::label_sequence> traved_elements = cm->getLabelStratum("traversal", 1);
            //PetscPrintf(cm->comm(), "%d\n", traved_elements->size());
            PETSC_MESH_TYPE::label_sequence::iterator tp_iter = traved_elements->begin();
            PETSC_MESH_TYPE::label_sequence::iterator tp_iter_end = traved_elements->end();
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
  delete [] fvCoords; delete [] nvCoords;
  PetscFunctionReturn(0);
}

bool PointIsInElement(ALE::Obj<PETSC_MESH_TYPE> mesh, PETSC_MESH_TYPE::point_type e, double * point) {
      int dim = mesh->getDimension();
      static double v0[3], J[9], invJ[9], detJ;
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

        if ((xi >= -0.00000000001) && (eta >= -0.00000000001) && (zeta >= -0.00000000001) && (xi + eta + zeta <= 2.00000000001)) { return true;
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




