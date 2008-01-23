/*
=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

Hierarchy_New.hh:

Routines for coarsening arbitrary simplicial meshes expressed as sieves using
a localized modification of the algorithm in Miller (1999)

- Peter Brune

=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
*/

#include <list>
#include <stdlib.h>

#include <Mesh.hh>
#include "SieveAlgorithms.hh"
#include "MeshSurgery.hh"
#include <SieveAlgorithms.hh>
#include <Selection.hh>


#undef __FUNCT__
#define __FUNCT__ "Hierarchy_createCoarsenVertexSet"

/*
=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
Hierarchy_createCoarsenVertexSet:

Inputs:

original_mesh: the mesh in question
spacing: the spacing function over the vertices of the original_mesh
interior_set: the points that could potentially be pruned
boundary_set: the points that are forced into the mesh
beta: the expansion of the spacing function
maxIndex: the maximum index in original_mesh, used for avoiding tag collisions

Returns:

The the set of vertices that are left standing after the coarsening procedure

Remarks:

Doesn't modify the original_mesh topology, works only with interactions between
the boundary set and the interior set and within the interior set.  Use for all
coarsening and then feed the output into a remesher.  Can be used for each 
layer of boundary coarsening for 2D/3D/WhateverD meshes.

Builds a rather expensive connection_sieve structure in the process.  This
has no labels associated with it, so it's less expensive than the mesh is.
Subsets of the mesh could be coarsened at a time with an internal skeleton 
boundary if this is too expensive.


=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
 */

ALE::Obj<ALE::Mesh::sieve_type::supportSet> Hierarchy_CoarsenVertexSet(ALE::Obj<ALE::Mesh> original_mesh, 
                                                                       ALE::Obj<ALE::Mesh::real_section_type> spacing, 
                                                                       ALE::Obj<ALE::Mesh::sieve_type::supportSet> interior_set, 
                                                                       ALE::Obj<ALE::Mesh::sieve_type::supportSet> boundary_set, 
                                                                       double beta, 
                                                                       ALE::Mesh::point_type maxIndex,
                                                                       int * comparisons = PETSC_NULL) {
  
  PetscErrorCode ierr;

  ALE::Obj<ALE::Mesh::sieve_type> original_sieve = original_mesh->getSieve();
  ALE::Obj<ALE::Mesh::real_section_type> coordinates = original_mesh->getRealSection("coordinates");  

  ALE::Mesh::point_type cur_new_index = maxIndex+1;

  //build the coarsening structure with the following links: links within the interior_set and links between the boundary_set and interior_set

  ALE::Obj<ALE::Mesh::sieve_type> connection_sieve = new ALE::Mesh::sieve_type(original_sieve->comm(), original_sieve->debug());
  ALE::Obj<ALE::Mesh> connection_mesh = new ALE::Mesh(original_mesh->comm(), original_mesh->debug());
  connection_mesh->setSieve(connection_sieve);


  ALE::Mesh::sieve_type::supportSet::iterator is_iter = interior_set->begin();
  ALE::Mesh::sieve_type::supportSet::iterator is_iter_end = interior_set->end();
  ALE::Mesh::sieve_type::supportSet::iterator bs_iter_end = boundary_set->end();

  ALE::Obj<ALE::Mesh::sieve_type::supportSet> output_vertices = new ALE::Mesh::sieve_type::supportSet();

  std::list<ALE::Mesh::point_type> comparison_queue; //edges go here

  int dim = original_mesh->getDimension();

  double *a_coords = new double[dim], *b_coords = new double[dim];

  ALE::Mesh::sieve_type::supportSet candidate_vertices;

  ALE::Obj<ALE::Mesh::sieve_type::supportSet> link;
  ALE::Obj<ALE::Mesh::sieve_type::supportSet> line = ALE::Mesh::sieve_type::supportSet();

  while (is_iter != is_iter_end) {
    //create the set of candidate points
    if (boundary_set->find(*is_iter) == bs_iter_end) {
      candidate_vertices.insert(*is_iter);
    } else {
      output_vertices->insert(*is_iter);
    }

    ALE::Obj<ALE::Mesh::sieve_type::coneSet> neighbors = original_sieve->cone(original_sieve->support(*is_iter));
    ALE::Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
    ALE::Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
    while (n_iter != n_iter_end) {
      if (*n_iter != *is_iter)
      if (boundary_set->find(*n_iter) != bs_iter_end) {
          connection_sieve->addArrow(*n_iter, cur_new_index);
          connection_sieve->addArrow(*is_iter, cur_new_index);
          comparison_queue.push_back(cur_new_index);
          cur_new_index++;
      } else if (interior_set->find(*n_iter) != interior_set->end()) {
        if (candidate_vertices.find(*n_iter) == candidate_vertices.end()) {
          connection_sieve->addArrow(*n_iter, cur_new_index);
          connection_sieve->addArrow(*is_iter, cur_new_index);
          cur_new_index++;
        }
      }
      n_iter++;
    }
    is_iter++;
  }
  // PetscPrintf(original_mesh->comm(), "current new index: %d\n", cur_new_index);
  //PetscPrintf(original_mesh->comm(), "trying to stratify\n");
  connection_mesh->stratify();
  output_vertices->clear();
  //PetscPrintf(original_mesh->comm(), "onto decimation!\n");
  //decimate the new structure
  while (!candidate_vertices.empty()) {
    //PetscPrintf(original_mesh->comm(), "starting with this round.\n");
    while (!comparison_queue.empty()) {
      ALE::Mesh::point_type current_edge = comparison_queue.front();
      comparison_queue.pop_front();
      //logic for comparisons
      ALE::Obj<ALE::Mesh::sieve_type::coneSequence> endpoints = connection_sieve->cone(current_edge);
      //PetscPrintf(original_mesh->comm(), "got the endpoints: size %d\n", endpoints->size());
      //      if (endpoints->size() != 2) throw ALE::Exception("not the connection sieve?");
      ALE::Mesh::point_type a, b;
      bool a_included = false, b_included = false;
      if (endpoints->size() == 2) {
        ALE::Mesh::sieve_type::coneSequence::iterator ep_iter = endpoints->begin();
        a = *ep_iter;
        ep_iter++;
        b = *ep_iter;
        if (output_vertices->find(a) != output_vertices->end()) {
          a_included = true;
          //PetscPrintf(original_mesh->comm(), "%d is in output\n", a);
        } else if (boundary_set->find(a) != boundary_set->end()) {
          a_included = true;
          //PetscPrintf(original_mesh->comm(), "%d is in boundary\n ", a);
        }
        if (output_vertices->find(b) != output_vertices->end()) {
          //PetscPrintf(original_mesh->comm(), "%d is in output\n",b);
          b_included = true;
        } else if (boundary_set->find(b) != boundary_set->end()) {
          //PetscPrintf(original_mesh->comm(), "%d is in boundary\n",b);
          b_included = true;
        }
      }
      if ((a_included && b_included) || ((!b_included) && (!a_included))) {
        //PetscPrintf(original_mesh->comm(), "edge is irrelevant\n");
        //either both are included or both are not included so this comparison doesn't need to be done
      } else {
      //a should be the point that is included in the mesh already; if it is not; swap it
        if (b_included) {
	  ALE::Mesh::point_type swap_vertex = a;
          a = b;
          b = swap_vertex;
        }
        //do the comparison
        ierr = PetscMemcpy(a_coords, coordinates->restrictPoint(a), dim*sizeof(double));
	ierr = PetscMemcpy(b_coords, coordinates->restrictPoint(b), dim*sizeof(double));
        double space_a = *spacing->restrictPoint(a);
        double space_b = *spacing->restrictPoint(b);
        double space = beta*(space_a + space_b);
        double dist = 0;
        for (int i = 0; i < dim; i++) {
          dist += (a_coords[i] - b_coords[i])*(a_coords[i] - b_coords[i]);
        }
        dist = sqrt(dist);
        if (dist < space) { //collision case; remove b from the candidate points and the sieve.
          //PetscPrintf(original_mesh->comm(), "removal criterion satisfied\n");
	  ALE::Mesh::sieve_type::supportSet::iterator removal_iter = candidate_vertices.find(b); 
	  if (removal_iter != candidate_vertices.end()) {
            //remove b from the queue, reconnect link vertices to a
	    ALE::Obj<ALE::Mesh::sieve_type::supportSequence> b_edges = connection_sieve->support(b);
	    ALE::Mesh::sieve_type::supportSequence::iterator be_iter = b_edges->begin();
	    ALE::Mesh::sieve_type::supportSequence::iterator be_iter_end = b_edges->end();
	    ALE::Mesh::sieve_type::supportSet remove_these_points;
            while (be_iter != be_iter_end) {
              connection_sieve->addArrow(a, *be_iter);
              if (connection_sieve->cone(*be_iter)->size() != 3) { //SHOULD contain a, b, and the associated link vertex; can be a and b, or just 
                remove_these_points.insert(*be_iter);
                //connection_sieve->removeBasePoint(*be_iter);
                //connection_sieve->removeCapPoint(*be_iter);
               
              }
              be_iter++;
            }
	    for (ALE::Mesh::sieve_type::supportSet::iterator rtp_iter = remove_these_points.begin(); rtp_iter != remove_these_points.end(); rtp_iter++) {
	      connection_sieve->removeBasePoint(*rtp_iter);
              connection_sieve->removeCapPoint(*rtp_iter);
	    }
            connection_sieve->removeBasePoint(b);
            connection_sieve->removeCapPoint(b);
            //remove it from the candidate points
            candidate_vertices.erase(removal_iter);
            //PetscPrintf(original_mesh->comm(), "removed a vertex\n");
          } //end of removal_iter if
        } //end of spacing if
      } //end of else denoting that this comparison matters
    } //end of !comparison_queue.empty() while
    //the comparison queue is empty; the present output_vertices is compatible
    //take one point from the candidate_vertices and move it to the output_vertices, adding its edges to the comparison queue
    if (!candidate_vertices.empty()) { //we could have removed the final vertex in the previous coarsening.  check
      ALE::Mesh::sieve_type::supportSet::iterator new_added_vertex_iter = candidate_vertices.begin();
      ALE::Mesh::point_type new_added_vertex = *new_added_vertex_iter;
      candidate_vertices.erase(new_added_vertex_iter);
      output_vertices->insert(new_added_vertex); //add this new vertex to the set of vertices in the output.
      ALE::Obj<ALE::Mesh::sieve_type::supportSequence> nav_support = connection_sieve->support(new_added_vertex);
      ALE::Mesh::sieve_type::supportSequence::iterator nav_iter = nav_support->begin();
      ALE::Mesh::sieve_type::supportSequence::iterator nav_iter_end = nav_support->end();
      while (nav_iter != nav_iter_end) {
        comparison_queue.push_back(*nav_iter);
        nav_iter++;
      }
    }  //end of candidate_vertices.empty() if for adding a new vertex
  }  //end of !candidate_vertices.empty() while
  //PetscPrintf(original_mesh->comm(), "%d vertices included in coarse set\n", output_vertices->size());
  delete [] a_coords; delete [] b_coords;
  return output_vertices;
}
/*
=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
Hierarchy_createEffective2DBoundary:

Use:

Returns a noninterpolated 2D boundary mesh, including faults, for use in coarsening

Inputs:

original_mesh: an effectively 3D simplicial mesh
maxIndex: the beginning of the free indices for use in the boundary mesh
forced_bound_mesh: an optional internal fault mesh (uninterpolated)

Output:

An uninterpolated boundary mesh containing all exterior faces in the 3D mesh, as well
as the additional internal fault mesh given by forced_bound_mesh

Assumptions:

forced_bound_mesh is uninterpolated and distinct from the exterior

=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
*/
#undef __FUNCT__
#define __FUNCT__ "Hierarchy_createEffective2DBoundary"

ALE::Obj<ALE::Mesh> Hierarchy_createEffective2DBoundary (ALE::Obj<ALE::Mesh> original_mesh, ALE::Mesh::point_type maxIndex, ALE::Obj<ALE::Mesh> forced_bound_mesh = PETSC_NULL) {
  //create a set of "essential" faces
  //from a given 3D mesh
  int depth = original_mesh->depth();
  int dim = original_mesh->getDimension();
  int cur_available_index = maxIndex+1;
  ALE::Obj<ALE::Mesh::sieve_type::supportSet> line = new ALE::Mesh::sieve_type::supportSet();
  ALE::Obj<ALE::Mesh::sieve_type::supportSet> link;
  ALE::Obj<ALE::Mesh::sieve_type> original_sieve = original_mesh->getSieve();

  //setup the output

  //build the exterior boundary
  ALE::Obj<ALE::Mesh> output_mesh = new ALE::Mesh(original_mesh->comm(), dim, original_mesh->debug());
  ALE::Obj<ALE::Mesh::sieve_type> output_sieve = new ALE::Mesh::sieve_type(original_mesh->comm(), original_mesh->debug());
  output_mesh->setSieve(output_sieve);

  if (depth == 1) { //noninterpolated case; for each cell look at the associated vertex subsets and assume simplicial
    //PetscPrintf(original_mesh->comm(), "Uninterpolated case\n");
    ALE::Obj<ALE::Mesh::label_sequence> cells = original_mesh->heightStratum(0);
    ALE::Mesh::label_sequence::iterator c_iter = cells->begin();
    ALE::Mesh::label_sequence::iterator c_iter_end = cells->end();
    while (c_iter != c_iter_end) {
      ALE::Obj<ALE::Mesh::sieve_type::coneSequence> cell_corners = original_sieve->cone(*c_iter);
      ALE::Mesh::sieve_type::coneSequence::iterator cc_iter = cell_corners->begin();
      ALE::Mesh::sieve_type::coneSequence::iterator cc_iter_end = cell_corners->end();
      line->clear();
      while (cc_iter != cc_iter_end) {
        line->insert(*cc_iter);
        cc_iter++;
      }
      //PetscPrintf(original_mesh->comm(), "%d vertices in the cone\n", cell_corners->size());
      //assumed simplicial; each n-1 subset is a valid face
      cc_iter = cell_corners->begin();
      while (cc_iter != cc_iter_end) {
        line->erase(line->find(*cc_iter));
        link = original_sieve->nJoin1(line);
	// PetscPrintf(original_mesh->comm(), "%d vertices in the face\n", line->size());
        if (line->size() != 3) throw ALE::Exception("bad line");
        if (link->size() == 1) {
          for (ALE::Mesh::sieve_type::supportSet::iterator f_iter = line->begin(); f_iter != line->end(); f_iter++) {
            output_sieve->addArrow(*f_iter, cur_available_index);
          }
          cur_available_index++;
        }
        line->insert(*cc_iter);
        cc_iter++;
      }
      c_iter++;
    }
  } else { //interpolated case; we have the faces, just add them
    //PetscPrintf(original_mesh->comm(), "Interpolated case\n");
    ALE::Obj<ALE::Mesh::label_sequence> outward_faces = original_mesh->heightStratum(1); //assume wlog the faces are heightstratum 1
    ALE::Mesh::label_sequence::iterator f_iter = outward_faces->begin();
    ALE::Mesh::label_sequence::iterator f_iter_end = outward_faces->end();
    while(f_iter != f_iter_end) {
      //PetscPrintf(original_mesh->comm(), "testing...\n");
      if (original_sieve->support(*f_iter)->size() != 2) {//add the uninterpolated face to the boundary.
        //if (cur_available_index <= *f_iter) cur_available-index = *f_iter+1;
	ALE::Obj<ALE::Mesh::sieve_type::coneArray> corners = original_sieve->nCone(*f_iter, depth-1);
	ALE::Mesh::sieve_type::coneArray::iterator c_iter = corners->begin();
	ALE::Mesh::sieve_type::coneArray::iterator c_iter_end = corners->end();
        while (c_iter != c_iter_end) {
          output_sieve->addArrow(*c_iter, cur_available_index);
          c_iter++;
        }
        cur_available_index++;
      }
      f_iter++;
    }
  }
  //PetscPrintf(original_mesh->comm(), "Done with finding the boundary\n");
  //force in the forced_bound_mesh, which will include any included faults

  if (forced_bound_mesh) {
    if (forced_bound_mesh->depth() != 1) throw ALE::Exception("Needs noninterpolated boundary meshes");
    ALE::Obj<ALE::Mesh::label_sequence> forced_boundary_cells = forced_bound_mesh->heightStratum(0);
    ALE::Mesh::label_sequence::iterator fbc_iter = forced_boundary_cells->begin();
    ALE::Mesh::label_sequence::iterator fbc_iter_end = forced_boundary_cells->end();
    while (fbc_iter != fbc_iter_end) {
      // if (*fbc_iter > cur_available_index) cur_available_index = *fbc_iter+1;
      ALE::Obj<ALE::Mesh::sieve_type::coneSequence> corners = forced_bound_mesh->getSieve()->cone(*fbc_iter);
      ALE::Mesh::sieve_type::coneSequence::iterator c_iter = corners->begin();
      ALE::Mesh::sieve_type::coneSequence::iterator c_iter_end = corners->end();
      while (c_iter != c_iter_end) {
        output_sieve->addArrow(*c_iter, cur_available_index);
        c_iter++;
      }
      cur_available_index++;
      fbc_iter++;
    }
  }
  output_mesh->stratify();
  output_mesh->setRealSection("coordinates", original_mesh->getRealSection("coordinates"));
  //PetscPrintf(output_mesh->comm(), "done with 2D boundary find.\n");
  return output_mesh;
}

/*
=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
Hierarchy_createEffective1DBoundary:

Use:

Returns a noninterpolated 1D boundary mesh, including faults, for use in coarsening

Inputs:

original_mesh: an effectively 2D simplicial mesh
maxIndex: the beginning of the free indices for use in the boundary mesh
forced_bound_mesh: an optional internal fault mesh (uninterpolated)

Output:

An uninterpolated boundary mesh containing all exterior edges in the 2D mesh, as well
as the additional internal fault mesh given by forced_bound_mesh

Assumptions:

forced_bound_mesh is uninterpolated and distinct from the exterior

=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
*/
#undef __FUNCT__
#define __FUNCT__ "Hierarchy_createEffective1DBoundary"

ALE::Obj<ALE::Mesh> Hierarchy_createEffective1DBoundary (ALE::Obj<ALE::Mesh> original_mesh, ALE::Mesh::point_type maxIndex, ALE::Obj<ALE::Mesh> forced_bound_mesh = PETSC_NULL) {
  //create a set of "essential" edges
  //from a given 2D mesh or 3D boundary mesh
  int dim = original_mesh->getDimension();
  ALE::Obj<ALE::Mesh::sieve_type> original_sieve = original_mesh->getSieve();
  ALE::Obj<ALE::Mesh::real_section_type> coordinates = original_mesh->getRealSection("coordinates");
  double *a_coords = new double[dim], *b_coords = new double[dim], *c_coords = new double[dim], *d_coords = new double[dim];
  int depth = original_mesh->depth();
  ALE::Mesh::point_type cur_available_index = maxIndex+1;
  ALE::Obj<ALE::Mesh> output_mesh = new ALE::Mesh(original_mesh->comm(), dim, original_mesh->debug());
  ALE::Obj<ALE::Mesh::sieve_type> output_sieve = new ALE::Mesh::sieve_type(original_mesh->comm(), original_mesh->debug());
  output_mesh->setSieve(output_sieve);
  ALE::Obj<ALE::Mesh::sieve_type::supportSet> line = new ALE::Mesh::sieve_type::supportSet();
  ALE::Obj<ALE::Mesh::sieve_type::supportSet> link;

  //force in the forced boundary mesh (2D)
  if (forced_bound_mesh) {
    //PetscPrintf(original_mesh->comm(), "forcing in the boundary mesh\n");
    if (forced_bound_mesh->depth() != 1) throw ALE::Exception("Needs noninterpolated boundary mesh"); //haha doesn't matter, but still
    ALE::Obj<ALE::Mesh::label_sequence> bound_edges = forced_bound_mesh->heightStratum(0);
    ALE::Mesh::label_sequence::iterator be_iter = bound_edges->begin(); 
    ALE::Mesh::label_sequence::iterator be_iter_end = bound_edges->end(); 
    while (be_iter != be_iter_end) {
      ALE::Obj<ALE::Mesh::sieve_type::coneSequence> edge_ends = forced_bound_mesh->getSieve()->cone(*be_iter);
      ALE::Mesh::sieve_type::coneSequence::iterator ee_iter = edge_ends->begin();
      ALE::Mesh::sieve_type::coneSequence::iterator ee_iter_end = edge_ends->end();
      while (ee_iter != ee_iter_end) {
        output_sieve->addArrow(*ee_iter, cur_available_index);
        ee_iter++;
      }
      cur_available_index++;
      be_iter++;
    }
  }
  //PetscPrintf(original_mesh->comm(), "Forcing in edges\n");
  //find topologically needed edges -- interpolated means support != 2; noninterpolated means njoin1 != 2
  //find geometrically attractive edges (3D); doublet-normals above a certain threshhold.
  if (depth == 1) { //uninterpolated case
    //PetscPrintf(original_mesh->comm(), "Uninterpolated\n");
    ALE::Mesh::sieve_type::supportSet seen;
    seen.clear();
    ALE::Obj<ALE::Mesh::label_sequence> vertices = original_mesh->depthStratum(0);
    ALE::Mesh::label_sequence::iterator v_iter = vertices->begin();
    ALE::Mesh::label_sequence::iterator v_iter_end = vertices->end();
    while (v_iter != v_iter_end) {
      seen.insert(*v_iter);
    ALE::Obj<ALE::Mesh::sieve_type::coneSet> neighbors = original_sieve->cone(original_sieve->support(*v_iter));
    ALE::Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
    ALE::Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
    while (n_iter != n_iter_end) { if (*n_iter != *v_iter) {
        if (seen.find(*n_iter) == seen.end()) { //if we've seen this vertex and therefore done this edge before..
          line->clear();
          line->insert(*n_iter);
          line->insert(*v_iter);
          link = original_sieve->nJoin1(line);
          if (link->size() != 2) {
            //either a boundary in 2D or a fault/mesh intersection in 3D
            output_sieve->addArrow(*v_iter, cur_available_index);
	    output_sieve->addArrow(*n_iter, cur_available_index);
            cur_available_index++;
          }else if (dim == 3) {  //do the extra tests for doublet-wide principal curvature
            //build the doublet
	    ALE::Mesh::point_type a, b, c, d;
	    a = *n_iter;
            b = *v_iter;
	    ALE::Obj<ALE::Mesh::sieve_type::coneSet> doublet_corners = original_sieve->cone(link);
	    ALE::Mesh::sieve_type::coneSet::iterator dc_iter = doublet_corners->begin();
	    ALE::Mesh::sieve_type::coneSet::iterator dc_iter_end = doublet_corners->end();
            int corner_number = 0;
            while (dc_iter != dc_iter_end) {
              if (*dc_iter != a && *dc_iter != b) {
                if (corner_number == 0) {
                  c = *dc_iter;
                  corner_number = 1;
                } else {
                  d = *dc_iter;
                }
              }
              dc_iter++;
            }
            PetscMemcpy(a_coords, coordinates->restrictPoint(a), dim*sizeof(double));
	    PetscMemcpy(b_coords, coordinates->restrictPoint(b), dim*sizeof(double));
            PetscMemcpy(c_coords, coordinates->restrictPoint(c), dim*sizeof(double));
            PetscMemcpy(d_coords, coordinates->restrictPoint(d), dim*sizeof(double));
            double d_angle = ALE::doublet_angle(dim, a_coords, b_coords, c_coords, d_coords);
            //PetscPrintf(original_mesh->comm(), "doublet angle: %f\n", d_angle);
            if (d_angle > 0.79) { //arbitrary cutoff of around 45 degrees 
              output_sieve->addArrow(*v_iter, cur_available_index);
              output_sieve->addArrow(*n_iter, cur_available_index);
              cur_available_index++;
            }
          }
        }
        }
        n_iter++;
      }
      v_iter++;
    }
  } else { //interpolated case
    //PetscPrintf(original_mesh->comm(), "Interpolated\n");
    ALE::Obj<ALE::Mesh::label_sequence> edges = original_mesh->depthStratum(1);
    ALE::Mesh::label_sequence::iterator e_iter = edges->begin();
    ALE::Mesh::label_sequence::iterator e_iter_end = edges->end();
    while (e_iter != e_iter_end) {
      //PetscPrintf(original_mesh->comm(), "in the loop...\n");
      ALE::Obj<ALE::Mesh::sieve_type::supportSequence> edge_support = original_sieve->support(*e_iter);
      if (edge_support->size() != 2) {
	ALE::Obj<ALE::Mesh::sieve_type::coneSequence> edge_cone = original_sieve->cone(*e_iter);
	ALE::Mesh::sieve_type::coneSequence::iterator ec_iter = edge_cone->begin();
	ALE::Mesh::sieve_type::coneSequence::iterator ec_iter_end = edge_cone->end();
        while (ec_iter != ec_iter_end) {
          output_sieve->addArrow(*ec_iter, cur_available_index);
          ec_iter++;
        }
	cur_available_index++;
      } else if (dim == 3) { //boundary meshes will be UNINTERPOLATED, so this won't come up
      }
      e_iter++;
    }
  }
  output_mesh->stratify();
  output_mesh->setRealSection("coordinates", original_mesh->getRealSection("coordinates"));
  //PetscPrintf(output_mesh->comm(), "leaving 1D Boundary Building\n");
  delete [] a_coords; delete [] b_coords; delete [] c_coords; delete [] d_coords;
  return output_mesh;
}

/*
=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
Hierarchy_createEffective0DBoundary:

Use:

Returns a set of topologically necessary vertices from an effectively 1D mesh

Inputs:

original_mesh: an effectively 1D simplicial mesh

Output:

A set of vertices from the 1D simplicial mesh that are topologically necessary


=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
*/

#undef __FUNCT__
#define __FUNCT__ "Hierarchy_createEffective0DBoundary"

ALE::Obj<ALE::Mesh::sieve_type::supportSet> Hierarchy_createEffective0DBoundary(ALE::Obj<ALE::Mesh> original_mesh) {
  ALE::Obj<ALE::Mesh::real_section_type> coordinates = original_mesh->getRealSection("coordinates");
  //PetscPrintf(original_mesh->comm(), "In 0D Boundary Building\n");
  int dim = original_mesh->getDimension();
  double *a_coords = new double[dim], *b_coords = new double[dim], *c_coords = new double[dim];
  //create a set of "essential" vertices from the 1D boundary meshes given
  //find the topologically necessary vertices in the 1D mesh -- this means vertices on which an edge terminates or is noncontractible.
  ALE::Obj<ALE::Mesh::sieve_type::supportSet> output_set = new ALE::Mesh::sieve_type::supportSet();
  ALE::Obj<ALE::Mesh::sieve_type> original_sieve = original_mesh->getSieve();
  //go through the vertices of this set looking for ones with support size greater than 3.
  ALE::Obj<ALE::Mesh::label_sequence> vertices = original_mesh->depthStratum(0);
  ALE::Mesh::label_sequence::iterator v_iter = vertices->begin();
  ALE::Mesh::label_sequence::iterator v_iter_end = vertices->end();
  while (v_iter != v_iter_end) {
    if (original_sieve->support(*v_iter)->size() != 2) {
      output_set->insert(*v_iter);
    } else if (dim > 1) { //angles
      //if the angle between eges is greater than say... 45 degrees, include it.
      ALE::Obj<ALE::Mesh::sieve_type::coneSet> neighbors = original_sieve->cone(original_sieve->support(*v_iter));
      ALE::Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
      ALE::Mesh::point_type b, c;
      //should have two neighbors
      if (*n_iter != *v_iter) b = *n_iter; 
      n_iter++;
      if (*n_iter != *v_iter) { c = b; b = *n_iter;}
      n_iter++;
      if (*n_iter != *v_iter) { c = b; b = *n_iter;}
      PetscMemcpy(a_coords, coordinates->restrictPoint(*v_iter), dim*sizeof(double));
      PetscMemcpy(b_coords, coordinates->restrictPoint(b), dim*sizeof(double));
      PetscMemcpy(c_coords, coordinates->restrictPoint(c), dim*sizeof(double));
      double angle = ALE::corner_angle(dim, a_coords, b_coords, c_coords);
      if (fabs(angle - 3.14159) > 0.78) output_set->insert(*v_iter);
    }
    v_iter++;
  }
  //we can't do curvatures here; leave this to another thing we'll include in this set later
  //PetscPrintf(original_mesh->comm(), "leaving 0D Boundary Building\n");
  delete [] a_coords; delete [] b_coords; delete [] c_coords;
  return output_set;
}

/*
=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
Hierarchy_curvatureSet:

Use:

Returns a set of geometrically attractive vertices from an effectively 1D or 2D mesh

Inputs:

original_mesh: an effectively 2D simplicial mesh


Output:

A set of vertices from the 2D simplicial mesh that are geometrically attractive

Remarks: computes the gaussian curvatures at each vertex on the boundary of the mesh


=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
*/

#undef __FUNCT__
#define __FUNCT__ "Hierarchy_curvatureSet"

ALE::Obj<ALE::Mesh::sieve_type::supportSet> Hierarchy_curvatureSet(ALE::Obj<ALE::Mesh> original_mesh) {
  //given a 2D mesh;
  ALE::Obj<ALE::Mesh::sieve_type::supportSet> output_set; 
  return output_set;
}
/*=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
Hierarchy_defineSpacingFunction


Use: 

Returns the real_section_type associated with "spacing"; which it defines if not already defined


=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
*/


#undef __FUNCT__
#define __FUNCT__ "Hierarchy_defineSpacingFunction"



ALE::Obj<ALE::Mesh::real_section_type> Hierarchy_defineSpacingFunction(ALE::Obj<ALE::Mesh> m, PetscTruth recalculate = PETSC_FALSE) {
  if (m->hasRealSection("spacing") && !recalculate) {
    return m->getRealSection("spacing");
  }  
  ALE::Obj<ALE::Mesh::sieve_type> s = m->getSieve();
  ALE::Obj<ALE::Mesh::real_section_type> coordinates = m->getRealSection("coordinates");
  ALE::Obj<ALE::Mesh::label_sequence> vertices = m->depthStratum(0);
  ALE::Mesh::label_sequence::iterator v_iter = vertices->begin();
  ALE::Mesh::label_sequence::iterator v_iter_end = vertices->end();
  ALE::Obj<ALE::Mesh::real_section_type> spacing;
  int dim = m->getDimension();
  double *v_coords = new double[dim];
  if (!m->hasRealSection("spacing")) { 
    spacing = m->getRealSection("spacing");
    spacing->setFiberDimension(vertices, 1);
    m->allocate(spacing);
  } else {
    spacing = m->getRealSection("spacing");
  }
  while (v_iter != v_iter_end) {  //standard nearest-neighbor calculation
    ALE::Obj<ALE::Mesh::sieve_type::coneSet> neighbors = s->cone(s->support(*v_iter));
    ALE::Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin(); 
    ALE::Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
    PetscMemcpy(v_coords, coordinates->restrictPoint(*v_iter), dim*sizeof(double)); 
    double min_dist;
    bool first = true;
    while (n_iter != n_iter_end) {
      if (*n_iter != *v_iter) {
        double dist = 0.;
        const double * n_coords = coordinates->restrictPoint(*n_iter);
        for (int i = 0; i < dim; i++) {
	  dist += (n_coords[i] - v_coords[i])*(n_coords[i] - v_coords[i]);
        }
        dist = sqrt(dist);
        //PetscPrintf(m->comm(), "%f\n", dist);
        if (dist < min_dist || first == true) {
          min_dist = dist;
          first = false;
        }
      }
      n_iter++;
    }
    //PetscPrintf(m->comm(), "%f\n", min_dist);
    spacing->updatePoint(*v_iter, &min_dist);
    v_iter++;
  }
  delete [] v_coords;
  return spacing;
}

/*
=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
Hierarchy_insertVerticesIntoMesh

Use: Inserts a bunch of unaffiliated vertices into a boundary mesh for use with
the meshbuilder routines

Inputs:

bound_mesh: the mesh
vertex_set: the set of vertices to be inserted

Returns: void

Remarks:

please please please only insert points that are already defined in the coordinate
section.

=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
*/

#undef __FUNCT__
#define __FUNCT__ "Hierarchy_insertVerticesIntoMesh"

void Hierarchy_insertVerticesIntoMesh(ALE::Obj<ALE::Mesh> bound_mesh, ALE::Obj<ALE::Mesh::sieve_type::supportSet> vertex_set) {
  ALE::Mesh::sieve_type::supportSet::iterator vs_iter = vertex_set->begin();
  ALE::Mesh::sieve_type::supportSet::iterator vs_iter_end = vertex_set->end();
  ALE::Obj<ALE::Mesh::sieve_type> s = bound_mesh->getSieve();
  while (vs_iter != vs_iter_end) {
    s->addCapPoint(*vs_iter);
    vs_iter++;
  }
  bound_mesh->stratify();
  //PetscPrintf(bound_mesh->comm(), "%d is the depth of the inserted points\n", bound_mesh->depth(*vertex_set->begin()));
}

/*
=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
Hierarchy_coarsenMesh:

Use:

Returns a coarser version of a 1D, 2D, or 3D mesh

Inputs:

original_mesh: an effectively 1D simplicial mesh
coarsen_factor: the expansion of the spacing function
boundary_mesh: a COMPLETE boundary mesh of effective dimension dim-1

Output:

A coarsened mesh

Remarks:

Uses routines dependent on having triangle and tetgen installed

=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
*/

#undef __FUNCT__
#define __FUNCT__ "Hierarchy_coarsenMesh"

ALE::Obj<ALE::Mesh> Hierarchy_coarsenMesh(ALE::Obj<ALE::Mesh> original_mesh, double coarsen_factor, ALE::Obj<ALE::Mesh> boundary_mesh = PETSC_NULL) {
  int dim = original_mesh->getDimension();
  ALE::Obj<ALE::Mesh::real_section_type> spacing = Hierarchy_defineSpacingFunction(original_mesh);
  //MPI_Comm comm = original_mesh->comm();
  ALE::Obj<ALE::Mesh::label_sequence> vertices = original_mesh->depthStratum(0);
  ALE::Mesh::label_sequence::iterator v_iter = vertices->begin();
  ALE::Mesh::label_sequence::iterator v_iter_end = vertices->end();
  ALE::Mesh::point_type maxIndex;
  bool first_maxIndex = true;
  while (v_iter != v_iter_end) {
    if (first_maxIndex) {
      maxIndex = *v_iter;
      first_maxIndex = false;
    } else if (maxIndex < *v_iter) {
      maxIndex = *v_iter;
    }
    v_iter++;
  }
  if (dim > 3) throw ALE::Exception("Cannot coarsen meshes in more than 3 dimensions");
  ALE::Obj<ALE::Mesh::sieve_type> s = original_mesh->getSieve();
  ALE::Obj<ALE::Mesh> output_mesh;
  if (dim == 1) {
    ALE::Obj<ALE::Mesh::sieve_type::supportSet> bound_0;
    if (!boundary_mesh) { //build the 0D boundary
      bound_0 = Hierarchy_createEffective0DBoundary(original_mesh);
    } else { //I know this case will never, ever come up, but whatever.
      ALE::Obj<ALE::Mesh::label_sequence> boundary_vertices = boundary_mesh->depthStratum(0);
      bound_0 = new ALE::Mesh::sieve_type::supportSet();
      ALE::Mesh::label_sequence::iterator bv_iter = boundary_vertices->begin();
      ALE::Mesh::label_sequence::iterator bv_iter_end = boundary_vertices->end();
      while (bv_iter != bv_iter_end) {
        bound_0->insert(*bv_iter);
        bv_iter++;
      }
    }
  }
  if (dim == 2) {
    ALE::Obj<ALE::Mesh::sieve_type::supportSet> bound_0;
    ALE::Obj<ALE::Mesh::sieve_type::supportSet> curvature_set;
    ALE::Obj<ALE::Mesh> bound_1;
    ALE::Obj<ALE::Mesh::sieve_type::supportSet> bound_1_vertices = new ALE::Mesh::sieve_type::supportSet();
    if (!boundary_mesh) {
      bound_1 = Hierarchy_createEffective1DBoundary(original_mesh, maxIndex);
      //PetscPrintf(original_mesh->comm(), "%d vertices, %d edges on the boundary\n", bound_1->depthStratum(0)->size(), bound_1->depthStratum(1)->size());
    }
    ALE::Obj<ALE::Mesh::label_sequence> bound_1_vert_label = bound_1->depthStratum(0);
    ALE::Mesh::label_sequence::iterator bv_iter = bound_1_vert_label->begin();
    ALE::Mesh::label_sequence::iterator bv_iter_end = bound_1_vert_label->end();
    while (bv_iter != bv_iter_end) {
      bound_1_vertices->insert(*bv_iter);
      bv_iter++;
    }
    bound_0 = Hierarchy_createEffective0DBoundary(bound_1);
    //PetscPrintf(comm, "%d vertices in the 0D effective boundary\n", bound_0->size());
    //coarsen the interior
    //come up with the interior set:
    ALE::Obj<ALE::Mesh::sieve_type::supportSet> interior_set = new ALE::Mesh::sieve_type::supportSet();
    vertices = original_mesh->depthStratum(0);
    v_iter = vertices->begin();
    v_iter_end = vertices->end();
    while (v_iter != v_iter_end) {
      if (bound_1_vertices->find(*v_iter) == bound_1_vertices->end()) {
        interior_set->insert(*v_iter);
      }
      v_iter++;
    }
    //PetscPrintf(original_mesh->comm(), "%d interior vertices\n", interior_set->size());
    ALE::Obj<ALE::Mesh::sieve_type::supportSet> coarse_interior = Hierarchy_CoarsenVertexSet(original_mesh, spacing, interior_set, bound_1_vertices, coarsen_factor, maxIndex);
    //coarsen the boundary
    //PetscPrintf(original_mesh->comm(), "%d vertices in coarsened interior\n", coarse_interior->size());
    ALE::Obj<ALE::Mesh::sieve_type::supportSet> coarse_bound_set = Hierarchy_CoarsenVertexSet(original_mesh, spacing, bound_1_vertices, bound_0, coarsen_factor, maxIndex);		     //coarsen the boundary mesh
    //insert bound_0 into the coarse bound set
    ALE::Mesh::sieve_type::supportSet::iterator b0_iter = bound_0->begin();
    ALE::Mesh::sieve_type::supportSet::iterator b0_iter_end = bound_0->end();
    while (b0_iter != b0_iter_end) {
      coarse_bound_set->insert(*b0_iter);
      b0_iter++;
    }
    ALE::Obj<ALE::Mesh> coarse_bound = ALE::Surgery_1D_Coarsen_Mesh(bound_1, coarse_bound_set);
    //PetscPrintf(original_mesh->comm(), "%d v, %d e in new coarse boundary\n", coarse_bound->depthStratum(0)->size(), coarse_bound->depthStratum(1)->size());			
    Hierarchy_insertVerticesIntoMesh(coarse_bound, coarse_interior);
    //PetscPrintf(original_mesh->comm(), "%d v, %d e in the thing we feed triangle\n", coarse_bound->depthStratum(0)->size(), coarse_bound->depthStratum(1)->size());
    //copy over the "marker" label
    ALE::Obj<ALE::Mesh::label_type> bound_marker_label = coarse_bound->createLabel("marker");
    ALE::Obj<ALE::Mesh::label_type> marker = original_mesh->getLabel("marker");
    vertices = coarse_bound->depthStratum(0);
    v_iter = vertices->begin();
    v_iter_end = vertices->end();
    while (v_iter != v_iter_end) {
      coarse_bound->setValue(bound_marker_label, *v_iter, original_mesh->getValue(marker, *v_iter));
      v_iter++;
    }
    //generate the mesh
    //this is screwy... we must do this differently
    coarse_bound->setDimension(1);
    output_mesh = ALE::Generator::generateMesh(coarse_bound, (original_mesh->depth() != 1), true);
    output_mesh->stratify();
    //PetscPrintf(original_mesh->comm(), "%d v, %d cells in the output mesh\n", output_mesh->depthStratum(0)->size(), output_mesh->heightStratum(0)->size());
  }
  if (dim == 3) {
    ALE::Obj<ALE::Mesh::sieve_type::supportSet> bound_0;
    ALE::Obj<ALE::Mesh> bound_1;
    ALE::Obj<ALE::Mesh::sieve_type::supportSet> bound_1_vertices = new ALE::Mesh::sieve_type::supportSet();
    ALE::Obj<ALE::Mesh> bound_2;
    ALE::Obj<ALE::Mesh::sieve_type::supportSet> bound_2_vertices = new ALE::Mesh::sieve_type::supportSet(); 
    ALE::Obj<ALE::Mesh::sieve_type::supportSet> interior_vertices = new ALE::Mesh::sieve_type::supportSet();
   if (!boundary_mesh) {
      bound_2 = Hierarchy_createEffective2DBoundary(original_mesh, maxIndex);
    } else {
      bound_2 = boundary_mesh;
    } 
   //build the boundary and interior vertex sets from this
   //PetscPrintf(original_mesh->comm(), "%d vertices, %d cells in 2D boundary\n", bound_2->depthStratum(0)->size(), bound_2->heightStratum(0)->size());
    ALE::Obj<ALE::Mesh::label_sequence> bound_2_vert_label = bound_2->depthStratum(0);
    ALE::Mesh::label_sequence::iterator bv_iter = bound_2_vert_label->begin();
    ALE::Mesh::label_sequence::iterator bv_iter_end = bound_2_vert_label->end();
    while (bv_iter != bv_iter_end) {
      bound_2_vertices->insert(*bv_iter);
      bv_iter++;
    }
    v_iter = vertices->begin();
    v_iter_end = vertices->end();
    while (v_iter != v_iter_end) {
      if (bound_2_vertices->find(*v_iter) == bound_2_vertices->end()) {
        interior_vertices->insert(*v_iter);
      }
      v_iter++;
    }
    //PetscPrintf(original_mesh->comm(), "%d vertices on the interior\n", interior_vertices->size());
    bound_1 = Hierarchy_createEffective1DBoundary(bound_2, maxIndex);
    //PetscPrintf(original_mesh->comm(), "%d vertices, %d edges in 1D boundary\n", bound_1->depthStratum(0)->size(), bound_1->heightStratum(0)->size());
    ALE::Obj<ALE::Mesh::label_sequence> bound_1_vert_label = bound_1->depthStratum(0);
    bv_iter = bound_1_vert_label->begin();
    bv_iter_end = bound_1_vert_label->end();
    while (bv_iter != bv_iter_end) {
      bound_1_vertices->insert(*bv_iter);
      bv_iter++;
    }
    bound_0 = Hierarchy_createEffective0DBoundary(bound_1);
    //PetscPrintf(comm, "%d vertices in the 0D effective boundary\n", bound_0->size());
    //ok.  Coarsen the interior set.
    ALE::Obj<ALE::Mesh::sieve_type::supportSet> coarse_interior_set = Hierarchy_CoarsenVertexSet(original_mesh, spacing, interior_vertices, bound_2_vertices, coarsen_factor, maxIndex);	
    //PetscPrintf(original_mesh->comm(), "%d vertices in new interior\n", coarse_interior_set->size());
    //coarsen the boundary mesh    
    ALE::Obj<ALE::Mesh::sieve_type::supportSet> coarse_bound_2_set = Hierarchy_CoarsenVertexSet(original_mesh, spacing, bound_2_vertices, bound_1_vertices, coarsen_factor, maxIndex);
    //PetscPrintf(original_mesh->comm(), "%d vertices in new boundary interior\n", coarse_bound_2_set->size());
    //coarsen the boundary skeleton    
    ALE::Obj<ALE::Mesh::sieve_type::supportSet> coarse_bound_1_set = Hierarchy_CoarsenVertexSet(original_mesh, spacing, bound_1_vertices, bound_0, coarsen_factor, maxIndex);
    //PetscPrintf(original_mesh->comm(), "%d vertices in new boundary skeleton\n", coarse_bound_1_set->size());
    //merge the coarse_bound_1, bound_0, and coarse_bound_2 sets
    ALE::Mesh::sieve_type::supportSet::iterator av_iter = coarse_bound_1_set->begin();
    ALE::Mesh::sieve_type::supportSet::iterator av_iter_end = coarse_bound_1_set->end();
    while (av_iter != av_iter_end) {
      coarse_bound_2_set->insert(*av_iter);
      av_iter++;
    }
    av_iter = bound_0->begin();
    av_iter_end = bound_0->end();
    while (av_iter != av_iter_end) {
      coarse_bound_2_set->insert(*av_iter);
      av_iter++;
    }
    //PetscPrintf(original_mesh->comm(), "%d total vertices in the coarse boundary: coarsening by flip.\n", coarse_bound_2_set->size());
    //ok.  Moment of truth; use the FLIPPING to coarsen the boundary mesh
    ALE::Obj<ALE::Mesh> coarse_bound_mesh = ALE::Surgery_2D_Coarsen_Mesh(bound_2, coarse_bound_2_set, bound_1);
    //PetscPrintf(original_mesh->comm(), "%d vertices, %d faces in new exterior boundary\n", coarse_bound_mesh->depthStratum(0)->size(), coarse_bound_mesh->heightStratum(0)->size());
    Hierarchy_insertVerticesIntoMesh(coarse_bound_mesh, coarse_interior_set);
    //PetscPrintf(original_mesh->comm(), "%d vertices, %d faces to tetgen\n", coarse_bound_mesh->depthStratum(0)->size(), coarse_bound_mesh->heightStratum(0)->size());
    ALE::Obj<ALE::Mesh::label_type> bound_marker_label = coarse_bound_mesh->createLabel("marker");
    ALE::Obj<ALE::Mesh::label_type> marker = original_mesh->getLabel("marker");
    vertices = coarse_bound_mesh->depthStratum(0);
    v_iter = vertices->begin();
    v_iter_end = vertices->end();
    while (v_iter != v_iter_end) {
      coarse_bound_mesh->setValue(bound_marker_label, *v_iter, original_mesh->getValue(marker, *v_iter));
      v_iter++;
    }
    //generate the mesh
    //this is screwy... we must do this differently
    coarse_bound_mesh->setDimension(2);
    output_mesh = ALE::Generator::generateMesh(coarse_bound_mesh, (original_mesh->depth() != 1), true);
    output_mesh->stratify();
    //PetscPrintf(original_mesh->comm(), "%d v, %d cells in the output mesh\n", output_mesh->depthStratum(0)->size(), output_mesh->heightStratum(0)->size());
  }
  //PetscPrintf(comm, "leaving overall coarsening\n");
  return output_mesh;
}

/*
=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
Hierarchy_createHierarchy:

Use:

Returns a coarser hierarchy of a 1D, 2D, or 3D mesh

Inputs:

original_mesh: an effectively 1D simplicial mesh
nLevels: the number of levels (including the fine level)
coarsen_factor: the expansion of the spacing function
boundary_mesh: an optional boundary mesh
CtF: do coarse-to-fine hierarchy building (O(ln)) instead of fine-to-coarse (O(n))

Output:

A coarsened mesh hierarchy

Remarks:

Uses routines dependent on having triangle and tetgen installed

=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
*/

ALE::Obj<ALE::Mesh> * Hierarchy_createHierarchy(ALE::Obj<ALE::Mesh> original_mesh, 
                                                int nLevels,
                                                double beta, 
                                                ALE::Obj<ALE::Mesh> boundary_mesh = PETSC_NULL,
                                                PetscTruth CtF = PETSC_FALSE) {

    ALE::Obj<ALE::Mesh> * meshes = new ALE::Obj<ALE::Mesh>[nLevels];
    if (CtF) {
      throw ALE::Exception("Coarse-to-Fine not yet implemented.");
    } else {
      //fine-to-coarse hierarchy creation
      meshes[0] = original_mesh;
      for (int i = 1; i < nLevels; i++) {
        meshes[i] = Hierarchy_coarsenMesh(meshes[i-1], beta);
      }
    }
    return meshes;
  }

  ALE::Obj<ALE::Mesh> * Hierarchy_createHierarchy_adaptive(ALE::Obj<ALE::Mesh> original_mesh,
                                                unsigned int nVertices,
						unsigned int max_meshes,
                                                double beta,
						int * nMeshes,
                                                ALE::Obj<ALE::Mesh> boundary_mesh = PETSC_NULL,
                                                PetscTruth CtF = PETSC_FALSE) {
    ALE::Obj<ALE::Mesh> * meshes;
    if (CtF) {
      throw ALE::Exception ("Coarse-to-Fine not yet implemented.");
    } else {
      //fine to coarse hierarchy creation AS LONG AS the mesh size is over nVertices.
      std::list<ALE::Obj<ALE::Mesh> > meshes_list;
      meshes_list.clear();
      meshes_list.push_front(original_mesh);
      ALE::Obj<ALE::Mesh> current_mesh = original_mesh;
      while (current_mesh->depthStratum(0)->size() > nVertices && meshes_list.size() <= max_meshes) {
         current_mesh = Hierarchy_coarsenMesh(current_mesh, beta);
         meshes_list.push_back(current_mesh);
      }
      meshes = new ALE::Obj<ALE::Mesh>[meshes_list.size()];
      *nMeshes = meshes_list.size();
      int i = 0;
      for (std::list<ALE::Obj<ALE::Mesh> >::iterator m_iter = meshes_list.begin(); m_iter != meshes_list.end(); m_iter++) {
        meshes[i] = *m_iter;
        i++;
      } 
    }
    return meshes;
  }
