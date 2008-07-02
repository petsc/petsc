//Sieve-centric meshing utilities -- arbitrary dimensions
#include "Mesh.hh"
#include "SieveAlgorithms.hh"

namespace ALE {

  namespace Meshing {
    namespace Geometry {
      
      
      double * Meshing_FaceNormal(Obj<Mesh> m, Mesh::point_type f) {
	const ALE::Obj<Mesh::sieve_type> & s = m->getSieve();
	int dim = m->getDimension();
	double normal[dim];
	for (int i = 0; i < dim; i++) {
	  for (int j = 0; j < dim; j++) {
	    
	  }
	}
      }
      
      double CellVolume(ALE::Obj<ALE::Mesh> m, ALE::Mesh::point_type c) {  //returns the SIGNED cell volume
	int celldim = m->getDimension() - m->height(c);
	if (m->depth(c) == 0) return 0.;
        const double 
      }

      bool InCircumcircle(ALE::Obj<Mesh> m, Mesh::point_type c, Mesh::point_type p) {
	
      }
      
      bool InCell(Obj<Mesh> m, Mesh:point_type c, Mesh::point_type p) {
	//express it in barycentric coordinates and do the dumbest thing.. FOR NOW!
	
      }

      double Circumcircle(Obj<ALE::Mesh> m, Mesh::point_type c, double * circumcenter) {
	//ok, to generalize this to n dimensions easily we are going to create an algorithm that iteratively finds the circumcenter 
	//through a steepest-descent scheme on the circumcenter starting from the center of mass.  
	//on each iteration:
	// d = (1/n)*sum_n(|p_n - c|^2)
	// c = c + sum_n((p_n - c)(|p_n - c|^2 - d)/(|p_n - c|^2)
	int dim = m->getDimension();
	double d, errdist;
	double * tmpcoords = m->restrict(m->getRealSection("coordinates"), c);
	for (int iteration = 0; iteration < 10; iteration++) {
	  PetscPrintf(m-comm(), "current radius: %f\n", d);
	  for (int vertex = 0; vertex < dim+1; vertex++) {
	    
	  }
	}
	double err = new double[dim];
	d = sqrt(d);
	delete err;
	return d;
      }
    }
    ALE::Mesh::point_type Meshing_ConeConstruct(ALE::Obj<ALE::Mesh> m, ALE::Mesh::point_type v, ALE::Mesh::point_type c, ALE::Mesh::point_type & max_index){
      //pretty commenting
      std::string a_string;
      const char * a;
      for (int i = 0; i < m->depth() - m->depth(c); i++) {
	a_string = a_string + "  ";
      }
      a = a_string.c_str();
      //glue vertex v to the closure of c creating a simplex of dimension dim(c)+1 in the center.
      //return the new simplex (which will be the new current_index-1 also)
      ALE::Obj<ALE::Mesh::sieve_type> s = m->getSieve();
      ALE::Obj<ALE::Mesh::label_type> depth = m->getLabel("depth");
      //it is ok to create copies of all things in the closure of c with the same local sieve, then extend them all such that:
      // 1. the lowest dimensional ones have v in their cone as well
      // 2. the copies support their original
      //do this through recursion: for 3D one would add the volume based upon the face, then add the faces based upon the edges, recursing to add the edges
      PetscPrintf(m->comm(), "%sentered cone construct with vertex %d, cell %d\n", a, v, c);
      ALE::Obj<ALE::Mesh::sieve_type::supportSet> current_join = s->nJoin(c, v, m->getDimension()); //base case
      PetscPrintf(m->comm(), "%sTOTAL nJoin size: %d\n", a, current_join->size()); 
      if (current_join->size() == 0) { //recurse
	//the created volume in the cone construction will always be covered by c
	max_index++;
	ALE::Mesh::point_type constructed_index = max_index;
	s->addArrow(c, constructed_index);
	m->setValue(depth, constructed_index, m->depth(c) + 1);
	PetscPrintf(m->comm(), "%sInserting %d of depth %d into the mesh\n", a, constructed_index, m->depth(c) + 1);
	if (m->depth(c) == 0) {
	  //base case: add an EDGE between this vertex and v as the simplest construction
	  s->addArrow(v, constructed_index);
	  PetscPrintf(m->comm(), "%s%d and %d are in the cone of %d\n", a,v, c, constructed_index);
	  return constructed_index;
	} else { //recursive case: add a face with surrounding edges 
	  ALE::Mesh::sieve_type::supportSet current_cone; //the recursively-derived cone of the present thing
	  ALE::Mesh::sieve_type::supportSet tmp_cone; //used to avoid static
	  ALE::Obj<ALE::Mesh::sieve_type::coneSequence> cone_c = s->cone(c);
	  ALE::Mesh::sieve_type::coneSequence::iterator cc_iter = cone_c->begin();
	  ALE::Mesh::sieve_type::coneSequence::iterator cc_iter_end = cone_c->end();
	  //because of STATIC everywhere operations on the sieve MUST be atomic
	  while(cc_iter != cc_iter_end) {
	    PetscPrintf(m->comm(), "%s%d is in the cone of %d\n", a,*cc_iter, c);
	    tmp_cone.insert(*cc_iter);
	    cc_iter++;
	  }
	  ALE::Mesh::sieve_type::supportSet::iterator curc_iter = tmp_cone.begin();
	  ALE::Mesh::sieve_type::supportSet::iterator curc_iter_end = tmp_cone.end();
	  while (curc_iter != curc_iter_end) {
	    PetscPrintf(m->comm(), "%srecursing on %d\n", a,*curc_iter);
	    ALE::Mesh::point_type cc_point = Meshing_ConeConstruct(m, v, *curc_iter, max_index);
	    current_cone.insert(cc_point);
	    curc_iter++;
	  }
	  //for all items in current_cone, set the arrows right.
	  curc_iter = current_cone.begin();
	  curc_iter_end = current_cone.end();
	  while (curc_iter != curc_iter_end) {
	    s->addArrow(*curc_iter, constructed_index);
	    curc_iter++;
	  }
	}
	return constructed_index;
      } else {
	ALE::Mesh::sieve_type::supportSet::iterator cj_iter = current_join->begin();
	ALE::Mesh::sieve_type::supportSet::iterator cj_iter_end = current_join->end();
	int min_depth_in_join = m->depth(c)+1; //the depth of the thing we want to connect up to
	cj_iter = current_join->begin();
	cj_iter_end = current_join->end();
	while (cj_iter != cj_iter_end) {
	  int cj_depth = m->depth(*cj_iter);
	  if (cj_depth != min_depth_in_join) {
	    ALE::Mesh::sieve_type::supportSet::iterator cj_erase = cj_iter;
	    cj_iter++;
	    current_join->erase(cj_erase);
	  } else {
	    cj_iter++;
	  }
	}
	//return the existing item in the join; if there is more than one item in the join something is wrong.
	if (current_join->size() != 1) throw ALE::Exception("ConeConstruct: bad join of size > 1");
	PetscPrintf(m->comm(), "%sjoin already exists, returning %d\n", a,*current_join->begin());
	return *current_join->begin();
      }
      return c;
    }
    
/*
  Meshing_FlattenCell
  
  Flatten the present cell to its support such that the resulting mesh is uninterpolated
  
  Use it when you've glued on both sides of a hyperface.
  
  Returns nothing!
  
*/
    
    void Meshing_FlattenCell(ALE::Obj<ALE::Mesh> m, ALE::Mesh::point_type c) {
      ALE::Obj<ALE::Mesh::sieve_type> s = m->getSieve();
      
    }
    
/*
Meshing_CoarsenMesh

Done through the cone construction of simplices based upon traversal.  so:
1. vertex finds vertex to make a line
2. line finds vertex to make a triangle
3. triangle finds vertex to make a tetrahedron
4. once a simplex is sufficiently surrounded, intermediate points may be collapsed to uninterpolate
5. preserves the vertex namespace 
6. based on traversal, so will preserve boundaries/internal boundaries implicitly.

*/
    
    ALE::Obj<ALE::Mesh> Meshing_CoarsenMesh(ALE::Obj<ALE::Mesh> m, ALE::Obj<ALE::Mesh::sieve_type::supportSet> includedVertices, bool interpolate = true) {
      //do this based upon SEARCH rather than anything else.  add Delaunay condition later.
      //int dim = m->getDimension();
      //build a series of sets for enqueuing new topological features;
      //find the maximum-valued vertex index
      ALE::Mesh::point_type cur_index;
      ALE::Obj<ALE::Mesh::label_sequence> all_vertices = m->depthStratum(0);
      ALE::Mesh::label_sequence::iterator av_iter = all_vertices->begin();
      ALE::Mesh::label_sequence::iterator av_iter_end = all_vertices->end();
      cur_index = *av_iter;
      while (av_iter != av_iter_end) {
	if (*av_iter > cur_index) cur_index = *av_iter;
	av_iter++;
      }
      //go through all vertices as an outer loop; might have discontinuous mesh regions
      ALE::Mesh::sieve_type::supportSet::iterator v_iter = includedVertices->begin();
      ALE::Mesh::sieve_type::supportSet::iterator v_iter_end = includedVertices->end();
      while (v_iter != v_iter_end) {
	
	v_iter++;
      }
    }
    

    Obj<Mesh> DelaunayCoarsen(Obj<Mesh> fine_mesh, Obj<Mesh::sieve_type::supportSet> vertex_subset) {
      //return a coarsened version of the present mesh; all meshing calculations done through adjacency.
      //the fine mesh gives us a GREAT DEAL of spacial information and allows us to approximate nearest neighbors readily.
      Obj<Mesh::sieve_type> f_s = fine_mesh->getSieve();
      //create the new mesh
      Obj<Mesh> m = new Mesh();
      Obj<Mesh::sieve_type> s = new Mesh::sieve_type();
      m->setSieve(s);
      
    }

    //ok we're obviously having problems with other things, but sieve just sped 
    //WAY up, so let's just implement some simple forms of space triangulation
    //I'm going to request that the boundaries be done in a bit of a different way than before.  

    
    
    Obj<Mesh> DelaunaySimplicialize_Recursive(Obj<Mesh> boundary, Obj<Mesh::sieve_type::supportSet> vertex_subset) {
      //implement a simple meshing thing over the boundary, inserting vertices when necessary (or not if constrained)
      Obj<Mesh> m = new Mesh();
      Obj<Mesh::sieve_type> s = new Mesh::sieve_type();
      
    }
    
    Obj<Mesh> DelaunayCoarsen(Obj<Mesh> fine_mesh, Obj<Mesh::supportSet> includedVertices) {
      //for each thing in the coarse set: outwardly construct its LINK from its nearest neighbors; respecting previously done work for other simplices
      //if each local link is delaunay then the whole thing is delaunay(?)
      Obj<Mesh::sieve_type> f_s = fine_mesh->getSieve();
      int dim = fine_mesh->getDimension();
      Obj<Mesh> n_m = new Mesh(fine_mesh->comm(), dim, 0);
      Obj<Mesh::sieve_type n_s = new Mesh::sieve_type(fine_mesh->comm(), 0);
      n_m->setSieve(n_s);
      //what do we have to do to add a simplex?  I'll tell you what:
      //1. Make sure that it's across a doublet from the edge it was created on.  
      //2. so, there will be a center simplex and its positive and negative suspension sides to dim-dimensional simplices
      //3. the orientations of these will be opposite, of course.
       //create a queue for the sake of components to remesh
      Obj<Mesh::supportSet> point_queue = new Mesh::supportSet();
      point_queue->insert(*includedVertices->begin());
      while (!point_queue->empty()) {
	Mesh::supportSet::iterator c_point_iter = point_queue->begin();
	Mesh::point_type c_point = *c_point_iter;
	point_queue->remove(*c_point_iter);
	int point_depth = m->depth(c_point);
	if (point_depth == dim) {
	  //nothing to do here; it shouldn't happen
	} else {
	  //we want to glue this simplex to a new vertex s.t. it doesn't collide with any other things and if we're building a dim-dim
	  //simplex, we want it to be delaunay. w.r.t. the "doublets" it invokes.
	  bool mesh_from_point = true;
	  if (n_s->hasPoint(c_point)) { //this case should only happen in the case of the includedVertices
	    if (point_depth == dim - 1) { //in this case it's a simplex "face" and we want it to be covered on each "side" only.
	    }
	  }
	  
	}
      } //end of the point_queue while.
    }
    
    Obj<Mesh> DelaunayTetrahedralize(Obj<Mesh> boundary, bool constrained = false) {
      
    }
    
    Obj<Mesh> DelaunaySimplicialize(Obj<Mesh> boundary, bool constrained = false) {
      
    }
    
    Obj<Mesh> SparseVoronoiRefine(Obj<Mesh> m, Obj<real_section_type> refinement_limits) {
      
    }
  }
}
