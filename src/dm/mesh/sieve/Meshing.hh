//Sieve-centric meshing utilities -- arbitrary dimensions
#include "Mesh.hh"


ALE::Mesh::point_type Meshing_ConeConstruct(ALE::Obj<ALE::Mesh> m, ALE::Mesh::point_type v, ALE::Mesh::point_type c, ALE::Mesh::point_type max_index){
  //glue vertex v to the closure of c creating a simplex of dimension dim(c)+1 in the center.
  //return the new simplex (which will be the new current_index-1 also)
  ALE::Obj<ALE::Mesh::sieve_type> s = m->getSieve();
  ALE::Obj<ALE::Mesh::label_type> depth = m->getLabel("depth");
  ALE::Mesh::point_type current_index = max_index;
  //it is ok to create copies of all things in the closure of c with the same local sieve, then extend them all such that:
  // 1. the lowest dimensional ones have v in their cone as well
  // 2. the copies support their original
  //do this through recursion: for 3D one would add the volume based upon the face, then add the faces based upon the edges, recursing to add the edges
  PetscPrintf(m->comm(), "entered cone construct with vertex %d, cell %d\n", v, c);
  ALE::Obj<ALE::Mesh::sieve_type::supportSet> current_join = s->join(v, c); //base case
  PetscPrintf(m->comm(), "join size: %d\n", current_join->size()); 
  if (current_join->size() == 0) { //recurse
    //the created volume in the cone construction will always be covered by c
    current_index++;
    ALE::Mesh::point_type constructed_index = current_index;
    s->addArrow(c, constructed_index);
    m->setValue(depth, constructed_index, m->depth(c) + 1);
    PetscPrintf(m->comm(), "Inserting %d of depth %d into the mesh\n", current_index, m->depth(c) + 1);
    if (m->depth(c) == 0) {
      //base case: add an EDGE between this vertex and v as the simplest construction
       s->addArrow(v, constructed_index);
       PetscPrintf(m->comm(), "%d and %d are in the cone of %d\n", v, c, current_index);
       return current_index;
    } else { //recursive case: add a face with surrounding edges 
      ALE::Mesh::sieve_type::supportSet current_cone; //the recursively-derived cone of the present thing
      ALE::Mesh::sieve_type::supportSet tmp_cone; //used to avoid static
      ALE::Obj<ALE::Mesh::sieve_type::coneSequence> cone_c = s->cone(c);
      ALE::Mesh::sieve_type::coneSequence::iterator cc_iter = cone_c->begin();
      ALE::Mesh::sieve_type::coneSequence::iterator cc_iter_end = cone_c->end();
      //because of STATIC everywhere operations on the sieve MUST be atomic
      while(cc_iter != cc_iter_end) {
	PetscPrintf(m->comm(), "%d is in the cone of %d\n", *cc_iter, c);
	tmp_cone.insert(*cc_iter);
	cc_iter++;
      }
      ALE::Mesh::sieve_type::supportSet::iterator curc_iter = tmp_cone.begin();
      ALE::Mesh::sieve_type::supportSet::iterator curc_iter_end = tmp_cone.end();
      while (curc_iter != curc_iter_end) {
	PetscPrintf(m->comm(), "recursing on %d\n", *curc_iter);
	ALE::Mesh::point_type cc_point = Meshing_ConeConstruct(m, v, *curc_iter, current_index);
	current_cone.insert(cc_point);
	if (cc_point > current_index) current_index = cc_point;
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
  } else {
    //return the existing item in the join; if there is more than one item in the join something is wrong.
    if (current_join->size() != 1) throw ALE::Exception("ConeConstruct: bad join of size > 1");
    PetscPrintf(m->comm(), "join already exists, returning %d" *current_join->begin);
    return *current_join->begin();
  }
  return current_index;
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
  //go through all vertices as an outer loop; might have discontinuous mesh regionss
  ALE::Mesh::sieve_type::supportSet::iterator v_iter = includedVertices->begin();
  ALE::Mesh::sieve_type::supportSet::iterator v_iter_end = includedVertices->end();
  while (v_iter != v_iter_end) {
    
    v_iter++;
  }
}

