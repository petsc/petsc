/*=============================================*\
  boundary discovery and categorizing functions
\*=============================================*/

namespace ALE {
  namespace Coarsener {
    struct bound_trav {  //structure with all the info needed to do a DFS on the boundary to find the PSLG
      ALE::Mesh::point_type lastCorn;  //the last corner seen.  as we string along we will connect these two in the new topology definition
      ALE::Mesh::point_type lastNode;  //the last node seen, so we do not backtrace.
      ALE::Mesh::point_type thisNode;  //the currently enqueued node.
      ALE::Mesh::point_type firstEdge;  //a HACK to name the edge something that most definitely isn't a point name.
      ALE::Mesh::point_type lastEdge;   //a HACK to keep edges from getting duplicated.
      int length;
    };
    
    PetscErrorCode make_coarsest_boundary (Obj<ALE::Mesh>& mesh, int dim, ALE::Mesh::real_section_type::patch_type patch) {
      //creates a 2-level (PSLG) representation of the boundary for feeding into triangle or tetgen.
      PetscFunctionBegin;
      Obj<ALE::Mesh::sieve_type> sieve = new ALE::Mesh::sieve_type(mesh->comm(), 0);
      ALE::Mesh::real_section_type::patch_type srcPatch = 0;
      const Obj<ALE::Mesh::topology_type>& topology = mesh->getTopology();
      int nEdges = 0; // just a counter for sanity checking.
      //const Obj<ALE::Mesh::topology_type::label_sequence>& vertices = topology->depthStratum(originalPatch, 0);
      //from here grab the corners
      Obj<ALE::Mesh::topology_type::label_sequence> corners = topology->getLabelStratum(srcPatch, "boundary", dim);
      ALE::Mesh::topology_type::label_sequence::iterator c_iter = corners->begin();
      ALE::Mesh::topology_type::label_sequence::iterator c_iter_end = corners->end();
      int nBoundaries = 0; //this loop should only run once if the space is a topological (dim-1)-sphere.
      std::list<bound_trav *> trav_queue;
      while (c_iter != c_iter_end) {
	if (!sieve->capContains(*c_iter)) {  //if it has not already been found and added to the new topology. (check to see we don't need to stratify)
	  // here we just create the first set of paths to travel on.
	  nBoundaries++;
	  ALE::Obj<ALE::Mesh::sieve_type::traits::supportSequence> support = topology->getPatch(srcPatch)->support(*c_iter);
	  ALE::Mesh::topology_type::label_sequence::iterator s_iter = support->begin();
	  ALE::Mesh::topology_type::label_sequence::iterator s_iter_end = support->end();
	  const Obj<ALE::Mesh::topology_type::patch_label_type>& boundary = topology->getLabel(srcPatch, "boundary");
	  //bool foundNeighbor = false;
	  while (s_iter != s_iter_end) {
	    ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence> neighbors = topology->getPatch(srcPatch)->cone(*s_iter);
	    ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter = neighbors->begin();
	    ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter_end = neighbors->end();
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
		ALE::Obj<ALE::Mesh::sieve_type::traits::supportSequence> support = topology->getPatch(srcPatch)->support(cur_trav->thisNode);
		ALE::Mesh::topology_type::label_sequence::iterator s_iter = support->begin();
		ALE::Mesh::topology_type::label_sequence::iterator s_iter_end = support->end();
		while (s_iter != s_iter_end) {
		  ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence> neighbors = topology->getPatch(srcPatch)->cone(*s_iter);
		  ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter = neighbors->begin();
		  ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter_end = neighbors->end();
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
	        PetscPrintf(mesh->comm(), "Added edge from %d to %d of length %d\n", cur_trav->thisNode, cur_trav->lastCorn, cur_trav->length);
	        nEdges++;
	      }
	      delete cur_trav; //we can get rid of this one.
	    } else {
	      //in this case we just continue travelling along the edge we already were at.  (assume that in 3D intersections DO NOT HAPPEN HERE or it would be essential.
	      //PetscPrintf(mesh->comm(), "|%d\n", cur_trav->thisNode);
	      ALE::Obj<ALE::Mesh::sieve_type::traits::supportSequence> support = topology->getPatch(srcPatch)->support(cur_trav->thisNode);
	      ALE::Mesh::topology_type::label_sequence::iterator s_iter = support->begin();
	      ALE::Mesh::topology_type::label_sequence::iterator s_iter_end = support->end();
	      bool foundPath = false;
	      while (s_iter != s_iter_end && !foundPath) {
		ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence> neighbors = topology->getPatch(srcPatch)->cone(*s_iter);
		ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter = neighbors->begin();
		ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter_end = neighbors->end();
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
      topology->setPatch(patch, sieve);
      topology->stratify();
      if (mesh->debug()) ;
	//PetscPrintf(mesh->comm(), "- Created %d segments in %d boundaries in the exterior PSLG\n", nEdges, nBoundaries);
      PetscFunctionReturn(0);
    }
  }
}
