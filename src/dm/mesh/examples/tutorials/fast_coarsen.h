/*=====================================*\
 A FAST approach to creating the coarsening hierarchy
\*=====================================*/

namespace ALE { namespace Coarsener {
  /*struct mis_node {
    bool isLeaf;
    mis_node * parent
    double * boundaries;
    double maxSpacing;  //we can only refine until the radius of the max spacing ball is hit.
    std::list<mis_node *> subspaces;
    int depth;
    std::list<ALE::Mesh::point_type> childPoints;
    //std::list<ALE::Mesh::point_type> childBoundPoints;
    std::list<ALE::Mesh::point_type> childColPoints;
};*/
  bool isOverlap(mis_node *, mis_node *, int);
  PetscErrorCode CreateCoarsenedHierarchyNew (Obj<ALE::Mesh>& mesh, int dim, int nMeshes, double beta = 1.41) {
    PetscFunctionBegin;
    //create the initial overhead comparison level.
    	  //build the root node;
    ALE::Mesh::section_type::patch_type rPatch = 0; //the patch on which everything is stored.. we restrict to this patch
    ALE::Mesh::section_type::patch_type boundPatch = nMeshes + 1; //the patch on which everything is stored.. we restrict to this patch
    Obj<ALE::Mesh::topology_type> topology = mesh->getTopology();
    const Obj<ALE::Mesh::topology_type::label_sequence>& vertices = topology->depthStratum(rPatch, 0);
    Obj<ALE::Mesh::section_type> coords = mesh->getSection("coordinates");
    Obj<ALE::Mesh::section_type> spacing = mesh->getSection("spacing");
    const Obj<ALE::Mesh::topology_type::patch_label_type>& boundary = topology->getLabel(rPatch, "boundary");
    std::list<ALE::Mesh::point_type> globalNodes; //the list of global nodes that have been accepted.
    mis_node * tmpPoint = new mis_node;
    tmpPoint->boundaries = new double[2*dim];
    tmpPoint->maxSpacing = 0;
    tmpPoint->depth = 0;
    tmpPoint->isLeaf = true;
    bool * bound_init = new bool[2*dim];
    for (int i = 0; i < 2*dim; i++) {
      bound_init[i] = false;
    }
    ALE::Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin();
    ALE::Mesh::topology_type::label_sequence::iterator v_iter_end = vertices->end();
    while (v_iter != v_iter_end) {
      double cur_space = *spacing->restrict(rPatch, *v_iter);
      const double * cur_coords = coords->restrict(rPatch, *v_iter); //I swear a lot when I code
	    //initialize the boundaries.
      for (int i = 0; i < dim; i++) {
	if (bound_init[2*i] == false || cur_coords[i] < tmpPoint->boundaries[2*i]) {
	  bound_init[2*i] = true; 
	  tmpPoint->boundaries[2*i] = cur_coords[i];
	}
	if (bound_init[2*i+1] == false || cur_coords[i] > tmpPoint->boundaries[2*i+1]) {
	  bound_init[2*i+1] = true;
	  tmpPoint->boundaries[2*i+1] = cur_coords[i];
	}
      }
	    //initialize the maximum spacing ball.
      if(tmpPoint->maxSpacing < cur_space) tmpPoint->maxSpacing = cur_space;

	    //if it's essential, push it to the ColPoints stack, which will be the pool that is compared with during the traversal-MIS algorithm.
      int boundRank = topology->getValue(boundary, *v_iter);
	if(boundRank == dim) { 
	  tmpPoint->childColPoints.push_front(*v_iter);
	  globalNodes.push_front(*v_iter);
	} else if (boundRank == 0) {  tmpPoint->childPoints.push_back(*v_iter);
	} else tmpPoint->childPoints.push_front(*v_iter);
      v_iter++;
    }

    std::list<mis_node *> mis_queue;
    std::list<mis_node *> leaf_list;
    mis_node * root = tmpPoint;
    for (int curLevel = nMeshes; curLevel > 0; curLevel--) {
      double pBeta = pow(beta, curLevel);
      //quadtree refinement phase.
      mis_queue.push_front(root);
      while (!mis_queue.empty()) {
	tmpPoint = *mis_queue.begin();
	mis_queue.pop_front();
	if (!tmpPoint->isLeaf) {
	  std::list<mis_node *>::iterator child_node_iter = tmpPoint->subspaces.begin();
	  std::list<mis_node *>::iterator child_node_end = tmpPoint->subspaces.end();
	  while (child_node_iter != child_node_end) {
	    mis_queue.push_front(*child_node_iter);
	    child_node_iter++;
	  }
	} else {  //check if we can refine the leaf.  If we can, do it!
	  bool canRefine = true;
	    //define the criterion under which we cannot refine this particular section
	  for (int i = 0; i < dim; i++) {
	    if((tmpPoint->boundaries[2*i+1] - tmpPoint->boundaries[2*i]) < 2*pBeta*tmpPoint->maxSpacing) { 
	      canRefine = false;
	     // PetscPrintf(mesh->comm(), "-- cannot refine: %f < %f\n", (tmpPoint->boundaries[2*i+1] - tmpPoint->boundaries[2*i]),2*pBeta*tmpPoint->maxSpacing);
	    }
	  }
	  if (tmpPoint->childPoints.size() + tmpPoint->childColPoints.size() < 20) canRefine = false;  //the threshhold at which we do not care to not do the greedy thing as comparison is cheap enough
	  if (canRefine) {
	  //PetscPrintf(mesh->comm(), "-- refining an area containing %d nodes..\n", tmpPoint->childPoints.size() + tmpPoint->childColPoints.size());
	    tmpPoint->isLeaf = false;
	    int nblocks = (int)pow(2, dim);
	    mis_node ** newBlocks = new mis_node*[nblocks];
	    for (int i = 0; i < nblocks; i++) {
	      newBlocks[i] = new mis_node;
	      tmpPoint->subspaces.push_front(newBlocks[i]);
	      newBlocks[i]->parent = tmpPoint;
	      newBlocks[i]->boundaries = new double[2*dim];
	      newBlocks[i]->maxSpacing = 0;
	      newBlocks[i]->depth = tmpPoint->depth + 1;
	      newBlocks[i]->isLeaf = true;
	    }
	    int curdigit = 1;
	    for (int d = 0; d < dim; d++) {
	      curdigit = curdigit * 2;
	      for (int i = 0; i < nblocks; i++) {
		if (i % curdigit == i) { //indicates that it is the one we'll have on the "left" boundary, otherwise is on the "right" boundary.
		  newBlocks[i]->boundaries[2*d] = tmpPoint->boundaries[2*d];
		  newBlocks[i]->boundaries[2*d+1] = (tmpPoint->boundaries[2*d] + tmpPoint->boundaries[2*d+1])/2;
		} else {
		  newBlocks[i]->boundaries[2*d] = (tmpPoint->boundaries[2*d] + tmpPoint->boundaries[2*d+1])/2;
		  newBlocks[i]->boundaries[2*d+1] = tmpPoint->boundaries[2*d+1];
		}
	      }
	    }
	    std::list<ALE::Mesh::point_type>::iterator p_iter = tmpPoint->childPoints.begin();
	    std::list<ALE::Mesh::point_type>::iterator p_iter_end = tmpPoint->childPoints.end();
	    while (p_iter != p_iter_end) {
	      double ch_space = *spacing->restrict(rPatch, *p_iter);
	      const double * cur_coords = coords->restrict(rPatch, *p_iter);
	      int index = 0, change = 1;
	      for (int d = 0; d < dim; d++) {
		if ((tmpPoint->boundaries[2*d] + tmpPoint->boundaries[2*d+1])/2 > cur_coords[d]) index += change;
		change = change * 2;
	      }
	      newBlocks[index]->childPoints.push_back(*p_iter);
	      if(ch_space > newBlocks[index]->maxSpacing) newBlocks[index]->maxSpacing = ch_space;
	      p_iter++;
	    }
	    p_iter = tmpPoint->childColPoints.begin();
	    p_iter_end = tmpPoint->childColPoints.end();
		//add the points to the 
	    while (p_iter != p_iter_end) {
	      double ch_space = *spacing->restrict(rPatch, *p_iter);
	      const double * cur_coords = coords->restrict(rPatch, *p_iter);
	      int index = 0, change = 1;
	      for (int d = 0; d < dim; d++) {
		if ((tmpPoint->boundaries[2*d] + tmpPoint->boundaries[2*d+1])/2 > cur_coords[d]) index += change;
		change = change * 2;
	      }
	      if(ch_space > newBlocks[index]->maxSpacing) newBlocks[index]->maxSpacing = ch_space;
	      newBlocks[index]->childColPoints.push_back(*p_iter);
	      p_iter++;
	    }
		//add all the new blocks to the refinement queue.
	    for (int i = 0; i < nblocks; i++) {
	      mis_queue.push_front(newBlocks[i]);
	    }

	  } else {  //ending refinement if
	    tmpPoint->isLeaf = true;
	    leaf_list.push_front(tmpPoint); 
	  } //ending canrefine else
	} //ending !isleaf else
      } //ending refinement while.
      //MIS picking phase
      PetscPrintf(mesh->comm(), "%d Refinement Regions created for this level.\n", leaf_list.size());
      std::list<mis_node *>::iterator leaf_iter = leaf_list.begin();
      std::list<mis_node *>::iterator leaf_iter_end = leaf_list.end();
      //PetscPrintf(mesh->comm(), "- created %d comparison spaces\n", leaf_list.size());
      while (leaf_iter != leaf_iter_end) {
	    //we must now traverse the tree in such a way as to determine what collides with this leaf and what to do about it.
	std::list<mis_node *> comparisons; //dump the spaces that will be directly compared to cur_point in here.
	std::list<mis_node *> mis_travQueue; //the traversal queue.
	    // go top-down with the comparisons.
	mis_node * cur_leaf = *leaf_iter;
	mis_travQueue.push_front(root);
	while(!mis_travQueue.empty()) {
	  mis_node * trav_node = *mis_travQueue.begin();
	  mis_travQueue.pop_front();
	  if (trav_node->isLeaf && (trav_node != *leaf_iter)) { //add this leaf to the comparison list.
	    comparisons.push_front(trav_node);
	  } else { //for non-leafs we compare the children using the same heuristic, namely if there could be any possible collision between the two.
	    std::list<mis_node *>::iterator child_iter = trav_node->subspaces.begin();
	    std::list<mis_node *>::iterator child_iter_end = trav_node->subspaces.end();
	    while(child_iter != child_iter_end) {
	      if(isOverlap(*child_iter, *leaf_iter, dim))mis_travQueue.push_front(*child_iter);
	      child_iter++;
	    }
	  } //end what to do for non-leafs
	} //end traversal of tree to determine adjacent sections
	//PetscPrintf(mesh->comm(), "Region has %d adjacent sections; comparing\n", comparisons.size());
	    //now loop over the adjacent areas we found to determine the MIS within *leaf_iter with respect to its neighbors.
	    //begin by looping over the vertices in the leaf.
	std::list<ALE::Mesh::point_type>::iterator l_points_iter = cur_leaf->childPoints.begin();
        //std::list<ALE::Mesh::point_type>::iterator l_points_intermed = cur_leaf->childBoundPoints.end();
	std::list<ALE::Mesh::point_type>::iterator l_points_iter_end = cur_leaf->childPoints.end();
	while (l_points_iter != l_points_iter_end) {
	  bool l_is_ok = true;
	  double l_coords[dim];
	  PetscMemcpy(l_coords, coords->restrict(rPatch, *l_points_iter), dim*sizeof(double));
	  double l_space = *spacing->restrict(rPatch, *l_points_iter);

		//internal consistency check; keeps us from having to go outside if we don't have to.
	  std::list<ALE::Mesh::point_type>::iterator int_iter = cur_leaf->childColPoints.begin();
	  std::list<ALE::Mesh::point_type>::iterator int_iter_end = cur_leaf->childColPoints.end();
	  while (int_iter != int_iter_end && l_is_ok) {
	    double i_coords[dim];
	    double dist = 0;
	    PetscMemcpy(i_coords, coords->restrict(rPatch, *int_iter), dim*sizeof(double));
	    double i_space = *spacing->restrict(rPatch, *int_iter);
	    for (int d = 0; d < dim; d++) {
	      dist += (i_coords[d] - l_coords[d])*(i_coords[d] - l_coords[d]);
	    }
	    double mdist = i_space + l_space;
	    if (dist < pBeta*pBeta*mdist*mdist/4) l_is_ok = false;
	    int_iter++;
	  }
		//now we must iterate over the adjacent spaces as determined before.
	  std::list<mis_node *>::iterator comp_iter = comparisons.begin();
	  std::list<mis_node *>::iterator comp_iter_end = comparisons.end();
	  while (comp_iter != comp_iter_end && l_is_ok) {
	    mis_node * cur_comp = *comp_iter;
	    std::list<ALE::Mesh::point_type>::iterator adj_iter = cur_comp->childColPoints.begin();
	    std::list<ALE::Mesh::point_type>::iterator adj_iter_end = cur_comp->childColPoints.end();
	    while (adj_iter != adj_iter_end && l_is_ok) {
	      double a_coords[dim];
	      double dist = 0;
	      PetscMemcpy(a_coords, coords->restrict(rPatch, *adj_iter), dim*sizeof(double));
	      double a_space = *spacing->restrict(rPatch, *adj_iter);
	      for (int d = 0; d < dim; d++) {
		dist += (a_coords[d] - l_coords[d])*(a_coords[d] - l_coords[d]);
	      }
	      double mdist = l_space + a_space;
	      if (dist < pBeta*pBeta*mdist*mdist/4) l_is_ok = false;
	      adj_iter++;
	    }
	    comp_iter++;
	  }
	  if (l_is_ok) {  //this point has run the gambit... cool.
	    cur_leaf->childColPoints.push_front(*l_points_iter);
	    globalNodes.push_front(*l_points_iter); //so we only need to run this once and can keep a tally! (node nested enforced by default)
	  }
	  l_points_iter++;
	} //end while over points
	comparisons.clear(); //we need to remake this list the next time around.
	leaf_iter++;
      } //end while over leaf spaces; after this point we have a complete MIS in globalNodes
      //Mesh building phase
      triangulateio * input = new triangulateio;
      triangulateio * output = new triangulateio;
  
      input->numberofpoints = globalNodes.size();
      PetscPrintf(mesh->comm(), "Accepted %d nodes; triangulating.\n", input->numberofpoints);
      input->numberofpointattributes = 0;
      input->pointlist = new double[dim*input->numberofpoints];

  //copy the points over
      std::list<ALE::Mesh::point_type>::iterator c_iter = globalNodes.begin(), c_iter_end = globalNodes.end();

      int index = 0;
      while (c_iter != c_iter_end) {
	PetscMemcpy(input->pointlist + dim*index, coords->restrict(rPatch, *c_iter), dim*sizeof(double));
	c_iter++;
	index++;
      }

  //ierr = PetscPrintf(srcMesh->comm(), "copy is ok\n");
      input->numberofpointattributes = 0;
      input->pointattributelist = NULL;

//set up the pointmarkerlist to hold the names of the points

      input->pointmarkerlist = new int[input->numberofpoints];
      c_iter = globalNodes.begin();
      c_iter_end = globalNodes.end();
      index = 0;
      while(c_iter != c_iter_end) {
	input->pointmarkerlist[index] = *c_iter;
	c_iter++;
	index++;
      }
      const Obj<ALE::Mesh::topology_type::label_sequence>& boundEdges = topology->heightStratum(boundPatch, 0);
      ALE::Mesh::topology_type::label_sequence::iterator be_iter = boundEdges->begin();
      ALE::Mesh::topology_type::label_sequence::iterator be_iter_end = boundEdges->end();
//set up the boundary segments
      
      input->numberofsegments = boundEdges->size();
      input->segmentlist = new int[2*input->numberofsegments];
      for (int i = 0; i < 2*input->numberofsegments; i++) {
	input->segmentlist[i] = -1;  //initialize
      }
      index = 0;
      while (be_iter != be_iter_end) { //loop over the boundary segments
	ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence> neighbors = topology->getPatch(boundPatch)->cone(*be_iter);
	ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter = neighbors->begin();
	ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter_end = neighbors->end();
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

      string triangleOptions = "-zpeQ"; //(z)ero indexing, output (e)dges, Quiet
      triangulate((char *)triangleOptions.c_str(), input, output, NULL);
      TriangleToMesh(mesh, output, curLevel);
      delete input->pointlist;
      delete output->pointlist;
      delete output->trianglelist;
      delete output->edgelist;
      delete input;
      delete output;
      leaf_list.clear();
    }  //end of for over the number of coarsening levels.
    PetscFunctionReturn(0);
  }  //end of CreateCoarsenedHierarchy
  
  
/*  bool isOverlap(mis_node * a, mis_node * b, int dim) { //see if any two balls in the two sections could overlap at all.
    int sharedDim = 0;
    for (int i = 0; i < dim; i++) {
      if((a->boundaries[2*i] - a->maxSpacing <= b->boundaries[2*i+1] + b->maxSpacing) && (b->boundaries[2*i] - b->maxSpacing <= a->boundaries[2*i+1] + a->maxSpacing)) sharedDim++;
    }
    if (sharedDim == dim) {return true;
    } else return false;
}*/
} }
