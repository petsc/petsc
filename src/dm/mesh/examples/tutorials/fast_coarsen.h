/*=====================================*\
 A FAST approach to creating the coarsening hierarchy
\*=====================================*/

namespace ALE { namespace Coarsener {

  struct coarsen_Stats {

    bool computeStats; //tell if to compute stats
    bool displayStats; //tell to display stats

    int nMeshes; //the number of meshes
    double beta; //the coarsening value
    int * nNodes; //the number of nodes
    int * nFaces; //the number of faces
    double * compPpoint; //average number of comparisons per point.
    int * regions; //the number of comparison regions
    double * adjPregion; //the average number of adjacent regions per region.
    double * minAngle; //the minimum angle in each level
    double * maxAngle; //the maximum angle in each level
    
    int edgeSegments; //
    int edgePoints;
  } coarsen_stats;

  void coarsen_CollectStats(bool stats, bool display) {
    coarsen_stats.computeStats = stats;
    coarsen_stats.displayStats = display;
  }

  void coarsen_DisplayStats() {
      printf("Data for: %d levels at %f coarsening factor\n", coarsen_stats.nMeshes, coarsen_stats.beta);
      printf("|Level          |Nodes          |Faces          |Regions        |Point Comp     |Reg Comp       |Max Ang. (rad) |Min Ang. (rad) |\n");
    for (int i = 0; i <= coarsen_stats.nMeshes; i++) {
      printf("| %13d | %13d | %13d | %13d | %13f | %13f | %13f | %13f |\n", i, coarsen_stats.nNodes[i], coarsen_stats.nFaces[i], coarsen_stats.regions[i], coarsen_stats.compPpoint[i], coarsen_stats.adjPregion[i], coarsen_stats.maxAngle[i], coarsen_stats.minAngle[i]);
    }
  }
  /*struct mis_node {
    bool isLeaf;
    mis_node * parent
    double * boundaries;
    double maxSpacing;  //we can only refine until the radius of the max spacing ball is hit.
    std::list<mis_node *> subspaces;
    int depth;
    std::list<ALE::Mesh::point_type> childPoints;
    //std::list<ALE::Mesh::point_type> childBoundPoints;
    std::list<ALE::Mesh::point_type> childColPoits;
};*/
  bool IsPointInElement (Obj<ALE::Mesh>, int, ALE::Mesh::real_section_type::patch_type, ALE::Mesh::point_type, ALE::Mesh::point_type);
  double * ComputeAngles(Obj<ALE::Mesh>, int, ALE::Mesh::patch_type);
  bool isOverlap(mis_node *, mis_node *, int, double);
  bool CompatibleWithEdge(Obj<ALE::Mesh>, int, ALE::Mesh::patch_type, ALE::Mesh::point_type, ALE::Mesh::point_type, double);
  PetscErrorCode CreateCoarsenedHierarchyNew (Obj<ALE::Mesh>& mesh, int dim, int nMeshes, double beta = 1.41) {
    coarsen_CollectStats(1, 1);
    PetscFunctionBegin;
    if (coarsen_stats.computeStats) {
      coarsen_stats.nMeshes = nMeshes;
      coarsen_stats.beta = beta;
      coarsen_stats.nNodes = new int[nMeshes + 1];
      coarsen_stats.nFaces = new int[nMeshes + 1];
      coarsen_stats.compPpoint = new double[nMeshes + 1];
      coarsen_stats.regions = new int[nMeshes + 1];
      coarsen_stats.adjPregion = new double[nMeshes + 1];
      coarsen_stats.minAngle = new double[nMeshes + 1];
      coarsen_stats.maxAngle = new double[nMeshes + 1];
      
    }
    //create the initial overhead comparison level.
    	  //build the root node;
    ALE::Mesh::real_section_type::patch_type rPatch = 0; //the patch on which everything is stored.. we restrict to this patch
    ALE::Mesh::real_section_type::patch_type boundPatch = nMeshes + 1; //the patch on which everything is stored.. we restrict to this patch
    Obj<ALE::Mesh::topology_type> topology = mesh->getTopology();
    const Obj<ALE::Mesh::topology_type::label_sequence>& vertices = topology->depthStratum(rPatch, 0);

    if (coarsen_stats.computeStats) {
      coarsen_stats.nNodes[0] = vertices->size();
      const Obj<ALE::Mesh::topology_type::label_sequence>& botFaces = topology->heightStratum(rPatch, 0);
      coarsen_stats.nFaces[0] = botFaces->size();
      double * tmpDat = ComputeAngles(mesh, 2, 0);
      coarsen_stats.minAngle[0] = tmpDat[0];
      coarsen_stats.maxAngle[0] = tmpDat[1];
      coarsen_stats.compPpoint[0] = 0;
      coarsen_stats.adjPregion[0] = 0.0;
      coarsen_stats.regions[0] = 0;
    }

    Obj<ALE::Mesh::real_section_type> coords = mesh->getRealSection("coordinates");
    Obj<ALE::Mesh::real_section_type> spacing = mesh->getRealSection("spacing");
    //Obj<ALE::Mesh::int_section_type> nearest = mesh->getIntSection("nearest");
    //nearest->setFiberDimensionByDepth(rPatch, 0, 1);
    //nearest->allocate();
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
	      //PetscPrintf(mesh->comm(), "-- cannot refine: %f < %f\n", (tmpPoint->boundaries[2*i+1] - tmpPoint->boundaries[2*i]),2*pBeta*tmpPoint->maxSpacing);
	    }
	  }
	  if (tmpPoint->childColPoints.size() <= 4) canRefine = false;  //allows us to compute NEAREST POINT at each stage, as SOME adjacent thing will have a point. 
          //if (tmpPoint->childColPoints.size() + tmpPoint->childPoints.size() < 8) canRefine = false;
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
      if(coarsen_stats.computeStats) coarsen_stats.regions[curLevel] = leaf_list.size();
      std::list<mis_node *>::iterator leaf_iter = leaf_list.begin();
      std::list<mis_node *>::iterator leaf_iter_end = leaf_list.end();
      //PetscPrintf(mesh->comm(), "- created %d comparison spaces\n", leaf_list.size());
      int regions_adjacent = 0; //the total number of comparisons between regions done.
      int point_comparisons = 0; //the total number of point-to-point comparisons performed.
      int visited_nodes = 0; //the total number of nodes considered.
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
	  if (trav_node->isLeaf && trav_node != cur_leaf) { //add this leaf to the comparison list.
              comparisons.push_front(trav_node);
	  } else { //for non-leafs we compare the children using the same heuristic, namely if there could be any possible collision between the two.
	    std::list<mis_node *>::iterator child_iter = trav_node->subspaces.begin();
	    std::list<mis_node *>::iterator child_iter_end = trav_node->subspaces.end();
	    while(child_iter != child_iter_end) {
	      if(isOverlap(*child_iter, *leaf_iter, dim, pBeta)) mis_travQueue.push_front(*child_iter);
	      child_iter++;
	    }
	  } //end what to do for non-leafs
	} //end traversal of tree to determine adjacent sections
        regions_adjacent += comparisons.size();
	//PetscPrintf(mesh->comm(), "Region has %d adjacent sections; comparing\n", comparisons.size());
	    //now loop over the adjacent areas we found to determine the MIS within *leaf_iter with respect to its neighbors.
	    //begin by looping over the vertices in the leaf.
	std::list<ALE::Mesh::point_type>::iterator l_points_iter = cur_leaf->childPoints.begin();
        //std::list<ALE::Mesh::point_type>::iterator l_points_intermed = cur_leaf->childBoundPoints.end();
	std::list<ALE::Mesh::point_type>::iterator l_points_iter_end = cur_leaf->childPoints.end();
	while (l_points_iter != l_points_iter_end) {
          visited_nodes++;
          //double nearPointDist = 100; //keep track of the minimum space between this point and a point in the next level up.
          //int whyset = 0; //DEBUG for the process.
          //ALE::Mesh::point_type nearPoint = -1;
	  bool l_is_ok = true;
	  double l_coords[dim];
	  PetscMemcpy(l_coords, coords->restrict(rPatch, *l_points_iter), dim*sizeof(double));
	  double l_space = *spacing->restrict(rPatch, *l_points_iter);
          //first, check it against all the coarsest boundary segments to protect them from ARBITRARILY PI-LIKE ANGLES!
          if (topology->getValue(boundary, *l_points_iter) == 0) {
            const Obj<ALE::Mesh::topology_type::label_sequence>& boundEdges = topology->heightStratum(boundPatch, 0);
            ALE::Mesh::topology_type::label_sequence::iterator be_iter = boundEdges->begin();
            ALE::Mesh::topology_type::label_sequence::iterator be_iter_end = boundEdges->end();
            while (be_iter != be_iter_end && l_is_ok) {
              if (!CompatibleWithEdge(mesh, dim, boundPatch, *be_iter, *l_points_iter, pBeta*l_space)) l_is_ok = false;
              be_iter++;
            }
          }
          
		//internal consistency check; keeps us from having to go outside if we don't have to.
	  std::list<ALE::Mesh::point_type>::iterator int_iter = cur_leaf->childColPoints.begin();
	  std::list<ALE::Mesh::point_type>::iterator int_iter_end = cur_leaf->childColPoints.end();
	  while (int_iter != int_iter_end && l_is_ok) {
	    double i_coords[dim];
	    double dist = 0;
	    PetscMemcpy(i_coords, coords->restrict(rPatch, *int_iter), dim*sizeof(double));
	    double i_space = *spacing->restrict(rPatch, *int_iter);
	    point_comparisons++;
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
              point_comparisons++;
	      for (int d = 0; d < dim; d++) {
		dist += (a_coords[d] - l_coords[d])*(a_coords[d] - l_coords[d]);
	      }
	      double mdist = l_space + a_space;
              /*if (curLevel != nMeshes && topology->getPatch(curLevel+1)->capContains(*adj_iter)) {
                if(nearPoint == -1 || dist < nearPointDist) {
                  whyset = 1;
                  nearPoint = *adj_iter;
                  nearPointDist = dist;
                }
              }*/
	      if (dist < pBeta*pBeta*mdist*mdist/4) l_is_ok = false;
	      adj_iter++;
	    }
	    comp_iter++;
	  }
	  if (l_is_ok) {  //this point has run the gambit... cool.
	    if(curLevel != 0) {
              cur_leaf->childColPoints.push_front(*l_points_iter);
	       globalNodes.push_front(*l_points_iter); //so we only need to run this once and can keep a tally! (node nested enforced by default)
            };
            /*ALE::Mesh::point_type contTri = -1;
            if (curLevel != nMeshes) {  //compute the triangle in the next level containing this point.
              Obj<ALE::Mesh::sieve_type::supportSet> pointStar = topology->getPatch(curLevel + 1)->star(nearPoint);
              Obj<ALE::Mesh::sieve_type::coneSet> setCone = topology->getPatch(curLevel + 1)->closure(pointStar);
              Obj<ALE::Mesh::sieve_type::supportSet> setStar = topology->getPatch(curLevel+1)->star(setCone);
              ALE::Mesh::sieve_type::supportSet::iterator ps_iter = setStar->begin();
              ALE::Mesh::sieve_type::supportSet::iterator ps_iter_end = setStar->end();
              while (ps_iter != ps_iter_end && contTri == -1) {
                if (topology->getPatch(curLevel+1)->height(*ps_iter) == 0) { //pull out the triangles
                  if (IsPointInElement(mesh, dim, curLevel + 1, *ps_iter, *l_points_iter)) {
                    contTri = *ps_iter;
                  }
                }
                ps_iter++;
              }
              if(contTri == -1) {
                const double * badNear = coords->restrict(0, nearPoint);
               //printf("ERROR: Couldn't find triangle for point %d: (%f, %f) - nearest is %d: (%f, %f) set for %d\n", *l_points_iter, l_coords[0], l_coords[1], nearPoint, badNear[0], badNear[1], whyset);
               //brute force find the actual nearest.
                const Obj<ALE::Mesh::topology_type::label_sequence>& why_verts = topology->depthStratum(curLevel+1, 0);
                ALE::Mesh::point_type newone = nearPoint;
                double newdist = nearPointDist;
                ALE::Mesh::topology_type::label_sequence::iterator why_iter = why_verts->begin();
                ALE::Mesh::topology_type::label_sequence::iterator why_iter_end = why_verts->end();
                while (why_iter != why_iter_end) {
                  badNear = coords->restrict(rPatch, *why_iter);
                  double dist = 0;
                  for (int d = 0; d < dim; d++) {
		    dist += (badNear[d] - l_coords[d])*(badNear[d] - l_coords[d]);
	          }
                  if (dist < newdist) {
                    newone = *why_iter;
                    newdist = dist;
                  }
                  why_iter++;
                }
                if (newone != nearPoint) {
                //printf("FOUND: %d is actually the nearest by %f\n", newone, sqrt(newdist) - sqrt(nearPointDist));
                pointStar = topology->getPatch(curLevel + 1)->star(newone);
                setCone = topology->getPatch(curLevel+1)->closure(pointStar);
                setStar = topology->getPatch(curLevel+1)->star(setCone);
                ps_iter = setStar->begin();
                ps_iter_end = setStar->end();

                while (ps_iter != ps_iter_end && contTri == -1) {
                  if (topology->getPatch(curLevel+1)->height(*ps_iter) == 0) { //pull out the triangles
                    if (IsPointInElement(mesh, dim, curLevel + 1, *ps_iter, *l_points_iter)) {
                      contTri = *ps_iter;
                    }
                  }
                  ps_iter++;
                }
                //if (contTri == -1) {
                  //printf("still broken\n");
                //} else printf("fixed\n");
                } 
              }
            }*/
            //nearest->update(rPatch, *l_points_iter, &contTri);
	    l_points_iter = cur_leaf->childPoints.erase(l_points_iter);
	  } else {
	    l_points_iter++;
	  }
	} //end while over points
	comparisons.clear(); //we need to remake this list the next time around.
	leaf_iter++;
      } //end while over leaf spaces; after this point we have a complete MIS in globalNodes
      //Mesh building phase
      //if (curLevel != 0) {
#ifdef PETSC_HAVE_TRIANGLE
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

      string triangleOptions = "-zpDQ"; //(z)ero indexing, output (e)dges, Quiet, Delaunay
      triangulate((char *)triangleOptions.c_str(), input, output, NULL);
      TriangleToMesh(mesh, output, curLevel);
      //printf("computing the angles\n");
      delete input->pointlist;
      delete output->pointlist;
      delete output->trianglelist;
      delete output->edgelist;
      delete input;
      delete output;
      leaf_list.clear();
      //}
      if (coarsen_stats.computeStats) {
        coarsen_stats.nNodes[curLevel] = output->numberofpoints;
        coarsen_stats.nFaces[curLevel] = output->numberoftriangles;
        

        coarsen_stats.adjPregion[curLevel] = ((float)regions_adjacent)/coarsen_stats.regions[curLevel];
        coarsen_stats.compPpoint[curLevel] = ((float)point_comparisons)/visited_nodes;
        
        double * tmp_stats = ComputeAngles(mesh, dim, curLevel);
        coarsen_stats.minAngle[curLevel] = tmp_stats[0];
        coarsen_stats.maxAngle[curLevel] = tmp_stats[1];
      }
#else
      SETERRQ(PETSC_ERR_SUP, "No mesh generator available.");
#endif
    }  //end of for over the number of coarsening levels.
    if (coarsen_stats.displayStats)coarsen_DisplayStats();
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
  bool IsPointInElement (Obj<ALE::Mesh> mesh, int dim, ALE::Mesh::real_section_type::patch_type cPatch, ALE::Mesh::point_type triangle, ALE::Mesh::point_type node) {
    Obj<ALE::Mesh::topology_type> topology = mesh->getTopology();
    Obj<ALE::Mesh::real_section_type> coords = mesh->getRealSection("coordinates");
    double v_coords[dim];
    PetscMemcpy(v_coords, coords->restrict(0, node), dim * sizeof(double));
    double p_coords[dim*(dim+1)]; //stores the points of the triangle
    //initial step: get the area of the triangle/tet using the parallelogram rule
    Obj<ALE::Mesh::coneArray> closure = ALE::Closure::closure(mesh, triangle);
    //if (closure->size() < 6) printf("ERROR! ERROR!\n");
    ALE::Mesh::sieve_type::coneArray::iterator c_iter = closure->begin();
    ALE::Mesh::sieve_type::coneArray::iterator c_iter_end = closure->end();
    int index = 0;
        //printf("%d: (%f, %f)\n", node, v_coords[0], v_coords[1]);
    while (c_iter != c_iter_end) {
      if (topology->getPatch(cPatch)->depth(*c_iter) == 0) {
        const double * tmpCoord = coords->restrict(0, *c_iter);
        for (int i = 0; i < dim; i++) {
          p_coords[index*dim + i] = tmpCoord[i];
        }
        //printf("%d: (%f, %f)\n", *c_iter, p_coords[index*dim], p_coords[index*dim + 1]);
        index++;
      }
      c_iter++;
    }
    //printf("found %d points on this triangle, the last being (%f, %f)\n", index, p_coords[4], p_coords[5]);
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
    if (t_area - area  > 0.00001*area) return false;
    return true;
  }
  bool CompatibleWithEdge(Obj<ALE::Mesh> mesh, int dim, ALE::Mesh::patch_type ePatch, ALE::Mesh::point_type edge, ALE::Mesh::point_type point, double region) {
    //If the point is within the region-sized area around the edge, then return false.
    Obj<ALE::Mesh::topology_type> topology = mesh->getTopology();
    Obj<ALE::Mesh::real_section_type> coords = mesh->getRealSection("coordinates");
    double e_coords[2*dim];
    double p_coords[dim];
    Obj<ALE::Mesh::sieve_type::coneSequence> endpoints = topology->getPatch(ePatch)->cone(edge);
    ALE::Mesh::sieve_type::coneSequence::iterator end_iter = endpoints->begin();
    ALE::Mesh::sieve_type::coneSequence::iterator end_iter_end = endpoints->end();
    int index = 0;
    while (end_iter != end_iter_end) {
      const double * tmpCoord = coords->restrict(0, *end_iter);
      PetscMemcpy(e_coords + dim*index*sizeof(double), tmpCoord, dim*sizeof(double));
      end_iter++;
      index++;
    }
    const double * tmpCoord = coords->restrict(0, point);
    PetscMemcpy(p_coords, tmpCoord, dim*sizeof(double));
    //we have the coordinates.  Now we must project the point on to the line and see if a) the projection is in the segment, and b) the residual is of greater magnitude than "region"
    double res_len = 0;
    double e_dot_e = 0;
    double p_dot_e = 0;
    for (int i = 0; i < dim; i++) { //put the first edge endpoint at (0, 0);
      p_coords[i] = p_coords[i] - e_coords[i];
      e_coords[dim + i] = e_coords[dim + i] - e_coords[i];
      e_dot_e += e_coords[dim + i] * e_coords[dim + i];
      p_dot_e += e_coords[dim + i] * p_coords[i];
    }
    double r = p_dot_e/e_dot_e;
    if (r > 1 || r < 0) return true; //it's outside of our juristiction.
    for (int i = 0; i < dim; i++) {
      double trm = p_coords[i] - r*e_coords[dim + i];
      res_len += trm*trm;
    }
    if (res_len < region*region) return false; //it's in the region surrounding the edge.
    return true;
  }
  double * ComputeAngles(Obj<ALE::Mesh> mesh, int dim, ALE::Mesh::patch_type patch) {
    //return the minimum and maximum angles for the given patch.
    Obj<ALE::Mesh::topology_type> topology = mesh->getTopology();
    Obj<ALE::Mesh::real_section_type> coords = mesh->getRealSection("coordinates");
    const Obj<ALE::Mesh::topology_type::label_sequence>& faces = topology->heightStratum(patch, dim - 2); //lets us do 3D
    ALE::Mesh::topology_type::label_sequence::iterator f_iter = faces->begin();
    ALE::Mesh::topology_type::label_sequence::iterator f_iter_end = faces->end();
    //printf("Faces: %d\n", faces->size());
    double * angle = new(double[2]);
     angle[0] = 6.28;
     angle[1] = 0.0;
    while (f_iter != f_iter_end) {
      //printf("Computing the angles for face %d\n", *f_iter);
      Obj<ALE::Mesh::coneArray> points = ALE::Closure::closure(mesh, *f_iter);
      ALE::Mesh::coneArray::iterator p_iter = points->begin();
      ALE::Mesh::coneArray::iterator p_iter_end = points->end();
      double point_coords[dim*3];
      int index = 0;
      while (p_iter != p_iter_end) {
        const double * tmpCoords;
        if(topology->getPatch(patch)->depth(*p_iter) == 0) {
          tmpCoords = coords->restrict(0, *p_iter);
          for (int i = 0; i < dim; i++) {
            point_coords[i + index*dim] = tmpCoords[i];
          }
          index++;
        }
        p_iter++;
      }
      //if (index != 3) printf("oops.");
      //got the points, now check the angles;
      double veca, vecb;
      for (int i = 0; i < 3; i++) {
        double norma = 0;
        double normb = 0;
        double dot = 0;
        for (int j = 0; j < dim; j++) {
          veca = (point_coords[((i+1)%3)*dim + j] - point_coords[i*dim + j]);
          vecb = (point_coords[((i+2)%3)*dim + j] - point_coords[i*dim + j]);
          norma += veca*veca;
          normb += vecb*vecb;
          dot += veca*vecb;
        }
        double tmpAngle = acos(dot/(sqrt(norma*normb)));
        if (tmpAngle > angle[1]) angle[1] = tmpAngle;
        if (tmpAngle < angle[0]) angle[0] = tmpAngle;
        //printf("%f\n", tmpAngle);
      }
      f_iter++;
    }
    return angle;
  }
} }
