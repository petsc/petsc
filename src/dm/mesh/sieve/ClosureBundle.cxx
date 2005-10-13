#define ALE_ClosureBundle_cxx

#ifndef included_ALE_ClosureBundle_hh
#include <ClosureBundle.hh>
#endif

#include <stack>
#include <queue>

namespace ALE {

  #undef  __FUNCT__
  #define __FUNCT__ "ClosureBundle::setAssemblyPolicy"
  ClosureBundle&   ClosureBundle::setAssemblyPolicy(BundleAssemblyPolicy policy){
    if((policy != ADDITION) && (policy != INSERTION)) {
      throw(Exception("Invalid BundleAssemblyPolicy value"));
    }
    this->_assemblyPolicy = policy;
    return *this;
  }//ClosureBundle::setAssemblyPolicy()

  #undef  __FUNCT__
  #define __FUNCT__ "ClosureBundle::setTopology"
  ClosureBundle&   ClosureBundle::setTopology(Obj<Sieve> topology){
    CHKCOMMS(*this, *topology.pointer());
    // we have to reset all other PreSieves/Stacks
    __reset(topology);
    return *this;
  }// ClosureBundle::setTopology()

  #undef  __FUNCT__
  #define __FUNCT__ "ClosureBundle::setFiberDimension"
  ClosureBundle&   ClosureBundle::setFiberDimension(Point element, int32_t d) {
    // Setting a fiber dimension makes the bundle dirty
    if(d < 0) {
      throw Exception("Invalid dimension");
    }
    if(this->getTopology()->spaceContains(element)) {
      Point dim(-1, d);
      // add a point with the dimension in the index to the dimensions PreSieve
      this->_dimensionsToElements->top()->addPoint(dim);
      // add a single arrow from dim to the element in the dimensionsToElements Stack
      this->_dimensionsToElements->setCone(dim, element);
      this->__markDirty();
      return *this;
    }
    else {
      throw ALE::Exception("Non-existent element");
    }
  }//ClosureBundle::setFiberDimension()
  
  #undef  __FUNCT__
  #define __FUNCT__ "ClosureBundle::__setFiberDimensionByStratum"
  ClosureBundle&   ClosureBundle::__setFiberDimensionByStratum(int stratumType, int32_t stratumIndex, int32_t dim) {
    Obj<Point_set> stratum;
    Obj<Sieve> topology = this->getTopology();
    switch(stratumType) {
    case stratumTypeDepth:
      stratum = topology->depthStratum(stratumIndex);
      break;
    case stratumTypeHeight:
      stratum = topology->heightStratum(stratumIndex);
      break;
    default:
      throw ALE::Exception("Unknown stratum type");
      break;
    }
    for(Point_set::iterator s_itor = stratum->begin(); s_itor != stratum->end(); s_itor++) {
      this->setFiberDimension(*s_itor,dim);
    }
    return *this;
  }//ClosureBundle::__setFiberDimensionByStratum()
  
  #undef  __FUNCT__
  #define __FUNCT__ "ClosureBundle::getFiberDimension"
  int32_t   ClosureBundle::getFiberDimension(Obj<Point_set> base) {
    int32_t dim = 0;
    base = this->__validateChain(base);
    for(Point_set::iterator base_itor = base->begin(); base_itor != base->end(); base_itor++) {
      Point e = *base_itor;
      Point_set dims = this->_dimensionsToElements->cone(e);
      // make sure there is only one dim element
      if(dims.size() > 1) {
        ostringstream ex;
        ex << "Multiple dimension designators found for element (" << e.prefix << ", " << e.index << ")";
        throw(Exception(ex.str().c_str()));
      }
      // no dimension designators means 'dimension zero'
      if(dims.size() != 0) {
        // The actual dimension is stored in the point's index
        Point dimPoint = *dims.begin();
        dim += dimPoint.index;
      }
    }// for(Point_set::iterator base_itor = base->begin(); base_itor != base->end(); base++) {
    return dim;

  }//ClosureBundle::getFiberDimension()

  #undef  __FUNCT__
  #define __FUNCT__ "ClosureBundle::getBundleDimension"
  int32_t   ClosureBundle::getBundleDimension(Obj<Point_set> base) {
    int32_t dim = 0;
    base = this->__validateChain(base);
    // Traverse the closure of base and add up fiber dimensions
    while(base->size() > 0){
      for(Point_set::iterator base_itor = base->begin(); base_itor != base->end(); base_itor++) {
        Point p = *base_itor;
        dim += this->getFiberDimension(p);
      }
      base = this->getTopology()->cone(base);
    }
    return dim;

  }//ClosureBundle::getBundleDimension()
  

  #undef  __FUNCT__
  #define __FUNCT__ "ClosureBundle::getFiberIndices"
  Obj<PreSieve>   ClosureBundle::getFiberIndices(Obj<Point_set> supports, Obj<Point_set> base) {
    base  = this->__validateChain(base);
    supports = this->__validateChain(supports);
    Obj<PreSieve> indices = this->__computeIndices(supports, base);
    return indices;
  }//ClosureBundle::getFiberIndices()


  #undef  __FUNCT__
  #define __FUNCT__ "ClosureBundle::getBundleIndices"
  Obj<PreSieve>   ClosureBundle::getBundleIndices(Obj<Point_set> supports, Obj<Point_set> base) {
    supports = this->__validateChain(supports);
    base  = this->__validateChain(base);
    bool includingBoundary = 1;
    Obj<PreSieve> indices = this->__computeIndices(supports, base, includingBoundary);
    return indices;
  }//ClosureBundle::getBundleIndices()
  

  #undef  __FUNCT__
  #define __FUNCT__ "ClosureBundle::getClosureIndices"
  Obj<Point_array>   ClosureBundle::getClosureIndices(Obj<Point_set> order, Obj<Point_set> base) {
    // IMPROVE:  the ordering algorithm employed here works only for 'MeshSieves' --
    //           Sieves with the property that any pair of elements of depth d > 0 share at most
    //           one element of depth d-1.  'MeshSieve' would check that this property is preserved
    //           under arrow additions.
    //           We should define class 'MeshBundle' that would take 'MeshSieves' as topology and
    //           move this method there.
    base  = this->__validateChain(base);
    // We expect that it contains a topology cell-tuple, so we order the points by depth.
    int__Point dElement;
    int32_t maxDepth = 0;
    int32_t minDepth = 0;
    for(Point_set::iterator ord_itor = order->begin(); ord_itor != order->end(); ord_itor++) {
      Point e = *ord_itor;
      int32_t depth = this->getTopology()->depth(e);
      if(depth < 0) {
        throw Exception("Invalid element: negative depth returned"); 
      }
      if(depth > maxDepth) {
        maxDepth = depth;
      }
      if(depth < minDepth) {
        minDepth = depth;
      }
      dElement[depth] = e;
    } //   for(Point_set::iterator ord_itor = order->begin(); ord_itor != order->end(); ord_itor++)

    // Verify that the chain is a "baricentric chain" 
    // Make sure it starts at depth 0
    if(minDepth != 0) {
      throw Exception("Invalid order chain: minimal depth is nonzero");
    }
    // Verify the chain has an element of every depth between 0 and maxDepth
    // and that each element at depth d is in the cone of the element at depth d+1
    for(int32_t d = 0; d <= maxDepth; d++) {
      int__Point::iterator d_itor = dElement.find(d);
      if(d_itor == dElement.end()){
        ostringstream ex;
        ex << "[" << this->getCommRank() << "]: ";
        ex << "Missing Point at depth " << d;
        throw Exception(ex.str().c_str());
      }
      if(d > 0) {
        if(!this->getTopology()->coneContains(dElement[d],dElement[d-1])){
          ostringstream ex;
          ex << "[" << this->getCommRank() << "]: ";
          ex << "point (" << dElement[d-1].prefix << ", " << dElement[d-1].index << ") at depth " << d-1 << " not in the cone of ";
          ex << "point (" << dElement[d].prefix << ", " << dElement[d].index << ") at depth " << d;
          throw Exception(ex.str().c_str());
        }
      }
    }// for(int32_t d = 0; d <= maxDepth; d++) {

    // Now we extract the subsieve which is the closure of the dElement[maxDepth]
    Obj<Sieve> closure = this->getTopology()->closureSieve(Point_set(dElement[maxDepth]));
    // Compute the bundle indices over the closure of dElement[maxDepth]
    Obj<PreSieve> indices = this->getBundleIndices(Point_set(dElement[maxDepth]), base);

    // Now we need to order 'indices' and store the cap in an array
    Obj<Point_array> indexArray(new Point_array);
      
    // We bootstrap the ordering process by ordering all vertices (depth 0) and storing them in a queue (ordered)
    Obj<std::queue<Point> > ordered(new std::queue<Point>);
    // Start with the queue oredered containing dElement[0]
    ordered->push(dElement[0]);
    Point e0, e1; // these are the "work" points -- a vertex and the latest pivot
    // The first work points are taken from dElement
    int32_t cntr = 0;    // keep track of oredered elements
    e0 = dElement[0];
    e1 = dElement[1];

    // Store e0's indices in the indexArray: there should be a single interval in indices->cone(e0)
    Point_set indCone = indices->cone(e0);
    if(indCone.size() > 0) {
      indexArray->push_back(*indCone.begin()); cntr++;
    }
    if (maxDepth == 0) {
      return indexArray;
    }
    bool zeroIsOrdered = 0; // we are done ordering when no new elements are yielded after pivoting
    while(!zeroIsOrdered) {
      // Pivot e0 around e1 (flip) within the closure Sieve
      Obj<Point_set> flip0 = closure->cone(e1);
      flip0->erase(e0);
      if(flip0->size() == 0) {
        ostringstream ex; 
        ex << "[" << this->getCommRank() << "]: Unable to flip down" << " (" << e0.prefix << "," << e0.index << ") ";
        ex << "around" << " (" << e1.prefix << "," << e1.index << "): empty flip set";
        throw Exception(ex.str().c_str());
      }
      // Take the first (assumed to be the only) of the remaining elements in the flip set as the next element
      e0 = *flip0->begin();
      // Check whether e0 is actually dElement[0], which means everything has been ordered.
      if(e0 == dElement[0]){
        zeroIsOrdered = 1;
        break;
      }
      // Add e0 to the queue
      ordered->push(e0);
      // Store e0's indices in the indexArray: there should be a single interval in indices->cone(e0)
      Point_set indCone = indices->cone(e0);
      if(indCone.size() > 0) {
        indexArray->push_back(*indCone.begin()); cntr++;
      }

      // Now flip e1 about e0
      Obj<Point_set> flip1 = closure->support(e0);
      flip1->erase(e1);
      if(flip1->size() == 0) {
        ostringstream ex; 
        ex << "[" << this->getCommRank() << "]: Unable to flip up" << " (" << e1.prefix << "," << e1.index << ") ";
        ex << "around" << " (" << e0.prefix << "," << e0.index << "): empty flip set";
        throw Exception(ex.str().c_str());
      }
      // Take the first of the remaining elements in the flip set and make it the new pivot
      e1 = *flip1->begin();
    }// while(!allOrdered)

    // Now order the non-zero depth elements
    // First we pop the queue, since the first 0-element is not use.  
    // This is only needed once after bootstrapping.
    ordered->pop();
    for(int32_t d = 1; d < maxDepth; d++) {
      e1 = dElement[d];
      // Store e1's indices in the indexArray: there should be a single interval in indices->cone(e1)
      Point_set indCone = indices->cone(e1);
      if(indCone.size() > 0) {
        indexArray->push_back(*indCone.begin()); cntr++;
      }

      // Enqueue e1 -- its popping will serve as the stopping criterion.
      ordered->push(e1);
      bool dIsOrdered = 0; // we are done ordering when no pivots exist.
      while(!dIsOrdered) {
        // Retrieve the next pivot
        e0 = ordered->front();
        ordered->pop();
        // Check whether e0 == dElement[d], which means we have exhausted all pivots.
        if(e0 == dElement[d]){
          dIsOrdered = 1;
          break;
        }
        // Pivot e1 around e0 (flip) within the closure Sieve
        Obj<Point_set> flip1 = closure->support(e0);
        flip1->erase(e1);
        if(flip1->size() == 0) {
          ostringstream ex; 
          ex << "[" << this->getCommRank() << "]: Unable to flip up" << " (" << e1.prefix << "," << e1.index << ") ";
          ex << "around" << " (" << e0.prefix << "," << e0.index << "): empty flip set";
          throw Exception(ex.str().c_str());
        }
        // Take the first (assumed to be the only) of the remaining elements in the flip set as the next element
        e1 = *flip1->begin();
        // Add e1 to the queue
        ordered->push(e1);
        // Store e1's indices in the indexArray: there should be a single interval in indices->cone(e1)
        Point_set indCone = indices->cone(e1);
        if(indCone.size() > 0) {
          indexArray->push_back(*indCone.begin()); cntr++;
        }
        
      }// while(!dIsOrdered)
    }// for(d = 1; d < maxDepth; d++) 
    // Now we have ordered all the elements in the closure except the one at maxDepth.
    // We add it explicitly
    e1 = dElement[maxDepth];
    // Store e1's indices in the indexArray: there should be a single interval in indices->cone(e1)
    indCone = indices->cone(e1);
    if(indCone.size() > 0) {
      indexArray->push_back(*indCone.begin()); cntr++;
    }

    return indexArray;
  }//ClosureBundle::getClosureIndices()
  

  #undef  __FUNCT__
  #define __FUNCT__ "ClosureBundle::__computeIndices"
  Obj<PreSieve>   ClosureBundle::__computeIndices(Obj<Point_set> supports, Obj<Point_set> base, bool includeBoundary) {
    base  = this->__validateChain(base);
    supports = this->__validateChain(supports);
    // IMPROVE: we could make this subroutine consult cache, if base is singleton
    // Traverse the boundary of s -- the closure of s except s itself -- in a depth-first search and compute the indices for the 
    // boundary fibers.  For an already seen point ss, its offset is in seen[ss].prefix, or in 'off' for a newly seen point.
    // 'off' is updated during the calculation by accumulating the dimensions of the fibers over all newly encountered elements.
    // An element ss  that has been seen AND indexed has a nonzero seen[ss].index equal to its dimension.
    // If requested, we will cache the results in arrows to e.

    Obj<PreSieve> indices(new PreSieve(this->getComm()));
    Point__Point   seen;
    // Traverse the closure of base in a depth-first search storing the offsets of each element ss we see during the search
    // in seen[ss].prefix.
    // Continue searching until we encounter s -- one of supports' points, lookup or compute s's offset and store its indices
    // in 'indices' as well as seen[ss].
    // If none of supports are not found in the closure of base, return an empty index set.

    // To enable the depth-first order traversal without recursion, we employ a stack.
    std::stack<Point> stk;
    int32_t off = 0;            // keeps track of the offset within the bundle 

    // We traverse the (sub)bundle over base until one of s is encountered
    while(1) { // use 'break' to exit the loop

      // First we stack base elements up on stk in the reverse order to ensure offsets 
      // are computed in the bundle order.
      for(Point_set::reverse_iterator base_ritor = base->rbegin(); base_ritor != base->rend(); base_ritor++) {
        stk.push(*base_ritor);
      }

      // If the stack is empty, we are done, albeit with a negative result.
      if(stk.empty()) { // if stk is empty
        break;
      }
      else { // if stk is not empty
        Point ss = stk.top(); stk.pop();
        int32_t ss_dim = this->getFiberDimension(ss);
        int32_t ss_off;
        // If ss has already been seen, we use the stored offset.
        if(seen.find(ss) != seen.end()){ // seen ss already
          ss_off = seen[ss].prefix;
        }
        else { // not seen ss yet
          // Compute ss_off
          ss_off = off;
          // Store ss_off, but ss_dim in 'seen'
          seen[ss] = Point(ss_off, 0);
          // off is only incremented when we encounter a not yet seen ss.
          off += ss_dim;
        }
        // ss in supports ?
        Point_set::iterator s_itor = supports->find(ss);
        if(s_itor != supports->end()) {
          Point s = *s_itor;
          // If s (aka ss) has a nonzero dimension but has not been indexed yet
          if((ss_dim > 0) && (seen[ss].index == 0)) {
            // store ss_off, ss_dim both in indices and seen[ss]
            Point ssPoint(ss_off, ss_dim);
            indices->addCone(ssPoint, ss);
            seen[ss] = Point(ss_off, ss_dim);
          }
          // If the boundary of s should be included in the calculation, add in the boundary indices
          if(includeBoundary) {
            Obj<PreSieve> boundary_indices = this->__computeBoundaryIndices(s, seen, off);
            indices->add(boundary_indices.object());
          }
        } // ss in supports
        // Regardless of whether ss is in supports or not, we need to index the elements in its closure to obtain a correct off.
        // Compute the cone over ss, which will replace ss on stk at the beginning of the next iteration.
        // IMPROVE: can probably reduce runtime by putting this line inside the 'if not seen' clause'.
        base = this->getTopology()->cone(ss);
                
      }// if stk is not empty
    }// while(1)     
    return indices;
  }//ClosureBundle::__computeIndices()


  #undef  __FUNCT__
  #define __FUNCT__ "ClosureBundle::__computeBoundaryIndices"
  Obj<PreSieve>      ClosureBundle::__computeBoundaryIndices(Point  s, Point__Point& seen, int32_t& off) {

    Obj<PreSieve> indices(new PreSieve(this->getComm()));
    // Traverse the boundary of s -- the closure of s except s itself -- in a depth-first search and compute the indices for the 
    // boundary fibers.  For an already seen point ss, its offset is in seen[ss].prefix, or in 'off' for a newly seen point.
    // 'off' is updated during the calculation by accumulating the dimensions of the fibers over all newly encountered elements.
    // An element ss  that has been seen AND indexed has a nonzero seen[ss].index equal to its dimension.
    // If requested, we will cache the results in arrows to e.


    // To enable the depth-first order traversal without recursion, we employ a stack.
    std::stack<Point> stk;
    // Initialize base with the cone over s
    Obj<Point_set> base = this->getTopology()->cone(s);
    while(1) { // use 'break' to exit the loop
      // First we stack base elements up on stk in the reverse order to ensure that offsets 
      // are computed in the bundle order.
      for(Point_set::reverse_iterator base_ritor = base->rbegin(); base_ritor != base->rend(); base_ritor++) {
        stk.push(*base_ritor);
      }
      // If the stack is empty, we are done.
      if(stk.empty()) { // if stk is empty
        break;
      }
      else { // if stk is not empty
        Point   ss = stk.top(); stk.pop();
        int32_t ss_dim = this->getFiberDimension(ss);
        int32_t ss_off;
        if(seen.find(ss) != seen.end()) { // already seen ss
          // use the stored offset
          ss_off = seen[ss].prefix;
          if(seen[ss].index == 0) {  // but not yet indexed ss
            seen[ss] = Point(ss_off, ss_dim);
            if(ss_dim > 0) {
              Point ssPoint = Point(ss_off, ss_dim);
              indices->addCone(ssPoint, ss);
            }
          } // not yet indexed ss
        } // already seen ss
        else // not seen ss yet
        {
          // use the computed offset
          ss_off = off;
          // Mark ss as already seen and store its offset and dimension
          seen[ss] = Point(ss_off, ss_dim);
          if(ss_dim > 0) {
            Point ssPoint(ss_off, ss_dim);
            indices->addCone(ssPoint, ss);
          }
          off += ss_dim;
        }    // not seen ss yet
        base = this->getTopology()->cone(ss);
      }// if stk is not empty
    }// while(1)     
    return indices;
  }//ClosureBundle::__computeBoundaryIndices()


  #undef  __FUNCT__
  #define __FUNCT__ "ClosureBundle::__getArrow"
  Point    ClosureBundle::__getArrow(Point e1, Point e) {
    Point_set arrows = this->__getArrows()->nMeet(this->_arrowsToStarts->cone(e1),this->_arrowsToEnds->cone(e),0);
    if(arrows.size() > 1) {
      throw(Exception("Multiple arrows attached to an element pair"));
    }
    Point arrow;
    if(arrows.size() == 0) {
      // We must add a new arrow.  How do we ensure it is indeed new?  
      // We insist of prefix == this->commRank; then all __getArrows()->cap() points are sorted by the index,
      // so we take the last + 1.
      arrow.prefix = this->commRank;
      arrow.index = (--(this->__getArrows()->cap().end()))->index + 1;
      this->__getArrows()->addPoint(arrow);
      this->_arrowsToStarts->addCone(arrow,e1);
      this->_arrowsToEnds->addCone(arrow,e);
    }
    else {// There already exists a unique arrow, so return it.
      arrow = *(arrows.begin());
    }
    return arrow;
    
  }// ClosureBundle::__getArrow()


  #undef  __FUNCT__
  #define __FUNCT__ "ClosureBundle::__getArrowInterval"
  Point    ClosureBundle::__getArrowInterval(Point e1, Point e) {
    // If the bundle is dirty, we reset all the indices first
    if(!this->__isClean()) {
      this->__resetArrowIndices();
      this->__markClean();
      return Point(-1,0);
    }
    // Retrieve the arrow between e1 and e
    Point arrow = this->__getArrow(e1,e);
    // Now retrieve the index set arrached to arrow
    Obj<Point_set> indices = this->_indicesToArrows->cone(arrow);
    if(indices->size() != 1) { // either nothing or too many things cached
      return Point(-1,0);
    }
    return *(indices->begin());
  }// ClosureBundle::__getArrowInterval()

  #undef  __FUNCT__
  #define __FUNCT__ "ClosureBundle::__setArrowInterval"
  void ClosureBundle::__setArrowInterval(Point e1, Point e, Point interval) {
    // If the bundle is dirty, we reset all the indices first
    if(!this->__isClean()) {
      this->__resetArrowIndices();
      this->__markClean();
    }
    // First we retrieve the arrow
    Point arrow = this->__getArrow(e1, e);
    // Now set the arrow index interval
    this->_indicesToArrows->setCone(interval, arrow);
  }// ClosureBundle::__setArrowInterval()

  #undef __FUNCT__
  #define __FUNCT__ "ClosureBundle::view"
  void ClosureBundle::view(const char *name) {
    CHKCOMM(*this);    
    MPI_Comm comm = this->comm;
    PetscErrorCode ierr;
    ostringstream txt, hdr;
    hdr << "Viewing ";
    if(name != NULL) {
      hdr << name;
    } 
    hdr << " ClosureBundle\n";
    // Print header
    ierr = PetscPrintf(comm, hdr.str().c_str()); CHKERROR(ierr, "Error in PetscPrintf");
    
    this->_dimensionsToElements->view("Dimension Assignment Stack");

  }// ClosureBundle::view()


} // namespace ALE

#undef ALE_ClosureBundle_cxx
