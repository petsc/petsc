#define ALE_IndexBundle_cxx

#ifndef included_ALE_IndexBundle_hh
#include <IndexBundle.hh>
#endif

#include <stack>
#include <queue>

namespace ALE {

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::setAssemblyPolicy"
  IndexBundle&   IndexBundle::setAssemblyPolicy(BundleAssemblyPolicy policy){
    if((policy != ADDITION) && (policy != INSERTION)) {
      throw(Exception("Invalid BundleAssemblyPolicy value"));
    }
    this->_assemblyPolicy = policy;
    return *this;
  }//IndexBundle::setAssemblyPolicy()

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::setTopology"
  IndexBundle&   IndexBundle::setTopology(Obj<Sieve> topology){
    CHKCOMMS(*this, *topology.pointer());
    // we have to reset all other PreSieves/Stacks
    __reset(topology);
    return *this;
  }// IndexBundle::setTopology()

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::setFiberDimension"
  IndexBundle&   IndexBundle::setFiberDimension(Point element, int32_t d) {
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
  }//IndexBundle::setFiberDimension()
  
  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__setFiberDimensionByStratum"
  IndexBundle&   IndexBundle::__setFiberDimensionByStratum(int stratumType, int32_t stratumIndex, int32_t dim) {
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
    }
    for(Point_set::iterator s_itor = stratum->begin(); s_itor != stratum->end(); s_itor++) {
      this->setFiberDimension(*s_itor,dim);
    }
    return *this;
  }//IndexBundle::__setFiberDimensionByStratum()
  
  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getFiberDimension"
  int32_t   IndexBundle::getFiberDimension(Obj<Point_set> base) {
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

  }//IndexBundle::getFiberDimension()

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getBundleDimension"
  int32_t   IndexBundle::getBundleDimension(Obj<Point_set> base) {
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

  }//IndexBundle::getBundleDimension()

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__computeIndices"
  Obj<PreSieve>   IndexBundle::__computeIndices(Obj<Point_set> supports, Obj<Point_set> base, bool includeBoundary, Obj<Point_set> exclusion) {
    base  = this->__validateChain(base);
    supports = this->__validateChain(supports);
    // IMPROVE: we could make this subroutine consult cache, if base is singleton
    // Traverse the boundary of s -- the closure of s except s itself -- in a depth-first search and compute the indices for the 
    // boundary fibers.  For an already seen point ss, its offset is in seen[ss].prefix, or in 'off' for a newly seen point.
    // 'off' is updated during the calculation by accumulating the dimensions of the fibers over all newly encountered elements.
    // An element ss  that has been seen AND indexed has a nonzero seen[ss].index equal to its dimension.
    // If requested, we will cache the results in arrows to e.

    Obj<PreSieve> indices(new PreSieve(MPI_COMM_SELF));
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
        //   Also, this can break the numbering if we have given a very specific base and od not want other points numbered
        base = this->getTopology()->cone(ss);
        if (!exclusion.isNull()) {
          base->subtract(exclusion);
        }
      }// if stk is not empty
    }// while(1)     
    return indices;
  }//IndexBundle::__computeIndices()


  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__computeBoundaryIndices"
  Obj<PreSieve>      IndexBundle::__computeBoundaryIndices(Point  s, Point__Point& seen, int32_t& off) {

    Obj<PreSieve> indices(new PreSieve(MPI_COMM_SELF));
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
  }//IndexBundle::__computeBoundaryIndices()


  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getFiberInterval"
  Point   IndexBundle::getFiberInterval(Point support, Obj<Point_set> base) {
    base  = this->__validateChain(base);
    Obj<PreSieve> indices = this->__computeIndices(Point_set(support), base);
    Point interval;
    if(indices->cap().size() == 0) {
      interval = Point(-1,0);
    } else {
      interval = *(indices->cap().begin());
    }
    return interval;
  }//IndexBundle::getFiberInterval()


  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getFiberIndices"
  Obj<PreSieve>   IndexBundle::getFiberIndices(Obj<Point_set> supports, Obj<Point_set> base, Obj<Point_set> exclusion) {
    base  = this->__validateChain(base);
    supports = this->__validateChain(supports);
    Obj<PreSieve> indices = this->__computeIndices(supports, base, false, exclusion);
    return indices;
  }//IndexBundle::getFiberIndices()


  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getClosureIndices"
  Obj<PreSieve>   IndexBundle::getClosureIndices(Obj<Point_set> supports, Obj<Point_set> base) {
    supports = this->__validateChain(supports);
    base  = this->__validateChain(base);
    bool includingBoundary = 1;
    Obj<PreSieve> indices = this->__computeIndices(supports, base, includingBoundary);
    return indices;
  }//IndexBundle::getClosureIndices()

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__getMaxDepthElement"
  ALE::Point IndexBundle::__getMaxDepthElement(Obj<Point_set> points) {
    ALE::Point max(-1, 0);
    int32_t    maxDepth = 0;

    for(ALE::Point_set::iterator e_itor = points->begin(); e_itor != points->end(); e_itor++) {
      int32_t depth = this->getTopology()->depth(*e_itor);

      if (depth > maxDepth) {
        maxDepth = depth;
        max = *e_itor;
      }
    }
    return max;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__checkOrderChain"
  ALE::int__Point IndexBundle::__checkOrderChain(Obj<Point_set> order, int& minDepth, int& maxDepth) {
    ALE::int__Point dElement;
    minDepth = 0;
    maxDepth = 0;

    // A topology cell-tuple contains one element per dimension, so we order the points by depth.
    for(Point_set::iterator ord_itor = order->begin(); ord_itor != order->end(); ord_itor++) {
      Point e = *ord_itor;
      int32_t depth = this->getTopology()->depth(e);

      if (depth < 0) {
        throw Exception("Invalid element: negative depth returned"); 
      }
      if (depth > maxDepth) {
        maxDepth = depth;
      }
      if (depth < minDepth) {
        minDepth = depth;
      }
      dElement[depth] = e;
    }
    // Verify that the chain is a "baricentric chain", i.e. it starts at depth 0
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
        if(!this->getTopology()->coneContains(dElement[d], dElement[d-1])){
          ostringstream ex;
          ex << "[" << this->getCommRank() << "]: ";
          ex << "point (" << dElement[d-1].prefix << ", " << dElement[d-1].index << ") at depth " << d-1 << " not in the cone of ";
          ex << "point (" << dElement[d].prefix << ", " << dElement[d].index << ") at depth " << d;
          throw Exception(ex.str().c_str());
        }
      }
    }
    return dElement;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__orderElement"
  void IndexBundle::__orderElement(int dim, ALE::Point element, std::map<int, std::queue<Point> > *ordered, ALE::Obj<ALE::Point_set> elementsOrdered) {
    if (elementsOrdered->find(element) != elementsOrdered->end()) return;
    (*ordered)[dim].push(element);
    elementsOrdered->insert(element);
    printf("  ordered element (%d, %d) dim %d\n", element.prefix, element.index, dim);
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__orderCell"
  ALE::Point IndexBundle::__orderCell(int dim, int__Point *orderChain, std::map<int, std::queue<Point> > *ordered, ALE::Obj<ALE::Point_set> elementsOrdered) {
    Obj<Sieve> closure = this->getTopology()->closureSieve(Point_set((*orderChain)[dim]));
    ALE::Point last;

    printf("Ordering cell (%d, %d) dim %d\n", (*orderChain)[dim].prefix, (*orderChain)[dim].index, dim);
    for(int d = 0; d < dim; d++) {
      printf("  orderChain[%d] (%d, %d)\n", d, (*orderChain)[d].prefix, (*orderChain)[d].index);
    }
    if (dim == 1) {
      Obj<Point_set> flip = closure->cone((*orderChain)[1]);

      this->__orderElement(0, (*orderChain)[0], ordered, elementsOrdered);
      flip->erase((*orderChain)[0]);
      last = *flip->begin();
      this->__orderElement(0, last, ordered, elementsOrdered);
      this->__orderElement(1, (*orderChain)[1], ordered, elementsOrdered);
      (*orderChain)[dim-1] = last;
      return last;
    }
    do {
      last = this->__orderCell(dim-1, orderChain, ordered, elementsOrdered);
      printf("    last (%d, %d)\n", last.prefix, last.index);
      Obj<Point_set> faces = closure->support(last);
      faces->erase((*orderChain)[dim-1]);
      if (faces->size() != 1) {
        throw Exception("Last edge did not separate two faces");
      }
      last = (*orderChain)[dim-1];
      (*orderChain)[dim-1] = *faces->begin();
    } while(elementsOrdered->find((*orderChain)[dim-1]) == elementsOrdered->end());
    printf("Finish ordering for cell (%d, %d)\n", (*orderChain)[dim].prefix, (*orderChain)[dim].index);
    printf("  with last (%d, %d)\n", last.prefix, last.index);
    (*orderChain)[dim-1] = last;
    this->__orderElement(dim, (*orderChain)[dim], ordered, elementsOrdered);
    return last;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getOrderedIndices"
  Obj<Point_array>   IndexBundle::getOrderedIndices(Obj<Point_set> order, Obj<PreSieve> indices) {
    // IMPROVE:  the ordering algorithm employed here works only for 'MeshSieves' --
    //           Sieves with the property that any pair of elements of depth d > 0 share at most
    //           one element of depth d-1.  'MeshSieve' would check that this property is preserved
    //           under arrow additions.
    //           We should define class 'MeshBundle' that would take 'MeshSieves' as topology and
    //           move this method there.
    int minDepth, maxDepth;
    ALE::int__Point dElement = this->__checkOrderChain(order, minDepth, maxDepth);
    // We store the elements ordered in each dimension
    std::map<int, std::queue<Point> > ordered;

    // Order elements in the closure
    printf("Ordering (%d, %d)\n", dElement[maxDepth].prefix, dElement[maxDepth].index);
    if (maxDepth == 0) {
      ordered[0].push(dElement[0]);
    } else if (maxDepth == 1) {
      // FIX: I think this is not necessary
      ALE::Point     face = dElement[1];
      Obj<Sieve>  closure = this->getTopology()->closureSieve(Point_set(face));
      Obj<Point_set> flip = closure->cone(face);

      ordered[0].push(dElement[0]);
      flip->erase(dElement[0]);
      ordered[0].push(*flip->begin());
      ordered[1].push(dElement[1]);
    } else {
      ALE::Point_set elementsOrdered;

      ALE::Point last = this->__orderCell(maxDepth, &dElement, &ordered, elementsOrdered);
    }

    // Generate indices from ordered elements
    Obj<Point_array> indexArray(new Point_array);
    for(int d = 0; d <= maxDepth; d++) {
      while(!ordered[d].empty()) {
        Obj<Point_set> indCone = indices->cone(ordered[d].front());

        ordered[d].pop();
        printf("  indices (%d, %d)\n", indCone->begin()->prefix, indCone->begin()->index);
        if (indCone->begin()->index > 0) {
          indexArray->push_back(*indCone->begin());
        }
      }
    }
    return indexArray;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::computeOverlapIndices"
  void   IndexBundle::computeOverlapIndices() {
    this->getTopology()->view("Topology");
    this->_overlapOwnership = this->getTopology()->baseFootprint(PreSieve::completionTypePoint, PreSieve::footprintTypeCone, NULL)->left();
    if (this->verbosity > 10) {this->_overlapOwnership->view("Overlap ownership");}
    this->_localOverlapIndices->setBottom(this->_overlapOwnership);
    this->_localOverlapIndices->setTop(new PreSieve(this->getComm()));
    // Traverse the points in the overlapOwnership base, compute the local indices over it and attach them using _localOverlapIndices
    ALE::Obj<ALE::Point_set> base = this->_overlapOwnership->base();
    for(Point_set::iterator o_itor = base->begin(); o_itor != base->end(); o_itor++) {
      Point e = *o_itor;
      Point interval = this->getFiberInterval(e);
      this->_localOverlapIndices->top()->addCapPoint(interval);
      this->_localOverlapIndices->addCone(interval, e);
    }
    if (this->verbosity > 10) {this->_localOverlapIndices->view("Local overlap indices");}
    // Now we do the completion
    this->_localOverlapIndices->coneCompletion(PreSieve::completionTypeArrow, PreSieve::footprintTypeCone, this->_remoteOverlapIndices);
    if (this->verbosity > 10) {this->_remoteOverlapIndices->view("Remote overlap indices");}
  }//IndexBundle::computeOverlapIndices()

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getOverlapSize"
  int32_t   IndexBundle::getOverlapSize() {
    return getFiberDimension(this->_overlapOwnership->base());
  }
  
  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getOverlapOwners"
  Obj<Point_set>   IndexBundle::getOverlapOwners(Point e) {
    return this->_overlapOwnership->cone(e);
  }//IndexBundle::getOverlapOwners()

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getOverlapFiberIndices"
  Obj<PreSieve>   IndexBundle::getOverlapFiberIndices(Obj<Point_set> supports, int32_t rank) {
    supports = this->__validateChain(supports);
    Obj<PreSieve> indices(new PreSieve(this->getComm()));
    if (rank == this->commRank) {
      for(ALE::Point_set::iterator e_itor = supports->begin(); e_itor != supports->end(); e_itor++) {
        ALE::Point e = *e_itor;
        //ALE::Obj<ALE::Point_set> ind = this->_localOverlapIndices->cone(e);
        ALE::Point_set ind = this->_localOverlapIndices->cone(e);

        indices->addCone(ind, e);
      }
    } else {
      throw Exception("Not supported: Remote overlap indices not done"); 
    }
    return indices;
  }//IndexBundle::getOverlapFiberIndices()

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getOverlapClosureIndices"
  Obj<PreSieve>   IndexBundle::getOverlapClosureIndices(Obj<Point_set> supports, int32_t rank) {
    supports = this->__validateChain(supports);
    Obj<PreSieve> indices(new PreSieve(this->getComm()));

    if (rank == this->commRank) {
      Obj<Point_set> points = this->getTopology()->closure(supports);

      for(ALE::Point_set::iterator e_itor = points->begin(); e_itor != points->end(); e_itor++) {
        ALE::Point e = *e_itor;
        //ALE::Obj<ALE::Point_set> ind = this->_localOverlapIndices->cone(e);
        ALE::Point_set ind = this->_localOverlapIndices->cone(e);

        indices->addCone(ind, e);
      }
    } else {
      throw Exception("Not supported: Remote overlap indices not done"); 
    }
    return indices;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__getPointType"
  ALE::PointType   IndexBundle::__getPointType(ALE::Point point) {
    ALE::Obj<ALE::Point_set> owners = getOverlapOwners(point);
    ALE::PointType pointType = localPoint;

    if (owners->size()) {
      for(ALE::Point_set::iterator o_itor = owners->begin(); o_itor != owners->end(); o_itor++) {
        if ((*o_itor).index < this->commRank) {
          pointType = rentedPoint;
          break;
        }
      }
      if (pointType == localPoint) {
        pointType = leasedPoint;
      }
    }
    return pointType;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__computePointTypes"
  /*
    This PreSieve classifies points in the topology. Marker points are in the base and topological points are in the cap.

      (rank, PointType)   is covered by   topological points of that type
      (   p, p)           is covered by   topological points leased from process p
      (-p-1, p)           is covered by   topological points rented to process p
  */
  ALE::Obj<ALE::PreSieve>   IndexBundle::__computePointTypes() {
    ALE::Obj<ALE::Point_set> space = this->getTopology()->space();
    ALE::Obj<ALE::PreSieve> pointTypes(new PreSieve(this->comm));

    for(ALE::Point_set::iterator e_itor = space->begin(); e_itor != space->end(); e_itor++) {
      ALE::Obj<ALE::Point_set> owners = getOverlapOwners(*e_itor);
      ALE::PointType pointType = localPoint;
      ALE::Point point = *e_itor;
      ALE::Point typePoint;

      if (owners->size()) {
        for(ALE::Point_set::iterator o_itor = owners->begin(); o_itor != owners->end(); o_itor++) {
          if ((*o_itor).index < this->commRank) {
            typePoint = ALE::Point(-(*o_itor).prefix-1, (*o_itor).index);

            pointType = rentedPoint;
            pointTypes->addSupport(point, typePoint);
            break;
          }
        }
        if (pointType == localPoint) {
          pointType = leasedPoint;
          pointTypes->addSupport(point, owners);
        }
      }
      typePoint = ALE::Point(this->commRank, pointType);
      pointTypes->addCone(point, typePoint);
    }
    this->_pointTypes = pointTypes;
    if (this->verbosity > 10) {pointTypes->view("Point types");}
    return pointTypes;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getPointTypes"
  ALE::Obj<ALE::PreSieve>   IndexBundle::getPointTypes() {
    return this->_pointTypes;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::computeGlobalIndices"
  void   IndexBundle::computeGlobalIndices() {
    // Make local indices
    ALE::Point_set localTypes;
    localTypes.insert(ALE::Point(this->commRank, ALE::localPoint));
    localTypes.insert(ALE::Point(this->commRank, ALE::leasedPoint));
    ALE::Obj<ALE::PreSieve> pointTypes = this->__computePointTypes();
    ALE::Obj<ALE::Point_set> localPoints = pointTypes->cone(localTypes);
    ALE::Obj<ALE::Point_set> rentedPoints = pointTypes->cone(ALE::Point(this->commRank, ALE::rentedPoint));
    if (this->verbosity > 10) {localPoints->view("Global local points");}
    ALE::Obj<ALE::PreSieve> localIndices = this->getFiberIndices(localPoints, localPoints, rentedPoints);
    if (this->verbosity > 10) {localIndices->view("Global local indices");}
    int localSize = this->getFiberDimension(localPoints);
    if (this->verbosity > 10) {
      PetscSynchronizedPrintf(this->comm, "[%d]Local size %d\n", this->commRank, localSize);
      PetscSynchronizedFlush(this->comm);
    }

    // Calculate global size
    ALE::Obj<ALE::PreSieve> globalIndices(new PreSieve(this->comm));
    int *firstIndex = new int[this->commSize+1];
    int ierr = MPI_Allgather(&localSize, 1, MPI_INT, &(firstIndex[1]), 1, MPI_INT, this->comm);
    CHKMPIERROR(ierr, ERRORMSG("Error in MPI_Allgather"));
    firstIndex[0] = 0;
    for(int p = 0; p < this->commSize; p++) {
      firstIndex[p+1] = firstIndex[p+1] + firstIndex[p];
    }

    // Add rented points
    for(ALE::Point_set::iterator r_itor = rentedPoints->begin(); r_itor != rentedPoints->end(); r_itor++) {
      ALE::Point p = *r_itor;
      int32_t dim = this->getFiberDimension(p);
      ALE::Point interval(localSize, dim);

      localIndices->addCone(interval, p);
      localSize += dim;
    }

    // Make global indices
    for(ALE::Point_set::iterator e_itor = localPoints->begin(); e_itor != localPoints->end(); e_itor++) {
      ALE::Obj<ALE::Point_set> cone = localIndices->cone(*e_itor);
      ALE::Point globalIndex(-1, 0);
      ALE::Point point = *e_itor;

      if (cone->size()) {
        globalIndex = *cone->begin();
        if (this->verbosity > 10) {
          PetscSynchronizedPrintf(this->comm, "[%d]   local interval (%d, %d) for point (%d, %d)\n", this->commRank, globalIndex.prefix, globalIndex.index, (*e_itor).prefix, (*e_itor).index);
        }
        globalIndex.prefix += firstIndex[this->commRank];
        if (this->verbosity > 10) {
          PetscSynchronizedPrintf(this->comm, "[%d]  global interval (%d, %d) for point (%d, %d)\n", this->commRank, globalIndex.prefix, globalIndex.index, (*e_itor).prefix, (*e_itor).index);
        }
      }
      globalIndices->addCone(globalIndex, point);
    }
    if (this->verbosity > 10) {
      PetscSynchronizedPrintf(this->comm, "[%d]Global size %d\n", this->commRank, firstIndex[this->commSize]);
      PetscSynchronizedFlush(this->comm);
    }
    delete firstIndex;

    // FIX: Communicate remote indices
    MPI_Request *requests = new MPI_Request[this->commSize];
    MPI_Status *statuses = new MPI_Status[this->commSize];
    int **recvIntervals = new int *[this->commSize];
    for(int p = 0; p < this->commSize; p++) {
      int size;
      if (p == this->commRank) {
        size = 0;
      } else {
        ALE::Obj<ALE::Point_set> rentedPoints = pointTypes->cone(ALE::Point(-p-1, p));
        size = rentedPoints->size()*2;
      }
      recvIntervals[p] = new int[size+1];

      ierr = MPI_Irecv(recvIntervals[p], size, MPI_INT, p, 1, this->comm, &(requests[p]));
      CHKMPIERROR(ierr, ERRORMSG("Error in MPI_Irecv"));
      if (this->verbosity > 10) {PetscSynchronizedPrintf(this->comm, "[%d]  rented size %d for proc %d\n", this->commRank, size, p);}
    }
    for(int p = 0; p < this->commSize; p++) {
      if (p == this->commRank) {
        ierr = MPI_Send(&p, 0, MPI_INT, p, 1, this->comm);
        CHKMPIERROR(ierr, ERRORMSG("Error in MPI_Send"));
        continue;
      }
      ALE::Obj<ALE::Point_set> leasedPoints = pointTypes->cone(ALE::Point(p, p));
      int size = leasedPoints->size()*2;
      int *intervals = new int[size+1];
      int i = 0;

      for(ALE::Point_set::iterator e_itor = leasedPoints->begin(); e_itor != leasedPoints->end(); e_itor++) {
        ALE::Obj<ALE::Point_set> cone = globalIndices->cone(*e_itor);
        ALE::Point interval(-1, 0);

        if (cone->size()) {
          interval = *cone->begin();
        }

        intervals[i++] = interval.prefix;
        intervals[i++] = interval.index;
      }
      ierr = MPI_Send(intervals, size, MPI_INT, p, 1, this->comm);
      CHKMPIERROR(ierr, ERRORMSG("Error in MPI_Send"));
      delete intervals;
      if (this->verbosity > 10) {PetscSynchronizedPrintf(this->comm, "[%d]  leased size %d for proc %d\n", this->commRank, size, p);}
    }
    ierr = MPI_Waitall(this->commSize, requests, statuses); CHKMPIERROR(ierr, ERRORMSG("Error in MPI_Waitall"));
    for(int p = 0; p < this->commSize; p++) {
      if (p == this->commRank) {
        delete recvIntervals[p];
        continue;
      }
      ALE::Obj<ALE::Point_set> rentedPoints = pointTypes->cone(ALE::Point(-p-1, p));
      int i = 0;

      for(ALE::Point_set::iterator e_itor = rentedPoints->begin(); e_itor != rentedPoints->end(); e_itor++) {
        ALE::Point interval(recvIntervals[p][i], recvIntervals[p][i+1]);
        ALE::Point point = *e_itor;

        globalIndices->addCone(interval, point);
        if (this->verbosity > 10) {PetscSynchronizedPrintf(this->comm, "[%d]Set global indices of (%d, %d) to (%d, %d)\n", this->commRank, point.prefix, point.index, interval.prefix, interval.index);}
        i += 2;
      }
      delete recvIntervals[p];
    }
    if (this->verbosity > 10) {PetscSynchronizedFlush(this->comm);}
    delete requests;
    delete statuses;
    delete recvIntervals;
    this->_localIndices = localIndices;
    this->_globalIndices = globalIndices;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getLocalSize"
  int32_t   IndexBundle::getLocalSize() {
    ALE::Point_set localTypes;
    localTypes.insert(ALE::Point(this->commRank, ALE::localPoint));
    localTypes.insert(ALE::Point(this->commRank, ALE::leasedPoint));
    return getFiberDimension(this->_pointTypes->cone(localTypes));
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getGlobalSize"
  int32_t   IndexBundle::getGlobalSize() {
    int localSize = getLocalSize(), globalSize;
    int ierr = MPI_Allreduce(&localSize, &globalSize, 1, MPI_INT, MPI_SUM, this->comm);
    CHKMPIERROR(ierr, ERRORMSG("Error in MPI_Allreduce"));
    return globalSize;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getRemoteSize"
  int32_t   IndexBundle::getRemoteSize() {
    return getFiberDimension(this->_pointTypes->cone(ALE::Point(this->commRank, ALE::rentedPoint)));
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getGlobalIndices"
  ALE::Obj<ALE::PreSieve>   IndexBundle::getGlobalIndices() {
    return this->_globalIndices;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getLocalIndices"
  ALE::Obj<ALE::PreSieve>   IndexBundle::getLocalIndices() {
    return this->_localIndices;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getGlobalFiberInterval"
  Point   IndexBundle::getGlobalFiberInterval(Point support) {
    ALE::Obj<ALE::Point_set> indices = this->_globalIndices->cone(support);
    Point interval;

    if(indices->size() == 0) {
      interval = Point(-1,0);
    } else {
      interval = *(indices->begin());
    }
    return interval;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getGlobalFiberIndices"
  ALE::Obj<ALE::PreSieve>   IndexBundle::getGlobalFiberIndices(Obj<Point_set> supports) {
    supports = this->__validateChain(supports);
    Obj<PreSieve> indices(new PreSieve(MPI_COMM_SELF));

    for(ALE::Point_set::iterator e_itor = supports->begin(); e_itor != supports->end(); e_itor++) {
      ALE::Point point = *e_itor;
      ALE::Point_set cone = this->_globalIndices->cone(point);

      indices->addCone(cone, point);
    }
    return indices;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getLocalFiberIndices"
  ALE::Obj<ALE::PreSieve>   IndexBundle::getLocalFiberIndices(Obj<Point_set> supports) {
    supports = this->__validateChain(supports);
    Obj<PreSieve> indices(new PreSieve(MPI_COMM_SELF));

    for(ALE::Point_set::iterator e_itor = supports->begin(); e_itor != supports->end(); e_itor++) {
      ALE::Point point = *e_itor;
      ALE::Point_set cone = this->_localIndices->cone(point);

      indices->addCone(cone, point);
    }
    return indices;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getGlobalClosureIndices"
  ALE::Obj<ALE::PreSieve>   IndexBundle::getGlobalClosureIndices(Obj<Point_set> supports) {
    supports = this->__validateChain(supports);
    Obj<PreSieve> indices(new PreSieve(MPI_COMM_SELF));
    Obj<Point_set> points = this->getTopology()->closure(supports);

    for(ALE::Point_set::iterator e_itor = points->begin(); e_itor != points->end(); e_itor++) {
      ALE::Point point = *e_itor;
      ALE::Point_set cone = this->_globalIndices->cone(point);

      indices->addCone(cone, point);
    }
    return indices;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::getLocalClosureIndices"
  ALE::Obj<ALE::PreSieve>   IndexBundle::getLocalClosureIndices(Obj<Point_set> supports) {
    supports = this->__validateChain(supports);
    Obj<PreSieve> indices(new PreSieve(MPI_COMM_SELF));
    Obj<Point_set> points = this->getTopology()->closure(supports);

    for(ALE::Point_set::iterator e_itor = points->begin(); e_itor != points->end(); e_itor++) {
      ALE::Point point = *e_itor;
      ALE::Point_set cone = this->_localIndices->cone(point);

      indices->addCone(cone, point);
    }
    return indices;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__postIntervalRequests"
  void   IndexBundle::__postIntervalRequests(ALE::Obj<ALE::PreSieve> pointTypes, int__Point rentMarkers, MPI_Request *intervalRequests[], int **receivedIntervals[]) {
    MPI_Request   *requests = new MPI_Request[this->commSize];
    int          **recvIntervals = new int *[this->commSize];
    PetscErrorCode ierr;

    for(int p = 0; p < this->commSize; p++) {
      int size;
      if (p == this->commRank) {
        size = 0;
      } else {
        ALE::Obj<ALE::Point_set> rentedPoints = pointTypes->cone(rentMarkers[p]);
        size = rentedPoints->size()*2;
      }
      recvIntervals[p] = new int[size+1];

      ierr = MPI_Irecv(recvIntervals[p], size, MPI_INT, p, 1, this->comm, &(requests[p]));
      CHKMPIERROR(ierr, ERRORMSG("Error in MPI_Irecv"));
      if (this->verbosity > 10) {PetscSynchronizedPrintf(this->comm, "[%d]  rented size %d for proc %d\n", this->commRank, size, p);}
    }
    PetscSynchronizedFlush(this->comm);
    *intervalRequests = requests;
    *receivedIntervals = recvIntervals;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__sendIntervals"
  void   IndexBundle::__sendIntervals(ALE::Obj<ALE::PreSieve> pointTypes, int__Point leaseMarkers, ALE::Obj<ALE::PreSieve> indices) {
    PetscErrorCode ierr;

    for(int p = 0; p < this->commSize; p++) {
      if (p == this->commRank) {
        ierr = MPI_Send(&p, 0, MPI_INT, p, 1, this->comm);
        CHKMPIERROR(ierr, ERRORMSG("Error in MPI_Send"));
        continue;
      }
      ALE::Obj<ALE::Point_set> leasedPoints = pointTypes->cone(leaseMarkers[p]);
      int size = leasedPoints->size()*2;
      int *intervals = new int[size+1];
      int i = 0;

      for(ALE::Point_set::iterator e_itor = leasedPoints->begin(); e_itor != leasedPoints->end(); e_itor++) {
        ALE::Obj<ALE::Point_set> cone = indices->cone(*e_itor);
        ALE::Point interval(-1, 0);

        if (cone->size()) {
          interval = *cone->begin();
        }
        intervals[i++] = interval.prefix;
        intervals[i++] = interval.index;
      }
      ierr = MPI_Send(intervals, size, MPI_INT, p, 1, this->comm);
      CHKMPIERROR(ierr, ERRORMSG("Error in MPI_Send"));
      delete intervals;
      if (this->verbosity > 10) {PetscSynchronizedPrintf(this->comm, "[%d]  leased size %d for proc %d\n", this->commRank, size, p);}
    }
    PetscSynchronizedFlush(this->comm);
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__receiveIntervals"
  void   IndexBundle::__receiveIntervals(ALE::Obj<ALE::PreSieve> pointTypes, int__Point rentMarkers, MPI_Request *requests, int *recvIntervals[], ALE::Obj<ALE::PreSieve> indices) {
    MPI_Status    *statuses = new MPI_Status[this->commSize];
    PetscErrorCode ierr;

    ierr = MPI_Waitall(this->commSize, requests, statuses); CHKMPIERROR(ierr, ERRORMSG("Error in MPI_Waitall"));
    for(int p = 0; p < this->commSize; p++) {
      if (p == this->commRank) {
        delete recvIntervals[p];
        continue;
      }
      ALE::Obj<ALE::Point_set> rentedPoints = pointTypes->cone(rentMarkers[p]);
      int i = 0;

      for(ALE::Point_set::iterator e_itor = rentedPoints->begin(); e_itor != rentedPoints->end(); e_itor++) {
        ALE::Point interval(recvIntervals[p][i], recvIntervals[p][i+1]);
        ALE::Point point = *e_itor;

        indices->addCone(interval, point);
        if (this->verbosity > 10) {
          PetscSynchronizedPrintf(this->comm, "[%d]Set indices of (%d, %d) to (%d, %d)\n", this->commRank, point.prefix, point.index, interval.prefix, interval.index);
        }
        i += 2;
      }
      delete recvIntervals[p];
    }
    PetscSynchronizedFlush(this->comm);
    delete requests;
    delete statuses;
    delete recvIntervals;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::computeMappingIndices"
  /*
    Creates a mapping (scatter) from this bundle to the target bundle

    This currently uses global indices to interface with VecScatter, but could
    use another communication structure with purely local indexing.
  */
  ALE::Obj<ALE::Stack>   IndexBundle::computeMappingIndices(ALE::Obj<ALE::PreSieve> pointTypes, ALE::Obj<ALE::IndexBundle> target) {
    if (this->verbosity > 10) {pointTypes->view("Mapping point types");}
    ALE::Obj<ALE::Stack> stack(new ALE::Stack(this->comm));

    // Make global source indices (local + leased)
    ALE::Point_set sourceTypes;
    sourceTypes.insert(ALE::Point(this->commRank, ALE::localPoint));
    sourceTypes.insert(ALE::Point(this->commRank, ALE::leasedPoint));
    ALE::Obj<ALE::Point_set> sourcePoints = pointTypes->cone(sourceTypes);
    // Need to implement copy(), then use restrictBase()
    ALE::Obj<ALE::PreSieve> sourceIndices(new ALE::PreSieve(this->comm));
    for(ALE::Point_set::iterator e_itor = sourcePoints->begin(); e_itor != sourcePoints->end(); e_itor++) {
      ALE::Point point = *e_itor;
      ALE::Point_set cone = this->_globalIndices->cone(*e_itor);
      sourceIndices->addCone(cone, point);
    }
    int sourceSize = this->getFiberDimension(sourcePoints);
    if (this->verbosity > 10) {
      PetscSynchronizedPrintf(this->comm, "[%d]Source size %d\n", this->commRank, sourceSize);
      PetscSynchronizedFlush(this->comm);
    }

    // Make initial global target indices (local)
    ALE::Point_set targetTypes;
    targetTypes.insert(ALE::Point(this->commRank, ALE::localPoint));
    ALE::Obj<ALE::Point_set> targetPoints = pointTypes->cone(targetTypes);
    // Need to implement copy(), then use restrictBase()
    ALE::Obj<ALE::PreSieve> targetGlobalIndices = target->getGlobalIndices();
    ALE::Obj<ALE::PreSieve> targetIndices(new ALE::PreSieve(this->comm));
    for(ALE::Point_set::iterator e_itor = targetPoints->begin(); e_itor != targetPoints->end(); e_itor++) {
      ALE::Point point = *e_itor;
      ALE::Point_set cone = targetGlobalIndices->cone(*e_itor);
      targetIndices->addCone(cone, point);
    }

    int__Point rentMarkers, leaseMarkers;
    MPI_Request *requests;
    int **recvIntervals;

    for(int p = 0; p < this->commSize; p++) {
      rentMarkers[p] = ALE::Point(-p-1, p);
      leaseMarkers[p] = ALE::Point(p, p);
    }
    // Send leased stuff from source to target, Accept rented stuff at target from source
    this->__postIntervalRequests(pointTypes, leaseMarkers, &requests, &recvIntervals);
    this->__sendIntervals(pointTypes, rentMarkers, targetGlobalIndices);
    this->__receiveIntervals(pointTypes, leaseMarkers, requests, recvIntervals, targetIndices);

    if (this->verbosity > 10) {
      sourceIndices->view("Source indices");
      targetIndices->view("Target indices");
    }
    stack->setTop(sourceIndices);
    stack->setBottom(targetIndices);
    return stack;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__getArrow"
  Point    IndexBundle::__getArrow(Point e1, Point e) {
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
    
  }// IndexBundle::__getArrow()


  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__getArrowInterval"
  Point    IndexBundle::__getArrowInterval(Point e1, Point e) {
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
  }// IndexBundle::__getArrowInterval()

  #undef  __FUNCT__
  #define __FUNCT__ "IndexBundle::__setArrowInterval"
  void IndexBundle::__setArrowInterval(Point e1, Point e, Point interval) {
    // If the bundle is dirty, we reset all the indices first
    if(!this->__isClean()) {
      this->__resetArrowIndices();
      this->__markClean();
    }
    // First we retrieve the arrow
    Point arrow = this->__getArrow(e1, e);
    // Now set the arrow index interval
    this->_indicesToArrows->setCone(interval, arrow);
  }// IndexBundle::__setArrowInterval()

  #undef __FUNCT__
  #define __FUNCT__ "IndexBundle::view"
  void IndexBundle::view(const char *name) {
    CHKCOMM(*this);    
    PetscErrorCode ierr;
    ostringstream txt, hdr;
    hdr << "Viewing ";
    if(name != NULL) {
      hdr << name;
    } 
    hdr << " IndexBundle\n";
    // Print header
    ierr = PetscPrintf(this->comm, hdr.str().c_str()); CHKERROR(ierr, "Error in PetscPrintf");
    
    this->_dimensionsToElements->view("Dimension Assignment Stack");

  }// IndexBundle::view()


} // namespace ALE

#undef ALE_IndexBundle_cxx
