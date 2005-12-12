#define ALE_Sieve_cxx

#ifndef included_ALE_Sieve_hh
#include <Sieve.hh>
#endif

#include <stack>

namespace ALE {
  
  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::Sieve()"
  Sieve::Sieve() : PreSieve(),_additionPolicy(additionPolicyAcyclic), _stratificationPolicy(stratificationPolicyOnMutation)  {};

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::Sieve(MPI_Comm)"
  Sieve::Sieve(MPI_Comm comm) : PreSieve(comm), _additionPolicy(additionPolicyAcyclic), _stratificationPolicy(stratificationPolicyOnMutation) {};

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::~Sieve"
  Sieve::~Sieve(){};

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::setComm"
  void Sieve::setComm(MPI_Comm c) {
    Coaster::setComm(c);
  }// Coaster::setComm()


  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::clear"
  Sieve& Sieve::clear(){
    PreSieve::clear();
    this->_depth.clear();
    this->_height.clear();
    return *this;
  }// Sieve::clear()
  

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::getLock"
  Sieve&     Sieve::getLock(){
    CHKCOMM(*this);
    this->_lock++;
    PetscErrorCode ierr = MPI_Barrier(this->getComm()); CHKERROR(ierr, "Error in MPI_Barrier");
    if(this->_stratificationPolicy == stratificationPolicyOnLocking) {
      this->__computeClosureHeights(this->cone(this->_leaves));
      this->__computeStarDepths(this->support(this->_roots));
    }
    return *this;
  };
  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::setAdditionPolicy"
  Sieve& Sieve::setAdditionPolicy(Sieve::AdditionPolicy policy) {
    this->__checkLock();
    if( (policy != additionPolicyAcyclic) && (policy != additionPolicyStratified) ) {
      throw ALE::Exception("Unknown AdditionPolicy value");
    }
    this->_additionPolicy = policy;
    return *this;
  }// Sieve::setAdditionPolicy()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::getAdditionPolicy"
  Sieve::AdditionPolicy Sieve::getAdditionPolicy() {
    return this->_additionPolicy;
  }// Sieve::getAdditionPolicy()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::setStratificationPolicy"
  Sieve& Sieve::setStratificationPolicy(Sieve::StratificationPolicy policy) {
    this->__checkLock();
    if( (policy != stratificationPolicyOnLocking) && (policy != stratificationPolicyOnMutation) ) {
      throw ALE::Exception("Unknown StratificationPolicy value");
    }
    this->_stratificationPolicy = policy;
    return *this;
  }// Sieve::setStratificationPolicy()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::getStratificationPolicy"
  Sieve::StratificationPolicy Sieve::getStratificationPolicy() {
    return this->_stratificationPolicy;
  }// Sieve::getStratificationPolicy()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::addArrow"
  Sieve& Sieve::addArrow(const Point& i, const Point& j) {
    ALE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    this->__checkLock();
    // Check whether the arrow addition would violate the addition policy.
    // This can be done only if the heights and depths are up-to-date, which means 'stratificationPolycyOnMutation'.
    if(this->_stratificationPolicy == stratificationPolicyOnMutation) {      
      Point_set iSet, jSet, iClosure, jStar;
      int32_t iDepth, jDepth;
      ostringstream txt;
      switch(this->_additionPolicy) {
      case additionPolicyAcyclic:
        if (i == j) {
          ostringstream ex;
          ex << "[" << this->getCommRank() << "]: ";
          ex << "Attempted arrow insertion (" << i.prefix << ", " << i.index << ") --> (" << j.prefix << ", " << j.index << ") ";
          ex << "would lead to a cycle";
          //throw Exception("Attempted arrow insertion would lead to a cycle");
          throw Exception(ex.str().c_str());
        };
        //jSet.insert(j);
        //iSet.insert(i);
        
        jStar = this->star(j);
        iClosure = this->closure(i);
        if((jStar.find(i) != jStar.end()) || (iClosure.find(j) != iClosure.end())) {
          printf("Adding (%d, %d)-->(%d, %d)\n", i.prefix, i.index, j.prefix, j.index);
          if (jStar.find(i) != jStar.end()) {
            printf("Head found in tail star\n");
          }
          if (iClosure.find(j) != iClosure.end()) {
            printf("Tail found in head closure\n");
          }
          ostringstream ex;
          ex << "[" << this->getCommRank() << "]: ";
          ex << "Attempted arrow insertion (" << i.prefix << ", " << i.index << ") --> (" << j.prefix << ", " << j.index << ") ";
          ex << "would lead to a cycle";
          //throw Exception("Attempted arrow insertion would lead to a cycle");
          throw Exception(ex.str().c_str());
        }
        break;
      case additionPolicyStratified:
        iDepth = this->depth(i);
        jDepth = this->depth(j);
        // Recall that iDepth < 0 means i is not in the Sieve; likewise for jDepth
        if( (iDepth >= 0) && (jDepth >= 0) && (iDepth >= jDepth) ){
          ostringstream ex;
          ex << "[" << this->getCommRank() << "]: ";
          ex << "Attempted arrow insertion would violate stratification";
          //throw Exception("Attempted arrow insertion would violate stratification");
          throw Exception(ex.str().c_str());
        }
        break;
      default:
        ostringstream ex;
        ex << "[" << this->getCommRank() << "]: ";
        ex << "Unknown addition policy";
        throw Exception("Unknown addition policy");
        //throw Exception(ex.str().c_str());
      }// switch(this->_additionPolicy)
    }// if(this->_stratificationPolicy == stratificationPolicyOnMutation)

    // Now add the points
    this->addCapPoint(i);
    this->addBasePoint(j);
    
    // Finally, add the arrow
    PreSieve::addArrow(i,j); // need to use PreSieve::addArrow to make sure that roots and leaves are consistent

    // The heights and depths are computed now or later, depending on the stratificationPolicy
    if(this->_stratificationPolicy == stratificationPolicyOnMutation) {
      // Only points in the star of j may have gotten their depth changed by introducing a path up through  i
      if(this->depth(j) != (this->depth(i)+1)) {
        this->__computeStarDepths(j);
      }
      // Only points in the closure of i may have gotten their height changed by introducing a path down through j
      if(this->height(i) != (this->height(j) + 1)) {
        this->__computeClosureHeights(i);
      }
    }
    ALE_LOG_STAGE_END;    
    return *this;
  }// Sieve::addArrow()


  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::removeArrow"
  Sieve& Sieve::removeArrow(const Point& i, const Point& j, bool removeSingleton) {
    ALE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    this->__checkLock();
    // need to use PreSieve::removeArrow to make sure that roots and leaves are consistent
    PreSieve::removeArrow(i,j, removeSingleton);
    // now we may need to recompute the heights and depths
    if(this->_stratificationPolicy == stratificationPolicyOnMutation) {
      if (this->spaceContains(i)) {
        // Only the points in the closure of i may have gotten their height changed, now that the path down through j is unavailable
        __computeClosureHeights(i);
      }
      if (this->spaceContains(j)) {
        // Only the points in the star of j may have gotten their depth changed, now that the path up through i is unavailable
        __computeStarDepths(j);
      }
    }
    ALE_LOG_STAGE_END;
    return *this;
  }// Sieve::removeArrow()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::removeBasePoint"
  Sieve& Sieve::removeBasePoint(const Point& p, bool removeSingleton) {
    ALE_LOG_STAGE_BEGIN;
    this->__checkLock();
    ALE::PreSieve::removeBasePoint(p, removeSingleton);
    if (!this->capContains(p)) {
      this->__setDepth(p, -1);
      this->__setHeight(p, -1);
    }
    ALE_LOG_STAGE_END;
    return *this;
  }
  
  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::addBasePoint"
  Sieve& Sieve::addBasePoint(const Point& p) {
    ALE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    this->__checkLock();
    // If the point is absent from the Sieve, after insertion it will have zero height/depth
    if(!(this->spaceContains(p))) {
      this->__setDepth(p,0);
      this->__setHeight(p,0);
    }
    // We must use PreSieve methods to make sure roots and leaves are maintained consistently
    PreSieve::addBasePoint(p);
    ALE_LOG_STAGE_END;
    return *this;
  }// Sieve::addBasePoint()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::removeCapPoint"
  Sieve& Sieve::removeCapPoint(const Point& q, bool removeSingleton) {
    ALE_LOG_STAGE_BEGIN;
    this->__checkLock();
    ALE::PreSieve::removeCapPoint(q, removeSingleton);
    if (!this->baseContains(q)) {
      this->__setDepth(q, -1);
      this->__setHeight(q, -1);
    }
    ALE_LOG_STAGE_END;
    return *this;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::addCapPoint"
  Sieve& Sieve::addCapPoint(const Point& p) {
    ALE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    this->__checkLock();
    // If the point is absent from the Sieve, after insertion it will have zero height/depth
    if(!(this->spaceContains(p))) {
        this->__setDepth(p,0);
        this->__setHeight(p,0);
    }
    // We must use PreSieve methods to make sure roots and leaves are maintained consistently
    PreSieve::addCapPoint(p);
    ALE_LOG_STAGE_END;
    return *this;
  }// Sieve::addCapPoint()


  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::__setHeight"
  void Sieve::__setHeight(Point p, int32_t h){
    ALE_LOG_STAGE_BEGIN;
    this->_height[p] = h;
    ALE_LOG_STAGE_END;
  }// Sieve::__setHeight()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::__setDepth"
  void Sieve::__setDepth(Point p, int32_t d){
    ALE_LOG_STAGE_BEGIN;
    this->_depth[p] = d;
    ALE_LOG_STAGE_END;
  }// Sieve::__setDepth()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::__computeClosureHeights"
  void Sieve::__computeClosureHeights(Obj<Point_set> points) {
    ALE_LOG_STAGE_BEGIN;
    ALE_LOG_EVENT_BEGIN
    // points contains points for the current height computation;
    // mpoints keeps track of 'modified' points identified at the current stage, 
    // and through which recursion propagates
    Obj<Point_set> mpoints = Point_set();

    for(Point_set::iterator p_itor = points->begin(); p_itor != points->end(); p_itor++) {
      // retrieve the current height of p
      int32_t h0 = this->height(*p_itor);
      // compute the max height of the points in the support of p
      int32_t maxH = this->maxHeight(this->support(*p_itor));
      // the height is maxH + 1
      int32_t h1 = maxH + 1;
      // if h0 differs from h1, set the height of p to h1 and add p to mpoints -- its height has been modified
      if(h1 != h0) {
        this->__setHeight(*p_itor,h1);
        mpoints->insert(*p_itor);
      }
    }
    // if the mpoints set is not empty, we recursively call __computeClosureHeights on points = cone(mpoints)
    if(mpoints->size() > 0) {
      this->__computeClosureHeights(this->cone(mpoints));
    }
    ALE_LOG_EVENT_END
    ALE_LOG_STAGE_END;
  }//Sieve::__computeClosureHeights()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::__computeStarDepths"
  void Sieve::__computeStarDepths(Obj<Point_set> points) {
    ALE_LOG_STAGE_BEGIN;
    ALE_LOG_EVENT_BEGIN
    // points contains points for the current depth computation;
    // mpoints keeps track of 'modified' points identified at the current stage, 
    // and through which recursion propagates
    Obj<Point_set> mpoints = Point_set();

    for(Point_set::iterator p_itor = points->begin(); p_itor != points->end(); p_itor++) {
      // retrieve the current depth of p
      int32_t d0 = this->depth(*p_itor);
      // compute the max depth of the points in the cone over p
      int32_t maxD = this->maxDepth(this->cone(*p_itor));
      // the new depth is maxD + 1
      int32_t d1 = maxD + 1;
      // if d0 differs from d1, set the depth of p to d1 and add p to mpoints -- its depth has been modified
      if(d1 != d0) {
        this->__setDepth(*p_itor,d1);
        mpoints->insert(*p_itor);
      }
    }
    // if the mpoints set is not empty, we recursively call __computeStarDepths on points = support(mpoints)
    if(mpoints->size() > 0) {
      this->__computeStarDepths(this->support(mpoints));
    }
    ALE_LOG_EVENT_END
    ALE_LOG_STAGE_END;
  }//Sieve::__computeStarDepths()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::depth"
  int32_t Sieve::depth(const Point& p) {
    int32_t depth;

    ALE_LOG_STAGE_BEGIN;
    ALE_LOG_EVENT_BEGIN
    CHKCOMM(*this);
    if(this->_depth.find(p) != this->_depth.end()) {
      depth = this->_depth[p];
    } else {
      /* This accomdates Stacks, since spaceContains() can return true before the point is added to the Stack itself */
      depth =  -1;
    }
    ALE_LOG_EVENT_END
    ALE_LOG_STAGE_END;
    return depth;
  }// Sieve::depth()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::maxDepth"
  int32_t Sieve::maxDepth(Point_set& points) {
    int32_t max = -1;
    for(Point_set::iterator p_itor = points.begin(); p_itor != points.end(); p_itor++) {
      int32_t m = this->depth(*p_itor);
      if(m > max) {
        max = m;
      }
    }
    return max;
  }// Sieve:maxDepth()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::maxDepth"
  int32_t Sieve::maxDepth(Obj<Point_set> points) {
    int32_t max = -1;
    for(Point_set::iterator p_itor = points->begin(); p_itor != points->end(); p_itor++) {
      int32_t m = this->depth(*p_itor);
      if(m > max) {
        max = m;
      }
    }
    return max;
  }// Sieve:maxDepth()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::maxHeight"
  int32_t Sieve::maxHeight(Point_set& points) {
    int32_t max = -1;
    for(Point_set::iterator p_itor = points.begin(); p_itor != points.end(); p_itor++) {
      int32_t m = this->height(*p_itor);
      if(m > max) {
        max = m;
      }
    }
    return max;
  }// Sieve:maxHeight()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::maxHeight"
  int32_t Sieve::maxHeight(Obj<Point_set> points) {
    int32_t max = -1;
    for(Point_set::iterator p_itor = points->begin(); p_itor != points->end(); p_itor++) {
      int32_t m = this->height(*p_itor);
      if(m > max) {
        max = m;
      }
    }
    return max;
  }// Sieve:maxHeight()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::height"
  int32_t Sieve::height(Point p) {
    int32_t height;
    ALE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    if(this->_height.find(p) != this->_height.end()) {
      height = this->_height[p];
    } else {
      /* This accomdates Stacks, since spaceContains() can return true before the point is added to the Stack itself */
      height =  -1;
    }
    ALE_LOG_STAGE_END;
    return height;
  }// Sieve::height()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::diameter"
  int32_t Sieve::diameter(Point p) {
    int32_t diameter;
    ALE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    if(this->spaceContains(p)) {
      diameter = this->depth(p) + this->height(p);
    }
    else {
      diameter = -1;
    }
    ALE_LOG_STAGE_END;
    return diameter;
  }// Sieve::diameter(Point)

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::diameter"
  int32_t Sieve::diameter() {
    int globalDiameter;
    ALE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    int32_t diameter = 0;
    // IMPROVE: PreSieve::space() should return an iterator instead
    Point_set space = this->space();
    for(Point_set::iterator s_itor = space.begin(); s_itor != space.end(); s_itor++) {
      Point p = *s_itor;
      int32_t pDiameter = this->diameter(p);
      if(pDiameter > diameter) {
        diameter = pDiameter;
      }
    }
    int ierr = MPI_Allreduce(&diameter, &globalDiameter, 1, MPI_INT, MPI_MAX, this->comm);
    CHKMPIERROR(ierr, ERRORMSG("Error in MPI_Allreduce"));
    ALE_LOG_STAGE_END;
    return globalDiameter;
  }// Sieve::diameter()


  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::closure"
  Obj<Point_set> Sieve::closure(Obj<Point_set> chain) {
    Obj<Point_set> closure = Point_set();

    ALE_LOG_STAGE_BEGIN;
    ALE_LOG_EVENT_BEGIN
    CHKCOMM(*this);
    int32_t depth = this->maxDepth(chain);
    if(depth >= 0) {
      closure = this->nClosure(chain, depth);
    }
    ALE_LOG_EVENT_END
    ALE_LOG_STAGE_END;
    return closure;
  }// Sieve::closure()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::closureSieve"
  Obj<Sieve> Sieve::closureSieve(Obj<Point_set> chain, Obj<Sieve> closure) {
    ALE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    if(closure.isNull()) {
      closure = Obj<Sieve>(new Sieve(this->getComm()));
    }
    int32_t depth = this->maxDepth(chain.object());
    if(depth >= 0) {
      this->nClosurePreSieve(chain,depth,closure);
    }
    ALE_LOG_STAGE_END;
    return closure;
  }// Sieve::closureSieve()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::star"
  Point_set Sieve::star(Point_set& chain) {
    Point_set star;
    ALE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    int32_t height = this->maxHeight(chain);
    if(height >= 0) {
      star = this->nStar(chain,height);
    }
    ALE_LOG_STAGE_END;
    return star;
  }// Sieve::star()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::starSieve"
  Obj<Sieve> Sieve::starSieve(Obj<Point_set> chain, Obj<Sieve> star) {
    ALE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    if(star.isNull()) {
      star = Obj<Sieve>(new Sieve(this->getComm()));
    }
    int32_t height = this->maxHeight(chain.object());
    if(height >= 0) {
      this->nStarPreSieve(chain,height,star);
    }
    ALE_LOG_STAGE_END;
    return star;
  }// Sieve::starSieve()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::meet"
  Point_set Sieve::meet(Point_set c0, Point_set c1) {
    // The strategy is to compute the intersection of cones over the chains, remove the intersection 
    // and use the remaining two parts -- two disjoined components of the symmetric difference of cones -- as the new chains.
    // The intersections at each stage are accumulated and their union is the meet.
    // The iteration stops when at least one of the chains is empty.
    Point_set meet; 
    ALE_LOG_STAGE_BEGIN;
    // Check if any the initial chains may be empty, so that we don't perform spurious iterations
    if((c0.size() == 0) || (c1.size() == 0)) {
      // return meet;
    }
    else {
    while(1) {
      Point_set *c  = &c0;
      Point_set *cc = &c1;
      // Traverse the smallest cone set
      if(cc->size() < c->size()) {
        Point_set *tmp = c; c = cc; cc = tmp;
      }
      // Compute the intersection of c & cc and put it in meet at the same time removing it from c and cc
      for(Point_set::iterator c_itor = c->begin(); c_itor != c->end(); c_itor++) {
        if(cc->find(*c_itor)!= cc->end()) {
          meet.insert(*c_itor);
          cc->erase(*c_itor);
          c->erase(c_itor);
        }
      }// for(Point_set::iterator c_itor = c->begin(); c_itor != c->end(); c_itor++)
      // Replace each of the cones with a cone over it, and check if either is empty; if so, return what's in meet at the moment.
      c0 = this->cone(c0);
      if(c0.size() == 0) {
        break;
      }
      c1 = this->cone(c1);
      if(c1.size() == 0) {
        break;
      }
    }// while(1)
    }
    ALE_LOG_STAGE_END;
    return meet;
  }// Sieve::meet()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::meetAll"
  Point_set Sieve::meetAll(Point_set& chain) {
    // The strategy is the same as in meet, except it is performed on an array of chains/cones--one per point in chain--simultaneously.
    // This may be faster than applying 'meet' recursively, since intersections may become empty faster.
    Point_set meets;
    // Populate the 'cones' map, while checking if any of the initial cones may be empty, 
    // so that we don't perform spurious iterations.  At the same time determine the cone of the smallest size.
    Point__Point_set cones;
    Point minp = *(chain.begin());
    for(Point_set::iterator chain_itor = chain.begin(); chain_itor != chain.end(); chain_itor++) {
      Point p = *chain_itor;
      cones[p] = this->cone(p);
      if(cones[p].size() == 0) {
        return meets;
      }
      if(cones[p].size() < cones[minp].size()) {
        minp = p;
      }
    }// for(Point_set::iterator chain_itor = chain.begin(); chain_itor != chain.end(); chain_itor++)
    
    while(1) {
      Point_set *c = &cones[minp];
      // Traverse the smallest cone set pointed to by c and compute the intersection of c with the other cones, 
      // putting it in meet at the same time removing it from c and the other cones.
      for(Point_set::iterator c_itor = c->begin(); c_itor != c->end(); c_itor++) {
        Point q = *c_itor;
        // Keep a flag indicating whether q belongs to the intersection of all cones.
        int qIsIn = 1;
        for(Point_set::iterator chain_itor = chain.begin(); chain_itor != chain.end(); chain_itor++) {
          Point p = *chain_itor;
          // skip p equal to minp
          if(p != minp) {
            if(cones[p].find(q) == cones[p].end()) {// q is absent from the cone over p
              qIsIn = 0;
              break;
            }
          }
        }// for(Point_set::iterator chain_itor = chain.begin(); chain_itor != chain.end(); chain_itor++)
        // if a is in, add it to the meet and remove from all the cones
        if(qIsIn) {
          meets.insert(q);
          for(Point_set::iterator chain_itor = chain.begin(); chain_itor != chain.end(); chain_itor++) {
            Point p = *chain_itor;
            cones[p].erase(q);
          }
        }
        // Recall that erase(q) should not invalidate c_itor
      }// for(Point_set::iterator c_itor = c->begin(); c_itor != c->end(); c_itor++)

      // Replace each of the cones with a cone over it, and check if either is empty; if so, we are done -- return meets.
      // At the same time, locate the point with the smallest cone over it.
      for(Point_set::iterator chain_itor = chain.begin(); chain_itor != chain.end(); chain_itor++) {
        Point p = *chain_itor;
        // We check before and after computing the cone
        if(cones[p].size() == 0) {
          return meets;
        }
        cones[p] = this->cone(cones[p]);
        if(cones[p].size() == 0) {
          return meets;
        }
        if(cones[p].size() < cones[minp].size()) {
          minp = p;
        }
      }// for(Point_set::iterator chain_itor = chain.begin(); chain_itor != chain.end(); chain_itor++)

    }// while(1)
    return meets;
  }// Sieve::meetAll()


  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::join"
  Point_set Sieve::join(Point_set c0, Point_set c1) {
    // The strategy is to compute the intersection of the supports of the two chains, remove the intersection 
    // and use the remaining two parts -- two disjoined components of the symmetric difference of the supports -- as the new chains.
    // The intersections at each stage are accumulated and their union is the join.
    // The iteration stops when at least one of the chains is empty.
    Point_set join; 
    ALE_LOG_STAGE_BEGIN;
    // Check if any the initial chains may be empty, so that we don't perform spurious iterations
    if((c0.size() == 0) || (c1.size() == 0)) {
      //return join;
    }
    else{
    while(1) {
      Point_set *s  = &c0;
      Point_set *ss = &c1;
      // Traverse the smallest supp set
      if(ss->size() < s->size()) {
        Point_set *tmp = s; s = ss; ss = tmp;
      }
      // Compute the intersection of s & ss and put it in join at the same time removing it from s and ss
      for(Point_set::iterator s_itor = s->begin(); s_itor != s->end(); s_itor++) {
        if(ss->find(*s_itor)!= ss->end()) {
          join.insert(*s_itor);
          ss->erase(*s_itor);
          s->erase(s_itor);
        }
      }// for(Point_set::iterator s_itor = s->begin(); s_itor != s->end(); s_itor++)
      // Replace each of the chains with its support, and check if either is empty; if so, stop
      c0 = this->support(c0);
      if(c0.size() == 0) {
        break;
      }
      c1 = this->support(c1);
      if(c1.size() == 0) {
        break;
      }
    }// while(1)
    }
    ALE_LOG_STAGE_END;
    return join;
  }// Sieve::join()


  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::joinAll"
  Point_set Sieve::joinAll(Point_set& chain) {
    // The strategy is the same as in join, except it is performed on an array of chains/supps--one per point in chain--simultaneously.
    // This may be faster than applying 'join' recursively, since intersections may become empty faster.
    Point_set joins;
    // Populate the 'supps' map, while checking if any of the initial supports may be empty, 
    // so that we don't perform spurious iterations.  At the same time determine the supp of the smallest size.
    Point__Point_set supps;
    Point minp = *(chain.begin());
    for(Point_set::iterator chain_itor = chain.begin(); chain_itor != chain.end(); chain_itor++) {
      Point p = *chain_itor;
      supps[p] = this->support(p);
      if(supps[p].size() == 0) {
        return joins;
      }
      if(supps[p].size() < supps[minp].size()) {
        minp = p;
      }
    }// for(Point_set::iterator chain_itor = chain.begin(); chain_itor != chain.end(); chain_itor++)
    
    while(1) {
      Point_set *s = &supps[minp];
      // Traverse the smallest supp set pointed to by s and compute the intersection of s with the other supps, 
      // putting it in join at the same time removing it from s and the other supps.
      for(Point_set::iterator s_itor = s->begin(); s_itor != s->end(); s_itor++) {
        Point q = *s_itor;
        // Keep a flag indicating whether q belongs to the intersection of all supps.
        int qIsIn = 1;
        for(Point_set::iterator chain_itor = chain.begin(); chain_itor != chain.end(); chain_itor++) {
          Point p = *chain_itor;
          // skip p equal to minp
          if(p != minp) {
            if(supps[p].find(q) == supps[p].end()) {// q is absent from the support of p
              qIsIn = 0;
              break;
            }
          }
        }// for(Point_set::iterator chain_itor = chain.begin(); chain_itor != chain.end(); chain_itor++)
        // if a is in, add it to the join and remove from all the supps
        if(qIsIn) {
          joins.insert(q);
          for(Point_set::iterator chain_itor = chain.begin(); chain_itor != chain.end(); chain_itor++) {
            Point p = *chain_itor;
            supps[p].erase(q);
          }
        }
        // Recall that erase(q) should not invalidate s_itor
      }// for(Point_set::iterator s_itor = s->begin(); s_itor != s->end(); s_itor++)

      // Replace each of the supps with its support, and check if any of the new supps is empty; if so, stop.
      // At the same time, locate the point with the smallest supp.
      for(Point_set::iterator chain_itor = chain.begin(); chain_itor != chain.end(); chain_itor++) {
        Point p = *chain_itor;
        // We check before and after computing the support
        if(supps[p].size() == 0) {
          return joins;
        }
        supps[p] = this->support(supps[p]);
        if(supps[p].size() == 0) {
          return joins;
        }
        if(supps[p].size() < supps[minp].size()) {
          minp = p;
        }
      }// for(Point_set::iterator chain_itor = chain.begin(); chain_itor != chain.end(); chain_itor++)

    }// while(1)
    return joins;
  }// Sieve::joinAll()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::roots"
  Point_set Sieve::roots(Point_set chain) {
    // Compute the roots (nodes with empty cones over them) from which chain is reacheable.
    // The strategy is to examine each node in the current chain, replace it by its cone, if the latter is non-empty,
    // or remove the node from the chain, if it is a root, while saving it in the roots set.
    Point_set roots;
    ALE_LOG_STAGE_BEGIN;
    Point_set cone;
    Point_set *c  = &chain;
    Point_set *cc = &cone;
    int stop = 0;
    while(!stop) {
      stop = 1; // presume that everything in c is a root
      // Traverse chain and check the size of the cone over each point.
      for(Point_set::iterator c_itor = c->begin(); c_itor != c->end(); c_itor++) {
        Point p = *c_itor;
        if(this->coneSize(p) == 0){
          roots.insert(p);
        }
        else {
          // add the cone over p to cc
          Point_set pCone = this->cone(p);
          for(Point_set::iterator pCone_itor = pCone.begin(); pCone_itor != pCone.end(); pCone_itor++) {
            cc->insert(*pCone_itor);
          }
          // turn off the stopping flag, as we are going to examine the just-added cone during the next iteration
          stop = 0;
        }// if(this->coneSize(p) == 0)
      }// for(Point_set::iterator c_itor = c->begin(); c_itor != c->end(); c_itor++)
      // swap c and cc
      Point_set *tmp = c; c = cc; cc = tmp;
    }// while(!stop)
    ALE_LOG_STAGE_END;
    return roots;
  }// Sieve::roots()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::leaves"
  Point_set Sieve::leaves(Point_set chain) {
    // Compute the leaves (nodes with empty supports) reacheable from chain.
    // The strategy is to examine each node in the current chain, replace it by its support, if the latter is non-empty,
    // or remove the node from the chain, if it is a leaf, while saving it in the leaves set.
    Point_set leaves;
    ALE_LOG_STAGE_BEGIN;
    Point_set supp;
    Point_set *c  = &chain;
    Point_set *cc = &supp;
    int stop = 0;
    while(!stop) {
      stop = 1; // presume that everything in c is a leaf
      // Traverse chain and check the size of each point's support.
      for(Point_set::iterator c_itor = c->begin(); c_itor != c->end(); c_itor++) {
        Point p = *c_itor;
        if(this->supportSize(p) == 0){
          leaves.insert(p);
        }
        else {
          // add the support of p to cc
          Point_set pSupp = this->support(p);
          for(Point_set::iterator pSupp_itor = pSupp.begin(); pSupp_itor != pSupp.end(); pSupp_itor++) {
            cc->insert(*pSupp_itor);
          }
          // turn off the stopping flag, as we are going to examine the just-added support during the next iteration
          stop = 0;
        }// if(this->supportSize(p) == 0)
      }// for(Point_set::iterator c_itor = c->begin(); c_itor != c->end(); c_itor++)
      // swap c and cc
      Point_set *tmp = c; c = cc; cc = tmp;
    }// while(!stop)
    ALE_LOG_STAGE_END;
    return leaves;
  }// Sieve::leaves()


  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::depthStratum"
  Point_set Sieve::depthStratum(int32_t depth) {
    Point_set stratum;
    ALE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    for(Point__int::iterator d_itor = this->_depth.begin(); d_itor != this->_depth.end(); d_itor++) {
      if (d_itor->second == depth) {
        stratum.insert(d_itor->first);
      }
    }
    ALE_LOG_STAGE_END;
    return stratum;
  }// Sieve::depthStratum()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::heightStratum"
  Point_set Sieve::heightStratum(int32_t height) {
    Point_set stratum;
    ALE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    for(Point__int::iterator h_itor = this->_height.begin(); h_itor != this->_height.end(); h_itor++) {
      if (h_itor->second == height) {
        stratum.insert(h_itor->first);
      }
    }
    ALE_LOG_STAGE_END;
    return stratum;
  }// Sieve::heightStratum()

  #undef __FUNCT__
  #define __FUNCT__ "Sieve::view"
  void Sieve::view(const char *name) {
    CHKCOMM(*this);    
    int32_t  rank = this->commRank;
    PetscErrorCode ierr;
    ostringstream txt, hdr;
    hdr << "Viewing ";
    if(name == NULL) {
      hdr << "a ";
    }
    if(!this->isLocked()) {
      hdr << "(locked) ";
    }
    if(name != NULL) {
      hdr << "sieve '" << name << "' (square brackets contain depth, height pairs)\n";
    }
    hdr << "\n";
    // Print header
    ierr = PetscSynchronizedPrintf(this->comm, hdr.str().c_str()); CHKERROR(ierr, "Error in PetscPrintf");
    // Use a string stream to accumulate output that is then submitted to PetscSynchronizedPrintf
    Point_set points = this->space();
    txt  << "[" << rank << "]: space of size " << points.size() << " : ";
    for(Point_set::iterator p_itor = points.begin(); p_itor != points.end(); p_itor++)
    {
      Point p = (*p_itor);
      if(p_itor != points.begin()) {
        txt << ", ";
      }
      txt  << "(" << p.prefix << ", " << p.index << ")["<<this->depth(p) << ", " << this->height(p) << "]";
    }
    txt  << "\n";
    //
    points = this->cap();
    txt  << "[" << rank << "]: cap   of size " << points.size() << " : ";
    for(Point_set::iterator p_itor = points.begin(); p_itor != points.end(); p_itor++)
    {
      Point p = (*p_itor);
      if(p_itor != points.begin()) {
        txt << ", ";
      }
      txt  << "(" << p.prefix << ", " << p.index << ")";
    }
    txt  << "\n";
    //
    points = this->base();
    txt  << "[" << rank << "]: base  of size " << points.size() << " : ";
    for(Point_set::iterator p_itor = points.begin(); p_itor != points.end(); p_itor++)
    {
      Point p = (*p_itor);
      if(p_itor != points.begin()) {
        txt << ", ";
      }
      txt  << "(" << p.prefix << ", " << p.index << ")";
    }
    txt  << "\n";
    //
    for(Point__Point_set::iterator cone_itor = this->_cone.begin(); cone_itor != this->_cone.end(); cone_itor++)
    {
      Point p = (*cone_itor).first;
      txt  << "[" << rank << "]: cone over ("<<p.prefix<<", "<<p.index<<"):  ";
      // Traverse the local cone over p
      for(Point_set::iterator pCone_itor = this->_cone[p].begin(); pCone_itor != this->_cone[p].end(); pCone_itor++) {
        Point q = *pCone_itor;
      if(pCone_itor != this->_cone[p].begin()) {
        txt << ", ";
      }
      txt  << "(" << q.prefix << ", " << q.index << ")";
      }
      txt  << "\n";
    }
    for(Point__Point_set::iterator support_itor = this->_support.begin(); support_itor != this->_support.end(); support_itor++)
    {
      Point p = (*support_itor).first;
      txt  << "[" << rank << "]: support of ("<<p.prefix<<", "<<p.index<<"):  ";
      // Traverse the local support of p
      for(Point_set::iterator pSupport_itor = this->_support[p].begin(); pSupport_itor != this->_support[p].end(); pSupport_itor++) {
        Point q = *pSupport_itor;
      if(pSupport_itor != this->_support[p].begin()) {
        txt << ", ";
      }
      txt  << "(" << q.prefix << ", " << q.index << ")";
      }
      txt  << "\n";
    }
    ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str());
    CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
    ierr = PetscSynchronizedFlush(this->comm);
    CHKERROR(ierr, "Error in PetscSynchronizedFlush");
#if 0
    ostringstream heighthdr;
    heighthdr << "Height of ";
    if (name == NULL) {
      heighthdr << "the Sieve";
    } else {
      heighthdr << name;
    }
    heighthdr << std::endl;
    this->_height.view(heighthdr.str().c_str());
#endif
  }// Sieve::view()
  
  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::baseRestriction"
  Sieve* Sieve::baseRestriction(Point_set& base) {
    Sieve *s;
    ALE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    s = new Sieve(this->getComm());
    for(Point_set::iterator b_itor = base.begin(); b_itor != base.end(); b_itor++){
      Point p = *b_itor;
      // is point p present in the base of *this?
      if(this->_cone.find(p) != this->_cone.end()){
        s->addCone(this->_cone[p],p);
      }
    }// for(Point_set::iterator b_itor = base.begin(); b_itor != base.end(); b_itor++){
    ALE_LOG_STAGE_END;
    return s;
  }// Sieve::baseRestriction()

  #undef  __FUNCT__
  #define __FUNCT__ "Sieve::capRestriction"
  Sieve* Sieve::capRestriction(Point_set& cap) {
    Sieve *s;
    ALE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    s = new Sieve(this->getComm());
    for(Point_set::iterator c_itor = cap.begin(); c_itor != cap.end(); c_itor++){
      Point q = *c_itor;
      // is point q present in the cap of *this?
      if(this->_support.find(q) == this->_support.end()){
        s->addSupport(q,this->_support[q]);
      }
    }// for(Point_set::iterator c_itor = cap.begin(); c_itor != cap.end(); c_itor++){
    ALE_LOG_STAGE_END;
    return s;
  }// Sieve::capRestriction()

} // namespace ALE

#undef ALE_Sieve_cxx
