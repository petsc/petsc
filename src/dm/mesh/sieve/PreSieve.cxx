#define ALE_PreSieve_cxx

#ifndef included_ALE_PreSieve_hh
#include <PreSieve.hh>
#endif

#ifndef included_ALE_Stack_hh
#include <Stack.hh>
#endif

#include <stack>
#include <map>

namespace ALE {

  std::map<std::string, int> ALE::PreSieve::_log_stage;

  
  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::PreSieve()"
  PreSieve::PreSieve() : Coaster(), _cone(), _support() {};

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::PreSieve(MPI_Comm)"
  PreSieve::PreSieve(MPI_Comm comm) : Coaster(comm), _cone(), _support() {};

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::~PreSieve"
  PreSieve::~PreSieve(){};


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::clear"
  PreSieve& PreSieve::clear(){
    /* This is not what we mean by clear: Coaster::clear(); */
    this->_cone.clear();
    this->_support.clear();
    return *this;
  }// PreSieve::clear()
  
  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::addBasePoint"
  PreSieve& PreSieve::addBasePoint(Point& p) {
    this->__checkLock();
    if(!this->baseContains(p)) {
      this->_cone[p] = Point_set();
      // This is a little counterintuitive, but after the initial addition of a base point,
      // it is initially a root -- no incoming arrows.
      this->_roots.insert(p);
      // If the point is not in the cap, it is also a leaf.
      // It could be a leaf even while being in the cap, but then addCapPoint would take care of that.
      if(!capContains(p)) {
        this->_leaves.insert(p);
      }
    }
    return *this;
  }// PreSieve::addBasePoint()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::removeBasePoint"
  PreSieve& PreSieve::removeBasePoint(Point& p, bool removeSingleton) {
    this->__checkLock();
    if (this->_cone.find(p) != this->_cone.end()) {
      // IMPROVE: use 'coneView' and iterate over it to avoid copying the set
      ALE::Obj<ALE::Point_set> cone = this->cone(p);
      for(ALE::Point_set::iterator c_itor = cone->begin(); c_itor != cone->end(); c_itor++) {
        ALE::Point cover = *c_itor;

        this->removeArrow(cover, p, removeSingleton);
      }
      // Remove the cone over p
      this->_cone.erase(p);
      // After the removal of p  from the base, if p is still in the cap, it  must necessarily be a root,
      // since there are no longer any arrows terminating at p.
      if(this->capContains(p)) {
        this->_roots.insert(p);
      } else {
        // otherwise the point is gone from the PreSieve, and is not a root, hence it better not be a leaf either
        this->_leaves.erase(p);
      }
    }
    return *this;
  }// PreSieve::removeBasePoint()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::addCapPoint"
  PreSieve& PreSieve::addCapPoint(Point& q) {
    CHKCOMM(*this);
    this->__checkLock();
    if(!this->capContains(q)) {
      this->_support[q] = Point_set();
      // This may appear counterintuitive, but upon initial addition to the cap the point is a leaf
      this->_leaves.insert(q);
      // If the point is not in the base, it is also a root.  It may be a root while being in the base as well,
      // but that is handled by addBasePoint
      if(!this->baseContains(q)) {
        this->_roots.insert(q);
      }
    }
    return *this;
  }// PreSieve::addCapPoint()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::removeCapPoint"
  PreSieve& PreSieve::removeCapPoint(Point& q, bool removeSingleton) {
    CHKCOMM(*this);
    this->__checkLock();
    if(this->capContains(q)) {
      // IMPROVE: use 'supportView' and iterate over that, instead of copying a set
      ALE::Obj<ALE::Point_set> support = this->support(q);
      for(ALE::Point_set::iterator s_itor = support->begin(); s_itor != support->end(); s_itor++) {
        ALE::Point s = *s_itor;

        this->removeArrow(q, s, removeSingleton);
        // We do not erase points from 'base' unless explicitly removed, even if no points terminate there
        // WARNING: the following code looks suspicious
        //         if (this->cone(s).size() == 0) {
        //           if (!this->capContains(s)) {
        //             this->removeBasePoint(s);
        //           } else {
        //             this->_cone.erase(s);
        //           }
        //         }
      }
      // Remove the point from the cap and delete it the support under q
      this->_support.erase(q);
      // After removal from the cap the point that is still in the base becomes a leaf -- no outgoing arrows.
      if(this->baseContains(q)) {
        this->_leaves.insert(q);
      } else {
        // otherwise the point is gone from the PreSieve, and is not a leaf, hence it must not be a root either
        this->_roots.erase(q);
      }
    }
    return *this;
  }// PreSieve::removeCapPoint()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::addArrow"
  PreSieve& PreSieve::addArrow(Point& i, Point& j) {
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    this->__checkLock();
    this->addBasePoint(j);
    this->addCapPoint(i);
    this->_cone[j].insert(i);
    this->_support[i].insert(j);
    this->_leaves.erase(i);
    this->_roots.erase(j);
    ALE_PRESIEVE_LOG_STAGE_END;
    return *this;
  }// PreSieve::addArrow()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::removeArrow"
  PreSieve& PreSieve::removeArrow(Point& i, Point& j, bool removeSingleton) {
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    this->__checkLock();
    this->_cone[j].erase(i);
    this->_support[i].erase(j);
    // If this was the last arrow terminating at j, it becomes a root
    if(this->coneSize(j) == 0) {
      if (removeSingleton && (this->_leaves.find(j) != this->_leaves.end())) {
        this->removePoint(j);
      } else {
        this->_roots.insert(j);
      }
    }
    // If this was the last arrow emanating from i, it becomes a root
    if(this->supportSize(i) == 0) {
      if (removeSingleton && (this->_roots.find(i) != this->_roots.end())) {
        this->removePoint(i);
      } else {
        this->_leaves.insert(i);
      }
    }
    ALE_PRESIEVE_LOG_STAGE_END;
    return *this;
  }// PreSieve::removeArrow()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::invert"
  PreSieve& PreSieve::invert() {
    CHKCOMM(*this);
    this->__checkLock();
    // We keep track of the inversion in an new PreSieve object
    PreSieve inv;

    // Traverse the base of 'this'
    for(Point__Point_set::iterator base_itor = this->_cone.begin(); base_itor != this->_cone.end(); base_itor++) {
      Point p = base_itor->first;
      // Traverse the cone over p in 'this'
      for(Point_set::iterator pCone_itor = this->_cone[p].begin(); pCone_itor != this->_cone[p].end(); pCone_itor++) {
        Point q = *pCone_itor;
        inv.addArrow(p,q); // inversion takes place here
      }
    }
    // Replace *this with inv
    *this = inv;
    return *this;
  }// PreSieve::invert()
  

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::stackBelow"
  PreSieve& PreSieve::stackBelow(PreSieve& s) {
    CHKCOMMS(*this,s);
    this->__checkLock();
    // Composition is by 'stacking': s is placed above 'this' so that the base of s is on par with the cap of 'this'.
    // Then the arrows from s' cap are composed with 'this's arrows below.
    // This can also be thought of as the 'right action by s': *this = *this o s 
    // so that the domain of the composition is the domain (cap) of s, and the range of the composition is the range (base) of 'this'. 
    // This is done so we can keep the base and hence the number of cones in this->_cone, although some cones may become
    // empty upon composition.

    // We keep the result of the composition in a new PreSieve C
    PreSieve C(this->getComm());
    // To construct a cone over the base points in the new PreSieve, we traverse the cones of *this.
    for(Point__Point_set::iterator base_itor = this->_cone.begin(); base_itor != this->_cone.end(); base_itor++) {
      Point p = (*base_itor).first;
      // Add p to the base of C ahead of time in case the cone ends up being empty
      C.addBasePoint(p);
      // Traverse the cone over p in this
      for(Point_set::iterator pCone_itor=this->_cone[p].begin(); pCone_itor != this->_cone[p].end(); pCone_itor++) {
        Point q = *pCone_itor;
        // For each q in the cone over p add it as the cone over p in C
        Point_set qCone = s.cone(q);
        C.addCone(qCone, p);
      }// for(Point_set::iterator pCone_itor = this->_cone[p].begin(); pCone_itor != this->_cone[p].end(); pCone_itor++)
    }// for(Point__Point_set::iterator base_itor = this->_cone.begin(); base_itor != this->_cone.end(); base_itor++)
    // We use the cap of s for C
    for(Point__Point_set::iterator cap_itor = s._support.begin(); cap_itor != s._support.end(); cap_itor++) {
      Point q = (*cap_itor).first;
      C.addCapPoint(q);
    }
    // Now C to *this
    *this = C;
    return *this;
  }// PreSieve::stackBelow()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::stackAbove"
  PreSieve& PreSieve::stackAbove(PreSieve& s) {
    CHKCOMMS(*this,s);
    this->__checkLock();
    // Composition is by 'stacking': s is placed below 'this' so that the cap of s is on par with the base of 'this'.
    // Then the arrows from this's cap are composed with s's arrows below.
    // This can also be thought of as the 'left action by s': *this = s o *this
    // so that the domain of the composition is the domain (cap) of 'this', and the range of the composition 
    // is the range (base) of s. 

    // We keep the result of the composition in a new PreSieve ss
    PreSieve ss;
    // Now traverse s's base.
    for(Point__Point_set::iterator base_itor = s._cone.begin(); base_itor != s._cone.end(); base_itor++) {
      Point p = (*base_itor).first;
      // Add p to the base of ss ahead of time in case the cone ends up being empty
      ss.addBasePoint(p);
      // Traverse the cone over p in s
      for(Point_set::iterator pCone_itor=s._cone[p].begin(); pCone_itor != s._cone[p].end(); pCone_itor++) {
        Point q = *pCone_itor;
        // For each q in the cone over p in s take the cone over q in *this and add it as the cone over p in ss
        Point_set qCone = this->cone(q);
        ss.addCone(qCone,p);
      }// for(Point_set::iterator pCone_itor = s._cone[p].begin(); pCone_itor != s._cone[p].end(); pCone_itor++)
    }// for(Point__Point_set::iterator base_itor = s._cone.begin(); base_itor != s._cone.end(); base_itor++)
    // We use the cap of *this for C
    for(Point__Point_set::iterator cap_itor = this->_support.begin(); cap_itor != this->_support.end(); cap_itor++) {
      Point q = (*cap_itor).first;
      ss.addCapPoint(q);
    }
    // Now replace the guts of *this with those of ss
    *this = ss;
    return *this;
  }// PreSieve::stackAbove()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::restrictBase"
  PreSieve& PreSieve::restrictBase(Point_set& base) {
    this->__checkLock();
    // IMPROVE: this should be improved to do the removal 'in-place', instead of creating a potentially huge 'removal' set
    Point_set removal;
    while(1) {
      for(Point__Point_set::iterator b_itor = this->_cone.begin(); b_itor != this->_cone.end(); b_itor++){
        Point p = b_itor->first;
        // is point p absent from base?
        if(base.find(p) == base.end()){
          removal.insert(p);
        }
      }
      if (!removal.size()) break;
      for(Point_set::iterator r_itor = removal.begin(); r_itor != removal.end(); r_itor++){
        Point p = *r_itor;
        this->removeBasePoint(p, true);
      }
      removal.clear();
    }
    return *this;
  }// PreSieve::restrictBase()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::excludeBase"
  PreSieve& PreSieve::excludeBase(Point_set& base) {
    this->__checkLock();
    // IMPROVE: this should be improved to do the removal 'in-place', instead of creating a potentially huge 'removal' set
    Point_set removal;
    for(Point__Point_set::iterator b_itor = this->_cone.begin(); b_itor != this->_cone.end(); b_itor++){
      Point p = b_itor->first;
      // is point p present in base?
      if(base.find(p) != base.end()){
        removal.insert(p);
      }
    }// for(Point_set::iterator b_itor = this->_cone.begin(); b_itor != this->_cone.end(); b_itor++){
    for(Point_set::iterator r_itor = removal.begin(); r_itor != removal.end(); r_itor++){
      Point p = *r_itor;
      this->removeBasePoint(p);
    }
    return *this;
  }// PreSieve::excludeBase()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::baseRestriction"
  PreSieve* PreSieve::baseRestriction(Point_set& base) {
    CHKCOMM(*this);
    PreSieve *s = new PreSieve(this->getComm());
    for(Point_set::iterator b_itor = base.begin(); b_itor != base.end(); b_itor++){
      Point p = *b_itor;
      // is point p present in the base of *this?
      if(this->_cone.find(p) != this->_cone.end()){
        s->addCone(this->_cone[p],p);
      }
    }// for(Point_set::iterator b_itor = base.begin(); b_itor != base.end(); b_itor++){
    return s;
  }// PreSieve::baseRestriction()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::baseExclusion"
  PreSieve* PreSieve::baseExclusion(Point_set& base) {
    CHKCOMM(*this);
    PreSieve *s = new PreSieve(this->getComm());
    this->__computeBaseExclusion(base, s);
    return s;
  }// PreSieve::baseExclusion()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::__computeBaseExclusion"
  void PreSieve::__computeBaseExclusion(Point_set& base, PreSieve *s) {
    // This function is used by Stack as well
    for(Point__Point_set::iterator cone_itor = this->_cone.begin(); cone_itor != this->_cone.end(); cone_itor++){
      Point p = cone_itor->first;
      Point_set pCone = cone_itor->second;
      // is point p absent from base?
      if(base.find(p) == base.end()){
        s->addCone(pCone,p);
      }
    }
  }// PreSieve::__computeBaseExclusion()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::restrictCap"
  PreSieve& PreSieve::restrictCap(Point_set& cap) {
    this->__checkLock();
    // IMPROVE: this should be improved to do the removal 'in-place', instead of creating a potentially huge 'removal' set
    Point_set removal;
    for(Point__Point_set::iterator c_itor = this->_support.begin(); c_itor != this->_support.end(); c_itor++){
      Point q = c_itor->first;
      // is point q absent from cap?
      if(cap.find(q) == cap.end()){
        removal.insert(q);
      }
    }// for(Point_set::iterator c_itor = this->_support.begin(); c_itor != this->_support.end(); c_itor++){
    for(Point_set::iterator r_itor = removal.begin(); r_itor != removal.end(); r_itor++){
      Point r = *r_itor;
      this->removeCapPoint(r, true);
    }
    return *this;
  }// PreSieve::restrictCap()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::excludeCap"
  PreSieve& PreSieve::excludeCap(Point_set& cap) {
    this->__checkLock();
    // IMPROVE: this should be improved to do the removal 'in-place', instead of creating a potentially huge 'removal' set
    Point_set removal;
    for(Point__Point_set::iterator c_itor = this->_support.begin(); c_itor != this->_support.end(); c_itor++){
      Point q = c_itor->first;
      // is point q present in cap?
      if(cap.find(q) != cap.end()){
        removal.insert(q);
      }
    }// for(Point_set::iterator c_itor = this->_support.begin(); c_itor != this->_support.end(); c_itor++){
    for(Point_set::iterator r_itor = removal.begin(); r_itor != removal.end(); r_itor++){
      Point r = *r_itor;
      this->removeCapPoint(r);
    }
    return *this;
  }// PreSieve::excludeCap()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::capRestriction"
  PreSieve* PreSieve::capRestriction(Point_set& cap) {
    CHKCOMM(*this);
    PreSieve *s = new PreSieve(this->getComm());
    for(Point_set::iterator c_itor = cap.begin(); c_itor != cap.end(); c_itor++){
      Point q = *c_itor;
      // is point q present in the cap of *this?
      if(this->_support.find(q) != this->_support.end()){
        s->addSupport(q,this->_support[q]);
      }
    }// for(Point_set::iterator c_itor = cap.begin(); c_itor != cap.end(); c_itor++){
    return s;
  }// PreSieve::capRestriction()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::capExclusion"
  PreSieve* PreSieve::capExclusion(Point_set& cap) {
    CHKCOMM(*this);
    PreSieve *s = new PreSieve(this->getComm());
    this->__computeCapExclusion(cap, s);
    return s;
  }// PreSieve::capExclusion()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::__computeCapExclusion"
  void PreSieve::__computeCapExclusion(Point_set& cap, PreSieve *s) {
    // This function is used by Stack as well
    for(Point__Point_set::iterator support_itor = this->_support.begin(); support_itor != this->_support.end(); support_itor++){
      Point q = support_itor->first;
      Point_set qSupport = support_itor->second;
      // is point q absent from cap?
      if(cap.find(q) == cap.end()){
        s->addSupport(q,qSupport);
      }
    }
  }// PreSieve::__computeCapExclusion()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::space"
  Point_set PreSieve::space() {
    Point_set space;
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    // Adding both cap and base

    for(Point__Point_set::iterator cap_itor = this->_support.begin(); cap_itor != this->_support.end(); cap_itor++){
      space.insert(cap_itor->first);
    }
    for(Point__Point_set::iterator base_itor = this->_cone.begin(); base_itor != this->_cone.end(); base_itor++){
      space.insert(base_itor->first);
    }
    ALE_PRESIEVE_LOG_STAGE_END;
    return space;
  }// PreSieve::space()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::base"
  Point_set PreSieve::base() {
    Point_set base;
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    for(Point__Point_set::iterator cone_itor = this->_cone.begin(); cone_itor != this->_cone.end(); cone_itor++) {
      base.insert((*cone_itor).first);
    }
    ALE_PRESIEVE_LOG_STAGE_END;
    return base;
  }// PreSieve::base()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::cap"
  Point_set PreSieve::cap() {
    Point_set cap;
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    for(Point__Point_set::iterator support_itor = this->_support.begin(); support_itor != this->_support.end(); support_itor++) {
      cap.insert((*support_itor).first);
    }
    ALE_PRESIEVE_LOG_STAGE_END;
    return cap;
  }// PreSieve::cap()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::spaceSize"
  int32_t PreSieve::spaceSize() {
    CHKCOMM(*this);
    return this->space().size();
  }// PreSieve::spaceSize()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::spaceSizes"
  int32_t *PreSieve::spaceSizes() {
    CHKCOMM(*this);
    // Allocate array of size commSize
    int32_t spaceSize = this->space().size();
    int32_t *spaceSizes;
    PetscErrorCode ierr;
    spaceSizes = (int32_t *) malloc(sizeof(int32_t)*this->commSize);
    ierr = MPI_Allgather(&spaceSize, 1, MPIU_INT, spaceSizes, 1, MPIU_INT, comm); CHKERROR(ierr, "Error in MPI_Allgather");
    return spaceSizes;
  }// PreSieve::spaceSizes()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::baseSize"
  int32_t PreSieve::baseSize() {
    CHKCOMM(*this);
    return this->_cone.size();
  }// PreSieve::baseSize()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::baseSizes"
  int32_t *PreSieve::baseSizes() {
    CHKCOMM(*this);
    // Allocate array of size commSize
    int32_t baseSize = this->baseSize();
    int32_t *baseSizes;
    PetscErrorCode ierr;
    baseSizes = (int32_t*)malloc(sizeof(int32_t)*this->commSize);
    ierr = MPI_Allgather(&baseSize, 1, MPIU_INT, baseSizes, 1, MPIU_INT, comm); CHKERROR(ierr, "Error in MPI_Allgather");
    return baseSizes;
  }// PreSieve::baseSizes()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::capSize"
  int32_t PreSieve::capSize() {
    CHKCOMM(*this);
    return this->_support.size();
  }// PreSieve::capSize()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::capSizes"
  int32_t *PreSieve::capSizes() {
    CHKCOMM(*this);
    // Allocate array of size commSize
    int32_t capSize = this->capSize();
    int32_t *capSizes;
    PetscErrorCode ierr;
    capSizes = (int32_t *)malloc(sizeof(int32_t)*this->commSize);
    ierr = MPI_Allgather(&capSize, 1, MPIU_INT, capSizes, 1, MPIU_INT, comm); CHKERROR(ierr, "Error in MPI_Allgather");
    return capSizes;
  }// PreSieve::capSizes()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::coneSize"
  int32_t PreSieve::coneSize(Point_set& c) {
    CHKCOMM(*this);
    int32_t coneSize = 0;
    for(Point_set::iterator c_itor = c.begin(); c_itor != c.end(); c_itor++) {
      Point p = *c_itor;

      if (this->_cone.find(p) != this->_cone.end()) {
        coneSize += this->_cone[p].size();
      }
    }
    return coneSize;
  }// PreSieve::coneSize()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::supportSize"
  int32_t PreSieve::supportSize(Point_set& c) {
    CHKCOMM(*this);
    int32_t supportSize = 0;
    for(Point_set::iterator c_itor = c.begin(); c_itor != c.end(); c_itor++) {
      Point q = *c_itor;
      supportSize += this->_support[q].size();
    }
    return supportSize;
  }// PreSieve::supportSize()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::spaceContains"
  int PreSieve::spaceContains(Point point) {
    CHKCOMM(*this);
    int flag;
    flag = (this->capContains(point) || this->baseContains(point));
    return flag;
  }// PreSieve::spaceContains()
  
  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::baseContains"
  int PreSieve::baseContains(Point point) {
    CHKCOMM(*this);
    int flag;
    //flag = ((this->_cone.find(point) != this->_cone.end()) && (this->_cone[point].size() > 0));
    // SEMANTICS: we assume that the point is in the base regardless of whether it actually has arrows terminating at it
    flag = ((this->_cone.find(point) != this->_cone.end()));
    return flag;
  }// PreSieve::baseContains()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::capContains"
  int PreSieve::capContains(Point point) {
    CHKCOMM(*this);
    int flag;
    // SEMANTICS: we assume that the point is in the cap regardless of whether it actually has arrows emanating from it
    //flag = ((this->_support.find(point) != this->_support.end()) && (this->_support[point].size() > 0));
    flag = ((this->_support.find(point) != this->_support.end()));
    return flag;
  }// PreSieve::capContains()
  

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::coneContains"
  int PreSieve::coneContains(Point& p, Point point) {
    CHKCOMM(*this);
    if (this->_cone.find(p) == this->_cone.end()) {
      return 0;
    }
    Point_set& pCone = this->_cone[p];
    int flag = 0;
    if(pCone.find(point) != pCone.end()) {
      flag = 1;
    }
    return flag;
  }// PreSieve::coneContains()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::supportContains"
  int PreSieve::supportContains(Point& q, Point point) {
    CHKCOMM(*this);
    int flag = 0;
    Point_set& qSupport = this->_support[q];
    if(qSupport.find(point) != qSupport.end()) {
      flag = 1;
    }
    return flag;
  }// PreSieve::supportContains()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::nCone"
  Obj<Point_set> PreSieve::nCone(Obj<Point_set> chain, int32_t n) {
    Obj<Point_set> top(new Point_set);
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    // Compute the point set obtained by taking the cone recursively on a set of points in the base
    // (i.e., the set of cap points resulting after each iteration is used again as the base for the next cone computation).
    // Note: a 0-cone is the chain itself.

    // We use two Point_set pointers and swap them at the beginning of each iteration
    Obj<Point_set> bottom(new Point_set);
    if(n == 0) {
      top.copy(chain);
    }
    else {
      top = chain;
    }
    // If no iterations are executed, chain is returned
    for(int32_t i = 0; i < n; i++) {
      // Swap pointers and clear top
      Obj<Point_set> tmp = top; top = bottom; bottom = tmp;
      // If top == chain, replace top->pointer() with another &Point_set;  this avoids having to copy *chain.
      if(top == chain) {
        top = new Point_set;
      }
      else {
        top->clear();
      }
      // Traverse the points in bottom
      for(Point_set::iterator b_itor = bottom->begin(); b_itor != bottom->end(); b_itor++) {
        Point p = *b_itor;
        if (this->_cone.find(p) != this->_cone.end()) {
          // Traverse the points in the cone over p
          for(Point_set::iterator pCone_itor = this->_cone[p].begin(); pCone_itor != this->_cone[p].end(); pCone_itor++) {
            Point q = *pCone_itor;
            top->insert(q);
          }
        }
      }
    }
    // IMPROVE: memory use can be imporoved (number of copies and alloc/dealloc reduced) 
    //          if pointers to Point_set are used
    ALE_PRESIEVE_LOG_STAGE_END;
    return top;
  }// PreSieve::nCone()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::nCone"
  Point_set PreSieve::nCone(Point_set& chain, int32_t n) {
    return (Point_set) nCone(Obj<Point_set>(chain), n);
  }// PreSieve::nCone()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::nClosure"
  Obj<Point_set> PreSieve::nClosure(Obj<Point_set> chain, int32_t n) {
    Obj<Point_set> closure(new Point_set);
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    // Compute the point set obtained by recursively accumulating the cone over all of points of a set in the base
    // (i.e., the set of cap points resulting after each iteration is both stored in the resulting set and 
    // used again as the base of a cone computation).
    // Note: a 0-closure is the chain itself.

    // If no iterations are executed, chain is returned
    closure.copy(chain);   // copy the initial set
    // We use two Point_set pointers and swap them at the beginning of each iteration
    Obj<Point_set> top(chain);
    Obj<Point_set> bottom(new Point_set);
    for(int32_t i = 0; i < n; i++) {
      // Swap pointers and clear top
      Obj<Point_set> tmp = top; top = bottom; bottom = tmp;
      // If top == chain, replace top->pointer() with another &Point_set;  this avoids having to copy *chain.
      if(top == chain) {
        top = new Point_set;
      }
      else {
        top->clear();
      }
      // Traverse the points in bottom
      for(Point_set::iterator b_itor = bottom->begin(); b_itor != bottom->end(); b_itor++) {
        Point p = *b_itor;
        if (this->_cone.find(p) != this->_cone.end()) {
          // Traverse the points in the cone over p
          for(Point_set::iterator pCone_itor = this->_cone[p].begin(); pCone_itor != this->_cone[p].end(); pCone_itor++) {
            Point q = *pCone_itor;
            top->insert(q);
            closure->insert(q);
          }
        }
      }
    }
    // IMPROVE: memory use can be imporoved (number of copies and alloc/dealloc reduced) 
    //          if pointers to Point_sets are used
    ALE_PRESIEVE_LOG_STAGE_END;
    return closure;
  }// PreSieve::nClosure()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::nClosure"
  Point_set PreSieve::nClosure(Point_set& chain, int32_t n) {
    return (Point_set) nClosure(Obj<Point_set>(chain), n);
  }// PreSieve::nClosure()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::nClosurePreSieve"
  Obj<PreSieve> PreSieve::nClosurePreSieve(Obj<Point_set> chain, int32_t n, Obj<PreSieve> closure) {
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    if(closure.isNull()) {
      closure = Obj<PreSieve>(new PreSieve(this->comm));
    }
    // Compute the PreSieve obtained by carving out the PreSieve 'over' a given chain up to height n.
    // Note: a 0-closure is a PreSieve with the chain itself in the base, an empty cap and no arrows.

    Obj<Point_set> base(new Point_set);
    Obj<Point_set> top = chain;;
    // Initially closure contains the chain only in the base, and top contains the chain itself
    for(Point_set::iterator c_itor = chain->begin(); c_itor != chain->end(); c_itor++) {
      Point p = *c_itor;
      closure->addBasePoint(p);
    }
    // If no iterations are executed, chain is returned
    for(int32_t i = 0; i < n; i++) {
      // Swap base and top
      Obj<Point_set> tmp = top; top = base; base = tmp;
      // Traverse the points in the base
      for(Point_set::iterator b_itor = base->begin(); b_itor != base->end(); b_itor++) {
        Point p = *b_itor;
        // Compute the cone over p and add it to closure
        // IMPROVE: memory use can be improve if nCone returned a pointer to Point_set
        Point_set pCone = this->nCone(p,1);
        // Add the cone to top
        top->join(pCone);
        // Set the cone over p in the closure
        closure->addCone(pCone,p);
      }
    }
    ALE_PRESIEVE_LOG_STAGE_END;
    return closure;
  }// PreSieve::nClosurePreSieve()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::nSupport"
  Obj<Point_set> PreSieve::nSupport(Obj<Point_set> chain, int32_t n) {
    Obj<Point_set> bottom(new Point_set);
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    // Compute the point set obtained by taking suppor recursively on a set of points in the cap
    // (i.e., the set of base points resulting after each iteration is used again as the cap for the next support computation).
    // Note: a 0-support is the chain itself.

    // We use two Point_set pointers and swap them at the beginning of each iteration
    Obj<Point_set> top(new Point_set);
    if(n == 0) {
      bottom.copy(chain);
    }
    else {
      bottom = chain;
    }
    // If no iterations are executed, chain is returned
    for(int32_t i = 0; i < n; i++) {
      // Swap pointers and clear bottom
      Obj<Point_set> tmp = top; top = bottom; bottom = tmp;
      // If bottom == chain, replace bottom->pointer() with another &Point_set;  this avoids having to copy *chain.
      if(bottom == chain) {
        bottom = new Point_set;
      }
      else {
        bottom->clear();
      }
      // Traverse the points in top
      for(Point_set::iterator t_itor = top->begin(); t_itor != top->end(); t_itor++) {
        Point q = *t_itor;
        // Must check since map automatically inserts missing keys
        if (this->_support.find(q) != this->_support.end()) {
          // Traverse the points in the support under q
          for(Point_set::iterator qSupport_itor = this->_support[q].begin(); qSupport_itor != this->_support[q].end(); qSupport_itor++) {
            Point p = *qSupport_itor;
            bottom->insert(p);
          }
        }
      }
    }
    // IMPROVE: memory use can be imporoved (number of copies and alloc/dealloc reduced) 
    //          if pointers to Point_set are used
    ALE_PRESIEVE_LOG_STAGE_END;
    return bottom;
  }// PreSieve::nSupport()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::nSupport"
  Point_set PreSieve::nSupport(Point_set& chain, int32_t n) {
    return (Point_set) nSupport(Obj<Point_set>(chain), n);
  }// PreSieve::nSupport()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::nStar"
  Obj<Point_set> PreSieve::nStar(Obj<Point_set> chain, int32_t n) {
    Obj<Point_set> star(new Point_set);
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    // Compute the point set obtained by recursively accumulating the support of all of points of a set in the cap
    // (i.e., the set of base points resulting after each iteration is both stored in the resulting set and 
    // used again as the cap of a support computation).
    // Note: a 0-star is the chain itself.

    // If no iterations are executed, chain is returned
    star.copy(chain);   // copy the initial set
    // We use two Point_set pointers and swap them at the beginning of each iteration
    Obj<Point_set> bottom(chain);
    Obj<Point_set> top(new Point_set);
    for(int32_t i = 0; i < n; i++) {
      // Swap pointers and clear bottom
      Obj<Point_set> tmp = top; top = bottom; bottom = tmp;
      // If bottom == chain, replace bottom->pointer() with another &Point_set;  this avoids having to copy *chain.
      if(bottom == chain) {
        bottom = new Point_set;
      }
      else {
        bottom->clear();
      }
      // Traverse the points in top
      for(Point_set::iterator t_itor = top->begin(); t_itor != top->end(); t_itor++) {
        Point q = *t_itor;
        // Traverse the points in the support of q
        for(Point_set::iterator qSupport_itor = this->_support[q].begin(); qSupport_itor != this->_support[q].end(); qSupport_itor++) {
          Point p = *qSupport_itor;
          bottom->insert(p);
          star->insert(p);
        }
      }
    }
    // IMPROVE: memory use can be imporoved (number of copies and alloc/dealloc reduced) 
    //          if pointers to Point_sets are used
    ALE_PRESIEVE_LOG_STAGE_END;
    return star;
  }// PreSieve::nStar()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::nStar"
  Point_set PreSieve::nStar(Point_set& chain, int32_t n) {
    return (Point_set) nStar(Obj<Point_set>(chain), n);
  }// PreSieve::nStar()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::nStarPreSieve"
  Obj<PreSieve> PreSieve::nStarPreSieve(Obj<Point_set> chain, int32_t n, Obj<PreSieve> star) {
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    // Compute the PreSieve obtained by accumulating intermediate kSupports (1<=k<=n) for a set of points in the cap.
    // Note: a 0-star is the PreSieve containing the chain itself in the cap, an empty base and no arrows.
    if(star.isNull()){
      star = Obj<PreSieve>(new PreSieve(this->comm));
    }
    // We use a 'PreSieve-light' -- a map from points in the chain to the bottom-most supports computed so far.
    Point__Point_set bottom;
    // Initially star contains only the chain in the cap, and bottom has a singleton {q} set under each q.
    for(Point_set::iterator c_itor = chain->begin(); c_itor != chain->end(); c_itor++) {
      Point q = *c_itor;
      Point_set qSet; qSet.insert(q);
      bottom[q] = qSet;
      Point_set emptySet;
      star->addSupport(q,emptySet);
    }
    for(int32_t i = 0; i < n; i++) {
      // Traverse the chain
      for(Point_set::iterator c_itor = chain->begin(); c_itor != chain->end(); c_itor++) {
        Point q = *c_itor;
        // Compute the new bottom set as the 1-support of the current bottom set in 'this'
        bottom[q] = this->nSupport(bottom[q],1);
        // Add the new bottom set to the support of q in the support PreSieve
        star->addSupport(q,bottom[q]);
      }// for(Point_set::iterator c_itor = chain.begin(); c_itor != chain.end(); c_itor++) 
    }// for(int32_t i = 0; i < n; i++)
    ALE_PRESIEVE_LOG_STAGE_END;
    return star;
  }// PreSieve::nStarPreSieve()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::nMeet"
  Point_set PreSieve::nMeet(Point_set c0, Point_set c1, int32_t n) {
    Point_set meet; 
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    // The strategy is to compute the intersection of cones over the chains, remove the intersection 
    // and use the remaining two parts -- two disjoined components of the symmetric difference of cones -- as the new chains.
    // The intersections at each stage are accumulated and their union is the meet.
    // The iteration stops after n steps in addition to the meet of the initial chains or sooner if at least one of the chains is empty.
    ALE::Obj<ALE::Point_set> cone;
    // Check if any the initial chains may be empty, so that we don't perform spurious iterations
    if((c0.size() == 0) || (c1.size() == 0)) {
      //return meet;
    }
    else { // nonzero sizes
      for(int32_t i = 0; i <= n; i++) {
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
        cone = this->cone(c0);
        c0.insert(cone->begin(), cone->end());
        if(c0.size() == 0) {
          //return meet;
          break;
        }
        cone = this->cone(c1);
        c1.insert(cone->begin(), cone->end());
        if(c1.size() == 0) {
          //return meet;
          break;
        }
      }// for(int32_t i = 0; i <= n; i++) 
    }// nonzero sizes
    ALE_PRESIEVE_LOG_STAGE_END;
    return meet;
  }// PreSieve::nMeet()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::nMeetAll"
  Point_set PreSieve::nMeetAll(Point_set& chain, int32_t n) {
    // The strategy is the same as in nMeet, except it is performed on an array of chains/cones--one per point in chain--simultaneously.
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
    
    for(int32_t i = 0; i <= n; i++) {
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

    }
    return meets;
  }// PreSieve::nMeetAll()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::nJoin"
  Point_set PreSieve::nJoin(Point_set c0, Point_set c1, int32_t n) {
    Point_set join; 
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    // The strategy is to compute the intersection of the supports of the two chains, remove the intersection 
    // and use the remaining two parts -- two disjoined components of the symmetric difference of the supports -- as the new chains.
    // The intersections at each stage are accumulated and their union is the join.
    // The iteration stops after n steps apart from the join of the initial chains or sooner if at least one of the chains is empty.
    ALE::Obj<ALE::Point_set> support;

    // Check if any the initial chains may be empty, so that we don't perform spurious iterations
    if((c0.size() == 0) || (c1.size() == 0)) {
      // return join;
    }
    else { // nonzero sizes
      for(int32_t i = 0; i <= n; i++) {
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
        support = this->support(c0);
        c0.insert(support->begin(), support->end());
        if(c0.size() == 0) {
          // return join;
          break;
        }
        support = this->support(c1);
        c1.insert(support->begin(), support->end());
        if(c1.size() == 0) {
          // return join;
          break;
        }
      }// for(int32_t i = 0; i <= n; i++) 
    }// nonzero sizes
    return join;
    ALE_PRESIEVE_LOG_STAGE_END;
  }// PreSieve::nJoin()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::nJoinAll"
  Point_set PreSieve::nJoinAll(Point_set& chain, int32_t n) {
    // The strategy is the same as in nJoin, except it is performed on an array of chains/supps--one per point in chain--simultaneously.
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
    
    for(int32_t i = 0; i <= n; i++) {
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

    }
    return joins;
  }// PreSieve::nJoinAll()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::add"
  PreSieve *PreSieve::add(PreSieve& s) {
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    CHKCOMMS(*this,s);
    this->__checkLock();
    // Add the points and arrows of s to those of 'this'
    Point_set emptySet;
    for(Point__Point_set::iterator sBase_itor = s._cone.begin(); sBase_itor != s._cone.end(); sBase_itor++) {
      Point p = sBase_itor->first;
      Point_set pCone = s._cone[p];
      this->addCone(pCone, p);
    }
    // Make sure all of the cap of s is added (some cap points have no arrows).
    for(Point__Point_set::iterator cap_itor = s._support.begin(); cap_itor != s._support.end(); cap_itor++) {
      Point q = cap_itor->first;
      this->addCapPoint(q);
    }
    ALE_PRESIEVE_LOG_STAGE_END;
    return this;
  }// PreSieve::add()


  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::add"
  PreSieve& PreSieve::add(Obj<PreSieve> s) {
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    // IMPROVE:  do something about CHKCOMMS so that I can use Obj<PreSieve> to do the checking.
    //CHKCOMMS(*this,s.object());
    this->__checkLock();
    // Add the points and arrows of s to those of 'this'
    Point_set emptySet;
    for(Point__Point_set::iterator sBase_itor = s->_cone.begin(); sBase_itor != s->_cone.end(); sBase_itor++) {
      Point p = sBase_itor->first;
      Point_set pCone = s->_cone[p];
      this->addCone(pCone, p);
    }
    // Make sure all of the cap of s is added (some cap points have no arrows).
    for(Point__Point_set::iterator cap_itor = s->_support.begin(); cap_itor != s->_support.end(); cap_itor++) {
      Point q = cap_itor->first;
      this->addCapPoint(q);
    }
    ALE_PRESIEVE_LOG_STAGE_END;
    return *this;
  }// PreSieve::add()


  #undef __FUNCT__
  #define __FUNCT__ "PreSieve::view"
  void PreSieve::view(const char *name) {
    CHKCOMM(*this);    
    int32_t  rank = this->commRank;
    PetscErrorCode ierr;
    ostringstream txt1, txt2, txt3, txt4, hdr;
    hdr << "[" << rank << "]:Viewing ";
    if(name == NULL) {
      hdr << "a ";
    }
    if(!this->isLocked()) {
      hdr << "(locked) ";
    }
    if(name != NULL) {
      hdr << "presieve '" << name << "'\n";
    }
    // Print header
    ierr = PetscSynchronizedPrintf(this->comm, hdr.str().c_str()); CHKERROR(ierr, "Error in PetscPrintf");
    // Use a string stream to accumulate output that is then submitted to PetscSynchronizedPrintf
    Point_set points = this->space();
    txt1  << "[" << rank << "]: space of size " << points.size() << " : ";
    for(Point_set::iterator p_itor = points.begin(); p_itor != points.end(); p_itor++)
    {
      Point p = (*p_itor);
      if(p_itor != points.begin()) {
        txt1 << ", ";
      }
      txt1  << "(" << p.prefix << ", " << p.index << ")";
    }
    txt1  << "\n";
    ierr = PetscSynchronizedPrintf(this->comm, txt1.str().c_str());
    //
    points = this->cap();
    txt2  << "[" << rank << "]: cap   of size " << points.size() << " : ";
    for(Point_set::iterator p_itor = points.begin(); p_itor != points.end(); p_itor++)
    {
      Point p = (*p_itor);
      if(p_itor != points.begin()) {
        txt2 << ", ";
      }
      txt2  << "(" << p.prefix << ", " << p.index << ")";
    }
    txt2  << "\n";
    ierr = PetscSynchronizedPrintf(this->comm, txt2.str().c_str());
    //
    points = this->base();
    txt3  << "[" << rank << "]: base  of size " << points.size() << " : ";
    for(Point_set::iterator p_itor = points.begin(); p_itor != points.end(); p_itor++)
    {
      Point p = (*p_itor);
      if(p_itor != points.begin()) {
        txt3 << ", ";
      }
      txt3  << "(" << p.prefix << ", " << p.index << ")";
    }
    txt3  << "\n";
    ierr = PetscSynchronizedPrintf(this->comm, txt3.str().c_str());
    //
    for(Point__Point_set::iterator cone_itor = this->_cone.begin(); cone_itor != this->_cone.end(); cone_itor++)
    {
      Point p = (*cone_itor).first;
      txt4  << "[" << rank << "]: cone over (" << p.prefix << ", " << p.index << "):  ";
      // Traverse the local cone over p
      for(Point_set::iterator pCone_itor = this->_cone[p].begin(); pCone_itor != this->_cone[p].end(); pCone_itor++) {
        Point q = *pCone_itor;
      if(pCone_itor != this->_cone[p].begin()) {
        txt4 << ", ";
      }
      txt4  << "(" << q.prefix << ", " << q.index << ")";
      }
      txt4  << "\n";
    }
#if 0
    for(Point__Point_set::iterator support_itor = this->_support.begin(); support_itor != this->_support.end(); support_itor++)
    {
      Point p = (*support_itor).first;
      txt4  << "[" << rank << "]: support of (" << p.prefix << ", " << p.index << "):  ";
      // Traverse the local support of p
      for(Point_set::iterator pSupport_itor = this->_support[p].begin(); pSupport_itor != this->_support[p].end(); pSupport_itor++) {
        Point q = *pSupport_itor;
      if(pSupport_itor != this->_support[p].begin()) {
        txt4 << ", ";
      }
      txt4  << "(" << q.prefix << ", " << q.index << ")";
      }
      txt4  << "\n";
    }
#endif
    ierr = PetscSynchronizedPrintf(this->comm, txt4.str().c_str());
    CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
    ierr = PetscSynchronizedFlush(this->comm);
    CHKERROR(ierr, "Error in PetscSynchronizedFlush");

  }// PreSieve::view()
  
  
  // -------------------------------------------------------------------------------------------------------------------------------- //
  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::coneCompletion"
  Stack* PreSieve::coneCompletion(int completionType, int footprintType, PreSieve *c) {
    Stack *s;
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    s = this->__computeCompletion(completedSetCap, completionType, footprintType, c);
    ALE_PRESIEVE_LOG_STAGE_END;
    return s;
  }// PreSieve::coneCompletion()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::supportCompletion"
  Stack* PreSieve::supportCompletion(int completionType, int footprintType, PreSieve *c) {
    Stack *s;
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    s = this->__computeCompletion(completedSetBase, completionType, footprintType, c);
    ALE_PRESIEVE_LOG_STAGE_END;
    return s;
  }// PreSieve::supportCompletion()

  #undef __FUNCT__
  #define __FUNCT__ "PreSieve::baseFootprint"
  Stack* PreSieve::baseFootprint(int completionType, int footprintType,  PreSieve *f) {
    PreSieve ownership(this->comm);
    Point self(this->commRank, this->commRank);
    if(footprintType == footprintTypeNone) {
      return NULL;
    }
    else if(footprintType == footprintTypeCone) {
      // Create a temporary 'ownership' PreSieve with this->base as the base and the process ID in the cone of all this->base points
      Point_set base = this->base();
      ownership.addSupport(self, base);
      // Now compute the cone completion of ownership and place it in footprint.
      // No footprint computation is required during the completion, since the completion itself carries that information.
      return ownership.coneCompletion(completionType, footprintTypeNone, f);
    }
    else if(footprintType == footprintTypeSupport) {
      // Create a temporary 'ownership' PreSieve with this->base as the cap and the process ID in the support of all this->base points
      Point_set base = this->base();
      ownership.addCone(base, self);
      // Now compute the support completion of ownership and place it in footprint.
      // No footprint computation is required during the completion, since the completion itself carries that information.
      return ownership.coneCompletion(completionType, footprintTypeNone, f);
     }
    else {
      throw Exception("Unknown footprintType");
    }
    
  }// PreSieve::baseFootprint()

  #undef __FUNCT__
  #define __FUNCT__ "PreSieve::capFootprint"
  Stack* PreSieve::capFootprint(int completionType, int footprintType,  PreSieve *f) {
    PreSieve ownership(this->comm);
    Point self(this->commRank, this->commRank);
    if(footprintType == footprintTypeNone) {
      return NULL;
    }
    else if(footprintType == footprintTypeCone) {
      // Create a temporary 'ownership' PreSieve with this->cap as the base and the process ID in the cone of all this->cap points.
      Point_set cap = this->cap();
      ownership.addSupport(self, cap);
      // Now compute the cone completion of ownership and place it in footprint.
      // No footprint computation is required during the completion, since the completion itself carries that information.
      return ownership.coneCompletion(completionType, footprintTypeNone, f);
    }
    else if(footprintType == footprintTypeSupport) {
      // Create a temporary 'ownership' PreSieve with this->cap as the cap and the process ID in the support of all this->cap points.
      Point_set cap = this->cap();
      ownership.addCone(cap, self);
      // Now compute the support completion of ownership and place it in footprint.
      // No footprint computation is required during the completion, since the completion itself carries that information.
      return ownership.coneCompletion(completionType, footprintTypeNone, f);
     }
    else {
      throw Exception("Unknown footprintType");
    }
    
  }// PreSieve::capFootprint()

  #undef __FUNCT__
  #define __FUNCT__ "PreSieve::spaceFootprint"
  Stack* PreSieve::spaceFootprint(int completionType, int footprintType,  PreSieve *f) {
    PreSieve ownership(this->comm);
    Point self(this->commRank, this->commRank);
    if(footprintType == footprintTypeNone) {
      return NULL;
    }
    else if(footprintType == footprintTypeCone) {
      // Create a temporary 'ownership' PreSieve with this->space as the base and the process ID in the cone of all this->space points.
      Point_set space = this->space();
      ownership.addSupport(self, space);
      // Now compute the cone completion of ownership and place it in footprint.
      // No footprint computation is required during the completion, since the completion itself carries that information.
      return ownership.coneCompletion(completionType, footprintTypeNone, f);
    }
    else if(footprintType == footprintTypeSupport) {
      //Create a temporary 'ownership' PreSieve with this->space as the cap and the process ID in the support of all this->space points.
      Point_set space = this->space();
      ownership.addCone(space, self);
      // Now compute the support completion of ownership and place it in footprint.
      // No footprint computation is required during the completion, since the completion itself carries that information.
      return ownership.coneCompletion(completionType, footprintTypeNone, f);
     }
    else {
      throw Exception("Unknown footprintType");
    }
    
  }// PreSieve::spaceFootprint()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::__determinePointOwners"
  void PreSieve::__determinePointOwners(Point_set& points, int32_t *LeaseData, std::map<Point, int32_t, Point::Cmp>& owner) {
    ALE_PRESIEVE_LOG_STAGE_BEGIN;
    CHKCOMM(*this);
    PetscErrorCode ierr;
    int32_t  size = this->commSize;
    int32_t  rank = this->commRank;

    // We need to partition global nodes among lessors, which we do by global prefix
    // First we determine the extent of global prefices and the bounds on the indices with each global prefix.
    int32_t minGlobalPrefix = 0;
    // Determine the local extent of global domains
    for(Point_set::iterator point_itor = points.begin(); point_itor != points.end(); point_itor++) {
      Point p = (*point_itor);
      if((p.prefix < 0) && (p.prefix < minGlobalPrefix)) {
        minGlobalPrefix = p.prefix;
      }
    }
    int32_t MinGlobalPrefix;
    ierr = MPI_Allreduce(&minGlobalPrefix, &MinGlobalPrefix, 1, MPIU_INT, MPI_MIN, this->comm); CHKERROR(ierr, "Error in MPI_Allreduce");

    int__int BaseLowerBound, BaseUpperBound; // global quantities computed from the local quantities below
    int__int BaseMaxSize;                    // the maximum size of the global base index space by global prefix
    int__int BaseSliceScale, BaseSliceSize, BaseSliceOffset;

    if(MinGlobalPrefix < 0) { // if we actually do have global base points
      // Determine the upper and lower bounds on the indices of base points with each global prefix.
      // We use maps to keep track of these quantities with different global prefices.
      int__int baseLowerBound, baseUpperBound; // local quantities
      // Initialize local bound maps with the upper below lower so we can later recognize omitted prefices.
      for(int d = -1; d >= MinGlobalPrefix; d--) {
        baseLowerBound[d] = 0; baseUpperBound[d] = -1;
      }
      // Compute local bounds
      for(Point_set::iterator point_itor = points.begin(); point_itor != points.end(); point_itor++) {
        Point p = (*point_itor);
        int32_t d = p.prefix;
        int32_t i = p.index;
        if(d < 0) { // it is indeed a global prefix
          if (i < baseLowerBound[d]) {
            baseLowerBound[d] = i;
          }
          if (i > baseUpperBound[d]) {
            baseUpperBound[d] = i;
          }
        }
      }
      // Compute global bounds
      for(int32_t d = -1; d >= MinGlobalPrefix; d--){
        int32_t lowerBound, upperBound, maxSize;
        ierr   = MPI_Allreduce(&baseLowerBound[d],&lowerBound,1,MPIU_INT,MPI_MIN,this->comm); 
        CHKERROR(ierr, "Error in MPI_Allreduce");
        ierr   = MPI_Allreduce(&baseUpperBound[d],&upperBound,1,MPIU_INT,MPI_MAX,this->comm); 
        CHKERROR(ierr, "Error in MPI_Allreduce");
        maxSize = upperBound - lowerBound + 1;
        if(maxSize > 0) { // there are actually some indices in this global prefix
          BaseLowerBound[d] = lowerBound;
          BaseUpperBound[d] = upperBound;
          BaseMaxSize[d]    = maxSize;

          // Each processor (at least potentially) owns a slice of the base indices with each global indices.
          // The size of the slice with global prefix d is BaseMaxSize[d]/size + 1 (except if rank == size-1, 
          // where the slice size can be smaller; +1 is for safety).
          
          // For a non-empty domain d we compute and store the slice size in BaseSliceScale[d] (the 'typical' slice size) and 
          // BaseSliceSize[d] (the 'actual' slice size, which only differs from 'typical' for processor with rank == size -1 ).
          // Likewise, each processor has to keep track of the index offset for each slice it owns and stores it in BaseSliceOffset[d].
          BaseSliceScale[d]  = BaseMaxSize[d]/size + 1;
          BaseSliceSize[d]   = BaseSliceScale[d]; 
          if (rank == size-1) {
            BaseSliceSize[d] =  BaseMaxSize[d] - BaseSliceScale[d]*(size-1); 
          }
          BaseSliceSize[d]   = PetscMax(1,BaseSliceSize[d]);
          BaseSliceOffset[d] = BaseLowerBound[d] + BaseSliceScale[d]*rank;
        }// for(int32_t d = -1; d >= MinGlobalPrefix; d--){
      }// 
    }// if(MinGlobalDomain < 0) 

    for (Point_set::iterator point_itor = points.begin(); point_itor != points.end(); point_itor++) {
      Point p = (*point_itor);
      // Determine which slice p falls into
      // ASSUMPTION on Point type
      int32_t d = p.prefix;
      int32_t i = p.index;
      int32_t proc;
      if(d < 0) { // global domain -- determine the owner by which slice p falls into
        proc = (i-BaseLowerBound[d])/BaseSliceScale[d];
      }
      else { // local domain -- must refer to a rank within the comm
        if(d >= size) {
          throw ALE::Exception("Local domain outside of comm size");
        }
        proc = d;
      }
      owner[p]     = proc;              
      LeaseData[2*proc+1] = 1;                 // processor owns at least one of ours (i.e., the number of leases from proc is 1)
      LeaseData[2*proc]++;                     // count of how many we lease from proc
    }
    ALE_PRESIEVE_LOG_STAGE_END;
  }// PreSieve::__determinePointOwners()

  #undef  __FUNCT__
  #define __FUNCT__ "PreSieve::__computeCompletion"
  Stack *PreSieve::__computeCompletion(int completedSet, int  completionType, int  footprintType, PreSieve *completion) {
    // Overall completion stack C
    Stack *C = new Stack(this->comm);

    // First, we set up the structure of the stack in which we return the completion information
    // This structure is controled by the flags in the calling sequence (completedSet, completionType, footprintType)
    // and is best described by a picture (see the Paper).

    
    // "Structure" presieve -- either the point presieve or the arrow stack (both defined below).
    Obj<PreSieve> S;

    // Point presieve P
    Obj<PreSieve> P(completion);
    if(completion != NULL) {
      // The completion sieve must have the same communicator
      CHKCOMMS(*this, *completion);
    }
    else {
      P = Obj<PreSieve>(new PreSieve(this->getComm()));
    }

    // Arrow stacks/presieve
    Obj<PreSieve> A;
    Obj<Stack> AAA, A0, A1;
    switch(completionType) {
    case completionTypePoint:
      S = P;
      break;
    case completionTypeArrow:
      A  = new PreSieve(this->getComm());
      A0  = new Stack(this->getComm());
      A0->setTop(P); A0->setBottom(A);
      A1  = new Stack(this->getComm());
      A1->setTop(A); A1->setBottom(P);
      AAA = new Stack(this->getComm());
      AAA->setTop(A0); AAA->setBottom(A1);
      S = AAA;
      break;
    default:
      throw(ALE::Exception("Unknown completionType"));
    }
    
    // Footprint presieve
    Obj<PreSieve> F; 
    // Depending on the footprintType flag, the structure presieve S sits either at the bottom or the top of C,
    // while F is null or not.
    switch(footprintType){
    case footprintTypeCone:
      F = Obj<PreSieve>(new PreSieve(this->getComm()));
      // S is the top of C
      C->setTop(S);
      // Footprint
      C->setBottom(F);
      break;
    case footprintTypeSupport:
      F = Obj<PreSieve>(new PreSieve(this->getComm()));
      // S is the bottom of C
      C->setBottom(S);
      // Footprint
      C->setTop(F);
      break;
    case footprintTypeNone:
      C->setTop(S);
      C->setBottom(S);
      break;
    default:
      throw(ALE::Exception("Unknown footprint type"));
    }
    
    // If this is a local sieve, there is no completion to be computed
    // Otherwise we have to compute the completion of the cone over each i from the cones over i
    // held by all the processes in the communicator, and at the same time contributing to whatever cones 
    // those processors may be computing.
    
    if( (this->comm == MPI_COMM_SELF) || (this->commSize == 1) ) {
      return C;
    }
    PetscErrorCode ierr;
    int32_t  size = this->commSize;
    int32_t  rank = this->commRank;
    
    bool debug = this->verbosity > 10;
    bool debug2 = this->verbosity > 11;
    
    /* Glossary:
       -- Node               -- a Point either in the base or in the cone; terms 'Node' and 'Point' (capitalized or not)
                               will be used interchangeably (sp?).
       
       -- Owned node         -- a that have been assigned to this processor just for this routine.
                               It is either a node with this processor's rank as the prefix, or a global node that has been 
                               assigned to this processor during the 'preprocessing' stage.
                               
       -- Domain             -- the set of all points in the base with a given prefix

       -- Global domain/node -- a domain corresponding to a global (i.e., negative) prefix; 
                                a global node is a node from a global domain, equivalently, a node with global prefix.

       -- Rented node        -- an owned node assigned to this processor that belongs to the base of another processor.

       -- Leased node        -- a node belonging to this processor's base that is owned by another processor.

       -- Lessor             -- a processor owning a node rented by this processor (hence leasing it to this processor).
       
       -- Renter             -- a processor that has leased a node owned by this processor.
       
       -- Rental             -- a renter--rented-node combination

       -- Sharer             -- (in relation to a given rented node) a renter sharing the node with another renter
       
       -- Neighbor           -- (in relation to a given leased node) a processor leasing the same node -- analogous to a sharer
                               from a renter's point of view

       -- Total              -- always refers to the cumulative number of entities across all processors 
                               (e.g., total number of nodes -- the sum of all nodes owned by each processor).
    */
    
    // ASSUMPTION: here we assume that Point == {int32_t prefix, index}; should the definition of Point change, so must this code.

    // Determine the owners of base nodes and collect the lease data for each processor:
    // the number of nodes leased and the number of leases (0 or 1).
    int32_t *LeaseData; // 2 ints per processor: number of leased nodes and number of leases (0 or 1).
    ierr = PetscMalloc(2*size*sizeof(PetscInt),&LeaseData);CHKERROR(ierr, "Error in PetscMalloc");
    ierr = PetscMemzero(LeaseData,2*size*sizeof(PetscInt));CHKERROR(ierr, "Error in PetscMemzero");
    
    // We also need to keep the set of points (base or cap) we use to determine the neighbors,
    // as well as a list of "cones" that are either cones or supports, depending on what the points are.
    Point_set *points;
    Point__Point_set *cones;
    if(completedSet == completedSetCap) {
      points = new Point_set();
      *points = this->base();
      cones  = &this->_cone;
    }
    else if(completedSet == completedSetBase){
      points = new Point_set();
      *points = this->cap();
      cones = &this->_support;
    }
    else {
      throw ALE::Exception("Unknown completedSet");
    }
    // determine owners of each base node and save it in a map
    std::map<Point, int32_t, Point::Cmp> owner;
    this->__determinePointOwners(*points, LeaseData, owner);
    
    // Now we accumulate the max lease size and the total number of renters
    int32_t MaxLeaseSize, RenterCount;
    ierr = PetscMaxSum(this->comm,LeaseData,&MaxLeaseSize,&RenterCount); 
    CHKERROR(ierr,"Error in PetscMaxSum");
    ierr = PetscVerboseInfo((0,"PreSieve::__computeCompletion: Number of renters %d\n", RenterCount)); 
    CHKERROR(ierr,"Error in PetscVerboseInfo");

    if(debug) { /* -------------------------------------------------------------- */
      ierr = PetscSynchronizedPrintf(this->comm, "[%d]: __computeCompletion: RenterCount = %d, MaxLeaseSize = %d\n", rank, RenterCount, MaxLeaseSize);
      CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm);
      CHKERROR(ierr, "Error in PetscSynchronizedFlush");
    } /* ----------------------------------------------------------------------- */
    
    // post receives for all Rented nodes; we will be receiving 3 data items per rented node, 
    // and at most MaxLeaseSize of nodes per renter
    PetscMPIInt    tag1;
    ierr = PetscObjectGetNewTag(this->petscObj, &tag1); CHKERROR(ierr, "Failded on PetscObjectGetNewTag");
    int32_t *RentedNodes;
    MPI_Request *Renter_waits;
    if(RenterCount){
      ierr = PetscMalloc((RenterCount)*(3*MaxLeaseSize+1)*sizeof(int32_t),&RentedNodes);  CHKERROR(ierr,"Error in PetscMalloc");
      ierr = PetscMemzero(RentedNodes,(RenterCount)*(3*MaxLeaseSize+1)*sizeof(int32_t));  CHKERROR(ierr,"Error in PetscMemzero");
      ierr = PetscMalloc((RenterCount)*sizeof(MPI_Request),&Renter_waits);                CHKERROR(ierr,"Error in PetscMalloc");
    }
    for (int32_t i=0; i<RenterCount; i++) {
      ierr = MPI_Irecv(RentedNodes+3*MaxLeaseSize*i,3*MaxLeaseSize,MPIU_INT,MPI_ANY_SOURCE,tag1,this->comm,Renter_waits+i);
      CHKERROR(ierr,"Error in MPI_Irecv");
    }
    
    int32_t LessorCount;
    LessorCount = 0; for (int32_t i=0; i<size; i++) LessorCount += LeaseData[2*i+1];
    ierr = PetscVerboseInfo((0,"PreSieve::__computeCompletion: Number of lessors %d\n",LessorCount));
    CHKERROR(ierr,"Error in PetscVerboseInfo");
    if(debug) { /* -------------------------------------------------------------- */
      ierr = PetscSynchronizedPrintf(this->comm, "[%d]: __computeCompletion: LessorCount = %d\n", rank, LessorCount);
      CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm);
      CHKERROR(ierr, "Error in PetscSynchronizedFlush");
    } /* ----------------------------------------------------------------------- */
    
    // We keep only the data about the real lessors -- those that own the nodes we lease
    int32_t *LeaseSizes, *Lessors;
    if(LessorCount) {
      ierr = PetscMalloc(sizeof(int32_t)*(LessorCount), &LeaseSizes); CHKERROR(ierr, "Error in PetscMalloc");
      ierr = PetscMalloc(sizeof(int32_t)*(LessorCount), &Lessors);    CHKERROR(ierr, "Error in PetscMalloc");
    }
    // We also need to compute the inverse to the Lessors array, since we need to be able to convert i into cntr
    // after using the owner array.  We use a map LessorIndex; it is likely to be small -- ASSUMPTION
    int__int LessorIndex;
    // Traverse all processes in ascending order
    int32_t cntr = 0; // keep track of entered records
    for(int32_t i = 0; i < size; i++) {
      if(LeaseData[2*i]) { // if there are nodes leased from process i, record it
        LeaseSizes[cntr] = LeaseData[2*i];
        Lessors[cntr] = i;
        LessorIndex[i] = cntr;
        cntr++;
      }
    }
    ierr = PetscFree(LeaseData); CHKERROR(ierr, "Error in PetscFree");
    if(debug2) { /* ----------------------------------- */
      ostringstream txt;
      txt << "[" << rank << "]: __computeCompletion: lessor data [index, rank, lease size]: ";
      for(int32_t i = 0; i < LessorCount; i++) {
        txt << "[" << i << ", " << Lessors[i] << ", " << LeaseSizes[i] << "] ";
      }
      txt << "\n";
      ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
    }/* -----------------------------------  */
    if(debug2) { /* ----------------------------------- */
      ostringstream txt;
      txt << "[" << rank << "]: __computeCompletion: LessorIndex: ";
      for(int__int::iterator li_itor = LessorIndex.begin(); li_itor!= LessorIndex.end(); li_itor++) {
        int32_t i = (*li_itor).first;
        int32_t j = (*li_itor).second;
        txt << i << "-->" << j << "; ";
      }
      txt << "\n";
      ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
    }/* -----------------------------------  */

    
    // pack messages containing lists of leased base nodes and their cone sizes to the lessors
    int32_t LeasedNodeCount = points->size(); // all points are considered leased from someone
    int32_t *LeasedNodes;
    int32_t *LessorOffsets;
    // We need 3 ints per leased node -- 2 per Point and 1 for the cone size
    if(LeasedNodeCount) {
      ierr       = PetscMalloc((3*LeasedNodeCount)*sizeof(PetscInt),&LeasedNodes); CHKERROR(ierr,"Error in PetscMalloc");
    }
    if(LessorCount) {
      ierr       = PetscMalloc((LessorCount)*sizeof(PetscInt),&LessorOffsets);     CHKERROR(ierr,"Error in PetscMalloc");
      LessorOffsets[0] = 0; 
    }
    for (int32_t i=1; i<LessorCount; i++) { LessorOffsets[i] = LessorOffsets[i-1] + 3*LeaseSizes[i-1];} 
    for (Point_set::iterator point_itor = points->begin(); point_itor != points->end(); point_itor++) {
      Point p = (*point_itor);
      int32_t ow = owner[p];
      int32_t ind  = LessorIndex[ow];
      LeasedNodes[LessorOffsets[ind]++] = p.prefix;
      LeasedNodes[LessorOffsets[ind]++] = p.index;      
      LeasedNodes[LessorOffsets[ind]++] = (*cones)[p].size();
    }
    if(LessorCount) {
      LessorOffsets[0] = 0; 
    }
    for (int32_t i=1; i<LessorCount; i++) { LessorOffsets[i] = LessorOffsets[i-1] + 3*LeaseSizes[i-1];} 
    
    // send the messages to the lessors
    MPI_Request *Lessor_waits;
    if(LessorCount) {
      ierr = PetscMalloc((LessorCount)*sizeof(MPI_Request),&Lessor_waits);CHKERROR(ierr,"Error in PetscMalloc");
    }
    for (int32_t i=0; i<LessorCount; i++) {
      ierr      = MPI_Isend(LeasedNodes+LessorOffsets[i],3*LeaseSizes[i],MPIU_INT,Lessors[i],tag1,this->comm,&Lessor_waits[i]);
      CHKERROR(ierr,"Error in MPI_Isend");
    }
    
    // wait on receive request and prepare to record the identities of the renters responding to the request and their lease sizes
    std::map<int32_t, int32_t> Renters, RenterLeaseSizes;
    // Prepare to compute the set of renters of each owned node along with the cone sizes held by those renters over the node.
    // Since we don't have a unique ordering on the owned nodes a priori, we will utilize a map.
    std::map<Point, int_pair_set, Point::Cmp> NodeRenters;
    cntr  = RenterCount; 
    while (cntr) {
      int32_t arrivalNumber;
      MPI_Status Renter_status;
      ierr = MPI_Waitany(RenterCount,Renter_waits,&arrivalNumber,&Renter_status);  
      CHKMPIERROR(ierr,ERRORMSG("Error in MPI_Waitany"));
      int32_t renter = Renter_status.MPI_SOURCE;
      Renters[arrivalNumber] = renter;
      ierr = MPI_Get_count(&Renter_status,MPIU_INT,&RenterLeaseSizes[arrivalNumber]); CHKERROR(ierr,"Error in MPI_Get_count");
      // Since there are 3 ints per leased node, the lease size is computed by dividing the received count by 3;
      RenterLeaseSizes[arrivalNumber] = RenterLeaseSizes[arrivalNumber]/3;
      // Record the renters for each node
      for (int32_t i=0; i<RenterLeaseSizes[arrivalNumber]; i++) {
        // Compute the offset into the RentedNodes array for the arrived lease.
        int32_t LeaseOffset = arrivalNumber*3*MaxLeaseSize;
        // ASSUMPTION on Point type
        Point node = Point(RentedNodes[LeaseOffset + 3*i], RentedNodes[LeaseOffset + 3*i+1]);
        int32_t coneSize = RentedNodes[LeaseOffset + 3*i + 2];
        NodeRenters[node].insert(int_pair(renter,coneSize));  
      }
      cntr--;
    }
    
    if (debug) { /* -----------------------------------  */
      // We need to collect all the data to be submitted to PetscSynchronizedPrintf
      // We use a C++ string streams for that
      ostringstream txt;
      for (std::map<Point, int_pair_set, Point::Cmp>::iterator nodeRenters_itor=NodeRenters.begin();nodeRenters_itor!= NodeRenters.end();nodeRenters_itor++) {
        Point node = (*nodeRenters_itor).first;
        int_pair_set renterSet   = (*nodeRenters_itor).second;
        // ASSUMPTION on point type
        txt << "[" << rank << "]: __computeCompletion: node (" << node.prefix << "," << node.index << ") is rented by " << renterSet.size() << " renters (renter, cone size):  ";
        for (int_pair_set::iterator renterSet_itor = renterSet.begin(); renterSet_itor != renterSet.end(); renterSet_itor++) 
        {
          txt << "(" << (*renterSet_itor).first << "," << (*renterSet_itor).second << ") ";
        }
        txt << "\n";
      }
      // Now send the C-string behind txt to PetscSynchronizedPrintf
      ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
    }/* -----------------------------------  */
    
    // wait on the original sends to the lessors
    MPI_Status *Lessor_status;
    if (LessorCount) {
      ierr = PetscMalloc((LessorCount)*sizeof(MPI_Status),&Lessor_status); CHKERROR(ierr,"Error in PetscMalloc");
      ierr = MPI_Waitall(LessorCount,Lessor_waits,Lessor_status);          CHKERROR(ierr,"Error in MPI_Waitall");
    }
    

    // Neighbor counts: here the renters receive from the lessors the number of other renters sharing each leased node.
    // Prepare to receive three integers per leased node: two for the node itself and one for the number of neighbors over that node.
    // The buffer has the same structure as LeasedNodes, hence LessorOffsets can be reused.
    // IMPROVE: can probably reduce the message size by a factor of 3 if we assume an ordering on the nodes received from each lessor.
    // ASSUMPTION on Point type
    int32_t *NeighborCounts;
    if(LeasedNodeCount) {
      ierr = PetscMalloc(3*(LeasedNodeCount)*sizeof(PetscInt),&NeighborCounts); CHKERROR(ierr,"Error in PetscMalloc");
    }
    // Post receives for NeighbornCounts
    PetscMPIInt    tag2;
    ierr = PetscObjectGetNewTag(this->petscObj, &tag2); CHKERROR(ierr, "Failded on PetscObjectGetNewTag");
    for (int32_t i=0; i<LessorCount; i++) {
      ierr = MPI_Irecv(NeighborCounts+LessorOffsets[i],3*LeaseSizes[i],MPIU_INT,Lessors[i],tag2,this->comm,&Lessor_waits[i]);
      CHKERROR(ierr,"Error in MPI_Irecv");
    }
    // pack and send messages back to renters; we need to send 3 integers per rental (2 for Point, 1 for sharer count) 
    // grouped by the renter
    // ASSUMPTION on Point type
    // first we compute the total number of rentals
    int32_t TotalRentalCount = 0;
    for(std::map<Point, int_pair_set, Point::Cmp>::iterator nodeRenters_itor=NodeRenters.begin();nodeRenters_itor!=NodeRenters.end();nodeRenters_itor++){
      TotalRentalCount += (*nodeRenters_itor).second.size();
    }
    if(debug2) {
      ierr = PetscSynchronizedPrintf(this->comm, "[%d]: TotalRentalCount %d\n", rank, TotalRentalCount); 
      CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm); CHKERROR(ierr, "PetscSynchronizedFlush");
    }/* -----------------------------------  */

    // Allocate sharer counts array for all rentals
    int32_t *SharerCounts;
    if(TotalRentalCount) {
      ierr = PetscMalloc(3*(TotalRentalCount)*sizeof(int32_t),&SharerCounts); CHKERROR(ierr,"Error in PetscMalloc");
    }
    // Renters are traversed in the order of their original arrival index by arrival number a
    int32_t RenterOffset = 0;
    cntr = 0;
    for(int32_t a = 0; a < RenterCount; a++) {
      // traverse the nodes leased by the renter
      int32_t RenterLeaseOffset = a*3*MaxLeaseSize;
      for(int32_t i = 0; i < RenterLeaseSizes[a]; i++) {
        // ASSUMPTION on Point type
        Point node;
        node.prefix = RentedNodes[RenterLeaseOffset + 3*i];
        node.index  = RentedNodes[RenterLeaseOffset + 3*i + 1];
        SharerCounts[cntr++]   = node.prefix;
        SharerCounts[cntr++]   = node.index;        
        // Decrement the sharer count by one not to count the current renter itself (with arrival number a).
        SharerCounts[cntr++] = NodeRenters[node].size()-1;
      }
      // Send message to renter
      ierr      = MPI_Isend(SharerCounts+RenterOffset,3*RenterLeaseSizes[a],MPIU_INT,Renters[a],tag2,this->comm,Renter_waits+a);
      CHKERROR(ierr, "Error in MPI_Isend");
      // Offset is advanced by thrice the number of leased nodes, since we store 3 integers per leased node: Point and cone size
      RenterOffset = cntr;
    }
    // Wait on receives from lessors with the neighbor counts
    if (LessorCount) {
      ierr = MPI_Waitall(LessorCount,Lessor_waits,Lessor_status); CHKERROR(ierr,"Error in MPI_Waitall");
    }
    // Wait on the original sends to the renters
    MPI_Status *Renter_status;
    ierr = PetscMalloc((RenterCount)*sizeof(MPI_Status),&Renter_status);CHKERROR(ierr,"Error in PetscMalloc");
    if(RenterCount) {
      ierr = MPI_Waitall(RenterCount, Renter_waits, Renter_status);CHKERROR(ierr,"Error in MPI_Waitall");
    }
    
    if (debug) { /* -----------------------------------  */
      // Use a C++ string stream to report the numbers of shared nodes leased from each lessor
      ostringstream txt;
      cntr = 0;
      txt << "[" << rank << "]: __computeCompletion: neighbor counts by lessor-node [lessor rank, (node), neighbor count]:  ";
      for(int32_t i = 0; i < LessorCount; i++) {
        // ASSUMPTION on point type
        for(int32_t j = 0; j < LeaseSizes[i]; j++) 
        {
          int32_t prefix, index, sharerCount;
          prefix      = NeighborCounts[cntr++];
          index       = NeighborCounts[cntr++];
          sharerCount = NeighborCounts[cntr++];
          txt << "[" << Lessors[i] <<", (" << prefix << "," << index << "), " << sharerCount << "] ";
        }
      }
      txt << "\n";
      ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm); CHKERROR(ierr, "PetscSynchronizedFlush");
    }/* -----------------------------------  */


    // Now we allocate an array to receive the neighbor ranks and the remote cone sizes for each leased node,
    // hence, the total array size is 2*TotalNeighborCount.
    // Note that the lessor offsets must be recalculated, since they are no longer based on the number of nodes 
    // leased from that lessor, but on the number of neighbor over the nodes leased from that lessor.

    // First we compute the numbers of neighbors over the nodes leased from a given lessor.
    int32_t TotalNeighborCount = 0;
    int32_t *NeighborCountsByLessor;
    if(LessorCount) {
      ierr = PetscMalloc((LessorCount)*sizeof(int32_t), &NeighborCountsByLessor); CHKERROR(ierr, "Error in PetscMalloc");
    }
    cntr = 0;
    for(int32_t i = 0; i < LessorCount; i++) {
      for(int32_t j = 0; j < LeaseSizes[i]; j++) {
        //ASSUMPTION on Point type affects NeighborCountsOffset size
        cntr += 2;
        TotalNeighborCount += NeighborCounts[cntr++];
      }
      if(i == 0) {
        NeighborCountsByLessor[i] = TotalNeighborCount;
      }
      else {
        NeighborCountsByLessor[i] = TotalNeighborCount - NeighborCountsByLessor[i-1];
      }
    }
    if (debug2) { /* -----------------------------------  */
      // Use a C++ string stream to report the numbers of shared nodes leased from each lessor
      ostringstream txt;
      cntr = 0;
      txt << "[" << rank << "]: __computeCompletion: NeighborCountsByLessor [rank, count]:  ";
      for(int32_t i = 0; i < LessorCount; i++) {
          txt << "[" << Lessors[i] <<","  <<  NeighborCountsByLessor[i] << "]; ";
      }
      txt << "\n";
      ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm); CHKERROR(ierr, "PetscSynchronizedFlush");
    }/* -----------------------------------  */
    int32_t *Neighbors = 0;
    if(TotalNeighborCount) {
      ierr = PetscMalloc((2*TotalNeighborCount)*sizeof(int32_t),&Neighbors); CHKERROR(ierr,"Error in PetscMalloc");
    }
  
    // Post receives for Neighbors
    PetscMPIInt    tag3;
    ierr = PetscObjectGetNewTag(this->petscObj, &tag3); CHKERROR(ierr, "Failded on PetscObjectGetNewTag");
    int32_t lessorOffset = 0;
    for(int32_t i=0; i<LessorCount; i++) {
      if(NeighborCountsByLessor[i]) { // We expect messages from lessors with a non-zero NeighborCountsByLessor entry only
        ierr = MPI_Irecv(Neighbors+lessorOffset,2*NeighborCountsByLessor[i],MPIU_INT,Lessors[i],tag3,this->comm,&Lessor_waits[i]);
        CHKERROR(ierr,"Error in MPI_Irecv");
        lessorOffset += 2*NeighborCountsByLessor[i];
      }
    }
    // Pack and send messages back to renters. 
    // For each node p and each renter r (hence for each rental (p,r)) we must send to r a segment consisting of the list of all
    // (rr,cc) such that (p,rr) is a share and cc is the cone size over p at rr.
    // ALTERNATIVE, SCALABILITY: 
    //             1. allocate an array capable of holding all messages to all renters and send one message per renter (more memory)
    //             2. allocate an array capable of holding all rentals for all nodes and send one message per share (more messages).
    // Here we choose 1 since we assume that the memory requirement is modest and the communication time is much more expensive, 
    // however, this is likely to be application-dependent, and a switch should be introduced to change this behavior at will.
    // The rental segments are grouped by the renter recepient and within the renter by the node in the same order as SharerCounts.

    // We need to compute the send buffer size using the SharerCounts array.
    // Traverse the renters in order of their original arrival, indexed by the arrival number a, and then by the nodes leased by a.
    // Add up all entries equal to 2 mod 3 in SharerCounts (0 & 1 mod 3 are node IDs, ASSUMPTION on Point type) and double that number 
    // to account for sharer ranks AND the cone sizes we are sending.
    int32_t SharersSize = 0; // 'Sharers' buffer size
    cntr = 0;
    for(int32_t a = 0; a < RenterCount; a++) {
      // traverse the number of nodes leased by the renter
      for(int32_t i = 0; i < RenterLeaseSizes[a]; i++) {
        SharersSize += SharerCounts[3*cntr+2];
        cntr++;
      }
    }
    SharersSize *= 2;
    // Allocate the Sharers array
    int32_t *Sharers;
    if(SharersSize) {
      ierr = PetscMalloc(SharersSize*sizeof(int32_t),&Sharers); CHKERROR(ierr,"Error in PetscMalloc");
    }
    // Now pack the messages and send them off.
    // Renters are traversed in the order of their original arrival index by arrival number a
    ostringstream txt; // DEBUG
    if(debug2) {
      txt << "[" << rank << "]: __computeCompletion: RenterCount = " << RenterCount << "\n";
    }
    RenterOffset = 0; // this is the current offset into Sharers needed for the send statement
    for(int32_t a = 0; a < RenterCount; a++) {//
      int32_t r = Renters[a];
      int32_t RenterLeaseOffset = a*3*MaxLeaseSize;
      int32_t SegmentSize = 0;
      // traverse the nodes leased by the renter
      for(int32_t i = 0; i < RenterLeaseSizes[a]; i++) {
        // Get a node p rented to r
        // ASSUMPTION on Point type
        Point p;
        p.prefix = RentedNodes[RenterLeaseOffset + 3*i];
        p.index  = RentedNodes[RenterLeaseOffset + 3*i + 1];
        if(debug) {
          txt << "[" << rank << "]: __computeCompletion: renters sharing with " << r << " of node  (" << p.prefix << "," << p.index << ")  [rank, cone size]:  ";
        }
        // now traverse the set of all the renters of p
        for(int_pair_set::iterator pRenters_itor=NodeRenters[p].begin(); pRenters_itor!=NodeRenters[p].end(); pRenters_itor++) {
          int32_t rr = (*pRenters_itor).first;  // rank of a pRenter 
          int32_t cc = (*pRenters_itor).second; // cone size over p at rr
          // skip r itself
          if(rr != r){
            Sharers[RenterOffset+SegmentSize++] = rr;
            Sharers[RenterOffset+SegmentSize++] = cc;
            if(debug) {
              txt << "[" << rr << ","  << cc << "]; ";
            }
          }
        }// for(int_pair_set::iterator pRenters_itor=NodeRenters[p].begin(); pRenters_itor!=NodeRenters[p].end(); pRenters_itor++) {
        if(debug) {
          txt << "\n";
        }
      }// for(int32_t i = 0; i < RenterLeaseSizes[a]; i++) {
      // Send message to renter only if the segment size is positive
      if(SegmentSize > 0) {
        ierr      = MPI_Isend(Sharers+RenterOffset,SegmentSize,MPIU_INT,Renters[a],tag3,this->comm,Renter_waits+a);
        CHKERROR(ierr, "Error in MPI_Isend");
      }
      // Offset is advanced by the segmentSize
      RenterOffset += SegmentSize;
    }//  for(int32_t a = 0; a < RenterCount; a++) {
    if(debug) {
      ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm); CHKERROR(ierr, "PetscSynchronizedFlush");      
    }
     
    // Wait on receives from lessors with the neighbor counts
    if (LessorCount) {
      ierr = MPI_Waitall(LessorCount,Lessor_waits,Lessor_status);CHKERROR(ierr,"Error in MPI_Waitall");
    }
    if (debug) { /* -----------------------------------  */
      // To report the neighbors at each lessor we use C++ a string stream
      ostringstream txt;
      int32_t cntr1 = 0;
      int32_t cntr2 = 0;
      for(int32_t i = 0; i < LessorCount; i++) {
        // ASSUMPTION on point type
        txt << "[" <<rank<< "]: __computeCompletion: neighbors over nodes leased from " <<Lessors[i]<< ":\n";
        int32_t activeLessor = 0;
        for(int32_t j = 0; j < LeaseSizes[i]; j++) 
        {
          int32_t prefix, index, sharerCount;
          prefix = NeighborCounts[cntr1++];
          index = NeighborCounts[cntr1++];
          sharerCount = NeighborCounts[cntr1++];
          if(sharerCount > 0) {
            txt <<"[" << rank << "]:\t(" << prefix <<","<<index<<"):  [rank, coneSize]: ";
            activeLessor++;
          }
          for(int32_t k = 0; k < sharerCount; k++) {
            int32_t sharer = Neighbors[cntr2++];
            int32_t coneSize = Neighbors[cntr2++];
            txt << "[" <<sharer <<", "<< coneSize << "] ";
          }
        }// for(int32_t j = 0; j < LeaseSizes[i]; j++) 
        if(!activeLessor) {
          txt <<"[" << rank << "]:\tnone";
        }
        txt << "\n";
      }// for(int32_t i = 0; i < LessorCount; i++)
      ierr = PetscSynchronizedPrintf(this->comm,txt.str().c_str());CHKERROR(ierr,"Error in PetscSynchronizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
    }/* -----------------------------------  */

    // This concludes the interaction of lessors and renters, and the exchange is completed by a peer-to-peer neighbor cone swap
    // (except we still have to wait on our last sends to the renters -- see below).
    // However, we don't free all of the arrays associated with the lessor-renter exchanges, since some of the data
    // still use those structures.  Here are the arrays we can get rid of:
    if(RenterCount) {
      ierr = PetscFree(RentedNodes);  CHKERROR(ierr, "Error in PetscFree");
    }
    if(SharersSize) {ierr = PetscFree(Sharers); CHKERROR(ierr, "Error in PetscFree");}
    if(LessorCount) {
      ierr = PetscFree(NeighborCountsByLessor); CHKERROR(ierr, "Error in PetscFree");
      ierr = PetscFree(Lessor_status);          CHKERROR(ierr,"Error in PetscFree");
      ierr = PetscFree(Lessor_waits);           CHKERROR(ierr,"Error in PetscFree");
      ierr = PetscFree(LessorOffsets);          CHKERROR(ierr,"Error in PetscFree");
      ierr = PetscFree(LeaseSizes);             CHKERROR(ierr,"Error in PetscFree");
      ierr = PetscFree(Lessors);                CHKERROR(ierr,"Error in PetscFree");
    }
    if(LeasedNodeCount) {
      ierr = PetscFree(LeasedNodes); CHKERROR(ierr,"Error in PetscFree");
    }

    // First we determine the neighbors and the cones over each node to be received from or sent to each neigbor.
    std::map<int32_t, Point__int > NeighborPointConeSizeIn, NeighborPointConeSizeOut;
    // Traverse Neighbors and separate the data by neighbor over each point: NeighborPointConeSizeIn/Out.
    // cntr keeps track of the current position within the Neighbors array, node boundaries are delineated using NeighborCounts.
    // ASSUMPTION: 'Neighbors' stores node renter segments in the same order as NeighborCounts stores the node data.
    cntr = 0;
    for(int32_t i = 0; i < LeasedNodeCount; i++) {
      // ASSUMPTION on Point type
      Point p;
      p.prefix = NeighborCounts[3*i];
      p.index  = NeighborCounts[3*i+1]; 
      int32_t pNeighborsCount = NeighborCounts[3*i+2]; // recall that NeighborCounts lists the number of neighbors after each node
      // extract the renters of p from Neighbors
      for(int32_t j = 0; j < pNeighborsCount; j++) {
        int32_t neighbor = Neighbors[cntr++];
        int32_t coneSize = Neighbors[cntr++];
        // Record the size of the cone over p coming in from neighbor
        NeighborPointConeSizeIn[neighbor][p]= coneSize;
        // Record the size of the cone over p going out to neighbor
        NeighborPointConeSizeOut[neighbor][p] = (*cones)[p].size();
      }
    }// for(int32_t i = 0; i < LeasedNodeCount; i++)


    // Compute total incoming cone sizes by neighbor and the total incomping cone size.
    // Also count the total number of neighbors we will be communicating with
    int32_t  NeighborCount = 0;
    int__int NeighborConeSizeIn;
    int32_t  ConeSizeIn = 0;
    ostringstream txt3;
    // Traverse all of the neighbors  from whom we will be receiving cones.
    for(std::map<int32_t, Point__int>::iterator np_itor = NeighborPointConeSizeIn.begin();np_itor!=NeighborPointConeSizeIn.end(); np_itor++){
      int32_t neighbor = (*np_itor).first;
      NeighborConeSizeIn[neighbor] = 0;
      if(debug2) {
        txt3 << "[" << rank << "]: " << "__computeCompletion: NeighborPointConeSizeIn[" << neighbor << "]: ";
      }
      // Traverse all the points the cones over which we are receiving and add the cone sizes
      for(Point__int::iterator pConeSize_itor = (*np_itor).second.begin(); pConeSize_itor != (*np_itor).second.end(); pConeSize_itor++){
        NeighborConeSizeIn[neighbor] = NeighborConeSizeIn[neighbor] + (*pConeSize_itor).second;
        txt3 << "(" << (*pConeSize_itor).first.prefix << "," << (*pConeSize_itor).first.index << ")" << "->" << (*pConeSize_itor).second << "; ";
      }
      // Accumulate the total cone size
      ConeSizeIn += NeighborConeSizeIn[neighbor];
      NeighborCount++;
      txt3 << "NeighborConeSizeIn[" << neighbor << "]: " << NeighborConeSizeIn[neighbor] << "\n";
    }//for(std::map<int32_t, Point__int>::iterator np_itor=NeighborPointConeSizeIn.begin();np_itor!=NeighborPointConeSizeIn.end(); np_itor++)
    if(debug2) {
      ierr = PetscSynchronizedPrintf(this->comm,txt3.str().c_str());CHKERROR(ierr,"Error in PetscSynchronizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
    }
    if(debug) {/* --------------------------------------------------------------------------------------------- */
      ostringstream txt;
      txt << "[" << rank << "]: __computeCompletion: total size of incoming cone: " << ConeSizeIn << "\n";
      for(int__int::iterator np_itor = NeighborConeSizeIn.begin();np_itor!=NeighborConeSizeIn.end();np_itor++)
      {
        int32_t neighbor = (*np_itor).first;
        int32_t coneSize = (*np_itor).second;
        txt << "[" << rank << "]: __computeCompletion: size of cone from " << neighbor << ": " << coneSize << "\n";

      }//int__int::iterator np_itor=NeighborConeSizeIn.begin();np_itor!=NeighborConeSizeIn.end();np_itor++)
      ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str());
      CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm);
      CHKERROR(ierr, "Error in PetscSynchronizedFlush");
    }/* --------------------------------------------------------------------------------------------- */
    // Now we can allocate a receive buffer to receive all of the remote cones from neighbors
    int32_t *ConesIn;
    // ASSUMPTION on Point type affects ConesIn size -- double ConeSizeIn if Point uses two ints
    if(ConeSizeIn) {
      ierr = PetscMalloc(2*ConeSizeIn*sizeof(PetscInt),&ConesIn); CHKERROR(ierr,"Error in PetscMalloc");
    }
  
    // Wait on the original sends to the renters (the last vestige of the lessor-renter exchange epoch; we delayed it to afford the
    // greatest opportunity for a communication-computation overlap).
    if(RenterCount) {
      ierr = MPI_Waitall(RenterCount, Renter_waits, Renter_status); CHKERROR(ierr,"Error in MPI_Waitall");
    }
    if(RenterCount) {
      ierr = PetscFree(Renter_waits); CHKERROR(ierr, "Error in PetscFree");
      ierr = PetscFree(Renter_status); CHKERROR(ierr, "Error in PetscFree");
    }

    // Allocate receive requests
    MPI_Request *NeighborsIn_waits;
    if(NeighborCount) {
      ierr = PetscMalloc((NeighborCount)*sizeof(MPI_Request),&NeighborsIn_waits);CHKERROR(ierr,"Error in PetscMalloc");
    }
    // Post receives for ConesIn
    PetscMPIInt    tag4;
    ierr = PetscObjectGetNewTag(this->petscObj, &tag4); CHKERROR(ierr, "Failded on PetscObjectGetNewTag");
    // Traverse all neighbors from whom we are receiving cones
    int32_t NeighborOffset = 0;
    int32_t n = 0;
    if(debug2) {
      ierr = PetscSynchronizedPrintf(this->comm, "[%d]: __computeCompletion: NeighborConeSizeIn.size() = %d\n",rank, NeighborConeSizeIn.size());
      CHKERROR(ierr, "Error in PetscSynchornizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm);
      CHKERROR(ierr, "Error in PetscSynchornizedFlush");
      if(NeighborConeSizeIn.size()) {
        ierr=PetscSynchronizedPrintf(this->comm, "[%d]: __computeCompletion: *NeighborConeSizeIn.begin() = (%d,%d)\n",
                                     rank, (*NeighborConeSizeIn.begin()).first, (*NeighborConeSizeIn.begin()).second);
        CHKERROR(ierr, "Error in PetscSynchornizedPrintf");
        ierr = PetscSynchronizedFlush(this->comm);
        CHKERROR(ierr, "Error in PetscSynchornizedFlush");
        
      }
    }
    for(std::map<int32_t, int32_t>::iterator n_itor = NeighborConeSizeIn.begin(); n_itor!=NeighborConeSizeIn.end(); n_itor++) {
      int32_t neighbor = (*n_itor).first;
      int32_t coneSize = (*n_itor).second;
      // ASSUMPTION on Point type affects NeighborOffset and coneSize
      ierr = MPI_Irecv(ConesIn+NeighborOffset,2*coneSize,MPIU_INT,neighbor,tag4,this->comm, NeighborsIn_waits+n);
      CHKERROR(ierr, "Error in MPI_Irecv");
      // ASSUMPTION on Point type affects NeighborOffset
      NeighborOffset += 2*coneSize;
      n++;
    }

    // Compute the total outgoing cone sizes by neighbor and the total outgoing cone size.
    int__int NeighborConeSizeOut;
    int32_t  ConeSizeOut = 0;
    // Traverse all the neighbors to whom we will be sending cones
    for(std::map<int32_t, Point__int>::iterator np_itor=NeighborPointConeSizeOut.begin();np_itor!=NeighborPointConeSizeOut.end();np_itor++){
      int32_t neighbor = (*np_itor).first;
      NeighborConeSizeOut[neighbor] = 0;
      // Traverse all the points cones over which we are sending and add up the cone sizes.
      for(Point__int::iterator pConeSize_itor = (*np_itor).second.begin(); pConeSize_itor != (*np_itor).second.end(); pConeSize_itor++){
        NeighborConeSizeOut[neighbor] = NeighborConeSizeOut[neighbor] + (*pConeSize_itor).second;
      }
      // Accumulate the total cone size
      ConeSizeOut += NeighborConeSizeOut[neighbor];
    }//for(std::map<int32_t,Point__int>::iterator np_itor=NeighborPointConeSizeOut.begin();np_itor!=NeighborPointConeSizeOut.end();np_itor++)


    // Now we can allocate a send buffer to send all of the remote cones to neighbors
    int32_t *ConesOut;
    // ASSUMPTION on Point type affects ConesOut size -- double ConeSizeOut if Point takes up 2 ints
    ierr = PetscMalloc(2*ConeSizeOut*sizeof(PetscInt),&ConesOut); CHKERROR(ierr,"Error in PetscMalloc");
    // Allocate send requests
    MPI_Request *NeighborsOut_waits;
    if(NeighborCount) {
      ierr = PetscMalloc((NeighborCount)*sizeof(MPI_Request),&NeighborsOut_waits);CHKERROR(ierr,"Error in PetscMalloc");
    }

    // Pack and send messages
    NeighborOffset = 0;
    cntr = 0;
    n = 0;
    ostringstream txt2;
    if(debug) {/* --------------------------------------------------------------------------------------------- */
      txt2 << "[" << rank << "]: __computeCompletion: total outgoing cone size: " << ConeSizeOut << "\n";
    }/* --------------------------------------------------------------------------------------------- */
    // Traverse all neighbors to whom we are sending cones
    for(std::map<int32_t, Point__int>::iterator np_itor=NeighborPointConeSizeOut.begin();np_itor!=NeighborPointConeSizeOut.end(); np_itor++){
      int32_t neighbor = (*np_itor).first;
      if(debug) { /* ------------------------------------------------------------ */
        txt2  << "[" << rank << "]: __computeCompletion: outgoing cones destined for " << neighbor << "\n";
      }/* ----------------------------------------------------------------------- */
      // ASSUMPTION: all Point__int maps are sorted the same way, so we safely can assume that 
      //             the receiver will be expecting points in the same order
      // Traverse all the points within this neighbor 
      for(Point__int::iterator pConeSize_itor = (*np_itor).second.begin(); pConeSize_itor != (*np_itor).second.end(); pConeSize_itor++)
      {
        Point p = (*pConeSize_itor).first;
        if(debug) { /* ------------------------------------------------------------ */
          txt2  << "[" << rank << "]: \t cone over (" << p.prefix << ", " << p.index << "):  ";
        }/* ----------------------------------------------------------------------- */
        // Traverse the local cone over p and store it in ConesOut
        for(Point_set::iterator cone_itor = (*cones)[p].begin(); cone_itor != (*cones)[p].end(); cone_itor++) {
          // ASSUMPTION on Point type affects storage of points in ConesOut
          Point q = *cone_itor;
          ConesOut[cntr++] =  q.prefix;
          ConesOut[cntr++] =  q.index;
          if(debug) { /* ------------------------------------------------------------ */
            txt2  << "(" << q.prefix << ", " << q.index << ") ";
          }/* ----------------------------------------------------------------------- */
        }
        if(debug) { /* ------------------------------------------------------------ */
          txt2  << "\n";
        }/* ----------------------------------------------------------------------- */
      }//for(Point__int::iterator pConeSize_itor = (*np_itor).second.begin(); pConeSize_itor!=(*np_itor).second.end();pConeSize_itor++)
      int32_t coneSize = NeighborConeSizeOut[neighbor];
      // ASSUMPTION on Point type  affects the usage of coneSize and the calculation of Neighbor offset
      ierr = MPI_Isend(ConesOut+NeighborOffset,2*coneSize,MPIU_INT,neighbor,tag4,this->comm, NeighborsOut_waits+n);
      CHKERROR(ierr, "Error in MPI_Isend");
      // ASSUMPTION on Point type affects the computation of NeighborOffset -- double coneSize if Point uses up 2 ints
      NeighborOffset += 2*coneSize; // keep track of offset
      n++;  // count neighbors
    }//for(std::map<int32_t,Point__int>::iterator np_itor=NeighborPointConeSizeIn.begin();np_itor!=NeighborPointConeSizeIn.end();np_itor++){
    if(debug) {/* --------------------------------------------------------------------------------------------- */
      ierr = PetscSynchronizedPrintf(this->comm, txt2.str().c_str());
      CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm);
      CHKERROR(ierr, "Error in PetscSynchronizedFlush");
    }/* --------------------------------------------------------------------------------------------- */

    // Allocate a status array
    MPI_Status *Neighbor_status;
    if(NeighborCount) {
      ierr = PetscMalloc((NeighborCount)*sizeof(MPI_Status),&Neighbor_status);CHKERROR(ierr,"Error in PetscMalloc");
    }

    // Wait on the receives
    if(NeighborCount) {
      ierr = MPI_Waitall(NeighborCount, NeighborsIn_waits, Neighbor_status); CHKERROR(ierr,"Error in MPI_Waitall");
    }
    
    // Now we unpack the received cones and store them in the completion stack C.
    // Traverse all neighbors  from whom we are expecting cones
    cntr = 0; // cone point counter
    for(std::map<int32_t,Point__int>::iterator np_itor=NeighborPointConeSizeIn.begin();np_itor!=NeighborPointConeSizeIn.end();np_itor++)
    {
      int32_t neighbor = (*np_itor).first;
      // Traverse all the points over which have received cones from neighbors
      // ASSUMPTION: points are sorted within each neighbor, so we are expecting points in the same order as they arrived in ConesIn
      for(Point__int::iterator pConeSize_itor = (*np_itor).second.begin(); pConeSize_itor != (*np_itor).second.end(); pConeSize_itor++){
        Point p = (*pConeSize_itor).first;
        Point s;  // this is the structure point whose footprint will be recorded
        int32_t coneSize = (*pConeSize_itor).second;
        //Traverse the cone points received in ConesIn
        for(int32_t i = 0; i < coneSize; i++) {
          // Insert into C an arrow from/to each arrived point q to/from p, depending on completedSet: cap/base.
          // ASSUMPTION on Point type affects the storage of nodes in ConesIn and the usage of cntr
          Point q = Point(ConesIn[2*cntr], ConesIn[2*cntr+1]);
          Point p0; // initial point of the completion arrow
          Point p1; // final point of the completion arrow
          cntr++;
          if(completedSet == completedSetCap) {
            p0 = q;
            p1 = p;
          }
          else if (completedSet == completedSetBase) {
            p0 = p;
            p1 = q;
          }
          else {
            throw ALE::Exception("Unknown completedSet");
          }
          //
          P->addArrow(p0,p1);
          if(completionType == completionTypeArrow) {
            // create an arrow node and insert it into A
            Point a(this->commRank, cntr);
            A->addPoint(a);
            // now add the arrows from p0 to a and from a to p1
            A0->addArrow(p0,a); A1->addArrow(a,p1);
            // save the footprint point
            s = a;
          }
          else {
            // save the footprint point
            s = q;
          }
          if(footprintType != footprintTypeNone) {
            // Record the footprint of s
            Point f;
            //f.prefix = -(rank+1);
            f.prefix = rank;
            f.index  = neighbor;
            // Insert the footprint point into F
            F->addPoint(f);
            // Add the arrow to/from f from/to s into S
            if(footprintType == footprintTypeSupport) {
              C->addArrow(f,s);
            }
            else if(footprintType == footprintTypeCone) {
              C->addArrow(s,f);
            }
            else {
              throw Exception("Unknown footprintType");
            }
          }
        }// for(int32_t i = 0; i < coneSize; i++)
      }
    }//for(std::map<int32_t, Point__int>::iterator np_itor=NeighborPointConeSizeIn.begin();np_itor!=NeighborPointConeSizeIn.end(); np_itor++)
    if (debug) { /* -----------------------------------  */
      ostringstream txt;
      cntr = 0;
      txt << "[" << rank << "]: debug begin\n";
      txt << "[" << rank << "]: __computeCompletion: completion cone:  base of size " << C->_cone.size() << "\n";
      for(std::map<Point,Point_set,Point::Cmp>::iterator Cbase_itor=C->_cone.begin(); Cbase_itor!=C->_cone.end(); Cbase_itor++) {
        Point Cb = (*Cbase_itor).first;
        txt << "[" << rank << "]: \t(" << Cb.prefix << ", " << Cb.index << "):  ";
        Point_set CbCone = (*Cbase_itor).second;
        txt << "cone of size " << CbCone.size() << ":  ";
        for(Point_set::iterator CbCone_itor = CbCone.begin(); CbCone_itor != CbCone.end(); CbCone_itor++)
        {
          Point q = *CbCone_itor;
          txt << "(" << q.prefix <<"," << q.index << ") ";
        }
        txt << "\n";
      }
      ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
      ierr = PetscSynchronizedFlush(this->comm); CHKERROR(ierr, "PetscSynchronizedFlush");
      txt << "[" << rank << "]: debug end\n";
    }/* -----------------------------------  */

    // Wait on the original sends
    if(NeighborCount) {
      ierr = MPI_Waitall(NeighborCount, NeighborsOut_waits, Neighbor_status); CHKERROR(ierr,"Error in MPI_Waitall");
    }

    // Computation complete; freeing memory.
    // Some of these can probably be freed earlier, if memory is needed.
    // However, be careful while freeing memory that may be in use implicitly.  
    // For instance, ConesOut is a send buffer and should probably be retained until all send requests have been waited on.
    if(NeighborCount){
      ierr = PetscFree(NeighborsOut_waits); CHKERROR(ierr, "Error in PetscFree");
      ierr = PetscFree(NeighborsIn_waits);  CHKERROR(ierr, "Error in PetscFree");
      ierr = PetscFree(Neighbor_status);    CHKERROR(ierr, "Error in PetscFree");
    }
    if(LeasedNodeCount) {
      ierr = PetscFree(NeighborCounts); CHKERROR(ierr,"Error in PetscFree");
    }

    if(TotalNeighborCount) {ierr = PetscFree(Neighbors); CHKERROR(ierr, "Error in PetscFree");}    
    if(ConeSizeIn) {ierr = PetscFree(ConesIn);           CHKERROR(ierr, "Error in PetscFree");}
    if(ConeSizeOut){ierr = PetscFree(ConesOut);          CHKERROR(ierr, "Error in PetscFree");}
    if(TotalRentalCount){ierr = PetscFree(SharerCounts); CHKERROR(ierr, "Error in PetscFree");}

    if(completedSet == completedSetCap) {
      delete points;
    }
    else if(completedSet == completedSetBase) {
      delete cones;
    }
    else {
      throw ALE::Exception("Unknown completionType");
    }
    return C;
    // Done! 
    
  }// PreSieve::__computeCompletion()


} // namespace ALE

#undef ALE_PreSieve_cxx
