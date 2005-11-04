#define ALE_Stack_cxx

#ifndef included_ALE_Stack_hh
#include <Stack.hh>
#endif

namespace ALE {

  #undef  __FUNCT__
  #define __FUNCT__ "Stack::top"
  Obj<PreSieve>      Stack::top() {
    return _top;
  }// Stack::top()

  #undef  __FUNCT__
  #define __FUNCT__ "Stack::setTop"
  Stack&             Stack::setTop(Obj<PreSieve> top) { 
    CHKCOMM(*this);
    this->__checkLock();
    PreSieve::clear(); 
    this->_top = top; 
    return *this;
  }// Stack::setTop()

  #undef  __FUNCT__
  #define __FUNCT__ "Stack::left"
  Obj<PreSieve>      Stack::left() {
    return top();
  }// Stack::left()

  #undef  __FUNCT__
  #define __FUNCT__ "Stack::setLeft"
  Stack&             Stack::setLeft(Obj<PreSieve> left) { 
    return this->setTop(left);
  }// Stack::setLeft()


  #undef  __FUNCT__
  #define __FUNCT__ "Stack::bottom"
  Obj<PreSieve>      Stack::bottom() {
    return _bottom;
  }// Stack::bottom()

  #undef  __FUNCT__
  #define __FUNCT__ "Stack::setBottom"
  Stack&             Stack::setBottom(Obj<PreSieve> bottom) {
    CHKCOMM(*this);
    this->__checkLock();
    this->_bottom = bottom; 
    PreSieve::clear(); 
    return *this;
  }// Stack::setBottom()

  #undef  __FUNCT__
  #define __FUNCT__ "Stack::right"
  Obj<PreSieve>      Stack::right() {
    return this->bottom();
  }// Stack::right()

  #undef  __FUNCT__
  #define __FUNCT__ "Stack::setRight"
  Stack&             Stack::setRight(Obj<PreSieve> right) {
    return this->setBottom(right);
  }// Stack::setBottom()
 

  #undef  __FUNCT__
  #define __FUNCT__ "Stack::baseRestriction"
  Stack*             Stack::baseRestriction(Point_set& base) {
    CHKCOMM(*this);
    Stack *s = new Stack(this->_top, this->_bottom);
    for(Point_set::iterator b_itor = base.begin(); b_itor != base.end(); b_itor++){
      Point p = *b_itor;
      // is point p present in the base of *this?
      if(this->_cone.find(p) == this->_cone.end()){
        s->addCone(this->_cone[p],p);
      }
    }// for(Point_set::iterator b_itor = base.begin(); b_itor != base.end(); b_itor++){
    return s;
  }// Stack::baseRestriction()

  #undef  __FUNCT__
  #define __FUNCT__ "Stack::baseExclusion"
  Stack*             Stack::baseExclusion(Point_set& base) {
    CHKCOMM(*this);
    Stack *s = new Stack(this->_top, this->_bottom);
    this->__computeBaseExclusion(base, s);
    return s;
  }// Stack::baseExclusion()

  #undef  __FUNCT__
  #define __FUNCT__ "Stack::capRestriction"
  Stack*             Stack::capRestriction(Point_set& cap) {
    CHKCOMM(*this);
    Stack *s = new Stack(this->_top, this->_bottom);
    for(Point_set::iterator c_itor = cap.begin(); c_itor != cap.end(); c_itor++){
      Point q = *c_itor;
      // is point q present in the cap of *this?
      if(this->_support.find(q) == this->_support.end()){
        s->addSupport(q,this->_support[q]);
      }
    }// for(Point_set::iterator c_itor = cap.begin(); c_itor != cap.end(); c_itor++){
    return s;
  }// Stack::capRestriction()


  #undef  __FUNCT__
  #define __FUNCT__ "Stack::capExclusion"
  Stack*             Stack::capExclusion(Point_set& cap) {
    CHKCOMM(*this);
    Stack *s = new Stack(this->_top, this->_bottom);
    this->__computeCapExclusion(cap, s);
    return s;
  }// Stack::capExclusion()

  // ---------------------------------------------------------------------------------------
  #undef  __FUNCT__
  #define __FUNCT__ "Stack::addBasePoint"
  Stack&             Stack::addBasePoint(Point& p) {
    this->__checkLock();
    // Check whether the point is in the _bottom, and reject the addition if it isn't
    if(!this->_bottom->spaceContains(p)) {
      throw ALE::Exception("Stack cannot add base points absent from its bottom Sieve");
    }
    // Invoke PreSieve's addition method
    PreSieve::addBasePoint(p);
    return *this;
  }// Stack::addBasePoint()

  #undef  __FUNCT__
  #define __FUNCT__ "Stack::addCapPoint"
  Stack&             Stack::addCapPoint(Point& p) {
    this->__checkLock();
    // Check whether the point is in the _top, and reject the addition if it isn't
    if(!this->_top->spaceContains(p)) {
      throw ALE::Exception("Stack cannot add cap points absent from its top Sieve");
    }
    // Invoke PreSieve's addition method
    PreSieve::addCapPoint(p);
    return *this;
  }// Stack::addCapPoint()

  #undef  __FUNCT__
  #define __FUNCT__ "Stack::stackAbove"
  Stack&             Stack::stackAbove(Stack& s) {
    PreSieve::stackAbove(s); 
    this->_bottom = s._bottom; 
    return *this;
  }// Stack::stackAbove()
 
  #undef  __FUNCT__
  #define __FUNCT__ "Stack::stackBelow"
  Stack&             Stack::stackBelow(Stack& s) {
    PreSieve::stackBelow(s); 
    this->_top = s._top; 
    return *this;
  }// Stack::stackAbove()
 

  #undef  __FUNCT__
  #define __FUNCT__ "Stack::invert"
  Stack&             Stack::invert() {
    PreSieve::invert(); 
    Obj<PreSieve> tmp = this->_top; 
    this->_top = this->_bottom; 
    this->_bottom = tmp; 
    return *this;
  }// Stack::invert()

  #undef  __FUNCT__
  #define __FUNCT__ "Stack::space()"
  Point_set          Stack::space() {
    // Take the union of the top and bottom spaces
    Point_set top = this->_top->space();
    Point_set bottom = this->_bottom->space();
    Point_set *s = &top, *ss = &bottom, *tmp;
    // Make s the smallest set to limit the iteration loop
    if(ss->size() < s->size()) {tmp = s; s = ss; ss = tmp;}
    for(Point_set::iterator s_itor = s->begin(); s_itor != s->end(); s_itor++) {
      Point p = *s_itor;
      ss->insert(p);
    }
    return *ss;
  }// Stack::space()

  
  #undef __FUNCT__
  #define __FUNCT__ "Stack::view"
  void               Stack::view(const char *name) {
    ostringstream vName, topName, bottomName, hdr;
    // Print header
    PetscErrorCode ierr;
    vName << "vertical part";
    hdr << "Viewing";
    if(this->isLocked()) {
      hdr << " (a locked)";
    }
    hdr << " Stack";
    if(name != NULL) {
      hdr << " " << name << std::endl;
      topName << name << "'s top";
      bottomName << name << "'s bottom";
      vName << " of " << name;
    }
    else {
      topName << "top";
      bottomName << "bottom";
    }
    // Print header
    ierr = PetscPrintf(comm, hdr.str().c_str()); CHKERROR(ierr, "Error in PetscPrintf");
        
    // First view the top and the bottom
    this->_top->view(topName.str().c_str());
    this->_bottom->view(bottomName.str().c_str());
    // Then the vertical part
    PreSieve::view(vName.str().c_str());

  }// Stack::view()

} // namespace ALE

#undef ALE_Stack_cxx
