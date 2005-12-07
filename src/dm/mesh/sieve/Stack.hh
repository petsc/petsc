#ifndef included_ALE_Stack_hh
#define included_ALE_Stack_hh

#ifndef  included_ALE_Sieve_hh
#include <Sieve.hh>
#endif

namespace ALE {


  class Stack : public PreSieve {
  protected:
    Obj<PreSieve>             _top;
    Obj<PreSieve>             _bottom;
  public:
    Stack() : PreSieve(){};
    Stack(MPI_Comm comm) : PreSieve(comm), _top((PreSieve*)NULL), _bottom((PreSieve *)NULL) {};
    Stack(Obj<PreSieve> top,Obj<PreSieve> bottom): PreSieve(top->getComm()),_top(top),_bottom(bottom){
      CHKCOMMS(*top.ptr(),*bottom.ptr());
    };
    Stack(PreSieve& top, PreSieve& bottom) : PreSieve(top.getComm()), _top(&top), _bottom(&bottom) {CHKCOMMS(top,bottom);};
    virtual                   ~Stack(){};
    virtual Stack&            clear() {PreSieve::clear(); this->_top = Obj<PreSieve>(); this->_bottom = Obj<PreSieve>(); return *this;};
    //----------------------------------------------------------------------
    // Virtual methods overloaded
    virtual Stack&                    getLock(){
      PreSieve::getLock();
      if(!this->_top.isNull()) {
        this->_top->getLock();
      }
      if(!this->_bottom.isNull()) {
        this->_bottom->getLock();
      }
      return *this;
    };
    virtual Stack&                    releaseLock(){
      PreSieve::releaseLock();
      if(!this->_top.isNull()) {
        this->_top->releaseLock();
      }
      if(!this->_bottom.isNull()) {
        this->_bottom->releaseLock();
      }
      return *this;
    };
    Stack&                    stackAbove(Stack& s);
    Stack&                    stackBelow(Stack& s);
    Stack&                    invert();
    Stack&                    addBasePoint(Point& p);
    Stack&                    addCapPoint(Point& q);
    // Point removal methods act only on the inherited (PreSieve) part of Stack, so they are inherited
    // Inherited arrow manipulation methods (add/removeArrow, add/setCone/Support, etc)
    // rely on addBasePoint/addCapPoint, so no need to redefine those.
    Obj<Point_set>            space();
    int32_t                   spaceSize(){return this->space()->size();};
    int32_t                   baseSize(){return this->_bottom->spaceSize();};
    Point_set                 base() {return this->_bottom->space();};
    int32_t                   capSize(){return this->_top->spaceSize();};
    Point_set                 cap()  {return this->_top->space();};
    int                       spaceContains(Point p) {return (this->_top->spaceContains(p) || this->_bottom->spaceContains(p));};
    int                       baseContains(Point p) {return this->_bottom->spaceContains(p);};
    int                       capContains(Point p) {return this->_top->spaceContains(p);};
    void                      view(const char *name);
    //----------------------------------------------------------------------
    // New and non-covariant methods
    Obj<PreSieve>             top();
    Obj<PreSieve>             left();
    Stack&                    setTop(Obj<PreSieve> top);
    Stack&                    setLeft(Obj<PreSieve> left);
    Obj<PreSieve>             bottom();
    Obj<PreSieve>             right();
    Stack&                    setBottom(Obj<PreSieve> bottom);
    Stack&                    setRight(Obj<PreSieve> right);
    // Non-covariant methods redefined with the correct return type
    virtual Stack*            baseRestriction(Point_set& base);
    virtual Stack*            capRestriction(Point_set& cap);
    // Exclusion methods act only on the inherited (PreSieve) part of Stack, so they could be inherited, but for the return type
    virtual Stack*            baseExclusion(Point_set& base);
    virtual Stack*            capExclusion(Point_set& cap);
  };

  
} // namespace ALE

#endif
