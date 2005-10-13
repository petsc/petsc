#ifndef included_ALE_PreSieve_hh
#define included_ALE_PreSieve_hh

#ifndef  included_ALE_ALE_hh
#include <ALE.hh>
#endif

namespace ALE {

  class Stack;
  class PreSieve : public Coaster {
  protected:
    Point_set                  _roots;
    Point_set                  _leaves;
    Point__Point_set           _cone;
    Point_set                  _cap;
    void                       __computeBaseExclusion(Point_set& base, PreSieve *s);
    void                       __computeCapExclusion(Point_set& cap, PreSieve *s);
    void                       __determinePointOwners(Point_set& points, int32_t *LeaseData, std::map<Point,int32_t,Point::Cmp>& owner);
    static const int           completedSetCap             = 0;
    static const int           completedSetBase            = 1;
    Stack *__computeCompletion(int32_t completedSet, int32_t completionType, int32_t footprintType, PreSieve *c);
  public:
    // 
    PreSieve();
    PreSieve(MPI_Comm comm);
    virtual ~PreSieve();
    virtual PreSieve&                 clear();
    virtual void                      view(const char *name);
    //----------------------------------------------------------------------
    // IMPROVE: implement deep copy when internals become Obj<X>
    virtual PreSieve&                 copy(PreSieve& s){*this = s; return *this;};  
    virtual PreSieve&                 addArrow(Point& i, Point& j);
    virtual PreSieve&                 removeArrow(Point& i, Point& j);
    virtual PreSieve&                 addBasePoint(Point& p);
    virtual PreSieve&                 removeBasePoint(Point& p);
    virtual PreSieve&                 addPoint(Point& p) {this->addBasePoint(p); this->addCapPoint(p); return *this;};
    virtual PreSieve&                 removePoint(Point& p) {this->removeBasePoint(p); this->removeCapPoint(p); return *this;};
    virtual PreSieve&                 addCapPoint(Point& q);
    virtual PreSieve&                 removeCapPoint(Point& q);
    virtual PreSieve&                 addSupport(Point& i, Point_set& suppSet) {
      CHKCOMM(*this);
      this->__checkLock();
      // Add i to the cap in case the supp set is empty and no addArrow are executed
      this->addCapPoint(i);
      // IMPROVE: keep the support size in _cap
      for(Point_set::iterator supp_itor = suppSet.begin(); supp_itor != suppSet.end(); supp_itor++) {
        Point j = (*supp_itor);
        this->addArrow(i,j);
      }
      return *this;
    };
    virtual PreSieve&                 addSupport(Point& i, Point& supp) {
      CHKCOMM(*this);
      this->__checkLock();
      // Add i to the cap in case the supp set is empty and no addArrow are executed
      this->addCapPoint(i);
      // IMPROVE: keep the support size in _cap
      this->addArrow(i,supp);
      return *this;
    };
    virtual PreSieve&                 setSupport(Point& i, Point_set& supp) {
      CHKCOMM(*this);
      this->__checkLock();
      // IMPROVE: use support iterator
      Point_set support = this->support(i);
      for(Point_set::iterator support_itor = support.begin(); support_itor != support.end(); support_itor++) {
        Point j = (*support_itor);
        this->removeArrow(i,j);
      }
      this->addSupport(i,supp);
      return *this;
    };
    virtual PreSieve&                 setSupport(Point& i, Point& s) {
      CHKCOMM(*this);
      this->__checkLock();
      // IMPROVE: use support iterator
      Point_set support = this->support(i);
      for(Point_set::iterator support_itor = support.begin(); support_itor != support.end(); support_itor++) {
        Point j = (*support_itor);
        this->removeArrow(i,j);
      }
      this->addSupport(i,s);
      return *this;
    };
    virtual PreSieve&                 addCone(Point_set& coneSet, Point& j) {
      CHKCOMM(*this);
      this->__checkLock();
      // Add j to the base in case coneSet is empty and no addArrow are executed
      this->addBasePoint(j);
      // IMPROVE: keep the support size in _cap
      for(Point_set::iterator cone_itor = coneSet.begin(); cone_itor != coneSet.end(); cone_itor++) {
        Point i = (*cone_itor);
        this->addArrow(i,j);
      }
      return *this;
    };
    virtual PreSieve&                 addCone(Point& cone, Point& j) {
      CHKCOMM(*this);
      this->__checkLock();
      // Add j to the base in case coneSet is empty and no addArrow are executed
      this->addBasePoint(j);
      // IMPROVE: keep the support size in _cap
      this->addArrow(cone,j);
      return *this;
    };
    virtual PreSieve&                 setCone(Point_set& coneSet, Point& j) {
      CHKCOMM(*this);
      this->__checkLock();
      // IMPROVE: use support iterator
      Point_set cone = this->cone(j);
      for(Point_set::iterator cone_itor = cone.begin(); cone_itor != cone.end(); cone_itor++) {
        Point i = (*cone_itor);
        this->removeArrow(i,j);
      }
      this->addCone(coneSet,j);
      return *this;
    };
    virtual PreSieve&                 setCone(Point& i, Point& j) {
      CHKCOMM(*this);
      this->__checkLock();
      // IMPROVE: use support iterator
      Point_set cone = this->cone(j);
      for(Point_set::iterator cone_itor = cone.begin(); cone_itor != cone.end(); cone_itor++) {
        Point i = (*cone_itor);
        this->removeArrow(i,j);
      }
      this->addCone(i,j);
      return *this;
    };
    virtual Point_set                 space();
    virtual int32_t                   spaceSize();
    virtual int32_t                   *spaceSizes();
    virtual int32_t                   baseSize();
    virtual int32_t                   *baseSizes();
    virtual Point_set                 base();
    virtual int32_t                   capSize();
    virtual int32_t                   *capSizes();
    virtual Point_set                 cap();

    virtual PreSieve&                 stackAbove(PreSieve& s);
    virtual PreSieve&                 stackBelow(PreSieve& s);
    virtual PreSieve&                 invert();
    virtual PreSieve&                 restrictBase(Point_set& base);
    virtual PreSieve&                 restrictCap(Point_set& cap);
    virtual PreSieve*                 baseRestriction(Point_set& base);
    virtual PreSieve*                 capRestriction(Point_set& cap);
    virtual PreSieve&                 excludeBase(Point_set& base);
    virtual PreSieve&                 excludeCap(Point_set& cap);
    virtual PreSieve*                 baseExclusion(Point_set& base);
    virtual PreSieve*                 capExclusion(Point_set& cap);
    virtual int                       spaceContains(Point point);
    virtual int                       baseContains(Point point);
    virtual int                       capContains(Point point);
    virtual int                       coneContains(Point& p, Point point);
    virtual int                       supportContains(Point& q, Point point);
    virtual Point_set                 roots(){return this->_roots;};
    virtual Point_set                 leaves(){return this->_leaves;};
    Obj<Point_set>                    cone(Obj<Point_set> chain) {return this->nCone(chain,1);};
    Point_set                         cone(Point& point) {
      CHKCOMM(*this);
      Point_set cone = this->nCone(point,1);
      return cone;
    };
    Point_set                         cone(Point_set& chain) {
      CHKCOMM(*this);
      Point_set cone = this->nCone(chain,1);
      return cone;
    };
    int32_t                           coneSize(Point& p) {
      Point_set pSet; pSet.insert(p);
      return coneSize(pSet);
    };
    int32_t                           coneSize(Point_set& chain);
    Obj<Point_set>                    nCone(Obj<Point_set> chain, int32_t n);
    Point_set                         nCone(Point_set& chain, int32_t n);
    Point_set                         nCone(Point& point, int32_t n) {
      CHKCOMM(*this);
      // Compute the point set obtained by taking the cone recursively on a point in the base
      // (i.e., the set of cap points resulting after each iteration is used again as the base for the next cone computation).
      // Note: a 0-cone is the point itself.
      Point_set chain; chain.insert(point);
      return this->nCone(chain,n);
    };
    Obj<Point_set>                    nClosure(Obj<Point_set> chain, int32_t n);
    Point_set                         nClosure(Point_set& chain, int32_t n);
    Point_set                         nClosure(Point& point, int32_t n) {
      CHKCOMM(*this);
      // Compute the point set obtained by recursively accumulating the cone over a point in the base
      // (i.e., the set of cap points resulting after each iteration is both stored in the resulting set and 
      // used again as the base of a cone computation).
      // Note: a 0-closure is the point itself.
      Point_set chain; chain.insert(point);
      return this->nClosure(chain,n);
    };
    Obj<PreSieve>                     nClosurePreSieve(Obj<Point_set> chain, int32_t n, Obj<PreSieve> closure = Obj<PreSieve>());
    //
    Obj<Point_set>                    support(Obj<Point_set> chain) {return this->nSupport(chain,1);};
    Point_set                         support(Point_set& chain) {
      CHKCOMM(*this);
      // IMPROVE: keep the support size in _cap
      Point_set supp = this->nSupport(chain,1);
      return supp;
    };
    Point_set                         support(Point& point) {
      CHKCOMM(*this);
      // IMPROVE: keep the support size in _cap
      Point_set supp = this->nSupport(point,1);
      return supp;
    };
    int32_t                           supportSize(Point_set& chain);
    int32_t                           supportSize(Point& p) {
      Point_set pSet; pSet.insert(p);
      return supportSize(pSet);
    };
    Obj<Point_set>                    nSupport(Obj<Point_set> chain, int32_t n);
    Point_set                         nSupport(Point_set& chain, int32_t n);
    Point_set                         nSupport(Point& point, int32_t n) {
      CHKCOMM(*this);
      // IMPROVE: keep the support size in _cap
      // Compute the point set obtained by taking the support recursively on a point in the cap
      // (i.e., the set of base points resulting after each iteration is used again as the cap for the next support computation).
      // Note: a 0-support is the point itself.
      Point_set chain; chain.insert(point);
      return this->nSupport(chain,n);
    };
    Obj<Point_set>                    nStar(Obj<Point_set> chain, int32_t n);
    Point_set                         nStar(Point_set& chain, int32_t n);
    Point_set                         nStar(Point& point, int32_t n) {
      CHKCOMM(*this);
      // IMPROVE: keep the support size in _cap
      // Compute the point set obtained by accumulating the recursively-computed support of a  point in the cap
      // (i.e., the set of base points resulting after each iteration is accumulated in the star AND used again as
      // the cap for the next support computation).
      // Note: a 0-star is the point itself.
      Point_set chain; chain.insert(point);
      return this->nStar(chain,n);
    };
    Obj<PreSieve> PreSieve::nStarPreSieve(Obj<Point_set> chain, int32_t n, Obj<PreSieve> star = Obj<PreSieve>());
    //
    Point_set                         nMeet(Point_set c1,  Point_set c2, int32_t n);
    Point_set                         nJoin(Point_set c1, Point_set c2, int32_t n);
    Point_set                         nMeetAll(Point_set& chain, int32_t n);
    Point_set                         nJoinAll(Point_set& chain, int32_t n);
    PreSieve*                         add(PreSieve& s);
    PreSieve&                         add(Obj<PreSieve> s);
    //
    static const int                  completionTypePoint         = 0;
    static const int                  completionTypeArrow         = 1;
    static const int                  footprintTypeNone           = 0;                 
    static const int                  footprintTypeCone           = 1;                 
    static const int                  footprintTypeSupport        = 2;                 
    virtual Stack*                    coneCompletion(   int completionType, int footprintType, PreSieve *c);
    virtual Stack*                    supportCompletion(int completionType, int footprintType, PreSieve *c);
    virtual Stack*                    baseFootprint( int completionType, int footprintType,  PreSieve *f);
    virtual Stack*                    capFootprint(  int completionType, int footprintType,  PreSieve *f);
    virtual Stack*                    spaceFootprint(int completionType, int footprintType,  PreSieve *f);
  };

  
} // namespace ALE

#endif
