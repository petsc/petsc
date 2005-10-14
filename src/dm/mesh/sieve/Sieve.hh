
#ifndef included_ALE_Sieve_hh
#define included_ALE_Sieve_hh

#ifndef  included_ALE_PreSieve_hh
#include <PreSieve.hh>
#endif

namespace ALE {


  class Sieve : public PreSieve {
  public:
    typedef enum {additionPolicyAcyclic = 0, additionPolicyStratified} AdditionPolicy;
    typedef enum {stratificationPolicyOnLocking = 0, stratificationPolicyOnMutation} StratificationPolicy;
  protected:
    PreSieve                     _depth;
    PreSieve                     _height;
    inline void                  __setHeight(Point p,int32_t h);       
    inline void                  __setDepth(Point p,int32_t d);       
    void                         __computeStarDepths(Point p){
      Point_set points; points.insert(p);__computeStarDepths(points);
    };
    void                         __computeClosureHeights(Point p){
      Point_set points;points.insert(p);__computeClosureHeights(points);
    };
    void                         __computeStarDepths(Obj<Point_set> points);
    void                         __computeClosureHeights(Obj<Point_set> points);
    AdditionPolicy               _additionPolicy;
    StratificationPolicy         _stratificationPolicy;
  public:
    Sieve();
    Sieve(MPI_Comm comm);
    virtual                   ~Sieve();
    virtual void              setComm(MPI_Comm comm);
    virtual Sieve&            clear();
    virtual Sieve&            getLock();
    //----------------------------------------------------------------------
    Sieve&                    setAdditionPolicy(AdditionPolicy policy);
    AdditionPolicy            getAdditionPolicy();
    Sieve&                    setStratificationPolicy(StratificationPolicy policy);
    StratificationPolicy      getStratificationPolicy();
    virtual Sieve&            addArrow(Point& i, Point& j);
    virtual Sieve&            removeArrow(Point& i, Point& j);
    virtual Sieve&            addBasePoint(Point& p);
    virtual Sieve&            removeBasePoint(Point& p);
    virtual Sieve&            addCapPoint(Point& q);
    virtual Point_set         closure(Point p){Point_set pSet(p); return this->closure(pSet);};
    virtual Point_set         closure(Point_set& chain);
    virtual Obj<Sieve>        closureSieve(Obj<Point_set> chain, Obj<Sieve> closure = Obj<Sieve>());
    virtual Point_set         star(Point p){Point_set pSet(p); return star(pSet);};
    virtual Point_set         star(Point_set& chain);
    virtual Obj<Sieve>        starSieve(Obj<Point_set> chain, Obj<Sieve> star = Obj<Sieve>());
    virtual Point_set         meet(Point_set c1, Point_set c2);
    virtual Point_set         meet(Point& p1, Point& p2){
      Point_set p1Set(p1), p2Set(p2); return this->meet(p1Set, p2Set);
    };
    virtual Point_set         meetAll(Point_set& chain);
    virtual Point_set         join(Point_set c1, Point_set c2);
    virtual Point_set         join(Point& p1, Point& p2){
      Point_set p1Set(p1), p2Set(p2); return this->join(p1Set, p2Set);
    };
    Point_set                 joinAll(Point_set& chain);
    virtual Point_set         roots(){return this->_roots;};
    virtual Point_set         leaves(){return this->_leaves;};
    virtual Point_set         roots(Point_set chain);
    virtual Point_set         roots(Point point){Point_set pSet; pSet.insert(point); return this->roots(pSet);};
    virtual Point_set         leaves(Point_set chain);
    virtual Point_set         leaves(Point point){Point_set pSet; pSet.insert(point); return this->leaves(pSet);};
    virtual int32_t           depth(Point p);
    virtual int32_t           height(Point p);
    virtual int32_t           maxDepth(Point_set &points);
    virtual int32_t           maxDepth(Obj<Point_set> points);
    virtual int32_t           maxHeight(Point_set &points);
    virtual int32_t           maxHeight(Obj<Point_set> points);
    virtual int32_t           diameter(Point p);
    virtual int32_t           diameter();
    virtual Point_set         depthStratum(int32_t depth);
    virtual Point_set         heightStratum(int32_t height);
    void                      view(const char *name);
    // Overloading methods to ensure the correct return type
    Sieve*                    baseRestriction(Point_set& base);
    Sieve*                    capRestriction(Point_set& cap);

  };

  
} // namespace ALE

#endif
