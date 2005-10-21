#ifndef included_ALE_ClosureBundle_hh
#define included_ALE_ClosureBundle_hh

#ifndef  included_ALE_Stack_hh
#include <Stack.hh>
#endif

#include <queue>

namespace ALE {

  typedef enum {INSERTION = 0, ADDITION = 1} BundleAssemblyPolicy;
  class ClosureBundle : public Coaster {
    int     _dirty;
    Obj<Stack>     _dimensionsToElements;
    Obj<Stack>     _indicesToArrows;
    Obj<Stack>     _arrowsToEnds;
    Obj<Stack>     _arrowsToStarts;
    //
    BundleAssemblyPolicy _assemblyPolicy;
    //
    void __reset(Obj<Sieve> topology = Obj<Sieve>()) {
      if(topology.isNull()) {
        topology = Obj<Sieve>(new Sieve(this->comm));
      }
      //
      this->_assemblyPolicy       = INSERTION;
      //
      this->_dimensionsToElements = Obj<Stack>(new Stack(this->comm));
      this->_dimensionsToElements->setTop(Obj<PreSieve>(new PreSieve(this->comm)));
      this->_dimensionsToElements->setBottom(topology);
      //
      this->_arrowsToStarts = Obj<Stack>(new Stack(this->comm));
      this->_arrowsToStarts->setTop(Obj<PreSieve>(new PreSieve(this->comm)));
      this->_arrowsToStarts->setBottom(topology);
      //
      this->_arrowsToEnds = Obj<Stack>(new Stack(this->comm));
      this->_arrowsToEnds->setTop(this->_arrowsToStarts->top());
      this->_arrowsToEnds->setBottom(topology);
      //
      this->__resetArrowIndices(); // this method depends on _arrowsToStarts having already been setup
      this->_cacheFiberIndices = 0;
    };
    //
    Obj<Sieve>     __getTopology(){return this->_dimensionsToElements->bottom();};
    Obj<PreSieve>  __getIndices(){return this->_indicesToArrows->top();};
    Obj<PreSieve>  __getArrows(){return this->_arrowsToStarts->top();};
    void           __resetArrowIndices(){
      this->_indicesToArrows = Obj<Stack>(new Stack(this->comm));
      this->_indicesToArrows->setTop(Obj<PreSieve>(new PreSieve(this->comm)));
      this->_indicesToArrows->setBottom(this->__getArrows());  // note that __getArrows does not rely on _indicesToArrows
    };
    //
    bool           _cacheFiberIndices;
    Obj<Point_set> __validateChain(Obj<Point_set> ee) {
      if(!ee->size()){
        return getTopology()->leaves();
      }
      else{
        return(ee);
      };
    };
    Point          __getArrow(Point e1, Point e);
    Point          __getArrowInterval(Point e1, Point e);
    void           __setArrowInterval(Point e1, Point e, Point interval);
    Obj<PreSieve> __computeIndices(Obj<Point_set> supports, Obj<Point_set> base, bool includeBoundary = 0);
    Obj<PreSieve> __computeBoundaryIndices(Point support, Point__Point& seen, int32_t& off);
    //
    void __markDirty(){this->_dirty = 1;};
    void __markClean(){this->_dirty = 0;};
    int  __isClean(){return !(this->_dirty);};
    void __assertClean(int flag){if((this->__isClean()) != flag) throw Exception("Clean assertion failed");};
    //
    static const int stratumTypeDepth  = 0;
    static const int stratumTypeHeight = 1;
    ClosureBundle& __setFiberDimensionByStratum(int stratumType, int32_t stratumIndex, int32_t dim);
    int__Point _checkOrderChain(Obj<Point_set> order, int& maxDepth, int& minDepth);
    void _orderElement(int dim, ALE::Obj<ALE::Point> element, std::map<int, std::queue<Point> > *ordered, ALE::Obj<ALE::Point_set> elementsOrdered);
    ALE::Obj<ALE::Point> _orderCell(int dim, int__Point *orderChain, std::map<int, std::queue<Point> > *ordered, ALE::Obj<ALE::Point_set> elementsOrdered);
  public:
    // constructors/destructors
    ClosureBundle()                    : Coaster()     {__reset();};
    ClosureBundle(MPI_Comm& comm)      : Coaster(comm) {__reset();};
    ClosureBundle(Obj<Sieve> topology) : Coaster(topology->getComm()) {__reset(topology);};
    virtual ~ClosureBundle(){};
    void view(const char *name);
    //
    virtual void            setComm(MPI_Comm c) {this->comm = c; __reset();};
    //
    ClosureBundle&          setAssemblyPolicy(BundleAssemblyPolicy policy);
    bool                    getFiberIndicesCachingPolicy() {return this->_cacheFiberIndices;};
    ClosureBundle&          setFiberIndicesCachingPolicy(bool policy){/*cannot cache (yet)*/this->_cacheFiberIndices = 0;return *this;};
    BundleAssemblyPolicy    getAssemblyPolicy() {return this->_assemblyPolicy;};
    ClosureBundle&          setTopology(Obj<Sieve> topology);
    Obj<Sieve>              getTopology(){return this->__getTopology();};
    ClosureBundle&          setFiberDimension(Point element, int32_t d);
    ClosureBundle&          setFiberDimensionByDepth(int32_t depth, int32_t dim){
      return __setFiberDimensionByStratum(stratumTypeDepth, depth, dim);
    };
    ClosureBundle&          setFiberDimensionByHeight(int32_t height, int32_t dim){
      return __setFiberDimensionByStratum(stratumTypeHeight, height, dim);
    };
    // Primary methods
    int32_t                 getFiberDimension(Obj<Point_set> ee);
    int32_t                 getBundleDimension(Obj<Point_set> ee);
    //
    Obj<PreSieve>           getFiberIndices(Obj<Point_set> support,  Obj<Point_set> base);
    Obj<PreSieve>           getBundleIndices(Obj<Point_set> support, Obj<Point_set> base);
    Obj<Point_array>        getClosureIndices(Obj<Point_set> order,  Obj<Point_set> base);
    // Convenience methods
    int32_t                 getFiberDimension(Point e){return getFiberDimension(Point_set(e));};
    int32_t                 getBundleDimension(Point e){return getBundleDimension(Point_set(e));};
    // Remote ordering methods
    void                    computeOverlapIndices(); // collective
    Obj<Point_set>          getOverlapOwners(Point e);
    Obj<PreSieve>           getOverlapFiberIndices(Point e, int32_t proc);
    Obj<PreSieve>           getOverlapBundleIndices(Point e, int32_t proc);
    Obj<PreSieve>           getOverlapOrderedIndices(Obj<Point_set> order, int32_t proc);
    // Global ordering methods
    void                    computeGlobalIndices(); // collective
    int32_t                 getGlobalOwner(Point e);
    Obj<PreSieve>           getGlobalFiberIndices(Point e, int32_t proc);
    Obj<PreSieve>           getGlobalBundleIndices(Point e, int32_t proc);
    Obj<PreSieve>           getGlobalOrderedIndices(Obj<Point_set> order, int32_t proc);
    
  };// class ClosureBundle



} // namespace ALE

#endif
