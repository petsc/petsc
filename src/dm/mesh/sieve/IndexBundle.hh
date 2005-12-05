#ifndef included_ALE_IndexBundle_hh
#define included_ALE_IndexBundle_hh

#ifndef  included_ALE_Stack_hh
#include <Stack.hh>
#endif

#include <queue>

namespace ALE {

  typedef enum {INSERTION = 0, ADDITION = 1} BundleAssemblyPolicy;
  typedef enum {localPoint, leasedPoint, rentedPoint} PointType;

  class IndexBundle : public Coaster {
    int     _dirty;
    Obj<Stack>     _dimensionsToElements;
    Obj<Stack>     _indicesToArrows;
    Obj<Stack>     _arrowsToEnds;
    Obj<Stack>     _arrowsToStarts;
    //
    Obj<PreSieve>  _overlapOwnership;     // a PreSieve supporting each overlap point at its owners
    Obj<Stack>     _localOverlapIndices;  // a stack with _overlapOwnership in the base, a discrete top contains the local indices 
                                          // attached to the overlap points by vertical arrows
    Obj<PreSieve>  _remoteOverlapIndices; // a completion stack with the remote overlap indices: completionTypeArrow, footprintTypeCone
    Obj<PreSieve> _pointTypes;
    Obj<PreSieve> _localIndices;
    Obj<PreSieve> _globalIndices;
    int *_firstGlobalIndex;
    //
    BundleAssemblyPolicy _assemblyPolicy;
    //
    void __reset(Obj<Sieve> topology = Obj<Sieve>()) {
      if(topology.isNull()) {
        topology = Obj<Sieve>(Sieve(this->comm));
      }
      //
      this->_assemblyPolicy       = INSERTION;
      //
      this->_dimensionsToElements = Obj<Stack>(Stack(this->comm));
      this->_dimensionsToElements->setTop(Obj<PreSieve>(PreSieve(this->comm)));
      this->_dimensionsToElements->setBottom(topology);
      //
      this->_arrowsToStarts = Obj<Stack>(Stack(this->comm));
      this->_arrowsToStarts->setTop(Obj<PreSieve>(PreSieve(this->comm)));
      this->_arrowsToStarts->setBottom(topology);
      //
      this->_arrowsToEnds = Obj<Stack>(Stack(this->comm));
      this->_arrowsToEnds->setTop(this->_arrowsToStarts->top());
      this->_arrowsToEnds->setBottom(topology);
      //
      this->_localOverlapIndices  = Obj<Stack>(Stack(this->comm));
      this->_remoteOverlapIndices = Obj<PreSieve>(PreSieve(this->comm));
      //
      this->__resetArrowIndices(); // this method depends on _arrowsToStarts having already been setup
      this->_cacheFiberIndices = 0;
      //
      _pointTypes = Obj<PreSieve>(PreSieve(this->comm));
      _localIndices = Obj<PreSieve>(PreSieve(this->comm));
      _globalIndices = Obj<PreSieve>(PreSieve(this->comm));
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
    Obj<PreSieve> __computeIndices(Obj<Point_set> supports, Obj<Point_set> base, bool includeBoundary = 0, Obj<Point_set> exclusion = NULL);
    Obj<PreSieve> __computeBoundaryIndices(Point support, Point__Point& seen, int32_t& off);
    //
    void __markDirty(){this->_dirty = 1;};
    void __markClean(){this->_dirty = 0;};
    int  __isClean(){return !(this->_dirty);};
    void __assertClean(int flag){if((this->__isClean()) != flag) throw Exception("Clean assertion failed");};
    //
    static const int stratumTypeDepth  = 0;
    static const int stratumTypeHeight = 1;
    IndexBundle& __setFiberDimensionByStratum(int stratumType, int32_t stratumIndex, int32_t dim);
    ALE::Point __getMaxDepthElement(Obj<Point_set> points);
    int__Point __checkOrderChain(Obj<Point_set> order, int& maxDepth, int& minDepth);
    void __orderElement(int dim, ALE::Point element, std::map<int, std::queue<Point> > *ordered, ALE::Obj<ALE::Point_set> elementsOrdered);
    ALE::Point __orderCell(int dim, int__Point *orderChain, std::map<int, std::queue<Point> > *ordered, ALE::Obj<ALE::Point_set> elementsOrdered);
    ALE::PointType __getPointType(ALE::Point point);
    ALE::Obj<ALE::PreSieve> __computePointTypes();
    void __postIntervalRequests(ALE::Obj<ALE::PreSieve> pointTypes, int__Point rentMarkers, MPI_Request *intervalRequests[], int **receivedIntervals[]);
    void __sendIntervals(ALE::Obj<ALE::PreSieve> pointTypes, int__Point leaseMarkers, ALE::Obj<ALE::PreSieve> indices);
    void __receiveIntervals(ALE::Obj<ALE::PreSieve> pointTypes, int__Point rentMarkers, MPI_Request *requests, int *recvIntervals[], ALE::Obj<ALE::PreSieve> indices);
  public:
    // constructors/destructors
    IndexBundle()                    : Coaster(MPI_COMM_SELF) {__reset();};
    IndexBundle(MPI_Comm& comm)      : Coaster(comm) {__reset();};
    IndexBundle(Obj<Sieve> topology) : Coaster(topology->getComm()) {__reset(topology);};
    virtual ~IndexBundle(){};
    void view(const char *name);
    //
    virtual void            setComm(MPI_Comm c) {this->comm = c; __reset();};
    //
    IndexBundle&            setAssemblyPolicy(BundleAssemblyPolicy policy);
    bool                    getFiberIndicesCachingPolicy() {return this->_cacheFiberIndices;};
    IndexBundle&            setFiberIndicesCachingPolicy(bool policy){/*cannot cache (yet)*/this->_cacheFiberIndices = 0;return *this;};
    BundleAssemblyPolicy    getAssemblyPolicy() {return this->_assemblyPolicy;};
    IndexBundle&            setTopology(Obj<Sieve> topology);
    Obj<Sieve>              getTopology(){return this->__getTopology();};
    IndexBundle&            setFiberDimension(Point element, int32_t d);
    IndexBundle&            setFiberDimensionByDepth(int32_t depth, int32_t dim){
      return __setFiberDimensionByStratum(stratumTypeDepth, depth, dim);
    };
    IndexBundle&            setFiberDimensionByHeight(int32_t height, int32_t dim){
      return __setFiberDimensionByStratum(stratumTypeHeight, height, dim);
    };
    // Primary methods
    int32_t                 getFiberDimension(Obj<Point_set> ee);
    int32_t                 getBundleDimension(Obj<Point_set> ee);
    Point                   getFiberInterval(Point support,  Obj<Point_set> base);
    Obj<PreSieve>           getFiberIndices(Obj<Point_set> support,  Obj<Point_set> base, Obj<Point_set> exclusion = NULL);
    Obj<PreSieve>           getClosureIndices(Obj<Point_set> support, Obj<Point_set> base);
    Obj<Point_array>        getOrderedIndices(Obj<Point_set> order,  Obj<PreSieve> indices);
    // Convenience methods
    int32_t                 getFiberDimension(Point e){return getFiberDimension(Point_set(e));};
    int32_t                 getBundleDimension(Point e){return getBundleDimension(Point_set(e));};
    Point                   getFiberInterval(Point support) {
      return getFiberInterval(support, Point_set());
    };
    Obj<Point_array>        getOrderedClosureIndices(Obj<Point_set> order, Obj<Point_set> base) {
      base = __validateChain(base);
      return getOrderedIndices(order, getClosureIndices(Point_set(__getMaxDepthElement(order)), base));
    };
    // Remote ordering methods
    void                    computeOverlapIndices(); // collective
    int32_t                 getOverlapSize();
    Obj<Point_set>          getOverlapOwners(Point e);
    Obj<PreSieve>           getOverlapFiberIndices(Obj<Point_set> supports, int32_t proc);
    Obj<PreSieve>           getOverlapClosureIndices(Obj<Point_set> supports, int32_t proc);
    // Convenience methods
    Obj<PreSieve>           getOverlapFiberIndices(Point e, int32_t proc) {return getOverlapFiberIndices(Point_set(e), proc);};
    Obj<PreSieve>           getOverlapClosureIndices(Point e, int32_t proc) {return getOverlapClosureIndices(Point_set(e), proc);};
    Obj<Point_array>        getOverlapOrderedClosureIndices(Obj<Point_set> order, int32_t proc) {
      Obj<PreSieve> indices = getOverlapClosureIndices(Point_set(__getMaxDepthElement(order)), proc);
      return getOrderedIndices(order, indices);
    };
    // Global ordering methods
    Obj<PreSieve>           getPointTypes();
    void                    computeGlobalIndices(); // collective
    int32_t                 getGlobalSize();
    int32_t                 getLocalSize();
    int*                    getLocalSizes();
    int32_t                 getRemoteSize();
    Obj<PreSieve>           getGlobalIndices();
    Obj<PreSieve>           getLocalIndices();
    int32_t                 getGlobalOwner(Point e);
    Point                   getGlobalFiberInterval(Point support);
    Obj<PreSieve>           getGlobalFiberIndices(Obj<Point_set> supports);
    Obj<PreSieve>           getLocalFiberIndices(Obj<Point_set> supports);
    Obj<PreSieve>           getGlobalClosureIndices(Obj<Point_set> supports);
    Obj<PreSieve>           getLocalClosureIndices(Obj<Point_set> supports);
    // Convenience methods
    Obj<PreSieve>           getGlobalFiberIndices(Point e) {return getGlobalFiberIndices(Point_set(e));};
    Obj<PreSieve>           getLocalFiberIndices(Point e) {return getLocalFiberIndices(Point_set(e));};
    Obj<PreSieve>           getGlobalClosureIndices(Point e) {return getGlobalClosureIndices(Point_set(e));};
    Obj<PreSieve>           getLocalClosureIndices(Point e) {return getLocalClosureIndices(Point_set(e));};
    Obj<Point_array>        getGlobalOrderedClosureIndices(Obj<Point_set> order) {
      Obj<PreSieve> indices = getGlobalClosureIndices(Point_set(__getMaxDepthElement(order)));
      return getOrderedIndices(order, indices);
    };
    Obj<Point_array>        getLocalOrderedClosureIndices(Obj<Point_set> order) {
      Obj<PreSieve> indices = getLocalClosureIndices(Point_set(__getMaxDepthElement(order)));
      return getOrderedIndices(order, indices);
    };
    // Mapping methods
    Obj<Stack>              computeMappingIndices(Obj<PreSieve> pointTypes, Obj<IndexBundle> target);
  };// class IndexBundle



} // namespace ALE

#endif
