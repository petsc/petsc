#ifndef included_ALE_CoSifter_hh
#define included_ALE_CoSifter_hh

#ifndef  included_ALE_Sieve_hh
#include <Sieve.hh>
#endif
#ifndef  included_ALE_ParDelta_hh
#include <ParDelta.hh>
#endif
#include <list>
#include <stack>
#include <queue>

// Dmitry's explanation:
//
// Okay, check out what I have put there.
// It's a rather high-level interface, but I think it sketches out the implementation idea.  I have also become a master of switching from 'public' to 'private' and back.

// The idea is to put more power into Sifters (bipartite graphs with color).  They are like Sieves but with two point types (source and target) and no recursive operations (nCone, closure, etc).
// I claim they should be parallel, so cone/support completions should be computable for them.  The footprint is incorporated into the color of the new Sifter, which is returned as a completion.
// It would be very natural to have Sieve<Point_, Color_> to extend Sifter<Point_, Point_, Color_> with the recursive operations.

// The reason for putting the completion functionality into Sifters is that patches and indices under and over a topology Sieve are Sifters and have to be completed:
// the new overlap_patches has to encode patch pairs along with the rank of the second patch (first is always local); likewise, overlap_indices must encode a pair of intervals with a rank
// -- the attached to the same Sieve point by two different processes -- one local and one (possibly) remote.  At any rate, the support completion of 'patches' contains all the information
// needed for 'overlap_patches' -- remember that struct with a triple {point, patch, number} you had on the board?   Likewise for 'overlap_indices' built out of the cone completion of 'indices'.

// Once the 'overlap_XXX' are computed, we can allocate the storage for the Delta data and post sends receives.
// We should be able to reuse the completion subroutine from the old Sieve.
// So you are right that perhaps Sieve completion gets us to the CoSieve completion, except I think it's the Sifter completion this time.
// I can do the completion when I come back if you get the serial Sifter/Sieve stuff going.
//
namespace ALE {
  namespace Two {
    template <typename Sieve_, typename Patch_, typename Index_, typename Value_>
    class CoSifter {
    public:
      // Basic types
      typedef Sieve_ sieve_type;
      typedef typename sieve_type::point_type point_type;
      typedef std::vector<point_type> PointArray;
      typedef Patch_ patch_type;
      typedef Index_ index_type;
      typedef std::vector<index_type> IndexArray;
      typedef Value_ value_type;
      typedef Sifter<point_type,patch_type,index_type, RecContainer<point_type,Rec<point_type> >,RecContainer<patch_type,Rec<patch_type> > > order_type;

      typedef RightSequenceDuplicator<ConeArraySequence<typename sieve_type::traits::arrow_type> > fuser;
      typedef ParConeDelta<sieve_type, fuser,
                           typename sieve_type::template rebind<typename fuser::fusion_source_type,
                                                                typename fuser::fusion_target_type,
                                                                typename fuser::fusion_color_type,
                                                                typename sieve_type::traits::cap_container_type::template rebind<typename fuser::fusion_source_type,
                                                                                                                                 typename sieve_type::traits::sourceRec_type::template rebind<typename fuser::fusion_source_type,
                                                                                                                                                                                              typename sieve_type::marker_type>::type>::type,
                                                                typename sieve_type::traits::base_container_type::template rebind<typename fuser::fusion_target_type,
                                                                                                                                  typename sieve_type::traits::targetRec_type::template rebind<typename fuser::fusion_target_type,
                                                                                                                                                                                               typename sieve_type::marker_type>::type>::type>::type> coneDelta_type;
      typedef ParSupportDelta<sieve_type, fuser,
                              typename sieve_type::template rebind<typename fuser::fusion_source_type,
                                                                   typename fuser::fusion_target_type,
                                                                   typename fuser::fusion_color_type,
                                                                   typename sieve_type::traits::cap_container_type::template rebind<typename fuser::fusion_source_type, typename sieve_type::traits::sourceRec_type::template rebind<typename fuser::fusion_source_type, typename sieve_type::marker_type>::type>::type,
                                                                   typename sieve_type::traits::base_container_type::template rebind<typename fuser::fusion_target_type, typename sieve_type::traits::targetRec_type::template rebind<typename fuser::fusion_target_type, typename sieve_type::marker_type>::type>::type
      >::type> supportDelta_type;
      typedef RightSequenceDuplicator<ConeArraySequence<typename order_type::traits::arrow_type> > orderFuser;
      typedef ParSupportDelta<order_type, orderFuser> supportOrderDelta_type;
      typedef CoSifter<sieve_type, patch_type, point_type, int> bundle_type;
    private:
      MPI_Comm        _comm;
      int             _commRank;
      int             _commSize;
      Obj<sieve_type> _topology;
      // We need an ordering, which should be patch<--order--point
      Obj<order_type> _order;
      // We need a reordering, which should be patch<--new order--old order
      std::map<std::string,Obj<order_type> > _reorders;
      // We can add fields to an ordering using <patch,field><--order--point
      // We need sequences that can return the color, or do it automatically
      // We allocate based upon a certain
      std::map<patch_type,int>          _storageSize;
      std::map<patch_type,value_type *> _storage;
      int *offsets;
      int  ghostVars;
      Obj<bundle_type> localOrder;
      Obj<bundle_type> globalOrder;
    public:
      int              debug;
      // OLD CRAP:
      // Breakdown of the base Sieve into patches
      //   the colors (int) order the points (point_type) over a patch (patch_type).
      // A patch is a member of the sheaf over the sieve which indicates a maximal
      // domain of definition for a function on the sieve. Furthermore, we use the
      // patch coloring to order the patch values, the analog of a coordinate system
      // on the patch. We use a Sifter here, but the object can properly be thought
      // of as a CoSieve over the topology sieve.
    public:
      CoSifter(MPI_Comm comm = PETSC_COMM_SELF, int debug = 0) : _comm(comm), debug(debug) {
        this->_order = order_type(this->_comm, debug);
        MPI_Comm_rank(this->_comm, &this->_commRank);
        MPI_Comm_size(this->_comm, &this->_commSize);
        this->offsets = NULL;
        this->ghostVars = 0;
      };
      ~CoSifter() {
        if (this->offsets) {
          delete [] this->offsets;
          this->offsets = NULL;
        }
      };

      MPI_Comm        comm() const {return this->_comm;};
      int             commRank() const {return this->_commRank;};
      int             commSize() const {return this->_commSize;};
      void            setTopology(const Obj<sieve_type>& topology) {this->_topology = topology;};
      Obj<sieve_type> getTopology() const {return this->_topology;};
      // -- Patch manipulation --
      // Creates a patch whose order is taken from the input point sequence
      template<typename pointSequence> void setPatch(const Obj<pointSequence>& points, const patch_type& patch) {
        int c = 1;

        for(typename pointSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          this->_order->addArrow(*p_iter, patch, point_type(c++, 0));
        }
        if (points->begin() == points->end()) {
          this->_order->addBasePoint(patch);
        }
      };
      Obj<bundle_type> getGlobalOrder() const {
        return this->globalOrder;
      };
      Obj<bundle_type> getLocalOrder() const {
        return this->localOrder;
      };
    public:
      Obj<order_type> __getOrder() {
        return this->_order;
      };
      Obj<order_type> __getOrder(const std::string& orderName) {
        if (this->_reorders.find(orderName) == this->_reorders.end()) {
          if (this->debug) {std::cout << "Creating new order: " << orderName << std::endl;}
          this->_reorders[orderName] = order_type(this->_comm, this->debug);
        }
        return this->_reorders[orderName];
      };
    public:
      // Creates a patch for a named reordering whose order is taken from the input point sequence
      void setPatch(const std::string& orderName, const point_type& point, const patch_type& patch) {
        Obj<order_type> reorder = this->__getOrder(orderName);

        reorder->addArrow(point, patch, point_type());
      }
      template<typename pointSequence> void setPatch(const std::string& orderName, const Obj<pointSequence>& points, const patch_type& patch) {
        Obj<order_type> reorder = this->__getOrder(orderName);
        int c = 1;

        for(typename pointSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          reorder->addArrow(*p_iter, patch, point_type(c++, 0));
        }
      };
      // Returns the points in the patch in order
      Obj<typename order_type::coneSequence> getPatch(const patch_type& patch) const {
        return this->_order->cone(patch);
      };
      Obj<typename order_type::baseSequence> getPatches() {
        return this->_order->base();
      };
    private:
      void __checkOrderName(const std::string& orderName) {
        if (this->_reorders.find(orderName) != this->_reorders.end()) return;
        //FIX: String memory management
        std::string msg("Invalid order name: ");

        msg += orderName;
        throw ALE::Exception(msg.c_str());
      };
    public:
      // Returns the points in the reorder patch in order
      Obj<typename order_type::coneSequence> getPatch(const std::string& orderName, const patch_type& patch) {
        this->__checkOrderName(orderName);
        return this->_reorders[orderName]->cone(patch);
      };
      Obj<typename order_type::baseSequence> getPatches(const std::string& orderName) {
        return this->_reorders[orderName]->base();
      };
      // -- Index manipulation --
    private:
      struct changeOffset {
        changeOffset(int newOffset) : newOffset(newOffset) {};

        void operator()(typename order_type::Arrow_& p) const {
          p.color.prefix = newOffset;
        }
      private:
        int newOffset;
      };
      struct incrementOffset {
        incrementOffset(int newOffset) : newOffset(newOffset) {};

        void operator()(typename order_type::Arrow_& p) const {
          p.color.prefix = p.color.prefix + newOffset;
        }
      private:
        int newOffset;
      };
      struct changeDim {
        changeDim(int newDim) : newDim(newDim) {};

        void operator()(typename order_type::Arrow_& p) const {
          p.color.index = newDim;
        }
      private:
        int newDim;
      };
      struct changeIndex {
        changeIndex(int newOffset, int newDim) : newOffset(newOffset), newDim(newDim) {};

        void operator()(typename order_type::Arrow_& p) const {
          p.color.prefix = newOffset;
          p.color.index  = newDim;
        }
      private:
        int newOffset;
        int newDim;
      };
    public:
      int getFiberDimension(const patch_type& patch, const point_type& p) const {
        return this->_order->getColor(p, patch, false).index;
      };
      int getFiberDimension(const std::string& orderName, const patch_type& patch, const point_type& p) const {
        this->__checkOrderName(orderName);
        return this->_reorders[orderName]->getColor(p, patch, false).index;
      };
      void setFiberDimension(const patch_type& patch, const point_type& p, int dim) {
        this->_order->modifyColor(p, patch, changeDim(-dim));
      };
      void setFiberDimensionByDepth(const patch_type& patch, int depth, int dim) {
        Obj<typename sieve_type::traits::depthSequence> points = this->_topology->depthStratum(depth);

        for(typename sieve_type::traits::depthSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          this->setFiberDimension(patch, *p_iter, dim);
        }
      };
      void setFiberDimensionByHeight(const patch_type& patch, int height, int dim) {
        Obj<typename sieve_type::traits::heightSequence> points = this->_topology->heightStratum(height);

        for(typename sieve_type::traits::heightSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          this->setFiberDimension(patch, *p_iter, dim);
        }
      };
      void setFiberDimension(const std::string& orderName, const patch_type& patch, const point_type& p, int dim) {
        this->__checkOrderName(orderName);
        this->_reorders[orderName]->modifyColor(p, patch, changeDim(-dim));
      };
      void setFiberDimensionByDepth(const std::string& orderName, const patch_type& patch, int depth, int dim) {
        Obj<typename sieve_type::depthSequence> points = this->_topology->depthStratum(depth);

        for(typename sieve_type::depthSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          this->setFiberDimension(orderName, patch, *p_iter, dim);
        }
      };
      int getFiberOffset(const patch_type& patch, const point_type& p) const {
        return this->_order->getColor(p, patch, false).prefix;
      };
      void setFiberOffset(const patch_type& patch, const point_type& p, int offset) {
        this->_order->modifyColor(p, patch, changeOffset(offset));
      };
      void addFiberOffset(const patch_type& patch, const point_type& p, int offset) {
        this->_order->modifyColor(p, patch, incrementOffset(offset));
      };
    private:
      struct trueTester {
      public:
        bool operator()(const point_type& p) const {
          return true;
        };
      };
      template<typename OrderTest>
      void __orderCell(const Obj<order_type>& order, const patch_type& patch, const point_type& cell, int& offset, const OrderTest& tester) {
        // Set the prefix to the current offset (this won't kill the topology iterator)
        Obj<typename sieve_type::coneSequence> cone = this->_topology->cone(cell);

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          this->__orderCell(order, patch, *p_iter, offset, tester);
        }

        int dim = order->getColor(cell, patch, false).index;

        if ((dim < 0) && tester(cell)) {
          order->modifyColor(cell, patch, changeIndex(offset, -dim));
          if (debug) {std::cout << "Order point " << cell << " of size " << -dim << " and offset " << offset << " color " << order->getColor(cell, patch) << std::endl;}
          offset -= dim;
        }
      };
      // This constructs an order on the patch by fusing the Ord CoSieve (embodied by the prefix number)
      // and the Count CoSieve (embodied by the index), turning the prefix into an offset.
      template<typename OrderTest>
      void __orderPatch(const Obj<order_type>& order, const patch_type& patch, bool allocate, const OrderTest& tester) {
        PointArray points;
        int        offset = 0;

        // Filter out newly added points
        Obj<typename order_type::coneSequence> cone = order->cone(patch);
        int rank = 1;
        for(typename order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          if (p_iter.color().prefix == rank) {
            points.push_back(*p_iter);
            rank++;
//           } else if (debug) {
//             std::cout << "Rejected patch point " << *p_iter << " with color " << p_iter.color() << std::endl;
          }
        }
        // Loop over patch members
        for(typename PointArray::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          // Traverse the closure of the member in the topology
          if (debug) {std::cout << "Ordering patch point " << *p_iter << std::endl;}
          this->__orderCell(order, patch, *p_iter, offset, tester);
        }
        if (allocate) {
          this->allocatePatch(patch, offset);
        }
      };
    public:
      void allocatePatch(const patch_type& patch, int size = -1) {
        if (size < 0) {
          Obj<typename order_type::coneSequence> cone = getPatch(patch);

          for(typename order_type::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
            const index_type& idx = this->getIndex(patch, *c_iter);

            if (size < idx.prefix + idx.index) size = idx.prefix + idx.index;
          }
        }
        if (this->_storage.find(patch) != this->_storage.end()) {
          delete [] this->_storage[patch];
        }
        if (debug) {std::cout << "Allocated patch " << patch << " of size " << size << std::endl;}
        this->_storage[patch] = new value_type[size];
        this->_storageSize[patch] = size;
        memset(this->_storage[patch], 0, size*sizeof(value_type));
      };
      void allocatePatches() {
        Obj<typename order_type::baseSequence> patches = this->getPatches();

        for(typename order_type::baseSequence::iterator p_iter = patches->begin(); p_iter != patches->end(); ++p_iter) {
          this->allocatePatch(*p_iter);
        }
      };
      void orderPatch(const patch_type& patch) {
        this->__orderPatch(this->_order, patch, true, trueTester());
      }
      #undef __FUNCT__
      #define __FUNCT__ "CoSifter::orderPatches"
      template<typename OrderTest>
      void orderPatches(const OrderTest& tester) {
        ALE_LOG_EVENT_BEGIN;
        Obj<typename order_type::baseSequence> base = this->_order->base();

        for(typename order_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          this->__orderPatch(this->_order, *b_iter, true, tester);
        }
        ALE_LOG_EVENT_END;
      };
      void orderPatches() {
        this->orderPatches(trueTester());
      };
      void orderPatches(const std::string& orderName) {
        ALE_LOG_EVENT_BEGIN;
        this->__checkOrderName(orderName);
        Obj<typename order_type::baseSequence> base = this->_reorders[orderName]->base();

        for(typename order_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          this->__orderPatch(this->_reorders[orderName], *b_iter, false, trueTester());
        }
//         std::cout << orderName << " ordering:" << std::endl;
//         for(typename order_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
//           Obj<typename order_type::coneSequence> cone = this->getPatch(orderName, *b_iter);

//           std::cout << "  patch " << *b_iter << std::endl;
//           for(typename order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
//             std::cout << "    " << *p_iter << std::endl;
//           }
//         };
        ALE_LOG_EVENT_END;
      };
      //Obj<IndexArray> getIndices(const patch_type& patch);
      //Obj<IndexArray> getIndices(const patch_type& patch, const point_type& p);
      const index_type& getIndex(const patch_type& patch, const point_type& p) {
        return this->_order->getColor(p, patch, false);
      };
      Obj<IndexArray> getIndices(const std::string& orderName, const patch_type& patch) {
        Obj<typename order_type::coneSequence> cone = getPatch(orderName, patch);
        Obj<IndexArray>                        array = IndexArray();
        patch_type                             oldPatch;

        // We have no way to map the the old patch yet
        // It would be better to map this through in a sequence to the original indices (like fusion)
        for(typename order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          array->push_back(this->getIndex(oldPatch, *p_iter));
        }
        return array;
      }
      // -- Value manipulation --
    private:
      void __checkPatch(const patch_type& patch) const {
        if (this->_storage.find(patch) != this->_storage.end()) return;
        ostringstream msg;

        msg << "Invalid patch: " << patch;
        throw ALE::Exception(msg.str().c_str());
      };
    public:
      int getSize(const patch_type& patch) {
        this->__checkPatch(patch);
        return this->_storageSize[patch];
      };
      const int *getGlobalOffsets() {return this->offsets;};
      const value_type *restrict(const patch_type& patch) {
        this->__checkPatch(patch);
        return this->_storage[patch];
      };
      const value_type *restrict(const patch_type& patch, const point_type& p) {
        this->__checkPatch(patch);
        return &this->_storage[patch][this->_order->getColor(p, patch, false).prefix];
      };
      // Can this be improved?
      const value_type *restrict(const std::string& orderName, const patch_type& patch) {
        Obj<typename order_type::coneSequence> cone = getPatch(orderName, patch);
        static value_type                     *values = NULL;
        static int                             size = 0;
        int                                    newSize = 0;
        int                                    newI = 0;
        patch_type                             oldPatch;

        for(typename order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          newSize += this->getIndex(oldPatch, *p_iter).index;
        }
        if (newSize != size) {
          if (!values) delete [] values;
          size = newSize;
          values = new value_type[size];
        }
        for(typename order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          const index_type& ind = this->getIndex(oldPatch, *p_iter);

          for(int i = ind.prefix; i < ind.prefix+ind.index; ++i) {
            values[newI++] = this->_storage[oldPatch][i];
          }
        }
        return values;
      };
      const value_type *restrict(const std::string& orderName, const patch_type& patch, const point_type& p);
      void              update(const patch_type& patch, const value_type values[]) {
        this->__checkPatch(patch);
        value_type *storage = this->_storage[patch];
        const int   size = this->_storageSize[patch];

        for(int i = 0; i < size; ++i) {
          storage[i] = values[i];
        }
      };
      void              update(const patch_type& patch, const point_type& p, const value_type values[]) {
        const index_type& idx = this->getIndex(patch, p);
        int offset = idx.prefix;
        value_type *storage = &(this->_storage[patch][offset]);

        for(int i = 0; i < idx.index; ++i) {
          if (debug) {std::cout << "Set a[" << offset+i << "] = " << values[i] << " on patch " << patch << std::endl;}
          storage[i] = values[i];
        }
      };
      // Can this be improved?
      void              update(const std::string& orderName, const patch_type& patch, const value_type values[]) {
        Obj<typename order_type::coneSequence> cone = getPatch(orderName, patch);
        int                                    newI = 0;
        patch_type                             oldPatch;

        for(typename order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          const index_type& ind = this->getIndex(oldPatch, *p_iter);

          for(int i = ind.prefix; i < ind.prefix+ind.index; ++i) {
            this->_storage[oldPatch][i] = values[newI++];
          }
        }
      };
      void              update(const std::string& orderName, const patch_type& patch, const point_type& p, const value_type values[]);
      void              updateAdd(const patch_type& patch, const value_type values[]);
      void              updateAdd(const patch_type& patch, const point_type& p, const value_type values[]) {
        const index_type& idx = this->getIndex(patch, p);
        int offset = idx.prefix;

        for(int i = 0; i < idx.index; ++i) {
          if (debug) {std::cout << "Set a[" << offset+i << "] = " << values[i] << " on patch " << patch << std::endl;}
          this->_storage[patch][offset+i] += values[i];
        }
      };
      void              updateAdd(const std::string& orderName, const patch_type& patch, const value_type values[]) {
        Obj<typename order_type::coneSequence> cone = getPatch(orderName, patch);
        int                                    newI = 0;
        patch_type                             oldPatch;

        for(typename order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          const index_type& ind = this->getIndex(oldPatch, *p_iter);

          for(int i = ind.prefix; i < ind.prefix+ind.index; ++i) {
            this->_storage[oldPatch][i] += values[newI++];
          }
        }
      };
      void              updateAdd(const std::string& orderName, const patch_type& patch, const point_type& p, const value_type values[]);

      void view(const char* label) const {
        ostringstream txt;

        if(label != NULL) {
          if(this->commRank() == 0) {
            txt << "viewing CoSifter :'" << label << "'" << std::endl;
          }
        } else {
          if(this->commRank() == 0) {
            txt << "viewing a CoSifter" << std::endl;
          }
        }
        for(typename std::map<patch_type,value_type *>::const_iterator s_iter = this->_storage.begin(); s_iter != this->_storage.end(); ++s_iter) {
          patch_type patch = s_iter->first;

          txt << "[" << this->commRank() << "]: Patch " << patch << std::endl;
          Obj<typename order_type::coneSequence> cone = this->getPatch(s_iter->first);
          const value_type *array = ((std::map<patch_type,value_type *>) _storage)[s_iter->first];

          for(typename order_type::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
            index_type color = this->_order->getColor(*c_iter, s_iter->first, false);

            if (color.index != 0) {
              txt << "[" << this->commRank() << "]:   " << *c_iter << " dim " << color.index << " offset " << color.prefix << "  ";
              for(int i = 0; i < color.index; i++) {
                txt << " " << array[color.prefix+i];
              }
              txt << std::endl;
            }
          }
        }
        PetscSynchronizedPrintf(this->comm(), txt.str().c_str());
        PetscSynchronizedFlush(this->comm());
      };
    protected:
      struct supportLocalizer {
        typedef typename supportDelta_type::overlap_type                      overlap_type;
        typedef typename supportDelta_type::overlap_type::traits::capSequence sequence_type;
        Obj<overlap_type>  overlap;
        Obj<sequence_type> points;
        int                rank;
        public:
        supportLocalizer(Obj<overlap_type> overlap, const int rank) : overlap(overlap), rank(rank) {points = overlap->cap();};
        bool isLocal(const Obj<overlap_type> overlap, const Obj<sequence_type> points, const typename sieve_type::point_type& p) const {
          if (points->contains(p)) {
            Obj<typename overlap_type::traits::supportSequence> neighbors = overlap->support(p);

            for(typename overlap_type::traits::supportSequence::iterator s_iter = neighbors->begin(); s_iter != neighbors->end(); ++s_iter) {
              if (s_iter.target() < rank)
                return false;
            }
          }
          return true;
        };

        bool operator()(const typename sieve_type::point_type& p) const {
          //std::cout << "Checking for local point " << p << std::endl;
          return this->isLocal(this->overlap, this->points, p);
        }
      };
      //FIX: Should just flip internals I think
      struct coneLocalizer {
        typedef typename coneDelta_type::overlap_type                       overlap_type;
        typedef typename coneDelta_type::overlap_type::traits::baseSequence sequence_type;
        Obj<overlap_type>  overlap;
        Obj<sequence_type> points;
        int                rank;
        public:
        coneLocalizer(Obj<overlap_type> overlap, const int rank) : overlap(overlap), rank(rank) {points = overlap->base();};
        bool isLocal(const Obj<overlap_type> overlap, const Obj<sequence_type> points, const typename sieve_type::point_type& p) const {
          if (points->contains(p)) {
            Obj<typename overlap_type::traits::coneSequence> neighbors = overlap->cone(p);

            for(typename overlap_type::traits::coneSequence::iterator c_iter = neighbors->begin(); c_iter != neighbors->end(); ++c_iter) {
              if (c_iter.source() < rank)
                return false;
            }
          }
          return true;
        };

        bool operator()(const typename sieve_type::point_type& p) const {
          //std::cout << "Checking for local point " << p << std::endl;
          return this->isLocal(this->overlap, this->points, p);
        }
      };
    public:
      void createGlobalOrder() {
        Obj<typename sieve_type::traits::depthSequence>  vertices = this->_topology->depthStratum(0);
        Obj<typename sieve_type::traits::heightSequence> cells    = this->_topology->heightStratum(0);
        int useLocBaseOverlap = 0, useLocCapOverlap = 0;
        int useBaseOverlap    = 0, useCapOverlap    = 0;
        typename bundle_type::patch_type patch;

        for(typename sieve_type::traits::depthSequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
          if (this->getFiberDimension(patch, *v_iter) > 0) {
            useLocCapOverlap = 1;
            break;
          }
        }
        for(typename sieve_type::traits::heightSequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
          if (this->getFiberDimension(patch, *c_iter) > 0) {
            useLocBaseOverlap = 1;
            break;
          }
        }
        // Get overlap
        Obj<typename supportDelta_type::overlap_type> capOverlap;
        Obj<typename coneDelta_type::overlap_type> baseOverlap;
        int rank = this->commRank();

        MPI_Allreduce(&useLocCapOverlap, &useCapOverlap, 1, MPI_INT, MPI_LOR, this->comm());
        if (useCapOverlap) {
          std::cout << "Doing cap overlap" << std::endl;
          capOverlap = supportDelta_type::overlap(this->_topology);
        }
        MPI_Allreduce(&useLocBaseOverlap, &useBaseOverlap, 1, MPI_INT, MPI_LOR, this->comm());
        if (useBaseOverlap) {
          std::cout << "Doing base overlap" << std::endl;
          baseOverlap = coneDelta_type::overlap(this->_topology);
        }
        if (useCapOverlap && useBaseOverlap) {
          throw ALE::Exception("Cannot have both kinds of overlap");
        }
        if (!useCapOverlap && !useBaseOverlap) {
          throw ALE::Exception("Cannot have no overlap");
        }
        // Give a local offset to each local element, continue sequential offsets for ghosts
        // Local order is a CoSifter<sieve_type, patch_type, point_type, int>
        //   which means localOrder->_order is a Sifter<point_type,patch_type,point_type>
        // SupportDelta::overlap_type is an ASifter<ALE::Point, int, ALE::pair<int,int>, uniColor>
        this->localOrder  = bundle_type(this->_comm, this->debug);
        this->globalOrder = bundle_type(this->_comm, this->debug);
        Obj<typename order_type::baseSequence> base = this->_order->base();

        this->localOrder->setTopology(this->_topology);
        this->globalOrder->setTopology(this->_topology);
        for(typename order_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          Obj<typename order_type::coneSequence> cone = this->getPatch(*b_iter);

          this->localOrder->setPatch(cone, *b_iter);
          this->globalOrder->setPatch(cone, *b_iter);
          for(typename order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            this->localOrder->setFiberDimension(patch, *p_iter, this->getFiberDimension(patch, *p_iter));
            this->globalOrder->setFiberDimension(patch, *p_iter, this->getFiberDimension(patch, *p_iter));
          }
        }
        if (useCapOverlap) {
          this->localOrder->orderPatches(supportLocalizer(capOverlap, this->commRank()));
          this->globalOrder->orderPatches(supportLocalizer(capOverlap, this->commRank()));
        } else {
          this->localOrder->orderPatches(coneLocalizer(baseOverlap, this->commRank()));
          this->globalOrder->orderPatches(coneLocalizer(baseOverlap, this->commRank()));
        }
        if (this->debug) {
          this->localOrder->view("Local order");
          this->globalOrder->view("Global order");
        }
        int ierr, localVars = 0;

        if (this->offsets) {
          delete [] this->offsets;
        }
        this->offsets = new int[this->_commSize+1];
        for(typename order_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          localVars += this->globalOrder->getSize(*b_iter);
        }
        ierr = MPI_Allgather(&localVars, 1, MPI_INT, &this->offsets[1], 1, MPI_INT, this->comm());CHKERROR(ierr, "Error in MPI_Allgather");
        this->offsets[0] = 0;
        this->ghostVars  = 0;
        for(int p = 1; p <= this->commSize(); p++) {
          this->offsets[p] += this->offsets[p-1];
        }
        // Create global numbering
        for(typename order_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          Obj<typename order_type::coneSequence> cone = getPatch(*b_iter);

          for(typename order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            this->globalOrder->addFiberOffset(*b_iter, *p_iter, this->offsets[rank]);
            this->localOrder->setFiberOffset(*b_iter, *p_iter, this->offsets[rank] + this->ghostVars++);
          }
        }
        if (this->debug) {
          this->localOrder->view("Local order with offset");
          this->globalOrder->view("Global order with offset");
        }
        // Complete order to get ghost offsets
        this->globalOrder->completeOrder();
        if (this->debug) {
          this->globalOrder->view("Global order after completion");
        }
      };
      void completeOrder() {
        typename bundle_type::patch_type patch;

        Obj<typename supportOrderDelta_type::overlap_type> overlap = supportOrderDelta_type::overlap(this->_order);
        Obj<typename supportOrderDelta_type::fusion_type>  fusion  = supportOrderDelta_type::fusion(this->_order, overlap);
        if (this->debug) {
          fusion->view("Fusion for global indices");
        }
        Obj<typename supportOrderDelta_type::fusion_type::coneSequence> fPatch = fusion->cone(patch);

        for(typename supportOrderDelta_type::fusion_type::coneSequence::iterator c_iter = fPatch->begin(); c_iter != fPatch->end(); ++c_iter) {
          if (c_iter.color().index > 0) {
            this->setFiberOffset(patch, *c_iter, c_iter.color().prefix);
          }
        }
      };
      void partitionOrder(const std::string& orderName) {
        Obj<typename sieve_type::traits::heightSequence> elements = this->_topology->heightStratum(0);
        Obj<order_type> reorder = this->__getOrder(orderName);
        
        if (this->debug) {
          reorder->view("Initial reorder");
        }
        ParConeDelta<order_type>::setDebug(this->debug);
        Obj<typename ParConeDelta<order_type>::overlap_type> overlap = ParConeDelta<order_type>::overlap(reorder);
        Obj<typename ParConeDelta<order_type>::fusion_type>  fusion  = ParConeDelta<order_type>::fusion(reorder, overlap);
        reorder->add(fusion);
        if (debug) {
          overlap->view("Reorder fusion");
          fusion->view("Reorder fusion");
          reorder->view("Reorder after adding fusion");
        }
        //FIX
        Obj<ALE::set<point_type> > points = ALE::set<point_type>();
        points->insert(elements->begin(), elements->end());
        reorder->restrictBase(points);
        if (debug) {
          reorder->view("Reorder after base restriction");
        }
      }
    };
  } // namespace Two
} // namespace ALE

#endif
