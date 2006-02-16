#ifndef included_ALE_CoSifter_hh
#define included_ALE_CoSifter_hh

#ifndef  included_ALE_Sifter_hh
#include <Sifter.hh>
#endif
#include <list>
#include <stack>
#include <queue>

// Dmitry's explanation:
//
// Okay, check out what I have put there.
// It's a rather high-level interface, but I think it sketches out the implementation idea.  I have also become a master of switching from 'public' to 'private' and back.

// The idea is to put more power into BiGraphs (BipartiteGraphs).  They are like Sieves but with two point types (source and target) and no recursive operations (nCone, closure, etc).
// I claim they should be parallel, so cone/support completions should be computable for them.  The footprint is incorporated into the color of the new BiGraph, which is returned as a completion.
// It would be very natural to have Sieve<Point_, Color_> to extend BiGraph<Point_, Point_, Color_> with the recursive operations.

// The reason for putting the completion functionality into BiGraphs is that patches and indices under and over a topology Sieve are BiGraphs and have to be completed:
// the new overlap_patches has to encode patch pairs along with the rank of the second patch (first is always local); likewise, overlap_indices must encode a pair of intervals with a rank
// -- the attached to the same Sieve point by two different processes -- one local and one (possibly) remote.  At any rate, the support completion of 'patches' contains all the information
// needed for 'overlap_patches' -- remember that struct with a triple {point, patch, number} you had on the board?   Likewise for 'overlap_indices' built out of the cone completion of 'indices'.

// Once the 'overlap_XXX' are computed, we can allocate the storage for the Delta data and post sends receives.
// We should be able to reuse the completion subroutine from the old Sieve.
// So you are right that perhaps Sieve completion gets us to the CoSieve completion, except I think it's the BiGraph completion this time.
// I can do the completion when I come back if you get the serial BiGraph/Sieve stuff going.
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
      typedef BiGraph<point_type,RecContainer<point_type,Rec<point_type> >,patch_type,RecContainer<patch_type,Rec<patch_type> >,index_type> order_type;
    private:
      int             debug;
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
    public:
      // OLD CRAP:
      // Breakdown of the base Sieve into patches
      //   the colors (int) order the points (point_type) over a patch (patch_type).
      // A patch is a member of the sheaf over the sieve which indicates a maximal
      // domain of definition for a function on the sieve. Furthermore, we use the
      // patch coloring to order the patch values, the analog of a coordinate system
      // on the patch. We use a BiGraph here, but the object can properly be thought
      // of as a CoSieve over the topology sieve.
    public:
      CoSifter(int debug = 0) : debug(debug) {
        _order = order_type(debug);
      };

      void            setTopology(const Obj<sieve_type>& topology) {this->_topology = topology;};
      Obj<sieve_type> getTopology() {return this->_topology;};
      // -- Patch manipulation --
      // Creates a patch whose order is taken from the input point sequence
      template<typename pointSequence> void setPatch(const Obj<pointSequence>& points, const patch_type& patch) {
        int c = 1;

        for(typename pointSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          this->_order->addArrow(*p_iter, patch, point_type(c++, 0));
        }
      };
    private:
      Obj<order_type> __getOrder(const std::string& orderName) {
        if (this->_reorders.find(orderName) == this->_reorders.end()) {
          this->_reorders[orderName] = order_type(this->debug);
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
      Obj<typename order_type::coneSequence> getPatch(const patch_type& patch) {
        return this->_order->cone(patch);
      };
    private:
      void __checkOrderName(const std::string& orderName) {
        if (this->_reorders.find(orderName) != this->_reorders.end()) return;
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
      int getFiberDimension(const patch_type& patch, const point_type& p) {
        return this->_order->getColor(p, patch, false).index;
      };
      int getFiberDimension(const std::string& orderName, const patch_type& patch, const point_type& p) {
        this->__checkOrderName(orderName);
        return this->_reorders[orderName]->getColor(p, patch, false).index;
      };
      void setFiberDimension(const patch_type& patch, const point_type& p, int dim) {
        this->_order->modifyColor(p, patch, changeDim(-dim));
      };
      void setFiberDimensionByDepth(const patch_type& patch, int depth, int dim) {
        Obj<typename sieve_type::depthSequence> points = this->_topology->depthStratum(depth);

        for(typename sieve_type::depthSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
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
    private:
      void __orderCell(const Obj<order_type>& order, const patch_type& patch, const point_type& cell, int& offset) {
        // Set the prefix to the current offset (this won't kill the topology iterator)
        Obj<typename sieve_type::coneSequence> cone = this->_topology->cone(cell);

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          this->__orderCell(order, patch, *p_iter, offset);
        }

        int dim = order->getColor(cell, patch, false).index;

        if (dim < 0) {
          order->modifyColor(cell, patch, changeIndex(offset, -dim));
          if (debug) {std::cout << "Order point " << cell << " of size " << -dim << " and offset " << offset << "(" << order->getColor(cell, patch) << ")" << std::endl;}
          offset -= dim;
        }
      };
      // This constructs an order on the patch by fusing the Ord CoSieve (embodied by the prefix number)
      // and the Count CoSieve (embodied by the index), turning the prefix into an offset.
      void __orderPatch(const Obj<order_type>& order, const patch_type& patch, bool allocate = true) {
        PointArray points;
        int        offset = 0;

        // Filter out newly added points
        Obj<typename order_type::coneSequence> cone = order->cone(patch);
        int rank = 1;
        for(typename order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          if (p_iter.color().prefix == rank) {
            points.push_back(*p_iter);
            rank++;
          }
        }
        // Loop over patch members
        for(typename PointArray::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          // Traverse the closure of the member in the topology
          if (debug) {std::cout << "Ordering patch point " << *p_iter << std::endl;}
          this->__orderCell(order, patch, *p_iter, offset);
        }
        if (allocate) {
          if (this->_storage.find(patch) != this->_storage.end()) {
            delete [] this->_storage[patch];
          }
          if (debug) {std::cout << "Allocated patch " << patch << " of size " << offset << std::endl;}
          this->_storage[patch] = new value_type[offset];
          this->_storageSize[patch] = offset;
          memset(this->_storage[patch], 0, offset*sizeof(value_type));
        }
      };
    public:
      void orderPatch(const patch_type& patch) {
        this->orderPatch(this->_order, patch);
      }
      #undef __FUNCT__
      #define __FUNCT__ "CoSifter::orderPatches"
      void orderPatches() {
        ALE_LOG_EVENT_BEGIN;
        Obj<typename order_type::baseSequence> base = this->_order->base();

        for(typename order_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          this->__orderPatch(this->_order, *b_iter);
        }
        ALE_LOG_EVENT_END;
      };
      void orderPatches(const std::string& orderName) {
        ALE_LOG_EVENT_BEGIN;
        this->__checkOrderName(orderName);
        Obj<typename order_type::baseSequence> base = this->_reorders[orderName]->base();

        for(typename order_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          this->__orderPatch(this->_reorders[orderName], *b_iter, false);
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
      int getSize(const patch_type& patch) {
        return this->_storageSize[patch];
      };
      const value_type *restrict(const patch_type& patch) {
        return this->_storage[patch];
      };
      const value_type *restrict(const patch_type& patch, const point_type& p) {
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
      void              update(const patch_type& patch, const value_type values[]);
      void              update(const patch_type& patch, const point_type& p, const value_type values[]) {
        const index_type& idx = this->getIndex(patch, p);
        int offset = idx.prefix;

        for(int i = 0; i < idx.index; ++i) {
          if (debug) {std::cout << "Set a[" << offset+i << "] = " << values[i] << " on patch " << patch << std::endl;}
          this->_storage[patch][offset+i] = values[i];
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
    };
  } // namespace Two
} // namespace ALE

#endif
