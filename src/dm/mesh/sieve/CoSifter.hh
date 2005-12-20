#ifndef included_ALE_CoSifter_hh
#define included_ALE_CoSifter_hh

#ifndef  included_ALE_Sifter_hh
#include <Sifter.hh>
#endif

namespace ALE {
  namespace def {
    //
    // CoSifter:
    // This object holds the data layout over a Sieve patitioned into supports.
    //
    template <typename Data, typename Color>
    class CoSieve {
    private:
      Obj<Sieve<Data, Color> > _topology;
      void __clear();
    public:
      //      Topology manipulation methods
      void                     setTopology(const Obj<Sieve<Data, Color> >& topology);
      Obj<Sieve<Data, Color> > getTopology();

      //      Fiber manipulation methods
      // typedef  ... coneSequence;
      //      Support manipulation methods
      // Identify a support point with a chain of topology points
      void          setSupport(const Obj<InputSequence>& chain, Data& support);
      // Retrieve the chain of topology points identified with a given support point
      supportChain  getSupport(Data& support);

      //      Data reduction helper methods
      // footprint anchors the points of a Delta at the pairs of support points with arrows colored by process rank
      Obj<Sieve<Data,Color> > footprint();
    }; // class CoSifter

    template<typname Data, typename Color, typename Value>
    CoSieve : public CoSifter<Data, Color> {
    };

  } // namespace def


} // namespace ALE

#endif
