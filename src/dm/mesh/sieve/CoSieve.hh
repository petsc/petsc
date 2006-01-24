#ifndef included_ALE_CoSieve_hh
#define included_ALE_CoSieve_hh

#ifndef  included_ALE_Sifter_hh
#include <Sifter.hh>
#endif

namespace ALE {
  namespace CoSieve {

    template <typename Sieve_, typname Color_, typename Value_>
    class CoSieve {
    public:
      typedef Sieve_ sieve_type;
      typedef Color_ color_type;
      typedef Value_ value_type;

      // Get the section value from the interior of a sieve point p
      value_type  getSectionI(const sieve_type::point_type& p, const color_type& c);
      // Set the section value to the interior of a sieve point p
      void setSectionI(const sieve_type::point_type& p, const color_type& c, const value_type& v);
      // Retrieve the section value from the closure of a sieve point p
      value_type  section(const sieve_type::point_type& p, const color_type& c);
    };


    // Count is a CoSieve whose section values and colors are integers.
    // This implements the idea of a cumulative count over a closure of a sieve point.
    // Example: the number degrees of freedom of a given field (fields encoded by color) over the closure of a given sieve point p;
    //          for a point q in the closure of p the count over q is not greater (less-than arrow) than the count over p.
    template <typename Sieve_>
    class Count : public CoSieve<Sieve_, int, int> {
    public:
      // Add up counts (point,color)-wise
      void fuse(const Count<sieve_type>& counter);
    };
      
    // Seq is a CoSieve whose section values are sequences (of some unspecified nature).
    // This implements the idea of a sheaf of preorders -- each sequence being one -- and order preserving maps
    // for each path between a pair of sieve points.
    // Example: the collection of int intervals of degree of freedom indices over the closure of a given sieve point p;
    //          for a point q in the closure p the intervals a subcollection (inclusion arrow) of those over p.
    // This can be used to implement Sieve completion.
    template <typename Sieve_, typename Color_, typename Sequence_>
    class Seq : public CoSieve<Sieve_, Color_, Sequence_> {
    public:
      typedef Sequence_ sequence_type;
    };
    
    // Ord is a CoSieve whose section values are integer interval sequences.
    // This implements the example in Seq.
    template <typename Sieve_, typename IntervalSequence_>
    class Ord : public Seq<Sieve_, std::pair<int,int>, IntervalSequence_ > {
    public:
      // Augment the ordering by using a counter
      void fuse(const Count<sieve_type>& counter);
    };

    

  }// namespace CoSieve
} // namespace ALE

#endif
