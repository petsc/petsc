#ifndef included_ALE_Sifter_hh
#define included_ALE_Sifter_hh

// ALE extensions

#ifndef  included_ALE_hh
#include <ALE.hh>
#endif

namespace ALE {

  namespace def {
    //
    // This is a set of abstract classes describing an interface to point containers.
    //

    // Basic object
    class point {
    public:
      int32_t prefix, index;
    };
    
    // point_iterator interface
    class point_iterator {
      virtual ~point_iterator() = 0;
      //
      virtual void                  operator++();
      virtual void                  operator++(int);
      virtual bool                  operator==(const point_iterator& itor);
      virtual bool                  operator!=(const point_iterator& itor);
      virtual const point&          operator*()  = 0;
    };

    // const_point_sequence interface:
    // a constant sequence of (not necesserily unique) points delineated by begin() & end() iterators; can be traversed linearly
    class const_point_sequence {
      typedef point_iterator iterator;
      virtual ~const_point_sequence() = 0;
      //
      virtual point_iterator& begin();
      virtual point_iterator& end();
      virtual std::size_t     size();
    };

    // const_point_collection interface:
    // a constant collection no particular order; can queried for containment of a given point
    class const_point_collection {
      virtual ~const_point_collection() = 0;
      //
      virtual bool contains(const point& p);
    };

    // point_set interface:
    // extends const_point_sequence & const_point_collection and allows point addition and removal
    class point_set : public const_point_sequence, const_point_collection {
      // conversion constructors
      point_set(const const_point_sequence&);
      point_set(const Obj<const_point_sequence>&);
      point_set(const const_point_collection&);
      point_set(const Obj<const_point_collection>&);
      // destructor
      virtual ~point_set();
      // mutating methods
      virtual void insert(const point& p);                     // post: contains(p) == true 
      virtual void remove(const point& p);                     // post: contains(p) == false
      virtual void add(const const_point_sequence& s);         // post: contains points from s and '*this before the call'
      virtual void add(const const_point_collection& s);       // post: contains points from s and '*this before the call'
      virtual void intersect(const const_point_sequence& s);   // post: contains points common to s and '*this before the call'
      virtual void intersect(const const_point_collection& s); // post: contains points common to s and '*this before the call'
      virtual void subtract(const const_point_sequence&  s);   // post: contains points of '*this before call' that are not in s
      virtual void subtract(const const_point_collection&  s); // post: contains points of '*this before call' that are not in s
    };
    
  } // namespace def



} // namespace ALE


#endif
