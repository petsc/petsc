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
    class Point {
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
      virtual const Point&          operator*()  = 0;
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
      virtual void insert(const Point& p);                     // post: contains(p) == true 
      virtual void remove(const Point& p);                     // post: contains(p) == false
      virtual void add(const const_point_sequence& s);         // post: contains points from s and '*this before the call'
      virtual void add(const const_point_collection& s);       // post: contains points from s and '*this before the call'
      virtual void intersect(const const_point_sequence& s);   // post: contains points common to s and '*this before the call'
      virtual void intersect(const const_point_collection& s); // post: contains points common to s and '*this before the call'
      virtual void subtract(const const_point_sequence&  s);   // post: contains points of '*this before call' that are not in s
      virtual void subtract(const const_point_collection&  s); // post: contains points of '*this before call' that are not in s
    };
    
    //
    // Sieve:  
    //      contains a set of points and a set of arrows between them (at most one arrow between any two points);
    //      -- 'cone','support','closure','star','meet','join' are as in the Paper
    //      -- each point has a 'height', 'depth' and a unique 'marker' (or 'color'), which is another 'point';
    //         a point is 'colored' or examined as to its color by 'set/getColor';
    //         all points of a given color are retrieved by 'isocolor'
    //      -- height and depth are not necessarily maintained up-to-date, unless 'stratification' is on (get/setStratification);
    //         stratification can be computed on demand by 'stratify';
    //         height and depth (up-to-date or not) allow to retrieve points in strata via 'isodepth/height' ("structural" color)
    //
    class Sieve {
      Obj<const_point_sequence> cone(const Point& p); 
      Obj<const_point_sequence> cone(const Obj<const_point_sequence>& p); 
      Obj<const_point_sequence> support(const Point& p); 
      Obj<const_point_sequence> support(const Obj<const_point_sequence>& p); 
      //
      Obj<const_point_sequence> closure(const Point& p); 
      Obj<const_point_sequence> closure(const Obj<const_point_sequence>& p); 
      Obj<const_point_sequence> star(const Point& p); 
      Obj<const_point_sequence> star(const Obj<const_point_sequence>& p); 
      //
      Obj<const_point_sequence> meet(const Point& p, const Point& q);
      Obj<const_point_sequence> meet(const const_point_sequence& pp);
      Obj<const_point_sequence> join(const Point& p, const Point& q);
      Obj<const_point_sequence> join(const const_point_sequence& pp);

      Point depth(const Point& p);
      Point height(const Point& p);
      //
      Obj<const_point_sequence> isodepth(const Point& p, const int& depth);
      Obj<const_point_sequence> isoheight(const Point& p, const int& height);
      //
      void  setColor(const Point& p, const Point& color);
      Point getColor(const Point& p);
      //
      Obj<const_point_sequence> isocolor(const Point& color);

      void setStratification(bool on);
      bool getStratification();
      void stratify();

      // Completion follows.
    }

  } // namespace def



} // namespace ALE


#endif
