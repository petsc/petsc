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
    
    // iterator interface
    template <typename Color>
    class const_iterator {
      virtual ~iterator() = 0;
      //
      virtual void                  operator++();
      virtual void                  operator++(int);
      virtual bool                  operator==(const iterator& itor);
      virtual bool                  operator!=(const iterator& itor);
      virtual const Color&          operator*()  = 0;
    };

    // const_sequence interface:
    // a constant sequence of (not necesserily unique) colors delineated by begin() & end() iterators; can be traversed linearly
    template <typename Color>
    class const_sequence {
      typedef const_iterator iterator;
      virtual ~const_sequence() = 0;
      //
      virtual const_iterator& begin();
      virtual const_iterator& end();
      virtual std::size_t     size();
    };

    // const_collection interface:
    // a constant collection no particular order; can queried for containment of a given color
    class const_collection {
      virtual ~const_collection() = 0;
      //
      virtual bool contains(const Color& p);
    };

    // const_set interface
    // combines const_sequence & const_collection interfaces
    template <typename Color>
    class const_set : public const_sequence<Color>, public const_collection<Color> {
      virtual ~const_set();
    };

    // set interface:
    // extends const_set interface to allows point addition and removal
    template <typename Color>
    class set : public const_set<Color> {
      // destructor
      virtual ~set();
      // mutating methods
      virtual void insert(const Color& p);                     // post: contains(p) == true 
      virtual void remove(const Color& p);                     // post: contains(p) == false
      virtual void add(const const_sequence& s);               // post: contains colors from s and '*this before the call'
      virtual void add(const const_collection& s);             // post: contains colors from s and '*this before the call'
      virtual void intersect(const const_sequence& s);         // post: contains colors common to s and '*this before the call'
      virtual void intersect(const const_collection& s);       // post: contains colors common to s and '*this before the call'
      virtual void subtract(const const_sequence&  s);         // post: contains colors of '*this before call' that are not in s
      virtual void subtract(const const_collection&  s);       // post: contains colors of '*this before call' that are not in s
    };
    
    //
    // Sieve:  
    //      contains a set of points and a set of arrows between them (at most one arrow between any two points);
    //      -- 'cone','support','closure','star','meet','join' are as in the Paper
    //      -- each point has a 'height', 'depth' 
    //         - height and depth are not necessarily maintained up-to-date, unless 'stratification' is on (get/setStratification);
    //         - stratification can be computed on demand by 'stratify';
    //         - height and depth (up-to-date or not) allow to retrieve points in strata via 'isodepth/isoheight'
    //      -- each point has a set of 'colors' of template type 'Color' attached to it
    //         - colors are added using 'addColor'
    //         - colors are retrieved using 'colors'
    //
    template <typename Color>
    class Sieve {
      Obj<const_sequence<Point> > cone(const Obj<const_sequence<Point> >& p); 
      Obj<const_sequence<Point> > support(const Obj<const_sequence<Point> >& p); 
      //
      Obj<const_sequence<Point> > closure(const Obj<const_sequence<Point> >& p); 
      Obj<const_sequence<Point> > star(const Obj<const_sequence<Point> >& p); 
      //
      Obj<const_sequence<Point> > meet(const Point& p, const Point& q);
      Obj<const_sequence<Point> > meet(const const_sequence<Point>& pp);
      Obj<const_sequence<Point> > join(const Point& p, const Point& q);
      Obj<const_sequence<Point> > join(const const_sequence<Point>& pp);

      Point depth(const Point& p);
      Point height(const Point& p);
      //
      Obj<const_sequence<Point> > isodepth(const Point& p, const int& depth);
      Obj<const_sequence<Point> > isoheight(const Point& p, const int& height);
      //
      void                        setStratification(bool on);
      bool                        getStratification();
      void                        stratify();

      void                        addColor(const Point& p, const Color& color);
      void                        addColor(const Point& p, const Obj<const_sequence<Color> >& colors);
      Obj<const_sequence<Color> > colors(const Point& p);
      Obj<const_sequence<Color> > colors(const Obj<const_sequence<Point> >& points);

      // Completion follows.
      void                        coneCompletion(const Sieve<Point>& base_itinerary);   // prescribes cones    to be exchanged
      void                        supportCompletion(const Sieve<Point>& cap_itinerary); // prescribes supports to be exchanged
    }

  } // namespace def



} // namespace ALE


#endif
