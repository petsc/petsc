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
    // Sifter:  
    //
    template <typename Color>
    class Sifter {
      Obj<const_sequence<Color> > cone(const Obj<const_sequence<Point> >& p); 
      Obj<const_sequence<Color> > nCone(const Obj<const_sequence<Point> >& p, const int& n); 
      Obj<const_sequence<Color> > support(const Obj<const_sequence<Point> >& p); 
      Obj<const_sequence<Color> > nSupport(const Obj<const_sequence<Point> >& p, const int& n); 
      //
      Obj<const_sequence<Color> > closure(const Obj<const_sequence<Point> >& p); 
      Obj<const_sequence<Color> > nClosure(const Obj<const_sequence<Point> >& p, const int& n); 
      Obj<const_sequence<Color> > star(const Obj<const_sequence<Point> >& p); 
      Obj<const_sequence<Color> > nStar(const Obj<const_sequence<Point> >& p, const int& n); 
      //
      Obj<const_sequence<Color> > meet(const const_sequence<Point>& pp);
      Obj<const_sequence<Color> > nMeet(const const_sequence<Point>& pp, const int& n);
      Obj<const_sequence<Color> > join(const const_sequence<Point>& pp);
      Obj<const_sequence<Color> > nJoin(const const_sequence<Point>& pp, const int& n);

      int depth(const Point& p);
      int height(const Point& p);
      int diameter(const Point& p);
      int diameter();
      
      //
      Obj<const_sequence<Point> > depthStratum(const int& depth);
      Obj<const_sequence<Point> > heightStratum(const int& height);
      //
      void                        setStratification(bool on);
      bool                        getStratification();
      void                        stratify();

      void                        addColor(const Obj<const_sequence<Color> >& colors, const Point& p);
      void                        addCone(const Obj<const_sequence<Point> >& points,  const Point& p);

      void                        add(const Obj<Sifter<Color> >& sifter);  // pointwise addition of fibers
      Sifter<Color>               completion(const Sifter<Point>& base);   // prescribes fibers to be exchanged
    };

  } // namespace def



} // namespace ALE


#endif
