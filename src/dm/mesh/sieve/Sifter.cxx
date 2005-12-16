#include <Sifter.hh>
#include <petscmesh.h>

namespace ALE {

  namespace def {
    template <typename Data>
    set<Data>::set() {
      this->delegate = new std::set<Data>;
    }

    template <typename Data>
    set<Data>::set(const std::set<Data>& delegate) {
      this->delegate = delegate;
    }

    template <typename Data>
    set<Data>::~set() {
      delete this->delegate;
    }

    template <typename Data, typename Color>
    Sieve<Data,Color>::Sieve() {
    }

    template <typename Data, typename Color>
    Obj<const_sequence<Data> > Sieve<Data,Color>::cone(const Obj<const_sequence<Data> >& p) {
      //return coneSequence(this->arrows.get<2>(), p);
      return coneSequence(::boost::multi_index::get<target>(this->arrows), p);
    }

    template <typename Data, typename Color>
    void Sieve<Data,Color>::addArrow(const Data& p, const Data& q) {
      this->arrows.insert(SieveArrow(p, q, Color()));
    }

    template <typename Data, typename Color>
    void Sieve<Data,Color>::addCone(const Obj<const_sequence<Data> >& points, const Data& p) {
      for(typename const_sequence<Data>::iterator iter = points.begin(); iter != points.end(); iter++) {
        this->addArrow(*iter, p);
      }
    }
  }
}
