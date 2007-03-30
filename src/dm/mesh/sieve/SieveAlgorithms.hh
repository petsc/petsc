#ifndef included_ALE_SieveAlgorithms_hh
#define included_ALE_SieveAlgorithms_hh

#ifndef  included_ALE_Distribution_hh
#include <CoSieve.hh>
#endif

namespace ALE {
  class Closure {
  public:
    template<typename Bundle_>
    static Obj<typename Bundle_::sieve_type::coneArray> closure(const Obj<Bundle_>& bundle, const typename Bundle_::point_type& p) {
      return closure(bundle.ptr(), bundle->getArrowSection("orientation"), p);
    };
    template<typename Bundle_>
    static Obj<typename Bundle_::sieve_type::coneArray> closure(const Bundle_ *bundle, const Obj<typename Bundle_::arrow_section_type>& orientation, const typename Bundle_::point_type& p) {
      typedef Bundle_                                  bundle_type;
      typedef typename bundle_type::sieve_type         sieve_type;
      typedef typename sieve_type::point_type          point_type;
      typedef typename sieve_type::coneArray           coneArray;
      typedef typename sieve_type::coneSet             coneSet;
      typedef typename bundle_type::arrow_section_type arrow_section_type;
      typedef MinimalArrow<point_type, point_type>     arrow_type;
      typedef typename ALE::array<arrow_type>          arrowArray;
      const Obj<sieve_type>&         sieve   = bundle->getSieve();
      const int                      depth   = bundle->depth();
      Obj<arrowArray>                cone    = new arrowArray();
      Obj<arrowArray>                base    = new arrowArray();
      Obj<coneArray>                 closure = new coneArray();
      coneSet                        seen;

      // Cone is guarateed to be ordered correctly
      const Obj<typename sieve_type::traits::coneSequence>& initCone = sieve->cone(p);

      closure->push_back(p);
      for(typename sieve_type::traits::coneSequence::iterator c_iter = initCone->begin(); c_iter != initCone->end(); ++c_iter) {
        cone->push_back(arrow_type(*c_iter, p));
        closure->push_back(*c_iter);
      }
      for(int i = 1; i < depth; ++i) {
        Obj<arrowArray> tmp = cone; cone = base; base = tmp;

        cone->clear();
        for(typename arrowArray::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          const Obj<typename sieve_type::traits::coneSequence>& pCone = sieve->cone(b_iter->source);
          const typename arrow_section_type::value_type         o     = orientation->restrictPoint(*b_iter)[0];

          if (o == -1) {
            for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter) {
              if (seen.find(*c_iter) == seen.end()) {
                seen.insert(*c_iter);
                cone->push_back(arrow_type(*c_iter, b_iter->source));
                closure->push_back(*c_iter);
              }
            }
          } else {
            for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter) {
              if (seen.find(*c_iter) == seen.end()) {
                seen.insert(*c_iter);
                cone->push_back(arrow_type(*c_iter, b_iter->source));
                closure->push_back(*c_iter);
              }
            }
          }
        }
      }
      return closure;
    };
    template<typename Bundle_>
    static Obj<typename Bundle_::sieve_type::coneArray> nCone(const Obj<Bundle_>& bundle, const typename Bundle_::point_type& p, const int n) {
      typedef Bundle_                                  bundle_type;
      typedef typename bundle_type::sieve_type         sieve_type;
      typedef typename sieve_type::point_type          point_type;
      typedef typename sieve_type::coneArray           coneArray;
      typedef typename sieve_type::coneSet             coneSet;
      typedef typename bundle_type::arrow_section_type arrow_section_type;
      typedef MinimalArrow<point_type, point_type>     arrow_type;
      typedef typename ALE::array<arrow_type>          arrowArray;
      const Obj<sieve_type>&         sieve       = bundle->getSieve();
      const Obj<arrow_section_type>& orientation = bundle->getArrowSection("orientation");
      const int                      depth       = std::min(n, bundle->depth());
      Obj<arrowArray>                cone        = new arrowArray();
      Obj<arrowArray>                base        = new arrowArray();
      Obj<coneArray>                 nCone       = new coneArray();
      coneSet                        seen;

      if (depth == 0) {
        nCone->push_back(p);
        return nCone;
      }

      // Cone is guarateed to be ordered correctly
      const Obj<typename sieve_type::traits::coneSequence>& initCone = sieve->cone(p);

      if (depth == 1) {
        for(typename sieve_type::traits::coneSequence::iterator c_iter = initCone->begin(); c_iter != initCone->end(); ++c_iter) {
          nCone->push_back(*c_iter);
        }
        return nCone;
      } else {
        for(typename sieve_type::traits::coneSequence::iterator c_iter = initCone->begin(); c_iter != initCone->end(); ++c_iter) {
          cone->push_back(arrow_type(*c_iter, p));
        }
      }
      for(int i = 1; i < depth; ++i) {
        Obj<arrowArray> tmp = cone; cone = base; base = tmp;

        cone->clear();
        for(typename arrowArray::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          const Obj<typename sieve_type::traits::coneSequence>& pCone = sieve->cone(b_iter->source);
          const typename arrow_section_type::value_type         o     = orientation->restrictPoint(*b_iter)[0];

          if (o == -1) {
            for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter) {
              if (seen.find(*c_iter) == seen.end()) {
                seen.insert(*c_iter);
                cone->push_back(arrow_type(*c_iter, b_iter->source));
                if (i == depth-1) nCone->push_back(*c_iter);
              }
            }
          } else {
            for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter) {
              if (seen.find(*c_iter) == seen.end()) {
                seen.insert(*c_iter);
                cone->push_back(arrow_type(*c_iter, b_iter->source));
                if (i == depth-1) nCone->push_back(*c_iter);
              }
            }
          }
        }
      }
      return nCone;
    };
    template<typename Bundle_>
    static Obj<typename Bundle_::sieve_type::supportArray> star(const Obj<Bundle_>& bundle, const typename Bundle_::point_type& p) {
      typedef Bundle_                                  bundle_type;
      typedef typename bundle_type::sieve_type         sieve_type;
      typedef typename sieve_type::point_type          point_type;
      typedef typename sieve_type::supportArray        supportArray;
      typedef typename sieve_type::supportSet          supportSet;
      typedef typename bundle_type::arrow_section_type arrow_section_type;
      typedef MinimalArrow<point_type, point_type>     arrow_type;
      typedef typename ALE::array<arrow_type>          arrowArray;
      const Obj<sieve_type>&         sieve       = bundle->getSieve();
      const Obj<arrow_section_type>& orientation = bundle->getArrowSection("orientation");
      const int                      height      = bundle->height();
      Obj<arrowArray>                support     = new arrowArray();
      Obj<arrowArray>                cap         = new arrowArray();
      Obj<supportArray>              star        = new supportArray();
      supportSet                     seen;

      // Support is guarateed to be ordered correctly
      const Obj<typename sieve_type::traits::supportSequence>& initSupport = sieve->support(p);

      star->push_back(p);
      for(typename sieve_type::traits::supportSequence::iterator s_iter = initSupport->begin(); s_iter != initSupport->end(); ++s_iter) {
        support->push_back(arrow_type(p, *s_iter));
        star->push_back(*s_iter);
      }
      for(int i = 1; i < height; ++i) {
        Obj<arrowArray> tmp = support; support = cap; cap = tmp;

        support->clear();
        for(typename arrowArray::iterator b_iter = cap->begin(); b_iter != cap->end(); ++b_iter) {
          const Obj<typename sieve_type::traits::supportSequence>& pSupport = sieve->support(b_iter->target);
          const typename arrow_section_type::value_type            o        = orientation->restrictPoint(*b_iter)[0];

          if (o == -1) {
            for(typename sieve_type::traits::supportSequence::reverse_iterator s_iter = pSupport->rbegin(); s_iter != pSupport->rend(); ++s_iter) {
              if (seen.find(*s_iter) == seen.end()) {
                seen.insert(*s_iter);
                support->push_back(arrow_type(b_iter->target, *s_iter));
                star->push_back(*s_iter);
              }
            }
          } else {
            for(typename sieve_type::traits::supportSequence::iterator s_iter = pSupport->begin(); s_iter != pSupport->end(); ++s_iter) {
              if (seen.find(*s_iter) == seen.end()) {
                seen.insert(*s_iter);
                support->push_back(arrow_type(b_iter->target, *s_iter));
                star->push_back(*s_iter);
              }
            }
          }
        }
      }
      return star;
    };
  };
}

#endif
