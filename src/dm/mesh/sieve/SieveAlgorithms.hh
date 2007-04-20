#ifndef included_ALE_SieveAlgorithms_hh
#define included_ALE_SieveAlgorithms_hh

#ifndef  included_ALE_Sieve_hh
#include <Sieve.hh>
#endif

namespace ALE {
  template<typename Bundle_>
  class SieveAlg {
  public:
    typedef Bundle_                                  bundle_type;
    typedef typename bundle_type::sieve_type         sieve_type;
    typedef typename sieve_type::point_type          point_type;
    typedef typename sieve_type::coneSet             coneSet;
    typedef typename sieve_type::coneArray           coneArray;
    typedef typename sieve_type::supportSet          supportSet;
    typedef typename sieve_type::supportArray        supportArray;
    typedef typename bundle_type::arrow_section_type arrow_section_type;
    typedef std::pair<point_type, int>               oriented_point_type;
    typedef ALE::array<oriented_point_type>          orientedConeArray;
  public:
    static Obj<coneArray> closure(const Obj<bundle_type>& bundle, const point_type& p) {
      return closure(bundle.ptr(), bundle->getArrowSection("orientation"), p);
    };
    static Obj<coneArray> closure(const Bundle_ *bundle, const Obj<arrow_section_type>& orientation, const point_type& p) {
      typedef MinimalArrow<point_type, point_type> arrow_type;
      typedef typename ALE::array<arrow_type>      arrowArray;
      const Obj<sieve_type>& sieve   = bundle->getSieve();
      const int              depth   = bundle->depth();
      Obj<arrowArray>        cone    = new arrowArray();
      Obj<arrowArray>        base    = new arrowArray();
      Obj<coneArray>         closure = new coneArray();
      coneSet                seen;

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
    static Obj<orientedConeArray> orientedClosure(const Obj<bundle_type>& bundle, const point_type& p) {
      return orientedClosure(bundle.ptr(), bundle->getArrowSection("orientation"), p);
    };
    static Obj<orientedConeArray> orientedClosure(const bundle_type *bundle, const Obj<arrow_section_type>& orientation, const point_type& p) {
      typedef MinimalArrow<point_type, point_type> arrow_type;
      typedef typename ALE::array<arrow_type>      arrowArray;
      const Obj<sieve_type>& sieve   = bundle->getSieve();
      const int              depth   = bundle->depth();
      Obj<arrowArray>        cone    = new arrowArray();
      Obj<arrowArray>        base    = new arrowArray();
      Obj<orientedConeArray> closure = new orientedConeArray();
      coneSet                seen;

      // Cone is guarateed to be ordered correctly
      const Obj<typename sieve_type::traits::coneSequence>& initCone = sieve->cone(p);

      closure->push_back(oriented_point_type(p, 0));
      for(typename sieve_type::traits::coneSequence::iterator c_iter = initCone->begin(); c_iter != initCone->end(); ++c_iter) {
        const arrow_type arrow(*c_iter, p);

        cone->push_back(arrow);
        closure->push_back(oriented_point_type(*c_iter, orientation->restrictPoint(arrow)[0]));
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
                const arrow_type arrow(*c_iter, b_iter->source);

                seen.insert(*c_iter);
                cone->push_back(arrow);
                closure->push_back(oriented_point_type(*c_iter, orientation->restrictPoint(arrow)[0]));
              }
            }
          } else {
            for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter) {
              if (seen.find(*c_iter) == seen.end()) {
                const arrow_type arrow(*c_iter, b_iter->source);

                seen.insert(*c_iter);
                cone->push_back(arrow);
                closure->push_back(oriented_point_type(*c_iter, orientation->restrictPoint(arrow)[0]));
              }
            }
          }
        }
      }
      return closure;
    };
    static Obj<coneArray> nCone(const Obj<bundle_type>& bundle, const point_type& p, const int n) {
      typedef MinimalArrow<point_type, point_type> arrow_type;
      typedef typename ALE::array<arrow_type>      arrowArray;
      const Obj<sieve_type>&         sieve       = bundle->getSieve();
      const Obj<arrow_section_type>& orientation = bundle->getArrowSection("orientation");
      const int                      height      = std::min(n, bundle->height());
      Obj<arrowArray>                cone        = new arrowArray();
      Obj<arrowArray>                base        = new arrowArray();
      Obj<coneArray>                 nCone       = new coneArray();
      coneSet                        seen;

      if (height == 0) {
        nCone->push_back(p);
        return nCone;
      }

      // Cone is guarateed to be ordered correctly
      const Obj<typename sieve_type::traits::coneSequence>& initCone = sieve->cone(p);

      if (height == 1) {
        for(typename sieve_type::traits::coneSequence::iterator c_iter = initCone->begin(); c_iter != initCone->end(); ++c_iter) {
          nCone->push_back(*c_iter);
        }
        return nCone;
      } else {
        for(typename sieve_type::traits::coneSequence::iterator c_iter = initCone->begin(); c_iter != initCone->end(); ++c_iter) {
          cone->push_back(arrow_type(*c_iter, p));
        }
      }
      for(int i = 1; i < height; ++i) {
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
                if (i == height-1) nCone->push_back(*c_iter);
              }
            }
          } else {
            for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter) {
              if (seen.find(*c_iter) == seen.end()) {
                seen.insert(*c_iter);
                cone->push_back(arrow_type(*c_iter, b_iter->source));
                if (i == height-1) nCone->push_back(*c_iter);
              }
            }
          }
        }
      }
      return nCone;
    };
    static Obj<supportArray> star(const Obj<bundle_type>& bundle, const point_type& p) {
      typedef MinimalArrow<point_type, point_type> arrow_type;
      typedef typename ALE::array<arrow_type>      arrowArray;
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
    static Obj<supportArray> nSupport(const Obj<bundle_type>& bundle, const point_type& p, const int n) {
      typedef MinimalArrow<point_type, point_type> arrow_type;
      typedef typename ALE::array<arrow_type>      arrowArray;
      const Obj<sieve_type>&         sieve       = bundle->getSieve();
      const Obj<arrow_section_type>& orientation = bundle->getArrowSection("orientation");
      const int                      depth       = std::min(n, bundle->depth());
      Obj<arrowArray>                support     = new arrowArray();
      Obj<arrowArray>                cap         = new arrowArray();
      Obj<coneArray>                 nSupport    = new supportArray();
      supportSet                     seen;

      if (depth == 0) {
        nSupport->push_back(p);
        return nSupport;
      }

      // Cone is guarateed to be ordered correctly
      const Obj<typename sieve_type::traits::supportSequence>& initSupport = sieve->support(p);

      if (depth == 1) {
        for(typename sieve_type::traits::supportSequence::iterator s_iter = initSupport->begin(); s_iter != initSupport->end(); ++s_iter) {
          nSupport->push_back(*s_iter);
        }
        return nSupport;
      } else {
        for(typename sieve_type::traits::supportSequence::iterator s_iter = initSupport->begin(); s_iter != initSupport->end(); ++s_iter) {
          support->push_back(arrow_type(*s_iter, p));
        }
      }
      for(int i = 1; i < depth; ++i) {
        Obj<arrowArray> tmp = support; support = cap; cap = tmp;

        support->clear();
        for(typename arrowArray::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
          const Obj<typename sieve_type::traits::supportSequence>& pSupport = sieve->support(c_iter->source);
          const typename arrow_section_type::value_type            o        = orientation->restrictPoint(*c_iter)[0];

          if (o == -1) {
            for(typename sieve_type::traits::supportSequence::reverse_iterator s_iter = pSupport->rbegin(); s_iter != pSupport->rend(); ++s_iter) {
              if (seen.find(*s_iter) == seen.end()) {
                seen.insert(*s_iter);
                support->push_back(arrow_type(*s_iter, c_iter->source));
                if (i == depth-1) nSupport->push_back(*s_iter);
              }
            }
          } else {
            for(typename sieve_type::traits::supportSequence::iterator s_iter = pSupport->begin(); s_iter != pSupport->end(); ++s_iter) {
              if (seen.find(*s_iter) == seen.end()) {
                seen.insert(*s_iter);
                support->push_back(arrow_type(*s_iter, c_iter->source));
                if (i == depth-1) nSupport->push_back(*s_iter);
              }
            }
          }
        }
      }
      return nSupport;
    };
  };
}

#endif
