#ifndef included_ALE_SieveAlgorithms_hh
#define included_ALE_SieveAlgorithms_hh

#ifndef  included_ALE_Sieve_hh
#include <sieve/Sieve.hh>
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
  protected:
    typedef MinimalArrow<point_type, point_type>     arrow_type;
    typedef std::pair<arrow_type, int>               oriented_arrow_type;
    typedef ALE::array<oriented_arrow_type>          orientedArrowArray;
  public:
    static Obj<coneArray> closure(const Obj<bundle_type>& bundle, const point_type& p) {
      return closure(bundle.ptr(), bundle->getArrowSection("orientation"), p);
    };
    static Obj<coneArray> closure(const Bundle_ *bundle, const Obj<arrow_section_type>& orientation, const point_type& p) {
      const Obj<sieve_type>&  sieve   = bundle->getSieve();
      const int               depth   = bundle->depth();
      Obj<orientedArrowArray> cone    = new orientedArrowArray();
      Obj<orientedArrowArray> base    = new orientedArrowArray();
      Obj<coneArray>          closure = new coneArray();
      coneSet                 seen;

      // Cone is guarateed to be ordered correctly
      const Obj<typename sieve_type::traits::coneSequence>& initCone = sieve->cone(p);

      closure->push_back(p);
      for(typename sieve_type::traits::coneSequence::iterator c_iter = initCone->begin(); c_iter != initCone->end(); ++c_iter) {
        cone->push_back(oriented_arrow_type(arrow_type(*c_iter, p), 1));
        closure->push_back(*c_iter);
      }
      for(int i = 1; i < depth; ++i) {
        Obj<orientedArrowArray> tmp = cone; cone = base; base = tmp;

        cone->clear();
        for(typename orientedArrowArray::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          const arrow_type&                                     arrow = b_iter->first;
          const Obj<typename sieve_type::traits::coneSequence>& pCone = sieve->cone(arrow.source);
          typename arrow_section_type::value_type               o     = orientation->restrictPoint(arrow)[0];

          if (b_iter->second < 0) {
            o = -(o+1);
          }
          if (o < 0) {
            const int size = pCone->size();

            if (o == -size) {
              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter) {
                if (seen.find(*c_iter) == seen.end()) {
                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(arrow_type(*c_iter, arrow.source), o));
                  closure->push_back(*c_iter);
                }
              }
            } else {
              const int numSkip = size + o;
              int       count   = 0;

              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter, ++count) {
                if (count < numSkip) continue;
                if (seen.find(*c_iter) == seen.end()) {
                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(arrow_type(*c_iter, arrow.source), o));
                  closure->push_back(*c_iter);
                }
              }
              count = 0;
              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter, ++count) {
                if (count >= numSkip) break;
                if (seen.find(*c_iter) == seen.end()) {
                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(arrow_type(*c_iter, arrow.source), o));
                  closure->push_back(*c_iter);
                }
              }
            }
          } else {
            if (o == 1) {
              for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter) {
                if (seen.find(*c_iter) == seen.end()) {
                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(arrow_type(*c_iter, arrow.source), o));
                  closure->push_back(*c_iter);
                }
              }
            } else {
              const int numSkip = o-1;
              int       count   = 0;

              for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter, ++count) {
                if (count < numSkip) continue;
                if (seen.find(*c_iter) == seen.end()) {
                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(arrow_type(*c_iter, arrow.source), o));
                  closure->push_back(*c_iter);
                }
              }
              count = 0;
              for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter, ++count) {
                if (count >= numSkip) break;
                if (seen.find(*c_iter) == seen.end()) {
                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(arrow_type(*c_iter, arrow.source), o));
                  closure->push_back(*c_iter);
                }
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
      const Obj<sieve_type>&  sieve   = bundle->getSieve();
      const int               depth   = bundle->depth();
      Obj<orientedArrowArray> cone    = new orientedArrowArray();
      Obj<orientedArrowArray> base    = new orientedArrowArray();
      Obj<orientedConeArray>  closure = new orientedConeArray();
      coneSet                 seen;

      // Cone is guarateed to be ordered correctly
      const Obj<typename sieve_type::traits::coneSequence>& initCone = sieve->cone(p);

      closure->push_back(oriented_point_type(p, 0));
      for(typename sieve_type::traits::coneSequence::iterator c_iter = initCone->begin(); c_iter != initCone->end(); ++c_iter) {
        const arrow_type arrow(*c_iter, p);

        cone->push_back(oriented_arrow_type(arrow, 1));
        closure->push_back(oriented_point_type(*c_iter, orientation->restrictPoint(arrow)[0]));
      }
      for(int i = 1; i < depth; ++i) {
        Obj<orientedArrowArray> tmp = cone; cone = base; base = tmp;

        cone->clear();
        for(typename orientedArrowArray::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          const arrow_type&                                     arrow = b_iter->first;
          const Obj<typename sieve_type::traits::coneSequence>& pCone = sieve->cone(arrow.source);
          typename arrow_section_type::value_type               o     = orientation->restrictPoint(arrow)[0];

          if (b_iter->second < 0) {
            o = -(o+1);
          }
          if (o < 0) {
            const int size = pCone->size();

            if (o == -size) {
              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter) {
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, arrow.source);
                  int              pointO = orientation->restrictPoint(newArrow)[0];

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, pointO ? -(pointO+1): pointO));
                }
              }
            } else {
              const int numSkip = size + o;
              int       count   = 0;

              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter, ++count) {
                if (count < numSkip) continue;
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, arrow.source);
                  int              pointO = orientation->restrictPoint(newArrow)[0];

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, pointO ? -(pointO+1): pointO));
                }
              }
              count = 0;
              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter, ++count) {
                if (count >= numSkip) break;
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, arrow.source);
                  int              pointO = orientation->restrictPoint(newArrow)[0];

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, pointO ? -(pointO+1): pointO));
                }
              }
            }
          } else {
            if (o == 1) {
              for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter) {
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, arrow.source);

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, orientation->restrictPoint(newArrow)[0]));
                }
              }
            } else {
              const int numSkip = o-1;
              int       count   = 0;

              for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter, ++count) {
                if (count < numSkip) continue;
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, arrow.source);

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, orientation->restrictPoint(newArrow)[0]));
                }
              }
              count = 0;
              for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter, ++count) {
                if (count >= numSkip) break;
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, arrow.source);

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, orientation->restrictPoint(newArrow)[0]));
                }
              }
            }
          }
        }
      }
      return closure;
    };
    static Obj<coneArray> nCone(const Obj<bundle_type>& bundle, const point_type& p, const int n) {
      const Obj<sieve_type>&         sieve       = bundle->getSieve();
      const Obj<arrow_section_type>& orientation = bundle->getArrowSection("orientation");
      const int                      height      = std::min(n, bundle->height());
      Obj<orientedArrowArray>        cone        = new orientedArrowArray();
      Obj<orientedArrowArray>        base        = new orientedArrowArray();
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
          cone->push_back(oriented_arrow_type(arrow_type(*c_iter, p), 1));
        }
      }
      for(int i = 1; i < height; ++i) {
        Obj<orientedArrowArray> tmp = cone; cone = base; base = tmp;

        cone->clear();
        for(typename orientedArrowArray::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          const arrow_type&                                     arrow = b_iter->first;
          const Obj<typename sieve_type::traits::coneSequence>& pCone = sieve->cone(arrow.source);
          typename arrow_section_type::value_type               o     = orientation->restrictPoint(arrow)[0];

          if (b_iter->second < 0) {
            o = -(o+1);
          }
          if (o < 0) {
            const int size = pCone->size();

            if (o == -size) {
              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter) {
                if (seen.find(*c_iter) == seen.end()) {
                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(arrow_type(*c_iter, arrow.source), o));
                  if (i == height-1) nCone->push_back(*c_iter);
                }
              }
            } else {
              const int numSkip = size + o;
              int       count   = 0;

              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter, ++count) {
                if (count < numSkip) continue;
                if (seen.find(*c_iter) == seen.end()) {
                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(arrow_type(*c_iter, arrow.source), o));
                  if (i == height-1) nCone->push_back(*c_iter);
                }
              }
              count = 0;
              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter, ++count) {
                if (count >= numSkip) break;
                if (seen.find(*c_iter) == seen.end()) {
                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(arrow_type(*c_iter, arrow.source), o));
                  if (i == height-1) nCone->push_back(*c_iter);
                }
              }
            }
          } else {
            if (o == 1) {
              for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter) {
                if (seen.find(*c_iter) == seen.end()) {
                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(arrow_type(*c_iter, arrow.source), o));
                  if (i == height-1) nCone->push_back(*c_iter);
                }
              }
            } else {
              const int numSkip = o-1;
              int       count   = 0;

              for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter, ++count) {
                if (count < numSkip) continue;
                if (seen.find(*c_iter) == seen.end()) {
                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(arrow_type(*c_iter, arrow.source), o));
                  if (i == height-1) nCone->push_back(*c_iter);
                }
              }
              count = 0;
              for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter, ++count) {
                if (count >= numSkip) break;
                if (seen.find(*c_iter) == seen.end()) {
                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(arrow_type(*c_iter, arrow.source), o));
                  if (i == height-1) nCone->push_back(*c_iter);
                }
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
    static Obj<sieve_type> Link(const Obj<bundle_type>& bundle, const point_type& p) {
      const Obj<sieve_type>&         link_sieve          = new sieve_type(bundle->comm(),  bundle->debug());
      const Obj<sieve_type>&         sieve               = bundle->getSieve();
      const int                      depth               = bundle->depth(p);
      const int                      interpolation_depth = bundle->height(p)+depth;
      Obj<coneArray>                 nSupport            = new supportArray();
      supportSet                     seen;
      //in either case, copy the closure of the cells surrounding the point to the new sieve
      static Obj<supportArray> neighboring_cells = sieve->nSupport(sieve->nCone(p, depth), interpolation_depth);
      static typename supportArray::iterator nc_iter = neighboring_cells->begin();
      static typename supportArray::iterator nc_iter_end = neighboring_cells->end();
      while (nc_iter != nc_iter_end) {
        addClosure(sieve, link_sieve, *nc_iter);
        nc_iter++;
      }
      if (interpolation_depth == 1) { //noninterpolated case
       //remove the point, allowing the copied closure to contract to a surface.

       if (depth != 0) {
         static Obj<coneArray> point_cone = sieve->cone(p);
         static typename coneArray::iterator pc_iter =  point_cone->begin();
         static typename coneArray::iterator pc_iter_end = point_cone->end();
         while (pc_iter != pc_iter_end) {
           link_sieve->removePoint(*pc_iter);
           pc_iter++;
         }
       }
       link_sieve->removePoint(p);

      } else { //interpolated case: remove the point, its closure, and the support of that closure,

        static Obj<supportArray> surrounding_support = sieve->Star(sieve->nCone(p, depth));
        static typename supportArray::iterator ss_iter = surrounding_support->begin();
        static typename supportArray::iterator ss_iter_end = surrounding_support->end();
        while (ss_iter != ss_iter_end) {
          link_sieve->removePoint(*ss_iter);
          ss_iter++;
        }
        link_sieve->removePoint(p);
      }
      link_sieve->stratify();
      return link_sieve;
    };
  };
}

#endif
