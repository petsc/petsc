#ifndef included_ALE_Delta_hh
#define included_ALE_Delta_hh

#ifndef  included_ALE_Sifter_hh
#include <Sifter.hh>
#endif

//
// This file contains classes and methods implementing  the fusion operation on a pair of ColorBiGraphs or similar objects.
//
namespace ALE {

  namespace Two {


      
//       template <typename LeftBiGraph_, typename RightBiGraph_, typename DeltaBiGraph_>
//       class Cone {
//         // Cone::operator() in various forms
//         void 
//         operator()(const left_type& l, const right_type& r, delta_type& d, const fuser_type& f = fuser_type()) {
//           // Compute the overlap of left and right bases and then call a 'based' version of the operator
//           operator()(overlap(l,r), l,r,d,f);
//         };
//         void 
//         operator()(const BaseOverlapSequence& overlap,const left_type& l,const right_type& r, delta_type& d, 
//                    const fuser_type& f = fuser_type()) {
//           for(typename BaseOverlapSequence::iterator i = overlap.begin(); i != overlap.end(); i++) {
//             typename left_type::traits::coneSequence lcone = l.cone(*i);
//             typename right_type::traits::coneSequence rcone = r.cone(*i);
//           }
//         }
//         Obj<delta_type> 
//         operator()(const left_type& l, const right_type& r, const fuser_type& f = fuser_type()) {
//           Obj<delta_type> d = delta_type();
//           operator()(l,r,d,f);
//           return d;
//         };
//         Obj<delta_type> 
//         operator()(const BaseOverlapSequence& overlap, const left_type& l, const right_type& r, const fuser_type& f = fuser_type()) {
//           Obj<delta_type> d = delta_type();
//           operator()(overlap,l,r,d,f);
//           return d;
//         };
//       }; // class Cone
    


//       template <typename LeftBiGraph_, typename RightBiGraph_, typename DeltaBiGraph_>
//       class ConeProductFuser {
//       public:
//         //Encapsulated types
//         struct traits {
//           typedef LeftBiGraph_  left_type;
//           typedef RightBiGraph_ right_type;
//           typedef DeltaBiGraph_ delta_type;
//           typedef std::pair<typename left_type::traits::source_type,typename right_type::traits::source_type> source_type;
//           typedef typename left_type::traits::target_type                                                     target_type;
//           typedef std::pair<typename left_type::traits::color_type,typename right_type::traits::color_type>   color_type;
//         };        
//         void
//         fuseCones(const typename traits::left_type::traits::coneSequence&  lcone, 
//                   const typename traits::right_type::traits::coneSequence& rcone, 
//                   typename typename traits::delta_type& delta) {
//           // This Fuser traverses both left cone and right cone, forming an arrow from each pair of arrows -- 
//           // one from each of the cones --  and inserting it into the delta BiGraph.
//           for(typename left_type::traits::coneSequence::iterator lci = lcone.begin(); lci != lcone.end(); lci++) {
//             for(typename left_type::traits::coneSequence::iterator lci = lcone.begin(); lci != lcone.end(); lci++) {
//               delta.addArrow(this->fuseArrows(lci.arrow(), rci.arrow()));
//             }
//           }
//         }
//         typename traits::delta_type::arrow_type
//         fuseArrows(const typename traits::left_type::traits::arrow_type& larrow, 
//                    const typename traits::right_type::traits::arrow_type& rarrow) {
//           return typename traits::arrow_type(traits::source_type(*lci,*rci), lci.target(), 
//                                              typename traits::color_type(lci.color(),rci.color()));
//         }
//       }; // struct ConeProductFuser


//     template <typename LeftBiGraph_, typename RightBiGraph_>
//     class BaseOverlapSequence : public LeftBiGraph_::traits::baseSequence {
//       // There is an assumption that LeftBiGraph_ and RightBiGraph_ have equivalent baseSequence types
//     public:
//       //
//       // Encapsulted types
//       //
//       typedef LeftBiGraph_  left_type;
//       typedef RightBiGraph_ right_type;
//       typedef typename left_type::traits::baseSequence::traits traits;
      
//       // Overloaded iterator
//       class iterator : public traits::iterator {
//       };
//       //
//       // Basic interface
//       //
//       BaseOverlapSequence(const left_type& l, const right_type& r) : left_type::traits::baseSequence(l.base()), _left(l), _right(r){};
      
//     protected:
//       const typename traits::left_type&  _left;
//       const typename traits::right_type& _right;
      
//     };// class BaseOverlapSequence
    


  } // namespace Two
    
} // namespace ALE

#endif
