#ifndef included_ALE_XSieve_hh
#define included_ALE_XSieve_hh


#include <XSifter.hh>


namespace ALE { 
  
  //
  // XSieve definition
  //
  template<typename Arrow_, 
    typename TailOrder_  = XSifterDef::SourceColorOrder<Arrow_>,
    int SliceDepth = 1>
  struct XSieve : public XSifter<Arrow_,TailOrder_,SliceDepth> { // struct XSieve
    //
    typedef XSifter<Arrow_,TailOrder_,SliceDepth> xsifter_type;
    typedef XSieve                                xsieve_type;
    //
    // Encapsulated types: re-export types and/or bind parameterized types
    //
    //
    typedef Arrow_                                                 arrow_type;
    typedef typename arrow_type::source_type                       source_type;
    typedef typename arrow_type::target_type                       target_type;
    typedef typename xsifter_type::rec_type                        rec_type;
    //
    // Slicing
    //
    typedef typename xsifter_type::slicer_type                     slicer_type;
    typedef typename slicer_type::slice_type                       slice_type;


    //
    // Sequence types
    //
    typedef typename xsifter_type::BaseSequence                    BaseSequence;
    typedef typename xsifter_type::ConeSequence                    ConeSequence;
    typedef typename xsifter_type::SourceSlice                     SourceSlice;
    typedef typename xsifter_type::SourceSlice                     ClosureSlice;
    //
    //
    // Basic interface
    //
    //
    XSieve(const MPI_Comm comm, int debug = 0) : xsifter_type(comm, debug) {};
    //
    // Extended interface
    //
    //
    ClosureSlice closureSlice(const target_type& t) {
        ClosureSlice cl(this->slice());
        this->__closureSlice(cl,t);
        return cl;
    };//closureSlice()
    //
  protected:
    // aux function: called recursively
    #undef  __FUNCT__
    #define __FUNCT__ "__closureSlice"
    #undef  __ALE_XDEBUG__ 
    #define __ALE_XDEBUG__ 7
    void __closureSlice(ClosureSlice& cl, const target_type& t) {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": >>>";
        };
#endif      //
      // check each arrow in the cone above t and add each unseen arrow
      // the proceed to the cone above that arrow's source recursively
      static ConeSequence cone;
      static typename ConeSequence::iterator cbegin, cend, citer;
      static typename ClosureSlice::marker_type blank = 0, marked = 1;
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "closure over " << t << "\n";
        };
#endif      //
      try {
        this->cone(t,cone);
        cbegin = cone.begin(); cend = cone.end();
        for(citer = cbegin; citer != cend; ++citer) {
#ifdef ALE_USE_DEBUGGING
          if(ALE_XDEBUG) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
            std::cout << "examining arrow rec: " << citer.rec() << " ... ";
          };
#endif      //
          if(cl.marker(citer.rec()) == blank) { // if arrow has not been marked
            cl.add(citer.rec(), marked); // add this arrow and mark it at the same time
#ifdef ALE_USE_DEBUGGING
            if(ALE_XDEBUG) {
              std::cout << "blank, added: " << citer.rec() << ", making recursive call\n";
            };
#endif      
            __closureSlice(cl, citer.source()); // recursively compute closure over s
          }// blank
          else {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << "marked\n";
        };
#endif 
          }// marked
        }    
      } catch(...) {
        std::cout << "Unknown exception caught in __closureSlice\n";
        throw;
      };
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": <<<";
        };
#endif      //
    };//__closureSlice()
  public:
    //
    //
    template <typename Stream_>
    friend Stream_& operator<<(Stream_& os, const XSieve& xsieve) {
      os << "\n";
      os << "Cone index: (";
      for(typename xsifter_type::cone_index_type::iterator itor = xsieve._cone_index.begin(); itor != xsieve._cone_index.end(); ++itor) {
        os << *itor << " ";
      }
      os << ")";
      os << "\n";
      return os;
    };
    //
    template<typename Stream_>
    void view(Stream_& os, const char* label = NULL){
      if(label != NULL) {
        os << "Viewing " << label << " XSieve (debug: " << this->debug() << "): " << "\n";
      } 
      else {
        os << "Viewing a XSieve (debug: " << this->debug() << "): " << "\n";
      } 
      os << *this;
    };// view()

  }; // struct XSieve

  
} // namespace ALE

#endif
