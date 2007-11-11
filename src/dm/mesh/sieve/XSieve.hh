#ifndef included_ALE_XSieve_hh
#define included_ALE_XSieve_hh


#include <XSifter.hh>


namespace ALE { 
  
  //
  // XSieve definition
  //
  #undef  __CLASS__
  #define __CLASS__ "XSieve"
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
    typedef typename xsifter_type::SourceSlice                     BoundarySlice;
    //
    typedef typename ALE::Set<source_type>                         BoundarySet;
    //
    //
    // Basic interface
    //
    //
    XSieve(const MPI_Comm comm, int debug = 0) : xsifter_type(comm, debug) {};
    //
    // Main interface
    //
    //
    BoundarySlice boundarySlice(const target_type& t) {
        BoundarySlice bd(this->slice());
        this->__boundarySlice(bd,t);
        return bd;
    };//boundarySlice()
    //
  protected:
    // aux function: called recursively
    #undef  __FUNCT__
    #define __FUNCT__ "__boundarySlice"
    #undef  __ALE_XDEBUG__ 
    #define __ALE_XDEBUG__ 7
    void __boundarySlice(BoundarySlice& bd, const target_type& t) {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": >>>";
        };
#endif      
        // CANNOT make cone, cbegin, cend and citer static in this RECURSIVE function
        ConeSequence cone;
        typename ConeSequence::iterator cbegin, cend, citer;
        static typename BoundarySlice::marker_type blank = 0, marked = 1;
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "boundary of " << t << "\n";
        };
#endif      //
      try {
        // check each arrow in the cone above t and add each unseen arrow
        // the proceed to the cone above that arrow's source recursively
        this->cone(t,cone);
        cbegin = cone.begin(); cend = cone.end();
        for(citer = cbegin; citer != cend; ++citer) {
#ifdef ALE_USE_DEBUGGING
          if(ALE_XDEBUG) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
            std::cout << "examining arrow rec: " << citer.rec() << " ... ";
          };
#endif      //
          if(bd.marker(citer.rec()) == blank) { // if arrow has not been marked
            bd.add(citer.rec(), marked); // add this arrow and mark it at the same time
#ifdef ALE_USE_DEBUGGING
            if(ALE_XDEBUG) {
              std::cout << "blank, added: " << citer.rec() << ", making recursive call\n";
            };
#endif      
            __boundarySlice(bd, citer.source()); // recursively compute boundary of s
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
        std::cout << "Unknown exception caught in " << __CLASS__ << "::" << __FUNCT__ << "\n";
        throw;
      };
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": <<<";
        };
#endif      //
    };//__boundarySlice()
  public:
    //
    #undef  __FUNCT__
    #define __FUNCT__ "boundarySet"
    #undef  __ALE_XDEBUG__ 
    #define __ALE_XDEBUG__ 7
    Obj<BoundarySet> boundarySet(const target_type& t) {
#ifdef ALE_USE_DEBUGGING
      if(ALE_XDEBUG) {
        std::cout << __CLASS__ << "::" << __FUNCT__ << ": >>>";
      };
#endif            
      Obj<BoundarySet> bd(new BoundarySet());
      this->__boundarySet(bd,t);
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": <<<";
        };
#endif      
      return bd;
    };//boundarSet()
    //
  protected:
    // aux function: called recursively
    #undef  __FUNCT__
    #define __FUNCT__ "__boundarySet"
    #undef  __ALE_XDEBUG__ 
    #define __ALE_XDEBUG__ 7
    void __boundarySet(Obj<BoundarySet>& bd, const target_type& t) {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": >>>";
        };
#endif      
        // CANNOT make cone, cbegin, cend and citer static in this RECURSIVE function
        ConeSequence cone;
        typename ConeSequence::iterator cbegin, cend, citer;
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "boundary of " << t << "\n";
        };
#endif      //
        try {
          // check each arrow in the cone above t and add each unseen source
          // the proceed to the cone above that arrow's source recursively
          this->cone(t,cone);
          cbegin = cone.begin(); cend = cone.end();
          for(citer = cbegin; citer != cend; ++citer) {
#ifdef ALE_USE_DEBUGGING
            if(ALE_XDEBUG) {
              std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
              std::cout << "examining arrow rec: " << citer.rec() << " ... ";
            };
#endif      //
            if(bd->find(citer.arrow().source()) == bd->end()) { // don't have this point yet
              bd->insert(citer.arrow().source()); // add the source of the arrow 
#ifdef ALE_USE_DEBUGGING
              if(ALE_XDEBUG) {
                std::cout << "source not seen yet, added: " << citer.rec().source() << ", making recursive call\n";
              };
#endif      
              __boundarySet(bd, citer.source()); // recursively compute boundary of s
            }
          }    
        } catch(...) {
          std::cout << "Unknown exception caught in " << __CLASS__ << "::" << __FUNCT__ << "\n";
          throw;
        };
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": <<<";
        };
#endif      //
    };//__boundarySet()
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
