// 
// File:          src/BS/LinkCheckerI/Checker_Impl/BS_LinkCheckerI_Checker_Impl.hh
// Symbol:        BS.LinkCheckerI.Checker-v3.0
// Symbol Type:   class
// Babel Version: 0.6.1
// Description:   Server-side implementation for BS.LinkCheckerI.Checker
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_BS_LinkCheckerI_Checker_Impl_hh
#define included_BS_LinkCheckerI_Checker_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_BS_LinkCheckerI_Checker_IOR_h
#include "BS_LinkCheckerI_Checker_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_BS_LinkCheckerI_Checker_hh
#include "src/BS/LinkCheckerI/Checker/BS_LinkCheckerI_Checker.hh"
#endif
#ifndef included_BS_LinkError_hh
#include "src/BS/LinkError/BS_LinkError.hh"
#endif
#ifndef included_SIDL_BaseInterface_hh
#include "src/SIDL/BaseInterface/SIDL_BaseInterface.hh"
#endif


// DO-NOT-DELETE splicer.begin(BS.LinkCheckerI.Checker._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(BS.LinkCheckerI.Checker._includes)

namespace BS { 
  namespace LinkCheckerI { 

    /**
     * Symbol "BS.LinkCheckerI.Checker" (version 3.0)
     */
    class Checker_impl {

    private:
      // Pointer back to IOR.
      // Use this to dispatch back through IOR vtable.
      Checker self;

      // DO-NOT-DELETE splicer.begin(BS.LinkCheckerI.Checker._implementation)
      // Put additional implementation details here...
      // DO-NOT-DELETE splicer.end(BS.LinkCheckerI.Checker._implementation)

    private:
      // private default constructor (required)
      Checker_impl() {} 

    public:
      // SIDL constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
      Checker_impl( struct BS_LinkCheckerI_Checker__object * s ) : self(s,
        true) { _ctor(); }

      // user defined construction
      void _ctor();

      // virtual destructor (required)
      virtual ~Checker_impl() { _dtor(); }

      // user defined destruction
      void _dtor();

    public:

      /**
       * user defined non-static method.
       */
      void
      openLibrary (
        /*in*/ std::string fullpath
      )
      throw ( 
        BS::LinkError
      );

    };  // end class Checker_impl

  } // end namespace LinkCheckerI
} // end namespace BS

// DO-NOT-DELETE splicer.begin(BS.LinkCheckerI.Checker._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(BS.LinkCheckerI.Checker._misc)

#endif
