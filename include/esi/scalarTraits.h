#ifndef __ESI_scalarTraits_h
#define __ESI_scalarTraits_h

#include <math.h> //for the fabs function...

namespace esi {

/** The ESI scalarTraits file.

    For the most general type 'class T', we define aborting functions, 
    which should restrict implementations from using traits other than the
    specializations defined below.
*/

template<class T>
struct scalarTraits {

  typedef T magnitude_type;

  static inline magnitude_type magnitude(T a) { 
    cout << "esi::scalarTraits: unsupported scalar type." << endl; 
    abort(); 
    return(a);
  };

  static inline T random() {
    cout << "esi::scalarTraits: unsupported scalar type." << endl; 
    abort(); 
    return(a);
  };
  
  static inline const char * name() {
    cout << "esi::scalarTraits: unsupported scalar type." << endl; 
    abort(); 
    return(0); 
  };
};

template<>
struct scalarTraits<real4> {

  typedef real4 magnitude_type;
  
  static inline magnitude_type magnitude(real4 a) { 
    return(fabs(a)); 
  };

  static inline real4 random() {
    real4 rnd = (real4)rand()/RAND_MAX;
    return( (real4)(-1.0 + 2.0*rnd) );
  };

  static inline const char * name() { 
    return("esi::real4"); 
  };
};


template<>
struct scalarTraits<real8> {

  typedef real8 magnitude_type;

  static magnitude_type magnitude(real8 a) {
    return(fabs(a));
  };

  static inline real8 random() {
    real8 rnd = (real8)rand()/RAND_MAX;
    return( (real8)(-1.0 + 2.0*rnd) );
  };

  static inline const char * name() {
    return("esi::real8");
  };
};

/*
  If we're using a Sun compiler, version earlier than 5.0,
  then complex isn't available.
*/
#if defined(__SUNPRO_CC) && __SUNPRO_CC < 0x500
#define NO_COMPLEX
#endif

/*
  If we're using the tflops Portland Group compiler, then complex isn't
  available. (As of July 21, 2000. abw)
*/
#if defined(__PGI) && defined(__i386)
#define NO_COMPLEX
#endif

#define NO_COMPLEX
#ifndef NO_COMPLEX

#include <complex>

template<>
struct scalarTraits< std::complex<real4> > {

  typedef real4 magnitude_type;

  static magnitude_type magnitude(std::complex<real4> a) {
    return(std::abs(a));
  };

  static inline std::complex<real4> random() {
    real4 rnd1 = (real4)rand()/RAND_MAX;
    real4 rnd2 = (real4)rand()/RAND_MAX;
    return( std::complex<real4>(-1.0+2.0*rnd1, -1.0+2.0*rnd2) );
  };

  static inline const char * name() {
    return("complex<esi::real4>");
  };
};

template<>
struct scalarTraits< std::complex<real8> > {

  typedef real8 magnitude_type;

  static magnitude_type magnitude(std::complex<real8> a) {
    return(std::abs(a));
  };
  
  static inline std::complex<real8> random() {
    real8 rnd1 = (real8)rand()/RAND_MAX;
    real8 rnd2 = (real8)rand()/RAND_MAX;
    return( std::complex<real8>(-1.0+2.0*rnd1, -1.0+2.0*rnd2) );
  };

  static inline const char * name() {
    return("complex<esi::real8>");
  };
};

#endif  /* NO_COMPLEX */


/*
  If we're using a Sun compiler, version earlier than 5.0,
  then 'typename' isn't available.
*/
#if defined(__SUNPRO_CC) && __SUNPRO_CC < 0x500
#define TYPENAME
#else
#define TYPENAME typename
#endif

};     // esi namespace
#endif

