#ifndef __ESI_MPI_traits_h
#define __ESI_MPI_traits_h

namespace esi {

/** The ESI MPI_traits file. */

template<class T>
struct MPI_traits {};

template<>
struct MPI_traits<real4> {
  static MPI_Datatype mpi_type() {return(ESI_MPI_REAL4);};
};

template<>
struct MPI_traits<real8> {
  static MPI_Datatype mpi_type() {return(ESI_MPI_REAL8);};
};

template<>
struct MPI_traits<int4> {
  static MPI_Datatype mpi_type() {return(ESI_MPI_INT4);};
};

template<>
struct MPI_traits<int8> {
  static MPI_Datatype mpi_type() {return(ESI_MPI_INT8);};
};

/*
  If we're using a Sun compiler, version earlier than 5.0,
  then complex isn't available.
*/
#if defined(__SUNPRO_CC) && __SUNPRO_CC < 0x500
#define NO_COMPLEX
#endif

#ifndef NO_COMPLEX

#include <complex>

template<>
struct MPI_traits<complex<real4> > {
  static MPI_Datatype mpi_type() {return(ESI_MPI_COMPLEX);};
};

template<>
struct MPI_traits<complex<real8> > {
  static MPI_Datatype mpi_type() {return(ESI_MPI_DOUBLE_COMPLEX);};
};

#endif /* NO_COMPLEX */

};     // esi namespace

#endif
