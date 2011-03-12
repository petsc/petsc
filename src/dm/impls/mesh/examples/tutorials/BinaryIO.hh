// -*- C++ -*-
//
// ======================================================================
//
//                           Brad T. Aagaard
//                        U.S. Geological Survey
//
// {LicenseText}
//
// ======================================================================
//

#if !defined(pylith_meshio_binaryio_hh)
#define pylith_meshio_binaryio_hh

#include <iosfwd>

namespace pylith {
  namespace meshio {
    class BinaryIO;
  } // meshio
} // pylith

class pylith::meshio::BinaryIO
{ // BinaryIO

// PUBLIC METHODS ///////////////////////////////////////////////////////
public :

  /** Read fixed length string from binary file.
   *
   * @param fin Input file stream
   * @param numChars Number of characters in string.
   */
  static
  std::string readString(std::ifstream& fin,
			 const int numChars);

  /** Change endian type by swapping byte order.
   *
   * @param vals Array of values
   * @param numVals Number of values
   * @param typesize Size of each value in bytes
   */
  static
  void swapByteOrder(char* vals,
		     const int numVals,
		     const int typesize);

}; // BinaryIO

#endif // pylith_meshio_binaryio_hh


// End of file 
