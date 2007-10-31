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

#if !defined(pylith_meshio_gmvfile_hh)
#define pylith_meshio_gmvfile_hh

#include <string> // HASA std::string

/// Forward declaration of STL vector
namespace std {
  // std::vector
  template<typename T> class allocator;
  template<typename T, typename U> class vector;

  // std::valarray
  template<typename T> class valarray;
} // std

/// Aliases 
namespace pylith {
  /// Alias for std::vector<int>
  typedef std::vector<int, std::allocator<int> > int_vector;

  /// Alias for std::vector<double>
  typedef std::vector<double, std::allocator<double> > double_vector;

  /// Alias for std::vector<std::string>
  typedef std::vector<std::string, std::allocator<std::string> > string_vector;

  /// Alias for std::valarray<int>
  typedef std::valarray<int> int_array;

  /// Alias for std::valarray<float>
  typedef std::valarray<float> float_array;

  /// Alias for std::valarray<double>
  typedef std::valarray<double> double_array;

} // pylith

#include <vector>
#include <valarray>

namespace pylith {
  namespace meshio {
  class GMVFile;
  } // meshio
} // pylith

class pylith::meshio::GMVFile
{ // GMVFile

// PUBLIC METHODS ///////////////////////////////////////////////////////
public :

  /** Constructor with name of GMV file.
   *
   * @param filename Name of GMV file
   */
  GMVFile(const char* name);

  /// Default destructor.
  ~GMVFile(void);

  /** Is GMV file ascii?
   *
   * @param filename Name of GMV file.
   *
   * @returns True if GMV file is ascii, false otherwise
   */
  static
  bool isAscii(const char* filename);

// PROTECTED MEMBERS ////////////////////////////////////////////////////
protected :

  std::string _filename; ///< Name of GMV file
  
}; // GMVFile

#endif // pylith_meshio_gmvfile_hh


// End of file 
