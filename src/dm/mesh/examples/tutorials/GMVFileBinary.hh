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

#if !defined(pylith_meshio_gmvfilebinary_hh)
#define pylith_meshio_gmvfilebinary_hh

#include "GMVFile.hh" // ISA GMVFile

#include <iosfwd>

namespace pylith {
  namespace meshio {
    class GMVFileBinary;
  } // meshio
} // pylith

class pylith::meshio::GMVFileBinary : public GMVFile
{ // GMVFileBinary

// PUBLIC METHODS ///////////////////////////////////////////////////////
public :

  /** Constructor with name of GMV file.
   *
   * @param filename Name of GMV file
   * @param flipEndian Flip endian type when reading/writing.
   */
  GMVFileBinary(const char* filename,
		const bool flipEndian =false);

  /// Default destructor 
  ~GMVFileBinary(void);

  /** Get header.
   *
   * @returns Header that appears in BINARY GMV file
   */
  static
  const char* header(void);

  /** Read BINARY GMV file.
   *
   * @coordinates Coordinates of vertices.
   * @param cells Indices of cell vertices.
   * @param materialIds Material identifiers for each cell.
   * @param meshDim Dimension of cells in mesh.
   * @param numVertices Number of vertices in mesh.
   * @param numCells Number of cells in mesh.
   * @param numCorners Number of vertices in each cell.
   */
  void read(double_array* coordinates,
	    int_array* cells,
	    int_array* materialIds,
	    int* meshDim,
	    int* spaceDim,
	    int* numVertices,
	    int* numCells,
	    int* numCorners);

  /** Write BINARY GMV file.
   *
   * @coordinates Coordinates of vertices.
   * @param cells Indices of cell vertices.
   * @param materialIds Material identifiers for each cell.
   * @param meshDim Dimension of cells in mesh.
   * @param spaceDim Number of coordinates per vertex.
   * @param numVertices Number of vertices in mesh.
   * @param numCells Number of cells in mesh.
   * @param numCorners Number of vertices in each cell.
   */
  void write(const double_array& coordinates,
	     const int_array& cells,
	     const int_array& materialIds,
	     const int meshDim,
	     const int spaceDim,
	     const int numVertices,
	     const int numCells,
	     const int numCorners);

// PRIVATE METHODS //////////////////////////////////////////////////////
private :
  
  /** Read header.
   *
   * @param fin Input file stream
   */
  void _readHeader(std::ifstream& fin);

  /** Read vertices.
   *
   * @param fin Input file stream.
   * @param coordinates Coordinates of vertices.
   * @param numVertices Number of vertices.
   */
  void _readVertices(std::ifstream& fin,
		     double_array* coordinates,
		     int* numVertices,
		     int* spaceDim);

  /** Read cells.
   *
   * @param fin Input file stream
   * @param cells Indices of cell vertices.
   * @param numCells Number of cells in mesh.
   * @param numCorners Number of vertices in each cell.
   */
  void _readCells(std::ifstream& fin,
		  int_array* cells,
		  int* numCells,
		  int* numCorners);

  /** Read and discard variables associated with vertices.
   *
   * @param fin Input file stream
   * @param numVertices Number of vertices in mesh.
   * @param numCells Number of cells in mesh.
   */
  void _readVariables(std::ifstream& fin,
		      const int numVertices,
		      const int numCells);

  /** Read and discard material flags for vertices.
   *
   * @param fin Input file stream
   * @param numVertices Number of vertices in mesh.
   * @param numCells Number of cells in mesh.
   */
  void _readFlags(std::ifstream& fin,
		  const int numVertices,
		  const int numCells);

  /** Read material values for cells.
   *
   * @param fin Input file stream
   * @param materialIds Material identifiers for each cell.
   * @param numVertices Number of vertices in mesh.
   * @param numCells Number of cells in mesh.
   */
  void _readMaterials(std::ifstream& fin,
		      int_array* materialIds,
		      const int numVertices,
		      const int numCells);

// PRIVATE MEMBERS //////////////////////////////////////////////////////
private :
  
  /// Header in binary GMV file.
  static const char* _HEADER;

  bool _flipEndian; ///< True if need to change endian when reading/writing

}; // GMVFileBinary

#endif // pylith_meshio_gmvfilebinary


// End of file 
