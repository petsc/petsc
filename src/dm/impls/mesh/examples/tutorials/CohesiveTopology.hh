// -*- C++ -*-
//
// ----------------------------------------------------------------------
//
//                           Brad T. Aagaard
//                        U.S. Geological Survey
//
// {LicenseText}
//
// ----------------------------------------------------------------------
//

/** @file libsrc/faults/CohesiveTopology.hh
 *
 * @brief C++ object to manage creation of cohesive cells.
 */

#if !defined(pylith_faults_cohesivetopology_hh)
#define pylith_faults_cohesivetopology_hh

#include <Mesh.hh> // Algorithms for submeshes
#include <Selection.hh> // Algorithms for submeshes

/// Namespace for pylith package
namespace pylith {
  namespace faults {
    class CohesiveTopology;
    typedef ALE::Mesh Mesh;
    typedef ALE::Mesh::sieve_type sieve_type;
    typedef ALE::Mesh::int_section_type int_section_type;
    typedef ALE::Mesh::real_section_type real_section_type;
  } // faults
} // pylith

/// C++ object to manage creation of cohesive cells.
class pylith::faults::CohesiveTopology
{ // class Fault

  // PUBLIC METHODS /////////////////////////////////////////////////////
public :
  typedef std::vector<Mesh::point_type> PointArray;

  /** Create cohesive cells.
   *
   * @param fault Finite-element mesh of fault (output)
   * @param mesh Finite-element mesh
   * @param faultVertices Vertices assocated with faces of cells defining 
   *   fault surface
   * @param materialId Material id for cohesive elements.
   * @param constraintCell True if creating cells constrained with 
   *   Lagrange multipliers that require extra vertices, false otherwise
   */
  static
  void create(ALE::Obj<Mesh>* fault,
              const ALE::Obj<Mesh>& mesh,
              const ALE::Obj<Mesh::int_section_type>& groupField,
              const int materialId,
              const bool constraintCell = false);

  // PRIVATE METHODS ////////////////////////////////////////////////////
private :
  /** Get number of vertices on face.
   *
   * @param cell Finite-element cell
   * @param mesh Finite-element mesh
   *
   * @returns Number of vertices on cell face
   */
  static
  unsigned int _numFaceVertices(const Mesh::point_type& cell,
                                const ALE::Obj<Mesh>& mesh,
                                const int depth = -1);

  /** Determine a face orientation
   *    We should really have an interpolated mesh, instead of
   *    calculating this on the fly.
   *
   * @param cell Finite-element cell
   * @param mesh Finite-element mesh
   *
   * @returns True for positive orientation, otherwise false
   */
  static
  bool _faceOrientation(const Mesh::point_type& cell,
                        const ALE::Obj<Mesh>& mesh,
                        const int numCorners,
                        const int indices[],
                        const int oppositeVertex,
                        PointArray *origVertices,
                        PointArray *faceVertices);

  template<typename FaceType>
  static
  bool _getOrientedFace(const ALE::Obj<Mesh>& mesh,
                        const Mesh::point_type& cell,
                        FaceType face,
                        const int numCorners,
                        int indices[],
                        PointArray *origVertices,
                        PointArray *faceVertices);

  template<class InputPoints>
  static
  bool _compatibleOrientation(const ALE::Obj<Mesh>& mesh,
                              const Mesh::point_type& p,
                              const Mesh::point_type& q,
                              const int numFaultCorners,
                              const int faultFaceSize,
                              const int faultDepth,
                              const ALE::Obj<InputPoints>& points,
                              int indices[],
                              PointArray *origVertices,
                              PointArray *faceVertices,
                              PointArray *neighborVertices);

  static
  void _replaceCell(const ALE::Obj<sieve_type>& sieve,
                    const Mesh::point_type cell,
                    std::map<int,int> *vertexRenumber,
                    const int debug = 0);

  template<class InputPoints>
  static
  void _computeCensoredDepth(const ALE::Obj<Mesh>& mesh,
                             const ALE::Obj<Mesh::label_type>& depth,
                             const ALE::Obj<Mesh::sieve_type>& sieve,
                             const ALE::Obj<InputPoints>& points,
                             const Mesh::point_type& firstCohesiveCell,
                             const ALE::Obj<std::set<Mesh::point_type> >& modifiedPoints);

}; // class CohesiveTopology

#endif // pylith_faults_cohesivetopology_hh


// End of file 
