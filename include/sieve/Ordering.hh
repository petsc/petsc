#ifndef included_ALE_Ordering_hh
#define included_ALE_Ordering_hh

#ifndef  included_ALE_Partitioner_hh
#include <Partitioner.hh>
#endif

PetscErrorCode SPARSEPACKgenrcm(PetscInt *neqns,PetscInt *xadj,PetscInt *adjncy,PetscInt *perm,PetscInt *mask,PetscInt *xls);

namespace ALE {
  template<typename Alloc_ = malloc_allocator<int> >
  class Ordering {
  public:
    typedef Alloc_                   alloc_type;
    typedef IUniformSection<int,int> perm_type;
  public:
    template<typename Mesh>
    static void calculateMeshReordering(const Obj<Mesh>& mesh, Obj<perm_type>& permutation) {
      int *start     = NULL;
      int *adjacency = NULL;
      int *perm      = NULL;
      int  numVertices;

      Partitioner<>::buildDualCSRV(mesh, &numVertices, &start, &adjacency, true);
      permutation->setChart(perm_type::chart_type(0, numVertices));
      for(int i = 0; i < numVertices; ++i) permutation->setFiberDimension(i, 1);
      permutation->allocate();
      perm = const_cast<int*>(permutation->restrictSpace());
      int *mask = alloc_type().allocate(numVertices);
      for(int i = 0; i < numVertices; ++i) {alloc_type().construct(mask+i, 1);}
      int *xls  = alloc_type().allocate(numVertices*2);
      for(int i = 0; i < numVertices*2; ++i) {alloc_type().construct(xls+i,  0);}
      // Correct for Fortran numbering
      for(int i = 0; i < start[numVertices]; ++i) ++adjacency[i];
      for(int i = 0; i <= numVertices; ++i) ++start[i];
      PetscErrorCode ierr = SPARSEPACKgenrcm(&numVertices, start, adjacency, perm, mask, xls);CHKERRXX(ierr);
      for(int i = 0; i < numVertices; ++i) {alloc_type().destroy(mask+i);}
      alloc_type().deallocate(mask, numVertices);
      for(int i = 0; i < numVertices*2; ++i) {alloc_type().destroy(xls+i);}
      alloc_type().deallocate(xls, numVertices*2);
      // Correct for Fortran numbering
      for(int i = 0; i < numVertices; ++i) --perm[i];
     };

    template<typename Section, typename Labeling>
    static void relabelSection(Section& section, Labeling& relabeling, Section& newSection) {
      newSection.setChart(section.getChart());

      for(typename Section::point_type p = section.getChart().min(); p < section.getChart().max(); ++p) {
	const typename Section::point_type newP = relabeling.restrictPoint(p)[0];

	newSection.setFiberDimension(newP, section.getFiberDimension(p));
      }
      newSection.allocatePoint();
      for(typename Section::point_type p = section.getChart().min(); p < section.getChart().max(); ++p) {
	const typename Section::point_type newP = relabeling.restrictPoint(p)[0];

	newSection.updatePoint(newP, section.restrictPoint(p));
      }
    };
  };
}
#endif /* included_ALE_Ordering_hh */
