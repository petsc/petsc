#ifndef included_ALE_Ordering_hh
#define included_ALE_Ordering_hh

#ifndef  included_ALE_Partitioner_hh
#include <Partitioner.hh>
#endif

namespace ALE {
  template<typename Alloc_ = malloc_allocator<int> >
  class Ordering {
  public:
    typedef Alloc_ alloc_type;
  public:
    template<typename Mesh>
    static reorderMesh(const Obj<Mesh>& mesh) {
      typedef IUniformSection<int,int> perm_type;
      int *start     = NULL;
      int *adjacency = NULL;
      int *perm      = NULL;
      int  numVertices;

      Partitioner<>::buildDualCSR(mesh, &numVertices, &start, &adjacency, true);
      Obj<perm_type> permutation;
      permutation->setChart(perm_type::chart_type(0, numVertices));
      perm = const_cst<int*>(permutation->restrictSpace());
      int *mask = alloc_type().allocate(numVertices);
      for(int i = 0; i < numVertices; ++i) {alloc_type().construct(mask+i, 1);}
      int *xls  = alloc_type().allocate(numVertices);
      for(int i = 0; i < numVertices; ++i) {alloc_type().construct(xls+i,  0);}
      PetscErrorCode ierr = SPARSEPACKgenrcm(numVertices, start, adjacency, perm, mask, xls);CHKERRXX(ierr);
      for(int i = 0; i < numVertices; ++i) {alloc_type().destroy(mask+i);}
      alloc_type().deallocate(mask, numVertices);
      for(int i = 0; i < numVertices; ++i) {alloc_type().destroy(xls+i);}
      alloc_type().deallocate(xls, numVertices);

      mesh->permute(*permutation);
    };
  };
}
#endif /* included_ALE_Ordering_hh */
