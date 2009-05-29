#if !defined(__PETSCMESH_SOLVERS_HH)
#define __PETSCMESH_SOLVERS_HH

#include <petscmesh.hh>
#include <petscpc.h>

using ALE::Obj;

template<typename Section>
void constructFieldSplit(const Obj<Section>& section, PC fieldSplit) {
  const typename Section::chart_type& chart = section->getChart();
  PetscInt                            space = 0;
  PetscErrorCode                      ierr;

  for(typename std::vector<Obj<typename Section::atlas_type> >::const_iterator s_iter = section->getSpaces()->begin(); s_iter != section->getSpaces()->end(); ++s_iter, ++space) {
    PetscInt  n = section->size(space);
    PetscInt  i = -1;
    PetscInt *idx;
    IS        is;

    ierr = PetscMalloc(n * sizeof(PetscInt), &idx);CHKERRXX(ierr);
    for(typename Section::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
      const int fDim = section->getFiberDimension(*c_iter, space);

      if (fDim) {
        const int off = s_iter->restrictPoint(*c_iter)[0].index;

        for(int d = 0; d < fDim; ++d) {
          // TODO: In parallel, we need remap this number to a global order
          idx[++i] = off+d;
        }
      }
    }
    if (i != n) {throw PETSc::Exception("Invalid fibration numbering");}
    ierr = ISCreateGeneralNC(section->comm(), n, idx, &is);CHKERRXX(ierr);
    ierr = PCFieldSplitSetIS(fieldSplit, is);CHKERRXX(ierr);
  }
};

#endif // __PETSCMESH_SOLVERS_HH
