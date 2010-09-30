#if !defined(__PETSCMESH_SOLVERS_HH)
#define __PETSCMESH_SOLVERS_HH

#include <petscmesh.hh>
#include <petscpc.h>

using ALE::Obj;

template<typename Section, typename Order>
void constructFieldSplit(const Obj<Section>& section, const Obj<Order>& globalOrder, Vec v, PC fieldSplit) {
  const typename Section::chart_type& chart = section->getChart();
  PetscInt                            space = 0;
  PetscInt                            *spaceSize = PETSC_NULL;
  char                                spaceName[2] = {'0', '\0'};
  const int                           debug        = 0;
  PetscErrorCode                      ierr;

  PetscInt total = 0;
  ierr = PetscMalloc(section->getNumSpaces() * sizeof(PetscInt), &spaceSize);CHKERRXX(ierr);
  for(typename std::vector<Obj<typename Section::atlas_type> >::const_iterator s_iter = section->getSpaces().begin(); s_iter != section->getSpaces().end(); ++s_iter, ++space) {
    PetscInt n = 0;

    for(typename Section::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
      const int dim  = section->getFiberDimension(*c_iter, space);
      const int cDim = section->getConstraintDimension(*c_iter, space);

      if ((dim > cDim) && globalOrder->isLocal(*c_iter)) {
        n += dim - cDim;
      }
    }
    if (debug) {std::cout << "Space " << space << ": size " << n << std::endl;}
    spaceSize[space] = n;
    total += n;
  }
  PetscInt localSize;
  ierr = VecGetLocalSize(v, &localSize);CHKERRXX(ierr);
  if (debug) {std::cout << "Vector local size " << localSize << std::endl;}
  assert(localSize == total);
  space = 0;
  for(typename std::vector<Obj<typename Section::atlas_type> >::const_iterator s_iter = section->getSpaces().begin(); s_iter != section->getSpaces().end(); ++s_iter, ++space) {
    PetscInt  n = spaceSize[space];
    PetscInt  i = -1;
    PetscInt *idx;
    IS        is;

    ierr = PetscMalloc(n * sizeof(PetscInt), &idx);CHKERRXX(ierr);
    for(typename Section::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
      const int dim  = section->getFiberDimension(*c_iter, space);
      const int cDim = section->getConstraintDimension(*c_iter, space);

      if (dim > cDim) {
        if (!globalOrder->isLocal(*c_iter)) continue;
        int off = globalOrder->getIndex(*c_iter);

        for(int s = 0; s < space; ++s) {
          off += section->getConstrainedFiberDimension(*c_iter, s);
        }
        if (cDim) {
          // This is potentially dangerous
          //   These constraints dofs are for SINGLE FIELDS, not the entire point (confusing)
          const int *cDofs = section->getConstraintDof(*c_iter, space);

          for(int d = 0, c = 0, k = 0; d < dim; ++d) {
            if ((c < cDim) && (cDofs[c] == d)) {
              if (debug) {std::cout << "  Ignored " << (off+k) << " at local pos " << d << " for point " << (*c_iter) << std::endl;}
              ++c;
              continue;
            }
            idx[++i] = off+k;
            if (debug) {std::cout << "Added " << (off+k) << " at pos " << i << " for point " << (*c_iter) << std::endl;}
            ++k;
          }
        } else {
          for(int d = 0; d < dim; ++d) {
            idx[++i] = off+d;
            if (debug) {std::cout << "Added " << (off+d) << " at pos " << i << " for point " << (*c_iter) << std::endl;}
          }
        }
      }
    }
    if (i != n-1) {throw PETSc::Exception("Invalid fibration numbering");}
    ierr = ISCreateGeneralNC(section->comm(), n, idx, &is);CHKERRXX(ierr);
    ierr = PCFieldSplitSetIS(fieldSplit, spaceName, is);CHKERRXX(ierr);
    ++spaceName[0];
  }
  ierr = PetscFree(spaceSize);CHKERRXX(ierr);
};

#endif // __PETSCMESH_SOLVERS_HH
							    
