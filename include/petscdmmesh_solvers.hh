#if !defined(__PETSCDMMESH_SOLVERS_HH)
#define __PETSCDMMESH_SOLVERS_HH

#include <petscdmmesh.hh>
#include <petscpc.h>

using ALE::Obj;

template<typename Section, typename Order>
void constructFieldSplit(const Obj<Section>& section, PetscInt numSplits, PetscInt numSplitFields[], PetscInt splitFields[], const Obj<Order>& globalOrder, Mat precon[], MatNullSpace nullsp[], Vec v, PC fieldSplit) {
  const typename Section::chart_type&  chart = section->getChart();
  PetscInt                            *numFields = PETSC_NULL;
  PetscInt                            *fields    = PETSC_NULL;
  PetscInt                            *splitSize = PETSC_NULL;
  char                                 splitName[2] = {'0', '\0'};
  const int                            debug        = 0;
  PetscErrorCode                       ierr;

  PetscInt total = 0;
  if (numSplits < 0) {numSplits = section->getNumSpaces();}
  ierr = PetscMalloc3(numSplits,PetscInt,&numFields,section->getNumSpaces(),PetscInt,&fields,numSplits,PetscInt,&splitSize);CHKERRXX(ierr);
  if (!numSplitFields) {
    for(PetscInt s = 0; s < numSplits; ++s) {numFields[s] = 1;}
  } else {
    for(PetscInt s = 0; s < numSplits; ++s) {numFields[s] = numSplitFields[s];}
  }
  if (!splitFields) {
    for(PetscInt s = 0, sp = 0; s < numSplits; ++s) {
      for(PetscInt f = 0; f < numFields[s]; ++f, ++sp) {
        fields[sp] = sp;
      }
    }
  } else {
    for(PetscInt s = 0, sp = 0; s < numSplits; ++s) {
      for(PetscInt f = 0; f < numFields[s]; ++f, ++sp) {
        fields[sp] = splitFields[sp];
      }
    }
  }
  for(PetscInt s = 0, q = 0; s < numSplits; ++s) {
    PetscInt n = 0;

    for(typename Section::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
      for(PetscInt f = 0, q2 = q; f < numFields[s]; ++f, ++q2) {
        const PetscInt space = fields[q2];
        const PetscInt dim   = section->getFiberDimension(*c_iter, space);
        const PetscInt cDim  = section->getConstraintDimension(*c_iter, space);

        if ((dim > cDim) && globalOrder->isLocal(*c_iter)) {
          n += dim - cDim;
        }
      }
    }
    if (debug) {std::cout << "Split " << s << ": size " << n << std::endl;}
    splitSize[s] = n;
    total       += n;
    q += numFields[s];
  }
  PetscInt localSize;
  ierr = VecGetLocalSize(v, &localSize);CHKERRXX(ierr);
  if (debug) {std::cout << "Vector local size " << localSize << std::endl;}
  assert(localSize == total);
  for(PetscInt s = 0, q = 0; s < numSplits; ++s) {
    PetscInt  n = splitSize[s];
    PetscInt  i = -1;
    PetscInt *idx = PETSC_NULL;
    IS        is;

    if (n) {
      ierr = PetscMalloc(n * sizeof(PetscInt), &idx);CHKERRXX(ierr);
      for(typename Section::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        for(PetscInt f = 0, q2 = q; f < numFields[s]; ++f, ++q2) {
          const PetscInt space = fields[q2];
          const PetscInt dim   = section->getFiberDimension(*c_iter, space);
          const PetscInt cDim  = section->getConstraintDimension(*c_iter, space);

          if (dim > cDim) {
            if (!globalOrder->isLocal(*c_iter)) continue;
            PetscInt off = globalOrder->getIndex(*c_iter);

            for(PetscInt sp = 0; sp < space; ++sp) {
              off += section->getConstrainedFiberDimension(*c_iter, sp);
            }
            if (cDim) {
              // This is potentially dangerous
              //   These constraints dofs are for SINGLE FIELDS, not the entire point (confusing)
              const PetscInt *cDofs = section->getConstraintDof(*c_iter, space);

              for(PetscInt d = 0, c = 0, k = 0; d < dim; ++d) {
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
              for(PetscInt d = 0; d < dim; ++d) {
                idx[++i] = off+d;
                if (debug) {std::cout << "Added " << (off+d) << " at pos " << i << " for point " << (*c_iter) << std::endl;}
              }
            }
          }
        }
      }
      if (i != n-1) {throw PETSc::Exception("Invalid fibration numbering");}
    }
    ierr = ISCreateGeneral(section->comm(), n, idx,PETSC_OWN_POINTER, &is);CHKERRXX(ierr);
    if (nullsp && nullsp[s]) {
      ierr = PetscObjectCompose((PetscObject) is, "nearnullspace", (PetscObject) nullsp[s]);CHKERRXX(ierr);
    }
    if (precon && precon[s]) {
      ierr = PetscObjectCompose((PetscObject) is, "pmat", (PetscObject) precon[s]);CHKERRXX(ierr);
    }
    ierr = PCFieldSplitSetIS(fieldSplit, splitName, is);CHKERRXX(ierr);
    ++splitName[0];
    q += numFields[s];
  }
  ierr = PetscFree3(numFields,fields,splitSize);CHKERRXX(ierr);
};

#endif // __PETSCDMMESH_SOLVERS_HH

