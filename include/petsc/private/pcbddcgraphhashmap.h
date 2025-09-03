#pragma once

#include <petsc/private/hashtable.h>
#include <petsc/private/pcbddcstructsimpl.h>

static inline PetscHash_t PCBDDCGraphNodeHash(const PCBDDCGraphNode *node)
{
  PetscHash_t hash;
  hash = PetscHashCombine(PetscHashInt(node->count), PetscHashInt(node->which_dof));
  hash = PetscHashCombine(hash, PetscHashInt(node->special_dof));
  for (PetscInt i = 0; i < node->count; i++) hash = PetscHashCombine(hash, PetscHashInt(node->neighbours_set[i]));
  hash = PetscHashCombine(hash, PetscHashInt(node->local_groups_count));
  if (!node->shared) {
    for (PetscInt i = 0; i < node->local_groups_count; i++) hash = PetscHashCombine(hash, PetscHashInt(node->local_groups[i]));
  }
  return hash;
}

static inline int PCBDDCGraphNodeEqual(const PCBDDCGraphNode *a, const PCBDDCGraphNode *b)
{
  if (a->count != b->count) return 0;
  if (a->which_dof != b->which_dof) return 0;
  if (a->special_dof != b->special_dof) return 0;
  /* check only for same local groups if not shared
     shared dofs at the process boundaries will be handled differently */
  PetscBool mpi_shared = a->shared;
  PetscBool same_set;
  PetscCallContinue(PetscArraycmp(a->neighbours_set, b->neighbours_set, a->count, &same_set));
  if (same_set && !mpi_shared) {
    if (a->local_groups_count != b->local_groups_count) same_set = PETSC_FALSE;
    else PetscCallContinue(PetscArraycmp(a->local_groups, b->local_groups, a->local_groups_count, &same_set));
  }
  return same_set ? 1 : 0;
}

PETSC_HASH_MAP(HMapPCBDDCGraphNode, PCBDDCGraphNode *, PetscInt, PCBDDCGraphNodeHash, PCBDDCGraphNodeEqual, -1)
