#ifndef included_ALE_ZOLTAN_hh
#define included_ALE_ZOLTAN_hh

#include <petsc.h>

#ifdef PETSC_HAVE_ZOLTAN
#include <zoltan.h>

extern "C" {
  // Inputs
  extern int  nvtxs_Zoltan;   // The number of vertices
  extern int  nhedges_Zoltan; // The number of hyperedges
  extern int *eptr_Zoltan;    // The offsets of each hyperedge
  extern int *eind_Zoltan;    // The vertices in each hyperedge, indexed by eptr

  int getNumVertices_Zoltan(void *, int *);

  void getLocalElements_Zoltan(void *, int, int, ZOLTAN_ID_PTR, ZOLTAN_ID_PTR, int, float *, int *);

  void getHgSizes_Zoltan(void *, int *, int *, int *, int *);

  void getHg_Zoltan(void *, int, int, int, int, ZOLTAN_ID_PTR, int *, ZOLTAN_ID_PTR, int *);
}

#endif

#endif
