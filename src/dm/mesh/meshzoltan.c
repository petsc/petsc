#include <petscmesh.hh>

extern "C" {
#ifdef PETSC_HAVE_ZOLTAN
  // Inputs
  int  nvtxs_Zoltan;   // The number of vertices
  int  nhedges_Zoltan; // The number of hyperedges
  int *eptr_Zoltan;    // The offsets of each hyperedge
  int *eind_Zoltan;    // The vertices in each hyperedge, indexed by eptr

  int getNumVertices_Zoltan(void *data, int *ierr) {
    *ierr = 0;
    return nvtxs_Zoltan;
  };

  void getLocalElements_Zoltan(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts, int *ierr) {
    if ((wgt_dim != 0) || (num_gid_entries != 1) || (num_lid_entries != 1)) {
      *ierr = 1;
      return;
    }
    *ierr = 0;
    for(int v = 0; v < nvtxs_Zoltan; ++v) {
      global_ids[v]= v;
      local_ids[v] = v;
    }
    return;
  };

  void getHgSizes_Zoltan(void *data, int *num_lists, int *num_pins, int *format, int *ierr) {
    *ierr = 0;
    *num_lists = nhedges_Zoltan;
    *num_pins  = eptr_Zoltan[nhedges_Zoltan];
    *format    = ZOLTAN_COMPRESSED_EDGE;
  };

  void getHg_Zoltan(void *data, int num_gid_entries, int num_row_or_col, int num_pins, int format, ZOLTAN_ID_PTR vtxedge_GID, int *vtxedge_ptr, ZOLTAN_ID_PTR pin_GID, int *ierr) {
    if ((num_gid_entries != 1) || (num_row_or_col != nhedges_Zoltan) || (num_pins != eptr_Zoltan[nhedges_Zoltan]) || (format != ZOLTAN_COMPRESSED_EDGE)) {
      *ierr = 1;
      return;
    }
    *ierr = 0;
    for(int e = 0; e < num_row_or_col; ++e) {
      vtxedge_GID[e] = e;
    }
    for(int e = 0; e < num_row_or_col; ++e) {
      vtxedge_ptr[e] = eptr_Zoltan[e];
    }
    for(int p = 0; p < num_pins; ++p) {
      pin_GID[p] = eind_Zoltan[p];
    }
  };
#endif
}
