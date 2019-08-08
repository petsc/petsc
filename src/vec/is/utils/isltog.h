
/*
     This is a terrible way of doing "templates" in C.
*/
#define PETSCMAP1_a(a,b)  a ## b
#define PETSCMAP1_b(a,b)  PETSCMAP1_a(a,b)
#define PETSCMAPNAME(a)   PETSCMAP1_b(a,GTOLNAME)
#define PETSCMAPTYPE(a)   PETSCMAP1_b(a,GTOLTYPE)

static PetscErrorCode PETSCMAPNAME(ISGlobalToLocalMappingApply)(ISLocalToGlobalMapping mapping,ISGlobalToLocalMappingMode type,
                                                      PetscInt n,const PetscInt idx[],PetscInt *nout,PetscInt idxout[])
{
  PetscInt                             i,nf = 0,tmp,start,end,bs;
  PETSCMAPTYPE(ISLocalToGlobalMapping) *map = (PETSCMAPTYPE(ISLocalToGlobalMapping)*)mapping->data;
  PetscErrorCode                       ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (!map) {
    ierr = ISGlobalToLocalMappingSetUp(mapping);CHKERRQ(ierr);
    map  = (PETSCMAPTYPE(ISLocalToGlobalMapping) *)mapping->data;
  }
  start = mapping->globalstart;
  end   = mapping->globalend;
  bs    = GTOLBS;

  if (type == IS_GTOLM_MASK) {
    if (idxout) {
      for (i=0; i<n; i++) {
        if (idx[i] < 0)                 idxout[i] = idx[i];
        else if (idx[i] < bs*start)     idxout[i] = -1;
        else if (idx[i] > bs*(end+1)-1) idxout[i] = -1;
        else                            GTOL(idx[i], idxout[i]);
      }
    }
    if (nout) *nout = n;
  } else {
    if (idxout) {
      for (i=0; i<n; i++) {
        if (idx[i] < 0) continue;
        if (idx[i] < bs*start) continue;
        if (idx[i] > bs*(end+1)-1) continue;
        GTOL(idx[i], tmp);
        if (tmp < 0) continue;
        idxout[nf++] = tmp;
      }
    } else {
      for (i=0; i<n; i++) {
        if (idx[i] < 0) continue;
        if (idx[i] < bs*start) continue;
        if (idx[i] > bs*(end+1)-1) continue;
        GTOL(idx[i], tmp);
        if (tmp < 0) continue;
        nf++;
      }
    }
    if (nout) *nout = nf;
  }
  PetscFunctionReturn(0);
}

#undef PETSCMAP1_a
#undef PETSCMAP1_b
#undef PETSCMAPTYPE
#undef PETSCMAPNAME
#undef GTOLTYPE
#undef GTOLNAME
#undef GTOLBS
#undef GTOL
