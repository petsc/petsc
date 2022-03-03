
/***********************************gs.c***************************************

Author: Henry M. Tufo III

e-mail: hmt@cs.brown.edu

snail-mail:
Division of Applied Mathematics
Brown University
Providence, RI 02912

Last Modification:
6.21.97
************************************gs.c**************************************/

/***********************************gs.c***************************************
File Description:
-----------------

************************************gs.c**************************************/

#include <../src/ksp/pc/impls/tfs/tfs.h>

/* default length of number of items via tree - doubles if exceeded */
#define TREE_BUF_SZ 2048;
#define GS_VEC_SZ   1

/***********************************gs.c***************************************
Type: struct gather_scatter_id
------------------------------

************************************gs.c**************************************/
typedef struct gather_scatter_id {
  PetscInt    id;
  PetscInt    nel_min;
  PetscInt    nel_max;
  PetscInt    nel_sum;
  PetscInt    negl;
  PetscInt    gl_max;
  PetscInt    gl_min;
  PetscInt    repeats;
  PetscInt    ordered;
  PetscInt    positive;
  PetscScalar *vals;

  /* bit mask info */
  PetscInt *my_proc_mask;
  PetscInt mask_sz;
  PetscInt *ngh_buf;
  PetscInt ngh_buf_sz;
  PetscInt *nghs;
  PetscInt num_nghs;
  PetscInt max_nghs;
  PetscInt *pw_nghs;
  PetscInt num_pw_nghs;
  PetscInt *tree_nghs;
  PetscInt num_tree_nghs;

  PetscInt num_loads;

  /* repeats == true -> local info */
  PetscInt nel;         /* number of unique elememts */
  PetscInt *elms;       /* of size nel */
  PetscInt nel_total;
  PetscInt *local_elms; /* of size nel_total */
  PetscInt *companion;  /* of size nel_total */

  /* local info */
  PetscInt num_local_total;
  PetscInt local_strength;
  PetscInt num_local;
  PetscInt *num_local_reduce;
  PetscInt **local_reduce;
  PetscInt num_local_gop;
  PetscInt *num_gop_local_reduce;
  PetscInt **gop_local_reduce;

  /* pairwise info */
  PetscInt    level;
  PetscInt    num_pairs;
  PetscInt    max_pairs;
  PetscInt    loc_node_pairs;
  PetscInt    max_node_pairs;
  PetscInt    min_node_pairs;
  PetscInt    avg_node_pairs;
  PetscInt    *pair_list;
  PetscInt    *msg_sizes;
  PetscInt    **node_list;
  PetscInt    len_pw_list;
  PetscInt    *pw_elm_list;
  PetscScalar *pw_vals;

  MPI_Request *msg_ids_in;
  MPI_Request *msg_ids_out;

  PetscScalar *out;
  PetscScalar *in;
  PetscInt    msg_total;

  /* tree - crystal accumulator info */
  PetscInt max_left_over;
  PetscInt *pre;
  PetscInt *in_num;
  PetscInt *out_num;
  PetscInt **in_list;
  PetscInt **out_list;

  /* new tree work*/
  PetscInt    tree_nel;
  PetscInt    *tree_elms;
  PetscScalar *tree_buf;
  PetscScalar *tree_work;

  PetscInt tree_map_sz;
  PetscInt *tree_map_in;
  PetscInt *tree_map_out;

  /* current memory status */
  PetscInt gl_bss_min;
  PetscInt gl_perm_min;

  /* max segment size for PCTFS_gs_gop_vec() */
  PetscInt vec_sz;

  /* hack to make paul happy */
  MPI_Comm PCTFS_gs_comm;

} PCTFS_gs_id;

static PCTFS_gs_id *gsi_check_args(PetscInt *elms, PetscInt nel, PetscInt level);
static PetscErrorCode gsi_via_bit_mask(PCTFS_gs_id *gs);
static PetscErrorCode get_ngh_buf(PCTFS_gs_id *gs);
static PetscErrorCode set_pairwise(PCTFS_gs_id *gs);
static PCTFS_gs_id *gsi_new(void);
static PetscErrorCode set_tree(PCTFS_gs_id *gs);

/* same for all but vector flavor */
static PetscErrorCode PCTFS_gs_gop_local_out(PCTFS_gs_id *gs, PetscScalar *vals);
/* vector flavor */
static PetscErrorCode PCTFS_gs_gop_vec_local_out(PCTFS_gs_id *gs, PetscScalar *vals, PetscInt step);

static PetscErrorCode PCTFS_gs_gop_vec_plus(PCTFS_gs_id *gs, PetscScalar *in_vals, PetscInt step);
static PetscErrorCode PCTFS_gs_gop_vec_pairwise_plus(PCTFS_gs_id *gs, PetscScalar *in_vals, PetscInt step);
static PetscErrorCode PCTFS_gs_gop_vec_local_plus(PCTFS_gs_id *gs, PetscScalar *vals, PetscInt step);
static PetscErrorCode PCTFS_gs_gop_vec_local_in_plus(PCTFS_gs_id *gs, PetscScalar *vals, PetscInt step);
static PetscErrorCode PCTFS_gs_gop_vec_tree_plus(PCTFS_gs_id *gs, PetscScalar *vals, PetscInt step);

static PetscErrorCode PCTFS_gs_gop_local_plus(PCTFS_gs_id *gs, PetscScalar *vals);
static PetscErrorCode PCTFS_gs_gop_local_in_plus(PCTFS_gs_id *gs, PetscScalar *vals);

static PetscErrorCode PCTFS_gs_gop_plus_hc(PCTFS_gs_id *gs, PetscScalar *in_vals, PetscInt dim);
static PetscErrorCode PCTFS_gs_gop_pairwise_plus_hc(PCTFS_gs_id *gs, PetscScalar *in_vals, PetscInt dim);
static PetscErrorCode PCTFS_gs_gop_tree_plus_hc(PCTFS_gs_id *gs, PetscScalar *vals, PetscInt dim);

/* global vars */
/* from comm.c module */

static PetscInt num_gs_ids = 0;

/* should make this dynamic ... later */
static PetscInt msg_buf    =MAX_MSG_BUF;
static PetscInt vec_sz     =GS_VEC_SZ;
static PetscInt *tree_buf  =NULL;
static PetscInt tree_buf_sz=0;
static PetscInt ntree      =0;

/***************************************************************************/
PetscErrorCode PCTFS_gs_init_vec_sz(PetscInt size)
{
  PetscFunctionBegin;
  vec_sz = size;
  PetscFunctionReturn(0);
}

/******************************************************************************/
PetscErrorCode PCTFS_gs_init_msg_buf_sz(PetscInt buf_size)
{
  PetscFunctionBegin;
  msg_buf = buf_size;
  PetscFunctionReturn(0);
}

/******************************************************************************/
PCTFS_gs_id *PCTFS_gs_init(PetscInt *elms, PetscInt nel, PetscInt level)
{
  PCTFS_gs_id    *gs;
  MPI_Group      PCTFS_gs_group;
  MPI_Comm       PCTFS_gs_comm;

  /* ensure that communication package has been initialized */
  PCTFS_comm_init();

  /* determines if we have enough dynamic/semi-static memory */
  /* checks input, allocs and sets gd_id template            */
  gs = gsi_check_args(elms,nel,level);

  /* only bit mask version up and working for the moment    */
  /* LATER :: get int list version working for sparse pblms */
  CHKERRABORT(PETSC_COMM_WORLD,gsi_via_bit_mask(gs));

  CHKERRABORT(PETSC_COMM_WORLD,MPI_Comm_group(MPI_COMM_WORLD,&PCTFS_gs_group));
  CHKERRABORT(PETSC_COMM_WORLD,MPI_Comm_create(MPI_COMM_WORLD,PCTFS_gs_group,&PCTFS_gs_comm));
  CHKERRABORT(PETSC_COMM_WORLD,MPI_Group_free(&PCTFS_gs_group));

  gs->PCTFS_gs_comm=PCTFS_gs_comm;

  return(gs);
}

/******************************************************************************/
static PCTFS_gs_id *gsi_new(void)
{
  PCTFS_gs_id    *gs;
  gs   = (PCTFS_gs_id*) malloc(sizeof(PCTFS_gs_id));
  CHKERRABORT(PETSC_COMM_WORLD,PetscMemzero(gs,sizeof(PCTFS_gs_id)));
  return(gs);
}

/******************************************************************************/
static PCTFS_gs_id *gsi_check_args(PetscInt *in_elms, PetscInt nel, PetscInt level)
{
  PetscInt       i, j, k, t2;
  PetscInt       *companion, *elms, *unique, *iptr;
  PetscInt       num_local=0, *num_to_reduce, **local_reduce;
  PetscInt       oprs[]   = {NON_UNIFORM,GL_MIN,GL_MAX,GL_ADD,GL_MIN,GL_MAX,GL_MIN,GL_B_AND};
  PetscInt       vals[sizeof(oprs)/sizeof(oprs[0])-1];
  PetscInt       work[sizeof(oprs)/sizeof(oprs[0])-1];
  PCTFS_gs_id    *gs;

  if (!in_elms) SETERRABORT(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"elms point to nothing!!!\n");
  if (nel<0)    SETERRABORT(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"can't have fewer than 0 elms!!!\n");

  if (nel==0) CHKERRABORT(PETSC_COMM_WORLD,PetscInfo(0,"I don't have any elements!!!\n"));

  /* get space for gs template */
  gs     = gsi_new();
  gs->id = ++num_gs_ids;

  /* hmt 6.4.99                                            */
  /* caller can set global ids that don't participate to 0 */
  /* PCTFS_gs_init ignores all zeros in elm list                 */
  /* negative global ids are still invalid                 */
  for (i=j=0; i<nel; i++) {
    if (in_elms[i]!=0) j++;
  }

  k=nel; nel=j;

  /* copy over in_elms list and create inverse map */
  elms      = (PetscInt*) malloc((nel+1)*sizeof(PetscInt));
  companion = (PetscInt*) malloc(nel*sizeof(PetscInt));

  for (i=j=0; i<k; i++) {
    if (in_elms[i]!=0) { elms[j] = in_elms[i]; companion[j++] = i; }
  }

  if (j!=nel) SETERRABORT(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"nel j mismatch!\n");

  /* pre-pass ... check to see if sorted */
  elms[nel] = INT_MAX;
  iptr      = elms;
  unique    = elms+1;
  j         =0;
  while (*iptr!=INT_MAX) {
    if (*iptr++>*unique++) { j=1; break; }
  }

  /* set up inverse map */
  if (j) {
    CHKERRABORT(PETSC_COMM_WORLD,PetscInfo(0,"gsi_check_args() :: elm list *not* sorted!\n"));
    CHKERRABORT(PETSC_COMM_WORLD,PCTFS_SMI_sort((void*)elms, (void*)companion, nel, SORT_INTEGER));
  } else CHKERRABORT(PETSC_COMM_WORLD,PetscInfo(0,"gsi_check_args() :: elm list sorted!\n"));
  elms[nel] = INT_MIN;

  /* first pass */
  /* determine number of unique elements, check pd */
  for (i=k=0; i<nel; i+=j) {
    t2 = elms[i];
    j  = ++i;

    /* clump 'em for now */
    while (elms[j]==t2) j++;

    /* how many together and num local */
    if (j-=i) { num_local++; k+=j; }
  }

  /* how many unique elements? */
  gs->repeats = k;
  gs->nel     = nel-k;

  /* number of repeats? */
  gs->num_local        = num_local;
  num_local           += 2;
  gs->local_reduce     = local_reduce=(PetscInt**)malloc(num_local*sizeof(PetscInt*));
  gs->num_local_reduce = num_to_reduce=(PetscInt*) malloc(num_local*sizeof(PetscInt));

  unique         = (PetscInt*) malloc((gs->nel+1)*sizeof(PetscInt));
  gs->elms       = unique;
  gs->nel_total  = nel;
  gs->local_elms = elms;
  gs->companion  = companion;

  /* compess map as well as keep track of local ops */
  for (num_local=i=j=0; i<gs->nel; i++) {
    k            = j;
    t2           = unique[i] = elms[j];
    companion[i] = companion[j];

    while (elms[j]==t2) j++;

    if ((t2=(j-k))>1) {
      /* number together */
      num_to_reduce[num_local] = t2++;

      iptr = local_reduce[num_local++] = (PetscInt*)malloc(t2*sizeof(PetscInt));

      /* to use binary searching don't remap until we check intersection */
      *iptr++ = i;

      /* note that we're skipping the first one */
      while (++k<j) *(iptr++) = companion[k];
      *iptr = -1;
    }
  }

  /* sentinel for ngh_buf */
  unique[gs->nel]=INT_MAX;

  /* for two partition sort hack */
  num_to_reduce[num_local]   = 0;
  local_reduce[num_local]    = NULL;
  num_to_reduce[++num_local] = 0;
  local_reduce[num_local]    = NULL;

  /* load 'em up */
  /* note one extra to hold NON_UNIFORM flag!!! */
  vals[2] = vals[1] = vals[0] = nel;
  if (gs->nel>0) {
    vals[3] = unique[0];
    vals[4] = unique[gs->nel-1];
  } else {
    vals[3] = INT_MAX;
    vals[4] = INT_MIN;
  }
  vals[5] = level;
  vals[6] = num_gs_ids;

  /* GLOBAL: send 'em out */
  CHKERRABORT(PETSC_COMM_WORLD,PCTFS_giop(vals,work,sizeof(oprs)/sizeof(oprs[0])-1,oprs));

  /* must be semi-pos def - only pairwise depends on this */
  /* LATER - remove this restriction */
  if (vals[3]<0) SETERRABORT(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"gsi_check_args() :: system not semi-pos def \n");
  if (vals[4]==INT_MAX) SETERRABORT(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"gsi_check_args() :: system ub too large !\n");

  gs->nel_min = vals[0];
  gs->nel_max = vals[1];
  gs->nel_sum = vals[2];
  gs->gl_min  = vals[3];
  gs->gl_max  = vals[4];
  gs->negl    = vals[4]-vals[3]+1;

  if (gs->negl<=0) SETERRABORT(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"gsi_check_args() :: system empty or neg :: %d\n");

  /* LATER :: add level == -1 -> program selects level */
  if (vals[5]<0) vals[5]=0;
  else if (vals[5]>PCTFS_num_nodes) vals[5]=PCTFS_num_nodes;
  gs->level = vals[5];

  return(gs);
}

/******************************************************************************/
static PetscErrorCode gsi_via_bit_mask(PCTFS_gs_id *gs)
{
  PetscInt       i, nel, *elms;
  PetscInt       t1;
  PetscInt       **reduce;
  PetscInt       *map;

  PetscFunctionBegin;
  /* totally local removes ... PCTFS_ct_bits == 0 */
  get_ngh_buf(gs);

  if (gs->level) set_pairwise(gs);
  if (gs->max_left_over) set_tree(gs);

  /* intersection local and pairwise/tree? */
  gs->num_local_total      = gs->num_local;
  gs->gop_local_reduce     = gs->local_reduce;
  gs->num_gop_local_reduce = gs->num_local_reduce;

  map = gs->companion;

  /* is there any local compression */
  if (!gs->num_local) {
    gs->local_strength = NONE;
    gs->num_local_gop  = 0;
  } else {
    /* ok find intersection */
    map    = gs->companion;
    reduce = gs->local_reduce;
    for (i=0, t1=0; i<gs->num_local; i++, reduce++) {
      if ((PCTFS_ivec_binary_search(**reduce,gs->pw_elm_list,gs->len_pw_list)>=0) || PCTFS_ivec_binary_search(**reduce,gs->tree_map_in,gs->tree_map_sz)>=0) {
        t1++;
        PetscCheckFalse(gs->num_local_reduce[i]<=0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"nobody in list?");
        gs->num_local_reduce[i] *= -1;
      }
      **reduce=map[**reduce];
    }

    /* intersection is empty */
    if (!t1) {
      gs->local_strength = FULL;
      gs->num_local_gop  = 0;
    } else { /* intersection not empty */
      gs->local_strength = PARTIAL;

      CHKERRQ(PCTFS_SMI_sort((void*)gs->num_local_reduce, (void*)gs->local_reduce, gs->num_local + 1, SORT_INT_PTR));

      gs->num_local_gop        = t1;
      gs->num_local_total      =  gs->num_local;
      gs->num_local           -= t1;
      gs->gop_local_reduce     = gs->local_reduce;
      gs->num_gop_local_reduce = gs->num_local_reduce;

      for (i=0; i<t1; i++) {
        PetscCheckFalse(gs->num_gop_local_reduce[i]>=0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"they aren't negative?");
        gs->num_gop_local_reduce[i] *= -1;
        gs->local_reduce++;
        gs->num_local_reduce++;
      }
      gs->local_reduce++;
      gs->num_local_reduce++;
    }
  }

  elms = gs->pw_elm_list;
  nel  = gs->len_pw_list;
  for (i=0; i<nel; i++) elms[i] = map[elms[i]];

  elms = gs->tree_map_in;
  nel  = gs->tree_map_sz;
  for (i=0; i<nel; i++) elms[i] = map[elms[i]];

  /* clean up */
  free((void*) gs->local_elms);
  free((void*) gs->companion);
  free((void*) gs->elms);
  free((void*) gs->ngh_buf);
  gs->local_elms = gs->companion = gs->elms = gs->ngh_buf = NULL;
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode place_in_tree(PetscInt elm)
{
  PetscInt *tp, n;

  PetscFunctionBegin;
  if (ntree==tree_buf_sz) {
    if (tree_buf_sz) {
      tp           = tree_buf;
      n            = tree_buf_sz;
      tree_buf_sz<<=1;
      tree_buf     = (PetscInt*)malloc(tree_buf_sz*sizeof(PetscInt));
      PCTFS_ivec_copy(tree_buf,tp,n);
      free(tp);
    } else {
      tree_buf_sz = TREE_BUF_SZ;
      tree_buf    = (PetscInt*)malloc(tree_buf_sz*sizeof(PetscInt));
    }
  }

  tree_buf[ntree++] = elm;
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode get_ngh_buf(PCTFS_gs_id *gs)
{
  PetscInt       i, j, npw=0, ntree_map=0;
  PetscInt       p_mask_size, ngh_buf_size, buf_size;
  PetscInt       *p_mask, *sh_proc_mask, *pw_sh_proc_mask;
  PetscInt       *ngh_buf, *buf1, *buf2;
  PetscInt       offset, per_load, num_loads, or_ct, start, end;
  PetscInt       *ptr1, *ptr2, i_start, negl, nel, *elms;
  PetscInt       oper=GL_B_OR;
  PetscInt       *ptr3, *t_mask, level, ct1, ct2;

  PetscFunctionBegin;
  /* to make life easier */
  nel   = gs->nel;
  elms  = gs->elms;
  level = gs->level;

  /* det #bytes needed for processor bit masks and init w/mask cor. to PCTFS_my_id */
  p_mask = (PetscInt*) malloc(p_mask_size=PCTFS_len_bit_mask(PCTFS_num_nodes));
  CHKERRQ(PCTFS_set_bit_mask(p_mask,p_mask_size,PCTFS_my_id));

  /* allocate space for masks and info bufs */
  gs->nghs       = sh_proc_mask = (PetscInt*) malloc(p_mask_size);
  gs->pw_nghs    = pw_sh_proc_mask = (PetscInt*) malloc(p_mask_size);
  gs->ngh_buf_sz = ngh_buf_size = p_mask_size*nel;
  t_mask         = (PetscInt*) malloc(p_mask_size);
  gs->ngh_buf    = ngh_buf = (PetscInt*) malloc(ngh_buf_size);

  /* comm buffer size ... memory usage bounded by ~2*msg_buf */
  /* had thought I could exploit rendezvous threshold */

  /* default is one pass */
  per_load      = negl  = gs->negl;
  gs->num_loads = num_loads = 1;
  i             = p_mask_size*negl;

  /* possible overflow on buffer size */
  /* overflow hack                    */
  if (i<0) i=INT_MAX;

  buf_size = PetscMin(msg_buf,i);

  /* can we do it? */
  PetscCheckFalse(p_mask_size>buf_size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"get_ngh_buf() :: buf<pms :: %d>%d",p_mask_size,buf_size);

  /* get PCTFS_giop buf space ... make *only* one malloc */
  buf1 = (PetscInt*) malloc(buf_size<<1);

  /* more than one gior exchange needed? */
  if (buf_size!=i) {
    per_load      = buf_size/p_mask_size;
    buf_size      = per_load*p_mask_size;
    gs->num_loads = num_loads = negl/per_load + (negl%per_load>0);
  }

  /* convert buf sizes from #bytes to #ints - 32 bit only! */
  p_mask_size/=sizeof(PetscInt); ngh_buf_size/=sizeof(PetscInt); buf_size/=sizeof(PetscInt);

  /* find PCTFS_giop work space */
  buf2 = buf1+buf_size;

  /* hold #ints needed for processor masks */
  gs->mask_sz=p_mask_size;

  /* init buffers */
  CHKERRQ(PCTFS_ivec_zero(sh_proc_mask,p_mask_size));
  CHKERRQ(PCTFS_ivec_zero(pw_sh_proc_mask,p_mask_size));
  CHKERRQ(PCTFS_ivec_zero(ngh_buf,ngh_buf_size));

  /* HACK reset tree info */
  tree_buf    = NULL;
  tree_buf_sz = ntree = 0;

  /* ok do it */
  for (ptr1=ngh_buf,ptr2=elms,end=gs->gl_min,or_ct=i=0; or_ct<num_loads; or_ct++) {
    /* identity for bitwise or is 000...000 */
    PCTFS_ivec_zero(buf1,buf_size);

    /* load msg buffer */
    for (start=end,end+=per_load,i_start=i; (offset=*ptr2)<end; i++, ptr2++) {
      offset = (offset-start)*p_mask_size;
      PCTFS_ivec_copy(buf1+offset,p_mask,p_mask_size);
    }

    /* GLOBAL: pass buffer */
    CHKERRQ(PCTFS_giop(buf1,buf2,buf_size,&oper));

    /* unload buffer into ngh_buf */
    ptr2=(elms+i_start);
    for (ptr3=buf1,j=start; j<end; ptr3+=p_mask_size,j++) {
      /* I own it ... may have to pairwise it */
      if (j==*ptr2) {
        /* do i share it w/anyone? */
        ct1 = PCTFS_ct_bits((char*)ptr3,p_mask_size*sizeof(PetscInt));
        /* guess not */
        if (ct1<2) { ptr2++; ptr1+=p_mask_size; continue; }

        /* i do ... so keep info and turn off my bit */
        PCTFS_ivec_copy(ptr1,ptr3,p_mask_size);
        CHKERRQ(PCTFS_ivec_xor(ptr1,p_mask,p_mask_size));
        CHKERRQ(PCTFS_ivec_or(sh_proc_mask,ptr1,p_mask_size));

        /* is it to be done pairwise? */
        if (--ct1<=level) {
          npw++;

          /* turn on high bit to indicate pw need to process */
          *ptr2++ |= TOP_BIT;
          CHKERRQ(PCTFS_ivec_or(pw_sh_proc_mask,ptr1,p_mask_size));
          ptr1    += p_mask_size;
          continue;
        }

        /* get set for next and note that I have a tree contribution */
        /* could save exact elm index for tree here -> save a search */
        ptr2++; ptr1+=p_mask_size; ntree_map++;
      } else { /* i don't but still might be involved in tree */

        /* shared by how many? */
        ct1 = PCTFS_ct_bits((char*)ptr3,p_mask_size*sizeof(PetscInt));

        /* none! */
        if (ct1<2) continue;

        /* is it going to be done pairwise? but not by me of course!*/
        if (--ct1<=level) continue;
      }
      /* LATER we're going to have to process it NOW */
      /* nope ... tree it */
      CHKERRQ(place_in_tree(j));
    }
  }

  free((void*)t_mask);
  free((void*)buf1);

  gs->len_pw_list = npw;
  gs->num_nghs    = PCTFS_ct_bits((char*)sh_proc_mask,p_mask_size*sizeof(PetscInt));

  /* expand from bit mask list to int list and save ngh list */
  gs->nghs = (PetscInt*) malloc(gs->num_nghs * sizeof(PetscInt));
  PCTFS_bm_to_proc((char*)sh_proc_mask,p_mask_size*sizeof(PetscInt),gs->nghs);

  gs->num_pw_nghs = PCTFS_ct_bits((char*)pw_sh_proc_mask,p_mask_size*sizeof(PetscInt));

  oper         = GL_MAX;
  ct1          = gs->num_nghs;
  CHKERRQ(PCTFS_giop(&ct1,&ct2,1,&oper));
  gs->max_nghs = ct1;

  gs->tree_map_sz  = ntree_map;
  gs->max_left_over=ntree;

  free((void*)p_mask);
  free((void*)sh_proc_mask);
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode set_pairwise(PCTFS_gs_id *gs)
{
  PetscInt       i, j;
  PetscInt       p_mask_size;
  PetscInt       *p_mask, *sh_proc_mask, *tmp_proc_mask;
  PetscInt       *ngh_buf, *buf2;
  PetscInt       offset;
  PetscInt       *msg_list, *msg_size, **msg_nodes, nprs;
  PetscInt       *pairwise_elm_list, len_pair_list=0;
  PetscInt       *iptr, t1, i_start, nel, *elms;
  PetscInt       ct;

  PetscFunctionBegin;
  /* to make life easier */
  nel          = gs->nel;
  elms         = gs->elms;
  ngh_buf      = gs->ngh_buf;
  sh_proc_mask = gs->pw_nghs;

  /* need a few temp masks */
  p_mask_size   = PCTFS_len_bit_mask(PCTFS_num_nodes);
  p_mask        = (PetscInt*) malloc(p_mask_size);
  tmp_proc_mask = (PetscInt*) malloc(p_mask_size);

  /* set mask to my PCTFS_my_id's bit mask */
  CHKERRQ(PCTFS_set_bit_mask(p_mask,p_mask_size,PCTFS_my_id));

  p_mask_size /= sizeof(PetscInt);

  len_pair_list   = gs->len_pw_list;
  gs->pw_elm_list = pairwise_elm_list=(PetscInt*)malloc((len_pair_list+1)*sizeof(PetscInt));

  /* how many processors (nghs) do we have to exchange with? */
  nprs = gs->num_pairs = PCTFS_ct_bits((char*)sh_proc_mask,p_mask_size*sizeof(PetscInt));

  /* allocate space for PCTFS_gs_gop() info */
  gs->pair_list = msg_list  = (PetscInt*)  malloc(sizeof(PetscInt)*nprs);
  gs->msg_sizes = msg_size  = (PetscInt*)  malloc(sizeof(PetscInt)*nprs);
  gs->node_list = msg_nodes = (PetscInt**) malloc(sizeof(PetscInt*)*(nprs+1));

  /* init msg_size list */
  CHKERRQ(PCTFS_ivec_zero(msg_size,nprs));

  /* expand from bit mask list to int list */
  CHKERRQ(PCTFS_bm_to_proc((char*)sh_proc_mask,p_mask_size*sizeof(PetscInt),msg_list));

  /* keep list of elements being handled pairwise */
  for (i=j=0; i<nel; i++) {
    if (elms[i] & TOP_BIT) { elms[i] ^= TOP_BIT; pairwise_elm_list[j++] = i; }
  }
  pairwise_elm_list[j] = -1;

  gs->msg_ids_out       = (MPI_Request*)  malloc(sizeof(MPI_Request)*(nprs+1));
  gs->msg_ids_out[nprs] = MPI_REQUEST_NULL;
  gs->msg_ids_in        = (MPI_Request*)  malloc(sizeof(MPI_Request)*(nprs+1));
  gs->msg_ids_in[nprs]  = MPI_REQUEST_NULL;
  gs->pw_vals           = (PetscScalar*) malloc(sizeof(PetscScalar)*len_pair_list*vec_sz);

  /* find who goes to each processor */
  for (i_start=i=0; i<nprs; i++) {
    /* processor i's mask */
    CHKERRQ(PCTFS_set_bit_mask(p_mask,p_mask_size*sizeof(PetscInt),msg_list[i]));

    /* det # going to processor i */
    for (ct=j=0; j<len_pair_list; j++) {
      buf2 = ngh_buf+(pairwise_elm_list[j]*p_mask_size);
      CHKERRQ(PCTFS_ivec_and3(tmp_proc_mask,p_mask,buf2,p_mask_size));
      if (PCTFS_ct_bits((char*)tmp_proc_mask,p_mask_size*sizeof(PetscInt))) ct++;
    }
    msg_size[i] = ct;
    i_start     = PetscMax(i_start,ct);

    /*space to hold nodes in message to first neighbor */
    msg_nodes[i] = iptr = (PetscInt*) malloc(sizeof(PetscInt)*(ct+1));

    for (j=0;j<len_pair_list;j++) {
      buf2 = ngh_buf+(pairwise_elm_list[j]*p_mask_size);
      CHKERRQ(PCTFS_ivec_and3(tmp_proc_mask,p_mask,buf2,p_mask_size));
      if (PCTFS_ct_bits((char*)tmp_proc_mask,p_mask_size*sizeof(PetscInt))) *iptr++ = j;
    }
    *iptr = -1;
  }
  msg_nodes[nprs] = NULL;

  j                  = gs->loc_node_pairs=i_start;
  t1                 = GL_MAX;
  CHKERRQ(PCTFS_giop(&i_start,&offset,1,&t1));
  gs->max_node_pairs = i_start;

  i_start            = j;
  t1                 = GL_MIN;
  CHKERRQ(PCTFS_giop(&i_start,&offset,1,&t1));
  gs->min_node_pairs = i_start;

  i_start            = j;
  t1                 = GL_ADD;
  CHKERRQ(PCTFS_giop(&i_start,&offset,1,&t1));
  gs->avg_node_pairs = i_start/PCTFS_num_nodes + 1;

  i_start = nprs;
  t1      = GL_MAX;
  PCTFS_giop(&i_start,&offset,1,&t1);
  gs->max_pairs = i_start;

  /* remap pairwise in tail of gsi_via_bit_mask() */
  gs->msg_total = PCTFS_ivec_sum(gs->msg_sizes,nprs);
  gs->out       = (PetscScalar*) malloc(sizeof(PetscScalar)*gs->msg_total*vec_sz);
  gs->in        = (PetscScalar*) malloc(sizeof(PetscScalar)*gs->msg_total*vec_sz);

  /* reset malloc pool */
  free((void*)p_mask);
  free((void*)tmp_proc_mask);
  PetscFunctionReturn(0);
}

/* to do pruned tree just save ngh buf copy for each one and decode here!
******************************************************************************/
static PetscErrorCode set_tree(PCTFS_gs_id *gs)
{
  PetscInt i, j, n, nel;
  PetscInt *iptr_in, *iptr_out, *tree_elms, *elms;

  PetscFunctionBegin;
  /* local work ptrs */
  elms = gs->elms;
  nel  = gs->nel;

  /* how many via tree */
  gs->tree_nel     = n = ntree;
  gs->tree_elms    = tree_elms = iptr_in = tree_buf;
  gs->tree_buf     = (PetscScalar*) malloc(sizeof(PetscScalar)*n*vec_sz);
  gs->tree_work    = (PetscScalar*) malloc(sizeof(PetscScalar)*n*vec_sz);
  j                = gs->tree_map_sz;
  gs->tree_map_in  = iptr_in  = (PetscInt*) malloc(sizeof(PetscInt)*(j+1));
  gs->tree_map_out = iptr_out = (PetscInt*) malloc(sizeof(PetscInt)*(j+1));

  /* search the longer of the two lists */
  /* note ... could save this info in get_ngh_buf and save searches */
  if (n<=nel) {
    /* bijective fct w/remap - search elm list */
    for (i=0; i<n; i++) {
      if ((j=PCTFS_ivec_binary_search(*tree_elms++,elms,nel))>=0) {*iptr_in++ = j; *iptr_out++ = i;}
    }
  } else {
    for (i=0; i<nel; i++) {
      if ((j=PCTFS_ivec_binary_search(*elms++,tree_elms,n))>=0) {*iptr_in++ = i; *iptr_out++ = j;}
    }
  }

  /* sentinel */
  *iptr_in = *iptr_out = -1;
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode PCTFS_gs_gop_local_out(PCTFS_gs_id *gs,  PetscScalar *vals)
{
  PetscInt    *num, *map, **reduce;
  PetscScalar tmp;

  PetscFunctionBegin;
  num    = gs->num_gop_local_reduce;
  reduce = gs->gop_local_reduce;
  while ((map = *reduce++)) {
    /* wall */
    if (*num == 2) {
      num++;
      vals[map[1]] = vals[map[0]];
    } else if (*num == 3) { /* corner shared by three elements */
      num++;
      vals[map[2]] = vals[map[1]] = vals[map[0]];
    } else if (*num == 4) { /* corner shared by four elements */
      num++;
      vals[map[3]] = vals[map[2]] = vals[map[1]] = vals[map[0]];
    } else { /* general case ... odd geoms ... 3D*/
      num++;
      tmp = *(vals + *map++);
      while (*map >= 0) *(vals + *map++) = tmp;
    }
  }
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode PCTFS_gs_gop_local_plus(PCTFS_gs_id *gs,  PetscScalar *vals)
{
  PetscInt    *num, *map, **reduce;
  PetscScalar tmp;

  PetscFunctionBegin;
  num    = gs->num_local_reduce;
  reduce = gs->local_reduce;
  while ((map = *reduce)) {
    /* wall */
    if (*num == 2) {
      num++; reduce++;
      vals[map[1]] = vals[map[0]] += vals[map[1]];
    } else if (*num == 3) { /* corner shared by three elements */
      num++; reduce++;
      vals[map[2]]=vals[map[1]]=vals[map[0]]+=(vals[map[1]]+vals[map[2]]);
    } else if (*num == 4) { /* corner shared by four elements */
      num++; reduce++;
      vals[map[1]]=vals[map[2]]=vals[map[3]]=vals[map[0]] += (vals[map[1]] + vals[map[2]] + vals[map[3]]);
    } else { /* general case ... odd geoms ... 3D*/
      num++;
      tmp = 0.0;
      while (*map >= 0) tmp += *(vals + *map++);

      map = *reduce++;
      while (*map >= 0) *(vals + *map++) = tmp;
    }
  }
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode PCTFS_gs_gop_local_in_plus(PCTFS_gs_id *gs,  PetscScalar *vals)
{
  PetscInt    *num, *map, **reduce;
  PetscScalar *base;

  PetscFunctionBegin;
  num    = gs->num_gop_local_reduce;
  reduce = gs->gop_local_reduce;
  while ((map = *reduce++)) {
    /* wall */
    if (*num == 2) {
      num++;
      vals[map[0]] += vals[map[1]];
    } else if (*num == 3) { /* corner shared by three elements */
      num++;
      vals[map[0]] += (vals[map[1]] + vals[map[2]]);
    } else if (*num == 4) { /* corner shared by four elements */
      num++;
      vals[map[0]] += (vals[map[1]] + vals[map[2]] + vals[map[3]]);
    } else { /* general case ... odd geoms ... 3D*/
      num++;
      base = vals + *map++;
      while (*map >= 0) *base += *(vals + *map++);
    }
  }
  PetscFunctionReturn(0);
}

/******************************************************************************/
PetscErrorCode PCTFS_gs_free(PCTFS_gs_id *gs)
{
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_free(&gs->PCTFS_gs_comm));
  if (gs->nghs) free((void*) gs->nghs);
  if (gs->pw_nghs) free((void*) gs->pw_nghs);

  /* tree */
  if (gs->max_left_over) {
    if (gs->tree_elms) free((void*) gs->tree_elms);
    if (gs->tree_buf) free((void*) gs->tree_buf);
    if (gs->tree_work) free((void*) gs->tree_work);
    if (gs->tree_map_in) free((void*) gs->tree_map_in);
    if (gs->tree_map_out) free((void*) gs->tree_map_out);
  }

  /* pairwise info */
  if (gs->num_pairs) {
    /* should be NULL already */
    if (gs->ngh_buf) free((void*) gs->ngh_buf);
    if (gs->elms) free((void*) gs->elms);
    if (gs->local_elms) free((void*) gs->local_elms);
    if (gs->companion) free((void*) gs->companion);

    /* only set if pairwise */
    if (gs->vals) free((void*) gs->vals);
    if (gs->in) free((void*) gs->in);
    if (gs->out) free((void*) gs->out);
    if (gs->msg_ids_in) free((void*) gs->msg_ids_in);
    if (gs->msg_ids_out) free((void*) gs->msg_ids_out);
    if (gs->pw_vals) free((void*) gs->pw_vals);
    if (gs->pw_elm_list) free((void*) gs->pw_elm_list);
    if (gs->node_list) {
      for (i=0;i<gs->num_pairs;i++) {
        if (gs->node_list[i])  {
          free((void*) gs->node_list[i]);
        }
      }
      free((void*) gs->node_list);
    }
    if (gs->msg_sizes) free((void*) gs->msg_sizes);
    if (gs->pair_list) free((void*) gs->pair_list);
  }

  /* local info */
  if (gs->num_local_total>=0) {
    for (i=0;i<gs->num_local_total+1;i++) {
      if (gs->num_gop_local_reduce[i]) free((void*) gs->gop_local_reduce[i]);
    }
  }

  /* if intersection tree/pairwise and local isn't empty */
  if (gs->gop_local_reduce) free((void*) gs->gop_local_reduce);
  if (gs->num_gop_local_reduce) free((void*) gs->num_gop_local_reduce);

  free((void*) gs);
  PetscFunctionReturn(0);
}

/******************************************************************************/
PetscErrorCode PCTFS_gs_gop_vec(PCTFS_gs_id *gs,  PetscScalar *vals,  const char *op,  PetscInt step)
{
  PetscFunctionBegin;
  switch (*op) {
  case '+':
    PCTFS_gs_gop_vec_plus(gs,vals,step);
    break;
  default:
    CHKERRQ(PetscInfo(0,"PCTFS_gs_gop_vec() :: %c is not a valid op\n",op[0]));
    CHKERRQ(PetscInfo(0,"PCTFS_gs_gop_vec() :: default :: plus\n"));
    PCTFS_gs_gop_vec_plus(gs,vals,step);
    break;
  }
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode PCTFS_gs_gop_vec_plus(PCTFS_gs_id *gs,  PetscScalar *vals,  PetscInt step)
{
  PetscFunctionBegin;
  PetscCheck(gs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"PCTFS_gs_gop_vec() passed NULL gs handle!!!");

  /* local only operations!!! */
  if (gs->num_local) PCTFS_gs_gop_vec_local_plus(gs,vals,step);

  /* if intersection tree/pairwise and local isn't empty */
  if (gs->num_local_gop) {
    PCTFS_gs_gop_vec_local_in_plus(gs,vals,step);

    /* pairwise */
    if (gs->num_pairs) PCTFS_gs_gop_vec_pairwise_plus(gs,vals,step);

    /* tree */
    else if (gs->max_left_over) PCTFS_gs_gop_vec_tree_plus(gs,vals,step);

    PCTFS_gs_gop_vec_local_out(gs,vals,step);
  } else { /* if intersection tree/pairwise and local is empty */
    /* pairwise */
    if (gs->num_pairs) PCTFS_gs_gop_vec_pairwise_plus(gs,vals,step);

    /* tree */
    else if (gs->max_left_over) PCTFS_gs_gop_vec_tree_plus(gs,vals,step);
  }
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode PCTFS_gs_gop_vec_local_plus(PCTFS_gs_id *gs,  PetscScalar *vals, PetscInt step)
{
  PetscInt    *num, *map, **reduce;
  PetscScalar *base;

  PetscFunctionBegin;
  num    = gs->num_local_reduce;
  reduce = gs->local_reduce;
  while ((map = *reduce)) {
    base = vals + map[0] * step;

    /* wall */
    if (*num == 2) {
      num++; reduce++;
      PCTFS_rvec_add (base,vals+map[1]*step,step);
      PCTFS_rvec_copy(vals+map[1]*step,base,step);
    } else if (*num == 3) { /* corner shared by three elements */
      num++; reduce++;
      PCTFS_rvec_add (base,vals+map[1]*step,step);
      PCTFS_rvec_add (base,vals+map[2]*step,step);
      PCTFS_rvec_copy(vals+map[2]*step,base,step);
      PCTFS_rvec_copy(vals+map[1]*step,base,step);
    } else if (*num == 4) { /* corner shared by four elements */
      num++; reduce++;
      PCTFS_rvec_add (base,vals+map[1]*step,step);
      PCTFS_rvec_add (base,vals+map[2]*step,step);
      PCTFS_rvec_add (base,vals+map[3]*step,step);
      PCTFS_rvec_copy(vals+map[3]*step,base,step);
      PCTFS_rvec_copy(vals+map[2]*step,base,step);
      PCTFS_rvec_copy(vals+map[1]*step,base,step);
    } else { /* general case ... odd geoms ... 3D */
      num++;
      while (*++map >= 0) PCTFS_rvec_add (base,vals+*map*step,step);

      map = *reduce;
      while (*++map >= 0) PCTFS_rvec_copy(vals+*map*step,base,step);

      reduce++;
    }
  }
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode PCTFS_gs_gop_vec_local_in_plus(PCTFS_gs_id *gs,  PetscScalar *vals, PetscInt step)
{
  PetscInt    *num, *map, **reduce;
  PetscScalar *base;

  PetscFunctionBegin;
  num    = gs->num_gop_local_reduce;
  reduce = gs->gop_local_reduce;
  while ((map = *reduce++)) {
    base = vals + map[0] * step;

    /* wall */
    if (*num == 2) {
      num++;
      PCTFS_rvec_add(base,vals+map[1]*step,step);
    } else if (*num == 3) { /* corner shared by three elements */
      num++;
      PCTFS_rvec_add(base,vals+map[1]*step,step);
      PCTFS_rvec_add(base,vals+map[2]*step,step);
    } else if (*num == 4) { /* corner shared by four elements */
      num++;
      PCTFS_rvec_add(base,vals+map[1]*step,step);
      PCTFS_rvec_add(base,vals+map[2]*step,step);
      PCTFS_rvec_add(base,vals+map[3]*step,step);
    } else { /* general case ... odd geoms ... 3D*/
      num++;
      while (*++map >= 0) PCTFS_rvec_add(base,vals+*map*step,step);
    }
  }
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode PCTFS_gs_gop_vec_local_out(PCTFS_gs_id *gs,  PetscScalar *vals, PetscInt step)
{
  PetscInt    *num, *map, **reduce;
  PetscScalar *base;

  PetscFunctionBegin;
  num    = gs->num_gop_local_reduce;
  reduce = gs->gop_local_reduce;
  while ((map = *reduce++)) {
    base = vals + map[0] * step;

    /* wall */
    if (*num == 2) {
      num++;
      PCTFS_rvec_copy(vals+map[1]*step,base,step);
    } else if (*num == 3) { /* corner shared by three elements */
      num++;
      PCTFS_rvec_copy(vals+map[1]*step,base,step);
      PCTFS_rvec_copy(vals+map[2]*step,base,step);
    } else if (*num == 4) { /* corner shared by four elements */
      num++;
      PCTFS_rvec_copy(vals+map[1]*step,base,step);
      PCTFS_rvec_copy(vals+map[2]*step,base,step);
      PCTFS_rvec_copy(vals+map[3]*step,base,step);
    } else { /* general case ... odd geoms ... 3D*/
      num++;
      while (*++map >= 0) PCTFS_rvec_copy(vals+*map*step,base,step);
    }
  }
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode PCTFS_gs_gop_vec_pairwise_plus(PCTFS_gs_id *gs,  PetscScalar *in_vals, PetscInt step)
{
  PetscScalar    *dptr1, *dptr2, *dptr3, *in1, *in2;
  PetscInt       *iptr, *msg_list, *msg_size, **msg_nodes;
  PetscInt       *pw, *list, *size, **nodes;
  MPI_Request    *msg_ids_in, *msg_ids_out, *ids_in, *ids_out;
  MPI_Status     status;
  PetscBLASInt   i1 = 1,dstep;

  PetscFunctionBegin;
  /* strip and load s */
  msg_list    = list     = gs->pair_list;
  msg_size    = size     = gs->msg_sizes;
  msg_nodes   = nodes    = gs->node_list;
  iptr        = pw       = gs->pw_elm_list;
  dptr1       = dptr3    = gs->pw_vals;
  msg_ids_in  = ids_in   = gs->msg_ids_in;
  msg_ids_out = ids_out  = gs->msg_ids_out;
  dptr2                  = gs->out;
  in1=in2                = gs->in;

  /* post the receives */
  /*  msg_nodes=nodes; */
  do {
    /* Should MPI_ANY_SOURCE be replaced by *list ? In that case do the
        second one *list and do list++ afterwards */
    CHKERRMPI(MPI_Irecv(in1, *size *step, MPIU_SCALAR, MPI_ANY_SOURCE, MSGTAG1 + *list, gs->PCTFS_gs_comm, msg_ids_in));
    list++;msg_ids_in++;
    in1 += *size++ *step;
  } while (*++msg_nodes);
  msg_nodes=nodes;

  /* load gs values into in out gs buffers */
  while (*iptr >= 0) {
    PCTFS_rvec_copy(dptr3,in_vals + *iptr*step,step);
    dptr3+=step;
    iptr++;
  }

  /* load out buffers and post the sends */
  while ((iptr = *msg_nodes++)) {
    dptr3 = dptr2;
    while (*iptr >= 0) {
      PCTFS_rvec_copy(dptr2,dptr1 + *iptr*step,step);
      dptr2+=step;
      iptr++;
    }
    CHKERRMPI(MPI_Isend(dptr3, *msg_size *step, MPIU_SCALAR, *msg_list, MSGTAG1+PCTFS_my_id, gs->PCTFS_gs_comm, msg_ids_out));
    msg_size++; msg_list++;msg_ids_out++;
  }

  /* tree */
  if (gs->max_left_over) PCTFS_gs_gop_vec_tree_plus(gs,in_vals,step);

  /* process the received data */
  msg_nodes=nodes;
  while ((iptr = *nodes++)) {
    PetscScalar d1 = 1.0;

    /* Should I check the return value of MPI_Wait() or status? */
    /* Can this loop be replaced by a call to MPI_Waitall()? */
    CHKERRMPI(MPI_Wait(ids_in, &status));
    ids_in++;
    while (*iptr >= 0) {
      CHKERRQ(PetscBLASIntCast(step,&dstep));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&dstep,&d1,in2,&i1,dptr1 + *iptr*step,&i1));
      in2+=step;
      iptr++;
    }
  }

  /* replace vals */
  while (*pw >= 0) {
    PCTFS_rvec_copy(in_vals + *pw*step,dptr1,step);
    dptr1+=step;
    pw++;
  }

  /* clear isend message handles */
  /* This changed for clarity though it could be the same */

  /* Should I check the return value of MPI_Wait() or status? */
  /* Can this loop be replaced by a call to MPI_Waitall()? */
  while (*msg_nodes++) {
    CHKERRMPI(MPI_Wait(ids_out, &status));
    ids_out++;
  }
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode PCTFS_gs_gop_vec_tree_plus(PCTFS_gs_id *gs,  PetscScalar *vals,  PetscInt step)
{
  PetscInt       size, *in, *out;
  PetscScalar    *buf, *work;
  PetscInt       op[] = {GL_ADD,0};
  PetscBLASInt   i1   = 1;
  PetscBLASInt   dstep;

  PetscFunctionBegin;
  /* copy over to local variables */
  in   = gs->tree_map_in;
  out  = gs->tree_map_out;
  buf  = gs->tree_buf;
  work = gs->tree_work;
  size = gs->tree_nel*step;

  /* zero out collection buffer */
  PCTFS_rvec_zero(buf,size);

  /* copy over my contributions */
  while (*in >= 0) {
    CHKERRQ(PetscBLASIntCast(step,&dstep));
    PetscStackCallBLAS("BLAScopy",BLAScopy_(&dstep,vals + *in++ * step,&i1,buf + *out++ * step,&i1));
  }

  /* perform fan in/out on full buffer */
  /* must change PCTFS_grop to handle the blas */
  PCTFS_grop(buf,work,size,op);

  /* reset */
  in  = gs->tree_map_in;
  out = gs->tree_map_out;

  /* get the portion of the results I need */
  while (*in >= 0) {
    CHKERRQ(PetscBLASIntCast(step,&dstep));
    PetscStackCallBLAS("BLAScopy",BLAScopy_(&dstep,buf + *out++ * step,&i1,vals + *in++ * step,&i1));
  }
  PetscFunctionReturn(0);
}

/******************************************************************************/
PetscErrorCode PCTFS_gs_gop_hc(PCTFS_gs_id *gs,  PetscScalar *vals,  const char *op,  PetscInt dim)
{
  PetscFunctionBegin;
  switch (*op) {
  case '+':
    PCTFS_gs_gop_plus_hc(gs,vals,dim);
    break;
  default:
    CHKERRQ(PetscInfo(0,"PCTFS_gs_gop_hc() :: %c is not a valid op\n",op[0]));
    CHKERRQ(PetscInfo(0,"PCTFS_gs_gop_hc() :: default :: plus\n"));
    PCTFS_gs_gop_plus_hc(gs,vals,dim);
    break;
  }
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode PCTFS_gs_gop_plus_hc(PCTFS_gs_id *gs,  PetscScalar *vals, PetscInt dim)
{
  PetscFunctionBegin;
  /* if there's nothing to do return */
  if (dim<=0) PetscFunctionReturn(0);

  /* can't do more dimensions then exist */
  dim = PetscMin(dim,PCTFS_i_log2_num_nodes);

  /* local only operations!!! */
  if (gs->num_local) PCTFS_gs_gop_local_plus(gs,vals);

  /* if intersection tree/pairwise and local isn't empty */
  if (gs->num_local_gop) {
    PCTFS_gs_gop_local_in_plus(gs,vals);

    /* pairwise will do tree inside ... */
    if (gs->num_pairs) PCTFS_gs_gop_pairwise_plus_hc(gs,vals,dim); /* tree only */
    else if (gs->max_left_over) PCTFS_gs_gop_tree_plus_hc(gs,vals,dim);

    PCTFS_gs_gop_local_out(gs,vals);
  } else { /* if intersection tree/pairwise and local is empty */
    /* pairwise will do tree inside */
    if (gs->num_pairs) PCTFS_gs_gop_pairwise_plus_hc(gs,vals,dim); /* tree */
    else if (gs->max_left_over) PCTFS_gs_gop_tree_plus_hc(gs,vals,dim);
  }
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode PCTFS_gs_gop_pairwise_plus_hc(PCTFS_gs_id *gs,  PetscScalar *in_vals, PetscInt dim)
{
  PetscScalar    *dptr1, *dptr2, *dptr3, *in1, *in2;
  PetscInt       *iptr, *msg_list, *msg_size, **msg_nodes;
  PetscInt       *pw, *list, *size, **nodes;
  MPI_Request    *msg_ids_in, *msg_ids_out, *ids_in, *ids_out;
  MPI_Status     status;
  PetscInt       i, mask=1;

  PetscFunctionBegin;
  for (i=1; i<dim; i++) { mask<<=1; mask++; }

  /* strip and load s */
  msg_list    = list     = gs->pair_list;
  msg_size    = size     = gs->msg_sizes;
  msg_nodes   = nodes    = gs->node_list;
  iptr        = pw       = gs->pw_elm_list;
  dptr1       = dptr3    = gs->pw_vals;
  msg_ids_in  = ids_in   = gs->msg_ids_in;
  msg_ids_out = ids_out  = gs->msg_ids_out;
  dptr2       = gs->out;
  in1         = in2      = gs->in;

  /* post the receives */
  /*  msg_nodes=nodes; */
  do {
    /* Should MPI_ANY_SOURCE be replaced by *list ? In that case do the
        second one *list and do list++ afterwards */
    if ((PCTFS_my_id|mask)==(*list|mask)) {
      CHKERRMPI(MPI_Irecv(in1, *size, MPIU_SCALAR, MPI_ANY_SOURCE, MSGTAG1 + *list, gs->PCTFS_gs_comm, msg_ids_in));
      list++; msg_ids_in++;in1 += *size++;
    } else { list++; size++; }
  } while (*++msg_nodes);

  /* load gs values into in out gs buffers */
  while (*iptr >= 0) *dptr3++ = *(in_vals + *iptr++);

  /* load out buffers and post the sends */
  msg_nodes=nodes;
  list     = msg_list;
  while ((iptr = *msg_nodes++)) {
    if ((PCTFS_my_id|mask)==(*list|mask)) {
      dptr3 = dptr2;
      while (*iptr >= 0) *dptr2++ = *(dptr1 + *iptr++);
      /* CHECK PERSISTENT COMMS MODE FOR ALL THIS STUFF */
      /* is msg_ids_out++ correct? */
      CHKERRMPI(MPI_Isend(dptr3, *msg_size, MPIU_SCALAR, *list, MSGTAG1+PCTFS_my_id, gs->PCTFS_gs_comm, msg_ids_out));
      msg_size++;list++;msg_ids_out++;
    } else {list++; msg_size++;}
  }

  /* do the tree while we're waiting */
  if (gs->max_left_over) PCTFS_gs_gop_tree_plus_hc(gs,in_vals,dim);

  /* process the received data */
  msg_nodes=nodes;
  list     = msg_list;
  while ((iptr = *nodes++)) {
    if ((PCTFS_my_id|mask)==(*list|mask)) {
      /* Should I check the return value of MPI_Wait() or status? */
      /* Can this loop be replaced by a call to MPI_Waitall()? */
      CHKERRMPI(MPI_Wait(ids_in, &status));
      ids_in++;
      while (*iptr >= 0) *(dptr1 + *iptr++) += *in2++;
    }
    list++;
  }

  /* replace vals */
  while (*pw >= 0) *(in_vals + *pw++) = *dptr1++;

  /* clear isend message handles */
  /* This changed for clarity though it could be the same */
  while (*msg_nodes++) {
    if ((PCTFS_my_id|mask)==(*msg_list|mask)) {
      /* Should I check the return value of MPI_Wait() or status? */
      /* Can this loop be replaced by a call to MPI_Waitall()? */
      CHKERRMPI(MPI_Wait(ids_out, &status));
      ids_out++;
    }
    msg_list++;
  }
  PetscFunctionReturn(0);
}

/******************************************************************************/
static PetscErrorCode PCTFS_gs_gop_tree_plus_hc(PCTFS_gs_id *gs, PetscScalar *vals, PetscInt dim)
{
  PetscInt    size;
  PetscInt    *in, *out;
  PetscScalar *buf, *work;
  PetscInt    op[] = {GL_ADD,0};

  PetscFunctionBegin;
  in   = gs->tree_map_in;
  out  = gs->tree_map_out;
  buf  = gs->tree_buf;
  work = gs->tree_work;
  size = gs->tree_nel;

  PCTFS_rvec_zero(buf,size);

  while (*in >= 0) *(buf + *out++) = *(vals + *in++);

  in  = gs->tree_map_in;
  out = gs->tree_map_out;

  PCTFS_grop_hc(buf,work,size,op,dim);

  while (*in >= 0) *(vals + *in++) = *(buf + *out++);
  PetscFunctionReturn(0);
}
