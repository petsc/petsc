#define PETSC_DLL

#include "petscsys.h"
#include "petscfwk.h"

PETSC_DLLEXPORT PetscClassId PETSC_FWK_CLASSID;
static char PETSC_FWK_CLASS_NAME[] = "PetscFwk";

static PetscTruth PetscFwkPackageInitialized = PETSC_FALSE;
typedef enum{PETSC_FWK_COMPONENT_SO, PETSC_FWK_COMPONENT_PY} PetscFwkComponentType;

typedef PetscErrorCode (*PetscFwkPythonImportConfigureFunction)(const char *url, const char *path, const char *name, void **configure);
typedef PetscErrorCode (*PetscFwkPythonConfigureComponentFunction)(void *configure, PetscFwk fwk, PetscInt state, PetscObject *component);
typedef PetscErrorCode (*PetscFwkPythonPrintErrorFunction)(void);

EXTERN_C_BEGIN
PetscFwkPythonImportConfigureFunction    PetscFwkPythonImportConfigure    = PETSC_NULL;
PetscFwkPythonConfigureComponentFunction PetscFwkPythonConfigureComponent = PETSC_NULL;
PetscFwkPythonPrintErrorFunction         PetscFwkPythonPrintError         = PETSC_NULL;
EXTERN_C_END

#define PETSC_FWK_CHECKINIT_PYTHON()					\
  if(PetscFwkPythonImportConfigure == PETSC_NULL) {			\
    PetscErrorCode ierr;						\
    ierr = PetscPythonInitialize(PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);	\
    if(PetscFwkPythonImportConfigure == PETSC_NULL) {			\
      SETERRQ(PETSC_ERR_LIB,						\
	      "Couldn't initialize Python support for PetscFwk");	\
    }									\
  }									
  
#define PETSC_FWK_CONFIGURE_PYTHON(fwk, id, state, component)		\
  PETSC_FWK_CHECKINIT_PYTHON();						\
  {									\
    PetscErrorCode ierr;						\
    const char *_url  = fwk->record[id].url;				\
    const char *_path = fwk->record[id].path;				\
    const char *_name = fwk->record[id].name;				\
    void *_configure = 0;						\
    ierr = PetscFwkPythonImportConfigure(_url, _path, _name, &_configure);	\
    if (ierr) { PetscFwkPythonPrintError(); SETERRQ(PETSC_ERR_LIB, "Python error"); } \
    ierr = PetscFwkPythonConfigureComponent(_configure, fwk, state, component); \
    if (ierr) { PetscFwkPythonPrintError(); SETERRQ(PETSC_ERR_LIB, "Python error"); } \
  }
  

struct _n_PetscFwkGraph {
  PetscInt vcount, vmax; /* actual and allocated number of vertices */
  PetscInt *i, *j, *outdegree; /* (A)IJ structure for the underlying matrix: 
                                  i[row]             the row offset into j, 
                                  i[row+1] - i[row]  allocated number of entries for row,
                                  outdegree[row]     actual number of entries in row
                               */
  PetscInt *indegree;
  PetscInt nz, maxnz;
  PetscInt rowreallocs, colreallocs;
};

typedef struct _n_PetscFwkGraph *PetscFwkGraph;

#define CHUNKSIZE 5
/*
    Inserts the (row,col) entry, allocating larger arrays, if necessary. 
    Does NOT check whether row and col are within range (< graph->vcount).
    Does NOT check whether the entry is already present.
*/
#undef  __FUNCT__
#define __FUNCT__ "PetscFwkGraphExpandRow_Private"
PetscErrorCode PetscFwkGraphExpandRow_Private(PetscFwkGraph graph, PetscInt row) {
  PetscErrorCode ierr;
  PetscInt rowlen, rowmax, rowoffset;
  PetscInt ii;
  PetscFunctionBegin;
  rowlen = graph->outdegree[row]; 
  rowmax = graph->i[row+1] - graph->i[row]; 
  rowoffset = graph->i[row];
  if (rowlen >= rowmax) {
    /* there is no extra room in row, therefore enlarge */              
    PetscInt   new_nz = graph->i[graph->vcount] + CHUNKSIZE;  
    PetscInt   *new_i=0,*new_j=0;                            
    
    /* malloc new storage space */ 
    ierr = PetscMalloc(new_nz*sizeof(PetscInt),&new_j); CHKERRQ(ierr);
    ierr = PetscMalloc((graph->vmax+1)*sizeof(PetscInt),&new_i);CHKERRQ(ierr);
    
    /* copy over old data into new slots */ 
    for (ii=0; ii<row+1; ii++) {new_i[ii] = graph->i[ii];} 
    for (ii=row+1; ii<graph->vmax+1; ii++) {new_i[ii] = graph->i[ii]+CHUNKSIZE;} 
    ierr = PetscMemcpy(new_j,graph->j,(rowoffset+rowlen)*sizeof(PetscInt));CHKERRQ(ierr); 
    ierr = PetscMemcpy(new_j+rowoffset+rowlen+CHUNKSIZE,graph->j+rowoffset+rowlen,(new_nz - CHUNKSIZE - rowoffset - rowlen)*sizeof(PetscInt));CHKERRQ(ierr); 
    /* free up old matrix storage */ 
    ierr = PetscFree(graph->j);CHKERRQ(ierr);  
    ierr = PetscFree(graph->i);CHKERRQ(ierr);    
    graph->i = new_i; graph->j = new_j;  
    graph->maxnz     += CHUNKSIZE; 
    graph->colreallocs++; 
  } 
  PetscFunctionReturn(0);
}/* PetscFwkGraphExpandRow_Private() */

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkGraphAddVertex"
PetscErrorCode PetscFwkGraphAddVertex(PetscFwkGraph graph, PetscInt *v) {
  PetscInt ii;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(graph->vcount >= graph->vmax) {
    /* Add rows */
    PetscInt   *new_i=0, *new_outdegree=0, *new_indegree;                            
    
    /* malloc new storage space */ 
    ierr = PetscMalloc((graph->vmax+CHUNKSIZE+1)*sizeof(PetscInt),&new_i);CHKERRQ(ierr);
    ierr = PetscMalloc((graph->vmax+CHUNKSIZE)*sizeof(PetscInt),&new_outdegree); CHKERRQ(ierr);
    ierr = PetscMalloc((graph->vmax+CHUNKSIZE)*sizeof(PetscInt),&new_indegree); CHKERRQ(ierr);
    ierr = PetscMemzero(new_outdegree, (graph->vmax+CHUNKSIZE)*sizeof(PetscInt)); CHKERRQ(ierr);
    ierr = PetscMemzero(new_indegree, (graph->vmax+CHUNKSIZE)*sizeof(PetscInt)); CHKERRQ(ierr);


    /* copy over old data into new slots */ 
    ierr = PetscMemcpy(new_i,graph->i,(graph->vmax+1)*sizeof(PetscInt));CHKERRQ(ierr); 
    ierr = PetscMemcpy(new_outdegree,graph->outdegree,(graph->vmax)*sizeof(PetscInt));CHKERRQ(ierr); 
    ierr = PetscMemcpy(new_indegree,graph->indegree,(graph->vmax)*sizeof(PetscInt));CHKERRQ(ierr); 
    for (ii=graph->vmax+1; ii<=graph->vmax+CHUNKSIZE; ++ii) {
      new_i[ii] = graph->i[graph->vmax];
    }

    /* free up old matrix storage */ 
    ierr = PetscFree(graph->i);CHKERRQ(ierr);  
    ierr = PetscFree(graph->outdegree);CHKERRQ(ierr);    
    ierr = PetscFree(graph->indegree);CHKERRQ(ierr);    
    graph->i = new_i; graph->outdegree = new_outdegree; graph->indegree = new_indegree;
    
    graph->vmax += CHUNKSIZE; 
    graph->rowreallocs++; 
  }
  if(v) {*v = graph->vcount;}
  ++(graph->vcount);
  PetscFunctionReturn(0);
}/* PetscFwkGraphAddVertex() */

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkGraphAddEdge"
PetscErrorCode PetscFwkGraphAddEdge(PetscFwkGraph graph, PetscInt row, PetscInt col) {
  PetscErrorCode        ierr;
  PetscInt              *rp,low,high,t,ii,i;
  PetscFunctionBegin;

  if(row < 0 || row >= graph->vcount) {
    SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Source vertex %D out of range: min %D max %D",row, 0, graph->vcount);
  }
  if(col < 0 || col >= graph->vcount) {
    SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Target vertex %D out of range: min %D max %D",col, 0, graph->vcount);
  }
  rp   = graph->j + graph->i[row];
  low  = 0;
  high = graph->outdegree[row];
  while (high-low > 5) {
    t = (low+high)/2;
    if (rp[t] > col) high = t;
    else             low  = t;
  }
  for (i=low; i<high; ++i) {
    if (rp[i] > col) break;
    if (rp[i] == col) {
      goto we_are_done;
    }
  } 
  ierr = PetscFwkGraphExpandRow_Private(graph, row); CHKERRQ(ierr);
  /* 
     If the graph was empty before, graph->j was NULL and rp was NULL as well.  
     Now that the row has been expanded, rp needs to be reset. 
  */
  rp = graph->j + graph->i[row];
  /* shift up all the later entries in this row */
  for (ii=graph->outdegree[row]; ii>=i; --ii) {
    rp[ii+1] = rp[ii];
  }
  rp[i] = col; 
  ++(graph->outdegree[row]);
  ++(graph->indegree[col]);
  ++(graph->nz);
  
 we_are_done:
  PetscFunctionReturn(0);
}/* PetscFwkGraphAddEdge() */


#undef  __FUNCT__
#define __FUNCT__ "PetscFwkGraphTopologicalSort"
PetscErrorCode PetscFwkGraphTopologicalSort(PetscFwkGraph graph, PetscInt *n, PetscInt **queue) {
  PetscTruth *queued;
  PetscInt   *indegree;
  PetscInt ii, k, jj, Nqueued = 0;
  PetscTruth progress;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(!n || !queue) {
    SETERRQ(PETSC_ERR_ARG_WRONG, "Invalid return argument pointers n or vertices");
  }
  *n = graph->vcount;
  ierr = PetscMalloc(sizeof(PetscInt)*graph->vcount, queue); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscTruth)*graph->vcount, &queued); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*graph->vcount, &indegree); CHKERRQ(ierr);
  for(ii = 0; ii < graph->vcount; ++ii) {
    queued[ii]   = PETSC_FALSE;
    indegree[ii] = graph->indegree[ii];
  }
  while(Nqueued < graph->vcount) {
    progress = PETSC_FALSE;
    for(ii = 0; ii < graph->vcount; ++ii) {
      /* If ii is not queued yet, and the indegree is 0, queue it. */ 
      if(!queued[ii] && !indegree[ii]) {
        (*queue)[Nqueued] = ii;
        queued[ii] = PETSC_TRUE;
        ++Nqueued;
        progress = PETSC_TRUE;
        /* Reduce the indegree of all vertices in row ii */
        for(k = 0; k < graph->outdegree[ii]; ++k) {
          jj = graph->j[graph->i[ii]+k];
          --(indegree[jj]);
          /* 
             It probably would be more efficient to make a recursive call to the body of the for loop 
             with the jj in place of ii, but we use a simple-minded algorithm instead, since the graphs
             we anticipate encountering are tiny. 
          */
        }/*for(k)*/
      }/* if(!queued) */
    }/* for(ii) */
    /* If no progress was made during this iteration, the graph must have a cycle */
    if(!progress) {
      SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Cycle detected in the dependency graph");
    }
  }/* while(Nqueued) */
  ierr = PetscFree(queued); CHKERRQ(ierr);
  ierr = PetscFree(indegree); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkGraphTopologicalSort() */

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkGraphDestroy"
PetscErrorCode PetscFwkGraphDestroy(PetscFwkGraph graph) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree(graph->i);        CHKERRQ(ierr);
  ierr = PetscFree(graph->j);        CHKERRQ(ierr);
  ierr = PetscFree(graph->outdegree);     CHKERRQ(ierr);
  ierr = PetscFree(graph->indegree); CHKERRQ(ierr);
  ierr = PetscFree(graph);           CHKERRQ(ierr);
  graph = PETSC_NULL;
  PetscFunctionReturn(0);
}/* PetscFwkGraphDestroy() */

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkGraphCreate"
PetscErrorCode PetscFwkGraphCreate(PetscFwkGraph *graph_p) {
  PetscFwkGraph graph;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscNew(struct _n_PetscFwkGraph, graph_p);   CHKERRQ(ierr);
  graph = *graph_p;
  graph->vcount = graph->vmax = graph->nz = graph->maxnz = graph->rowreallocs = graph->colreallocs = 0;
  ierr = PetscMalloc(sizeof(PetscInt), &(graph->i)); CHKERRQ(ierr);
  graph->j = graph->outdegree = graph->indegree = PETSC_NULL;
  PetscFunctionReturn(0);
}/* PetscFwkGraphCreate() */

/* ------------------------------------------------------------------------------------------------------- */



struct _n_PetscFwkRecord {
  char              *url, *path, *name;
  PetscFwkComponentType      type;
  PetscObject                component;
  PetscFwkComponentConfigure configure;
};

struct _p_PetscFwk {
  PETSCHEADER(int);
  PetscInt                  N, maxN;
  struct _n_PetscFwkRecord  *record;
  PetscFwkGraph             dep_graph;
};

static PetscFwk defaultFwk = PETSC_NULL;

PetscErrorCode PetscFwkCheck(PetscFwk *_fwk);

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkViewConfigurationOrder"
PetscErrorCode PetscFwkViewConfigurationOrder(PetscFwk fwk, PetscViewer viewerASCII){
  PetscInt *vertices, N;
  PetscInt i, id;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscFwkCheck(&fwk); CHKERRQ(ierr);

  ierr = PetscFwkGraphTopologicalSort(fwk->dep_graph, &N, &vertices); CHKERRQ(ierr);
  for(i = 0; i < N; ++i) {
    if(i) {
      ierr = PetscViewerASCIIPrintf(viewerASCII, ", "); CHKERRQ(ierr);
    }
    id = vertices[i];
    ierr = PetscViewerASCIIPrintf(viewerASCII, "%d: %s", id, fwk->record[id].url); CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewerASCII, "\n"); CHKERRQ(ierr);
  ierr = PetscFree(vertices); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkViewConfigurationOrder() */

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkConfigure"
PetscErrorCode PetscFwkConfigure(PetscFwk fwk, PetscInt state){
  PetscInt i, id, N, *vertices;
  PetscFwkComponentConfigure configure = PETSC_NULL;
  PetscObject component;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscFwkCheck(&fwk); CHKERRQ(ierr);

  ierr = PetscFwkGraphTopologicalSort(fwk->dep_graph, &N, &vertices); CHKERRQ(ierr);
  ierr = PetscObjectSetState((PetscObject)fwk,state); CHKERRQ(ierr);
  for(i = 0; i < N; ++i) {
    id = vertices[i];
    configure = PETSC_NULL;
    component = fwk->record[id].component;
    switch(fwk->record[id].type){
    case PETSC_FWK_COMPONENT_SO:
      configure = fwk->record[id].configure;
      if(configure != PETSC_NULL) {
        ierr = (*configure)(fwk,state,&component); CHKERRQ(ierr);
      }
      break;
    case PETSC_FWK_COMPONENT_PY:
      PETSC_FWK_CONFIGURE_PYTHON(fwk, id, state, &component);
      break;
    }
  }
  ierr = PetscFree(vertices); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkConfigure() */



static PetscDLLibrary PetscFwkDLList = PETSC_NULL;
#define PETSC_FWK_MAX_URL_LENGTH 1024

/* 
   Normalize the url (by truncating to PETSC_FWK_MAX_URL_LENGTH) and parse it to find out the component type and location.
   Warning: if nurl, npath, nname are passed in as NULL, the returned char pointers are borrowed and their contents
   must be copied elsewhere to be preserved 
*/
#undef  __FUNCT__
#define __FUNCT__ "PetscFwkParseURL_Private"
PetscErrorCode PETSC_DLLEXPORT PetscFwkParseURL_Private(PetscFwk fwk, const char inurl[], char url[], char path[], char name[], PetscFwkComponentType *type){
  char *n, *s;
  static PetscInt nlen = PETSC_FWK_MAX_URL_LENGTH;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* FIX: this routine should replace the filesystem path by an abolute path for real normalization */
  /* Copy the inurl so we can manipulate it inplace and also truncate to the max allowable length */
  ierr = PetscStrncpy(path, inurl, nlen); CHKERRQ(ierr);  
  /* Split url <path>:<name> into <path> and <name> */
  ierr = PetscStrrchr(path,':',&n); CHKERRQ(ierr);
  /* Make sure it's not the ":/" of the "://" separator */
  if(!n[0] || n[0] == '/') {
    SETERRQ2(PETSC_ERR_ARG_WRONG, 
           "Could not locate component name within the URL.\n"
           "Must have url = [<path/><library>:]<name>.\n"
           "Instead got %s\n"
           "Remember that URL is always truncated to the max allowed length of %d", 
           inurl, nlen);
  }
  /* Copy n to name */
  ierr = PetscStrcpy(name, n); CHKERRQ(ierr);
  /* If n isn't the whole path (i.e., there is a ':' separator), end 'path' right before the located ':' */
  if(n != path) {
    n[-1] = '\0';
  }
  /* Find and remove the library suffix */
  ierr = PetscStrrchr(path,'.',&s);CHKERRQ(ierr);
  /* Determine the component library type: .so or .py */
  /* FIX: we should really be using PETSc's internally defined suffices */
  if(s != path && s[-1] == '.') {
    if((s[0] == 'a' && s[1] == '\0') || (s[0] == 's' && s[1] == 'o' && s[2] == '\0')){
      *type = PETSC_FWK_COMPONENT_SO;
    }
    else if (s[0] == 'p' && s[1] == 'y' && s[2] == '\0'){
      *type = PETSC_FWK_COMPONENT_PY;
    }
    else {
      SETERRQ3(PETSC_ERR_ARG_WRONG, 
           "Unknown library suffix within the URL.\n"
           "Must have url = [<path/><library>:]<name>,\n"
           "where library = <libname>.<suffix>, suffix = .a || .so || .py.\n"
           "Instead got url %s and suffix %s\n"
           "Remember that URL is always truncated to the max allowed length of %d", 
               inurl, s,nlen);     
    }
    /* Remove the suffix from the library name */
    s[-1] = '\0';  
  }
  else {
    SETERRQ2(PETSC_ERR_ARG_WRONG, 
             "Could not locate library within the URL.\n"
             "Must have url = [<path/><library>:]<name>.\n"
             "Instead got %s\n"
             "Remember that URL is always truncated to the max allowed length of %d", 
             inurl, nlen);     
  }
  ierr = PetscStrcpy(url, path); CHKERRQ(ierr);
  ierr = PetscStrcat(url, ":");  CHKERRQ(ierr);
  ierr = PetscStrcat(url, name); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkParseURL_Private() */


#undef  __FUNCT__
#define __FUNCT__ "PetscFwkRegisterComponentID_Private"
PetscErrorCode PETSC_DLLEXPORT PetscFwkRegisterComponentID_Private(PetscFwk fwk, const char inurl[], PetscInt *_id){
  PetscFwkComponentType type;
  PetscInt i, v, id;
  size_t len;
  PetscTruth eq;
  char url[PETSC_FWK_MAX_URL_LENGTH+1], path[PETSC_FWK_MAX_URL_LENGTH+1], name[PETSC_FWK_MAX_URL_LENGTH+1];
  PetscObject component = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFwkParseURL_Private(fwk, inurl, url, path, name, &type); CHKERRQ(ierr);
  /* Check whether a component with the given url has already been registered.  If so, return its id, if it has been requested. */
  for(i = 0; i < fwk->N; ++i) {
    ierr = PetscStrcmp(url, fwk->record[i].url, &eq); CHKERRQ(ierr);
    if(eq) {
      if(_id) {*_id = i;}
      PetscFunctionReturn(0);
    }
  }
  /* No such url found. */
  /* Create a new record for this url. */
  if(fwk->N >= fwk->maxN) {
    /* No more empty records, therefore, expand the record array */
    struct _n_PetscFwkRecord *new_record;
    ierr = PetscMalloc(sizeof(struct _n_PetscFwkRecord)*(fwk->maxN+CHUNKSIZE), &new_record);   CHKERRQ(ierr);
    ierr = PetscMemcpy(new_record, fwk->record, sizeof(struct _n_PetscFwkRecord)*(fwk->maxN)); CHKERRQ(ierr);
    ierr = PetscMemzero(new_record+fwk->maxN,sizeof(struct _n_PetscFwkRecord)*(CHUNKSIZE));    CHKERRQ(ierr);
    ierr = PetscFree(fwk->record);                                                             CHKERRQ(ierr);
    fwk->record = new_record;
    fwk->maxN += CHUNKSIZE;
  }
  id = fwk->N;
  ++(fwk->N);
  /* Store url, path, name */
  ierr = PetscStrlen(url, &len);                                     CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(char)*(len+1), &(fwk->record[id].url));  CHKERRQ(ierr);
  ierr = PetscStrcpy(fwk->record[id].url, url);                      CHKERRQ(ierr);
  /**/
  ierr = PetscStrlen(name, &len);                                    CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(char)*(len+1), &(fwk->record[id].name)); CHKERRQ(ierr);
  ierr = PetscStrcpy(fwk->record[id].name, name);                    CHKERRQ(ierr);
  /**/
  ierr = PetscStrlen(path, &len);                                    CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(char)*(len+1), &(fwk->record[id].path)); CHKERRQ(ierr);
  ierr = PetscStrcpy(fwk->record[id].path, path);                    CHKERRQ(ierr);
  /* Set component type */
  fwk->record[id].type = type;
  /* The rest is NULL */
  fwk->record[id].component = PETSC_NULL;
  fwk->record[id].configure = PETSC_NULL;
  /* Add a new vertex to the dependence graph.  This vertex will correspond to the newly registered component. */
  ierr = PetscFwkGraphAddVertex(fwk->dep_graph, &v); CHKERRQ(ierr);
  /* v must equal id */
  if(v != id) {
    SETERRQ2(PETSC_ERR_ARG_CORRUPT, "New dependence graph vertex %d not the same as component id %d", v, id); 
  }
  switch(type) {
  case PETSC_FWK_COMPONENT_SO:
    {
      char sym[PETSC_FWK_MAX_URL_LENGTH+26+1];
      PetscFwkComponentConfigure configure = PETSC_NULL;
      /* Build the configure symbol from name and standard prefix */
      ierr = PetscStrcpy(sym, "PetscFwkComponentConfigure"); CHKERRQ(ierr);
      ierr = PetscStrcat(sym, name); CHKERRQ(ierr);
      /* Load the library designated by 'path' and retrieve from it the configure routine designated by the constructed symbol */
      ierr = PetscDLLibrarySym(((PetscObject)fwk)->comm, &PetscFwkDLList, path, sym, (void**)(&configure)); CHKERRQ(ierr);
      /* Run the configure routine, which should return a valid object or PETSC_NULL */
      ierr = (*configure)(fwk, 0, &component); CHKERRQ(ierr);
      fwk->record[id].component = component;
      fwk->record[id].configure = configure;
    }
    break;
  case PETSC_FWK_COMPONENT_PY:
    PETSC_FWK_CONFIGURE_PYTHON(fwk, id, 0, &component);
    fwk->record[id].component = component;
    /* configure field remains NULL for a Py component */
    break;
  default:
    SETERRQ2(PETSC_ERR_ARG_WRONG, 
             "Could not determine type of component with url %s.\n"
             "Remember: URL was truncated past the max allowed length of %d", 
             inurl, PETSC_FWK_MAX_URL_LENGTH);    
  }
  if(_id) {
    *_id = id;
  }
  PetscFunctionReturn(0);
}/* PetscFwkRegisterComponentID_Private()*/

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkRegisterComponent"
PetscErrorCode PETSC_DLLEXPORT PetscFwkRegisterComponent(PetscFwk fwk, const char url[]){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFwkCheck(&fwk); CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponentID_Private(fwk, url, PETSC_NULL); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkRegisterComponent() */


#undef  __FUNCT__
#define __FUNCT__ "PetscFwkRegisterDependence"
PetscErrorCode PETSC_DLLEXPORT PetscFwkRegisterDependence(PetscFwk fwk, const char clienturl[], const char serverurl[])
{
  PetscInt clientid, serverid;
  PetscErrorCode ierr; 
  PetscFunctionBegin; 
  ierr = PetscFwkCheck(&fwk); CHKERRQ(ierr);
  PetscValidCharPointer(clienturl,2);
  PetscValidCharPointer(serverurl,3);
  /* Register urls */
  ierr = PetscFwkRegisterComponentID_Private(fwk, clienturl, &clientid); CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponentID_Private(fwk, serverurl, &serverid); CHKERRQ(ierr);

  /*
    Add the dependency edge to the dependence_graph as follows (serverurl, clienturl): 
     this means "server preceeds client", so server should be configured first.
  */
  ierr = PetscFwkGraphAddEdge(fwk->dep_graph, clientid, serverid); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/*PetscFwkRegisterDependence()*/



#undef  __FUNCT__
#define __FUNCT__ "PetscFwkDestroy"
PetscErrorCode PETSC_DLLEXPORT PetscFwkDestroy(PetscFwk fwk)
{
  PetscInt i;
  PetscErrorCode ierr;
  if (fwk == PETSC_FWK_DEFAULT || --((PetscObject)fwk)->refct > 0) PetscFunctionReturn(0);
  for(i = 0; i < fwk->N; ++i){
    ierr = PetscFree(fwk->record[i].url);
    ierr = PetscFree(fwk->record[i].path);
    ierr = PetscFree(fwk->record[i].name);
    if(fwk->record[i].component != PETSC_NULL) {
      ierr = PetscObjectDestroy(fwk->record[i].component); CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(fwk->record); CHKERRQ(ierr);
  ierr = PetscFwkGraphDestroy(fwk->dep_graph); CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(fwk);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkDestroy()*/

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkCreate"
PetscErrorCode PETSC_DLLEXPORT PetscFwkCreate(MPI_Comm comm, PetscFwk *framework){
  PetscFwk fwk;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  #ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PetscFwkInitializePackage(PETSC_NULL);CHKERRQ(ierr);
  #endif
  PetscValidPointer(framework,2);
  ierr = PetscHeaderCreate(fwk,_p_PetscFwk,PetscInt,PETSC_FWK_CLASSID,0,"PetscFwk",comm,PetscFwkDestroy,0);CHKERRQ(ierr);
  fwk->record = PETSC_NULL;
  fwk->N = fwk->maxN = 0;
  ierr = PetscFwkGraphCreate(&fwk->dep_graph); CHKERRQ(ierr);
  *framework = fwk;
  PetscFunctionReturn(0);
}/* PetscFwkCreate() */


#undef  __FUNCT__
#define __FUNCT__ "PetscFwkGetComponent"
PetscErrorCode PETSC_DLLEXPORT PetscFwkGetComponent(PetscFwk fwk, const char url[], PetscObject *_component, PetscTruth *_found) {
  PetscInt i;
  PetscTruth eq;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFwkCheck(&fwk); CHKERRQ(ierr);
  PetscValidCharPointer(url,2);
  if(_found) {*_found = PETSC_FALSE;}
  if(_component) {*_component = PETSC_NULL;}
  for(i = 0; i < fwk->N; ++i) {
    ierr = PetscStrcmp(url, fwk->record[i].url, &eq); CHKERRQ(ierr);
    if(eq) {
      if(_component) {*_component = fwk->record[i].component;}
      if(_found) {*_found = PETSC_TRUE;};
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}/* PetscFwkGetComponent() */


#undef  __FUNCT__
#define __FUNCT__ "PetscFwkFinalizePackage"
PetscErrorCode PetscFwkFinalizePackage(void){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(PetscFwkDLList != PETSC_NULL) {
    ierr = PetscDLLibraryClose(PetscFwkDLList); CHKERRQ(ierr);
    PetscFwkDLList = PETSC_NULL;
  }
  if(defaultFwk) {
    ierr = PetscFwkDestroy(defaultFwk); CHKERRQ(ierr);
  }
  PetscFwkPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}/* PetscFwkFinalizePackage() */


#undef  __FUNCT__
#define __FUNCT__ "PetscFwkInitializePackage"
PetscErrorCode PetscFwkInitializePackage(const char path[]){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(PetscFwkPackageInitialized) PetscFunctionReturn(0);
  PetscFwkPackageInitialized = PETSC_TRUE;
  /* Register classes */
  ierr = PetscClassIdRegister(PETSC_FWK_CLASS_NAME, &PETSC_FWK_CLASSID); CHKERRQ(ierr);
  /* Register finalization routine */
  ierr = PetscRegisterFinalize(PetscFwkFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkInitializePackage() */



#undef  __FUNCT__
#define __FUNCT__ "PetscFwkCheck"
PetscErrorCode PetscFwkCheck(PetscFwk *_fwk) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(*_fwk == PETSC_FWK_DEFAULT) {
    if(!defaultFwk) {
      ierr = PetscFwkCreate(PETSC_COMM_WORLD, &defaultFwk); CHKERRQ(ierr);
    }
    *_fwk = defaultFwk;
  }
  PetscFunctionReturn(0);
}/* PetscFwkCheck() */

