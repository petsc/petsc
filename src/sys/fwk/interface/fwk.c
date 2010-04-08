#define PETSC_DLL

#include "petscsys.h"
#include "petscfwk.h"

PetscCookie PETSC_FWK_COOKIE;
static char PETSC_FWK_CLASS_NAME[] = "PetscFwk";

static PetscTruth PetscFwkPackageInitialized = PETSC_FALSE;


/* 
   Graph/topological sort stuff from BOOST.
   May need to be replaced with a custom, all-C implementation
*/
// >> C++
#include <iostream>
#include <map>
#include <vector>
#include <list>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>


struct vertex_id {
  PetscInt id;
};
//
// We use bundled properties to store id with the vertex.
typedef ::boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, vertex_id>             dependence_graph_type;
typedef boost::graph_traits<dependence_graph_type>::vertex_descriptor                                   vertex_type;
//
struct dependence_graph_cycle_detector_type : public boost::default_dfs_visitor {
  void back_edge(boost::graph_traits<dependence_graph_type>::edge_descriptor e, const dependence_graph_type& g) {
    std::basic_ostringstream<char> ss;
    ss << "Edge creates dependence loop: [" << (g[::boost::source(e,g)].id) << " --> " << (g[::boost::target(e,g)].id) << "]; ";
    throw ss.str();
  }//back_edge()
};// dependence_graph_cycle_detector_type
// << C++

struct _p_PetscFwk {
  PETSCHEADER(int);
  // >> C++
  // Since PetscFwk is allocated using the C-style PetscMalloc, the C++-object data members have to be 'newed', hance must be pointers
  std::map<std::string, PetscInt>         *id;
  std::vector<std::string>                *url;
  std::vector<PetscFwkComponentConfigure> *configure;
  std::vector<PetscObject>                *component;
  std::vector<vertex_type>                *vertex;
  dependence_graph_type                   *dependence_graph;
  // << C++
};

// >> C++
#undef  __FUNCT__
#define __FUNCT__ "PetscFwkConfigure_Sort"
PetscErrorCode PetscFwkConfigure_Sort(PetscFwk fwk, std::list<vertex_type>& antitopological_vertex_order){
  PetscFunctionBegin;
  // >> C++
  /* Run a cycle detector on the dependence graph */
  try {
    dependence_graph_cycle_detector_type cycle_detector;
    ::boost::depth_first_search(*(fwk->dependence_graph), boost::visitor(cycle_detector));
  }
  catch(const std::string& s) {
    SETERRQ1(PETSC_ERR_ARG_WRONGSTATE, "Component dependence graph has a loop: %s", s.c_str());
  }
  catch(...) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Exception caught while detecting loops in the dependence graph");
  }
  /* 
     Run topological sort on the dependence graph; remember that the vertices are returned in the "antitopological" order.
     This, in fact, is just what we need, since edges in the graph are entered as (client, server), implying that server 
     must be configured *before* the client and BOOSTs topological sort will in fact order server before client.
   */
  try{
    ::boost::topological_sort(*(fwk->dependence_graph), std::back_inserter(antitopological_vertex_order));
  }
  catch(...) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Exception caught while performing topological_sort in the dependence graph");
  }

  PetscFunctionReturn(0);
}/* PetscFwkConfigure_Sort() */
// << C++

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkViewConfigurationOrder"
PetscErrorCode PetscFwkViewConfigurationOrder(PetscFwk fwk, PetscViewer viewerASCII){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  // >> C++
  std::list<vertex_type> antitopological_vertex_order;
  ierr = PetscFwkConfigure_Sort(fwk, antitopological_vertex_order); CHKERRQ(ierr);
  /* traverse the vertices in their antitopological, extract component ids from vertices and  configure each corresponding component */
  /* current_state starts with 1, since 0 was used during Preconfigure */
  for(std::list<vertex_type>::iterator iter = antitopological_vertex_order.begin(); iter != antitopological_vertex_order.end(); ++iter) {
    PetscInt id = (*(fwk->dependence_graph))[*iter].id;
    std::string urlstring = (*fwk->url)[id];
    if(iter != antitopological_vertex_order.begin()) {
      ierr = PetscViewerASCIIPrintf(viewerASCII, ", "); CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewerASCII, "%d: %s", id, urlstring.c_str()); CHKERRQ(ierr);
  }
  // << C++
  ierr = PetscViewerASCIIPrintf(viewerASCII, "\n"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkViewConfigurationOrder() */

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkConfigure"
PetscErrorCode PetscFwkConfigure(PetscFwk fwk, PetscInt state){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  // >> C++
  std::list<vertex_type> antitopological_vertex_order;
  ierr = PetscFwkConfigure_Sort(fwk, antitopological_vertex_order); CHKERRQ(ierr);
  /* traverse the vertices in their antitopological, extract component ids from vertices and  configure each corresponding component */
  for(PetscInt current_state=0;current_state<=state; ++current_state) {
    ierr = PetscObjectSetState((PetscObject)fwk,current_state); CHKERRQ(ierr);
    for(std::list<vertex_type>::iterator iter = antitopological_vertex_order.begin(); iter != antitopological_vertex_order.end(); ++iter) {
      PetscInt id = (*(fwk->dependence_graph))[*iter].id;
      PetscObject component = (*fwk->component)[id];
      PetscFwkComponentConfigure configure = (*fwk->configure)[id];
      if(configure != PETSC_NULL) {
        ierr = (*configure)(fwk,current_state,&component); CHKERRQ(ierr);
      }
    }
  }
  // << C++
  PetscFunctionReturn(0);
}/* PetscFwkConfigure() */



static PetscDLLibrary PetscFwkDLList = PETSC_NULL;
#define PETSC_FWK_MAX_URL_LENGTH 1024

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkNormalizeURL_Private"
PetscErrorCode PETSC_DLLEXPORT PetscFwkNormalizeURL_Private(PetscFwk fwk, const char url[], char nurl[], char npath[], char nname[]){
  char *n, *s;
  char lpath[PETSC_FWK_MAX_URL_LENGTH+1], lname[PETSC_FWK_MAX_URL_LENGTH+1];
  char *path, *name;
  static PetscInt nlen = PETSC_FWK_MAX_URL_LENGTH;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* FIX: this should replace the filesystem path by an abolute path for real normalization */
  /* Use local buffers if npath or nname is NULL */
  if(npath == PETSC_NULL) {
    path = lpath;
  }
  else {
    path = npath;
  }
  if(nname == PETSC_NULL) {
    name = lname;
  }
  else {
    name = nname;
  }
  /* Copy the url so we can manipulate it inplace and also truncate to the max allowable length */
  ierr = PetscStrncpy(path, url, nlen); CHKERRQ(ierr);  
  /* Split url <path>:<name> into <path> and <name> */
  ierr = PetscStrrchr(path,':',&n); CHKERRQ(ierr);
  /* Make sure it's not the ":/" of the "://" separator */
  if(!n[0] || n[0] == '/') {
    SETERRQ2(PETSC_ERR_ARG_WRONG, 
             "Could not locate component name within the URL.\n"
             "Must have url = [<path>:]<name>, instead got %s\n"
             "Remember: URL was truncated past the max allowed length of %d", 
             path, nlen);
  }
  /* Copy n to name */
  ierr = PetscStrcpy(name, n); CHKERRQ(ierr);
  /* If n isn't the whole path (i.e., there is no ':' separator), end 'path' right before the located ':' */
  if(n != path) {
    n[-1] = '\0';
  }
  /* Find and remove the library suffix */
  ierr = PetscStrrchr(path,'.',&s);CHKERRQ(ierr);
  /* Make sure this isn't part of a relative path name (i.e.., "./" or "../") */
  /* FIX: we should really be using PETSc's internally defined suffices, because otherwise, 
     we might be removing names of hidden files (e.g., '${PETSC_DIR}/lib/${PETSC_ARCH}/.myhiddenlib.a') */
  if(s[0] != '/') {
    s[-1] = '\0';
  }
  ierr = PetscStrcpy(nurl, path); CHKERRQ(ierr);
  ierr = PetscStrcat(nurl, ":");  CHKERRQ(ierr);
  ierr = PetscStrcat(nurl, name); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkNormalizeURL_Private() */

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkInsertComponent_Private"
PetscErrorCode PETSC_DLLEXPORT PetscFwkInsertComponent_Private(PetscFwk fwk, const char url[], PetscObject component, PetscFwkComponentConfigure configure, PetscInt *_id){
  /* 
     WARNING: This routine will not check whether url has already been registered.  
     It will assign a new id to the url and insert the corresponding component and configure objects. 
  */
  PetscObject old_component;
  PetscInt id;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  // >> C++
  std::string urlstring(url);
  // Check whether url has already been registered
  if((*(fwk->id)).count(urlstring)) {
    // Retrieve the existing id
    id = (*(fwk->id))[urlstring];
    // Reset the existing component and configure objects
    old_component = (*(fwk->component))[id];
    if(old_component) {
      ierr = PetscObjectDereference(old_component); CHKERRQ(ierr);
    }
    if(component) {
      ierr = PetscObjectReference(component); CHKERRQ(ierr);
    }
    (*(fwk->component))[id] = component;
    (*(fwk->configure))[id] = configure;
  }
  else {
    // Assign and store the new id for urlstring.
    id = (*(fwk->id)).size();
    (*(fwk->id))[urlstring] = id;
    (*(fwk->url)).push_back(urlstring);
    /* Push new component and configure objects onto the list */
    if(component) {
      ierr = PetscObjectReference(component); CHKERRQ(ierr);
    }
    (*(fwk->component)).push_back(component);
    (*(fwk->configure)).push_back(configure); 
    /* Add a new vertex to the dependence graph.  This vertex will correspond to the newly registered component. */
    vertex_type v = ::boost::add_vertex(*(fwk->dependence_graph));
    /* Attach id to v */
    (*(fwk->dependence_graph))[v].id = id;
    /* Store v in fwk */
    (*(fwk->vertex)).push_back(v); 
  }
  // << C++
  if(_id) {
    *_id = id;
  }
  PetscFunctionReturn(0);
}/* PetscFwkInsertComponent_Private()*/


#undef  __FUNCT__
#define __FUNCT__ "PetscFwkRegisterComponentWithID"
PetscErrorCode PETSC_DLLEXPORT PetscFwkRegisterComponentWithID(PetscFwk fwk, const char url[], PetscInt *_id){
  PetscFwkComponentConfigure configure = PETSC_NULL;
  PetscObject component = PETSC_NULL;
  char nurl[PETSC_FWK_MAX_URL_LENGTH+1], path[PETSC_FWK_MAX_URL_LENGTH+1], name[PETSC_FWK_MAX_URL_LENGTH+1];
  char sym[PETSC_FWK_MAX_URL_LENGTH+26+1];
  //PetscInt id;
  MPI_Comm comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFwkNormalizeURL_Private(fwk, url, nurl, path, name); CHKERRQ(ierr);
  // >> C++
  // Check whether a component with the given url has already been registered.  If so, return its id, if it has been requested.
  std::string urlstring(nurl);
  if((*(fwk->id)).count(urlstring)) {
    if(_id) {
      *_id = (*(fwk->id))[urlstring];
    }
    PetscFunctionReturn(0);
  }
  // Insert a new url with empty corresponding component and configure objects
  ierr = PetscFwkInsertComponent_Private(fwk, nurl, PETSC_NULL, PETSC_NULL, PETSC_NULL); CHKERRQ(ierr);
  /* Build the configure symbol from name and standard prefix */
  ierr = PetscStrcpy(sym, "PetscFwkComponentConfigure"); CHKERRQ(ierr);
  ierr = PetscStrcat(sym, name); CHKERRQ(ierr);
  /* Load the library designated by 'path' and retrieve from it the configure routine designated by the constructed symbol */
  ierr = PetscObjectGetComm((PetscObject)fwk, &comm); CHKERRQ(ierr);
  ierr = PetscDLLibrarySym(comm, &PetscFwkDLList, path, sym, (void**)(&configure)); CHKERRQ(ierr);
  /* Run the configure routine, which should return a valid component object */
  ierr = (*configure)(fwk, 0, &component); CHKERRQ(ierr);
  if(component == PETSC_NULL) {
    SETERRQ2(PETSC_ERR_LIB, "Configure routine %s from library %s returned a NULL component", path, sym);
  }
  /* Reinsert nurl with the correct component and configure objects, and retrieve its id. */
  ierr = PetscFwkInsertComponent_Private(fwk, nurl, component, configure, _id); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkRegisterComponentWithID()*/

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkRegisterComponent"
PetscErrorCode PETSC_DLLEXPORT PetscFwkRegisterComponent(PetscFwk fwk, const char url[]){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFwkRegisterComponentWithID(fwk, url, PETSC_NULL); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkRegisterComponent()*/


#undef  __FUNCT__
#define __FUNCT__ "PetscFwkRegisterDependence"
PetscErrorCode PETSC_DLLEXPORT PetscFwkRegisterDependence(PetscFwk fwk, const char clienturl[], const char serverurl[])
{
  PetscInt clientid, serverid;
  PetscErrorCode ierr; 
  PetscFunctionBegin; 
  PetscValidCharPointer(clienturl,2);
  PetscValidCharPointer(serverurl,3);
  /* Register urls */
  ierr = PetscFwkRegisterComponentWithID(fwk, clienturl, &clientid); CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponentWithID(fwk, serverurl, &serverid); CHKERRQ(ierr);

  /*
    Add the dependency edge to the dependence_graph as follows (clienturl, serverurl): 
     this means "client depends on server", so server should be configured first.
    For this reason we need to order components using an "antitopological" sort of the dependence_graph.
    BOOST Graph Library does just that, but keep that in mind if reimplementing the graph/sort in C.
  */
  vertex_type c,s;
  c = (*(fwk->vertex))[clientid];
  s = (*(fwk->vertex))[serverid];
  ::boost::add_edge(c,s, *(fwk->dependence_graph));
  PetscFunctionReturn(0);
}/* PetscFwkRegisterDependence()*/



#undef  __FUNCT__
#define __FUNCT__ "PetscFwkDestroy"
PetscErrorCode PETSC_DLLEXPORT PetscFwkDestroy(PetscFwk fwk)
{
  PetscErrorCode ierr;
  // >> C++
  delete fwk->id;
  delete fwk->url;             
  delete fwk->configure;        
  delete fwk->component;        
  delete fwk->vertex;
  delete fwk->dependence_graph;
  // << C++
  ierr = PetscHeaderDestroy(fwk);CHKERRQ(ierr);
  fwk = PETSC_NULL;
  PetscFunctionReturn(0);
}/* PetscFwkDestroy()*/

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkCreate"
PetscErrorCode PETSC_DLLEXPORT PetscFwkCreate(MPI_Comm comm, PetscFwk *framework){
  PetscFwk fwk;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /*#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)*/
  ierr = PetscFwkInitializePackage(PETSC_NULL);CHKERRQ(ierr);
  /*#endif*/
  PetscValidPointer(framework,2);
  ierr = PetscHeaderCreate(fwk,_p_PetscFwk,PetscInt,PETSC_FWK_COOKIE,0,"PetscFwk",comm,PetscFwkDestroy,0);CHKERRQ(ierr);
  // >> C++
  fwk->id               = new std::map<std::string, PetscInt>;
  fwk->url              = new std::vector<std::string>;
  fwk->configure        = new std::vector<PetscFwkComponentConfigure>;
  fwk->component        = new std::vector<PetscObject>;
  fwk->vertex           = new std::vector<vertex_type>;
  fwk->dependence_graph = new dependence_graph_type;
  // << C++
  *framework = fwk;
  PetscFunctionReturn(0);
}/* PetscFwkCreate() */

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkGetComponentByID"
PetscErrorCode PETSC_DLLEXPORT PetscFwkGetComponentByID(PetscFwk fwk, PetscInt id, PetscObject *component) {
  PetscFunctionBegin;
  // >> C++
  try{
    *component = (*fwk->component)[id];
  }
  catch(...){
    SETERRQ1(PETSC_ERR_ARG_WRONG, "Couldn't retrieve PetscFwk component with id %d", id);
  }
  // << C++
  PetscFunctionReturn(0);
}/* PetscFwkGetComponentByID() */

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkGetComponent"
PetscErrorCode PETSC_DLLEXPORT PetscFwkGetComponent(PetscFwk fwk, const char url[], PetscObject *component) {
  PetscFunctionBegin;
  PetscValidCharPointer(url,2);
  // >> C++
  try{
    *component = (*fwk->component)[(*fwk->id)[url]];
  }
  catch(...){
    SETERRQ1(PETSC_ERR_ARG_WRONG, "Couldn't retrieve PetscFwk component with url %s", url);
  }
  // << C++
  PetscFunctionReturn(0);
}/* PetscFwkGetComponent() */

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkFinalizePackage"
PetscErrorCode PetscFwkFinalizePackage(void){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscFwkPackageInitialized = PETSC_FALSE;
  if(PetscFwkDLList != PETSC_NULL) {
    ierr = PetscDLLibraryClose(PetscFwkDLList); CHKERRQ(ierr);
    PetscFwkDLList = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}/* PetscFwkFinalizePackage() */



#undef  __FUNCT__
#define __FUNCT__ "PetscFwkInitializePackage"
PetscErrorCode PetscFwkInitializePackage(const char path[]){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(PetscFwkPackageInitialized) {
    PetscFwkPackageInitialized = PETSC_TRUE;
  }
  /* Regster classes */
  ierr = PetscCookieRegister(PETSC_FWK_CLASS_NAME, &PETSC_FWK_COOKIE); CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscFwkFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkInitializePackage() */



