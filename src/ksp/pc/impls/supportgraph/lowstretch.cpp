#define PETSCKSP_DLL

#include <math.h>
#include <queue>
#include "private/pcimpl.h"   /*I "petscpc.h" I*/
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/subgraph.hpp"

using namespace boost;

/*
  Boost Graph type definitions
*/

enum edge_length_t { edge_length };
enum edge_keep_t { edge_keep };
enum vertex_parent_t { vertex_parent };
enum vertex_children_t { vertex_children };
enum vertex_depth_t { vertex_depth };
namespace boost {
  BOOST_INSTALL_PROPERTY(edge, length);
  BOOST_INSTALL_PROPERTY(edge, keep);
  BOOST_INSTALL_PROPERTY(vertex, parent);
  BOOST_INSTALL_PROPERTY(vertex, children);
  BOOST_INSTALL_PROPERTY(vertex, depth);
}


typedef property<vertex_parent_t, PetscInt, 
		 property<vertex_children_t, std::vector<PetscInt>,
			  property<vertex_depth_t, PetscScalar, 
				   property<vertex_index_t, PetscInt> > > > VertexProperty;
typedef property<edge_length_t, PetscScalar,
		 property<edge_keep_t, PetscTruth, 
			  property<edge_index_t, PetscInt> > >  EdgeProperty2;
typedef property<edge_weight_t, PetscScalar, EdgeProperty2> EdgeProperty;
// I wish I knew a better way to make a convenient edge property constructor
#define EDGE_PROPERTY(WEIGHT,LENGTH,KEEP) EdgeProperty(WEIGHT,EdgeProperty2(LENGTH,KEEP))

typedef subgraph<adjacency_list<vecS, vecS, undirectedS, VertexProperty, EdgeProperty> > Graph; 
typedef graph_traits<Graph>::edge_descriptor Edge;

typedef property_map<Graph, edge_weight_t>::type EdgeWeight;
typedef property_map<Graph, edge_length_t>::type EdgeLength;
typedef property_map<Graph, edge_keep_t>::type EdgeKeep;
typedef property_map<Graph, edge_index_t>::type EdgeIndex;
typedef property_map<Graph, vertex_parent_t>::type VertexParent;
typedef property_map<Graph, vertex_children_t>::type VertexChildren;
typedef property_map<Graph, vertex_depth_t>::type VertexDepth;

typedef std::pair<PetscInt,PetscInt> PetscIntPair;
struct Component {
  //  static PetscInt next_id;
  //static PetscInt max_id;

  //PetscInt id;
  std::vector<PetscInt> vertices; /* ordered s.t. root is first; parent precedes child */
  /*
  Component() {
    id = next_id++;
  }
  */

};
struct ComponentPair {
  Component *first;
  Component *second;
  std::vector<Edge> edges; // pointing from first to second
  std::vector<std::pair<PetscScalar,PetscScalar> > lengthBelow;
  std::pair<PetscScalar,PetscScalar> rootLengthBelow;
  std::pair<PetscScalar,PetscScalar> rootCongestion;

  ComponentPair() {
    first = PETSC_NULL;
    second = PETSC_NULL;
  }

  int getIndex(Component *c) {
    if (first == c)
      return 0;
    else if (second == c)
      return 1;
    else
      return -1;
  }

  Component* get(int i) {
    return (i==0)?first:second;
  }

  void put(int i,Component *c) {
    if (i==0)
      first=c;
    else
      second=c;
  }

  bool match(Component *c1, Component *c2) {
    return (first == c1 && second == c2) || (first == c2 && second == c1);
  }
};

/* ShortestPathPriorityQueue is a priority queue in which each node (PQNode)
   represents a potential shortest path to a vertex.  Each node stores
   the terminal vertex, the distance along the path, and (optionally) 
   the vertex (pred) adjacent to the terminal vertex.
   The top node of the queue is the shortest path in the queue.
*/

struct PQNode {
  PetscInt vertex;
  PetscInt pred;
  PetscReal dist;

  PQNode() {}

  PQNode(const PetscInt v,const PetscReal d) {
    vertex = v;
    dist = d;
  }

  PQNode(const PetscInt v,const PetscInt p,const PetscReal d) {
    vertex = v;
    pred = p;
    dist = d;
  }

  bool operator<( const PQNode &a ) const {
    return dist > a.dist;
  }
};
typedef std::priority_queue<PQNode> ShortestPathPriorityQueue;

/*
  Function headers
*/
PetscErrorCode LowStretchSpanningTree(Mat mat,Mat *pre,
				      PetscReal tol,PetscReal& maxCong);
PetscErrorCode AugmentedLowStretchSpanningTree(Mat mat,Mat *pre,PetscTruth augment,
					       PetscReal tol,PetscReal& maxCong);
PetscErrorCode LowStretchSpanningTreeHelper(Graph& g,const PetscInt root,const PetscScalar alpha,PetscInt perm[]);
PetscErrorCode StarDecomp(const Graph g,const PetscInt root,const PetscScalar delta,const PetscScalar epsilon,
			  PetscInt& k,std::vector<PetscInt>& size,std::vector<std::vector<PetscInt> >& idx,
			  std::vector<PetscInt>& x,std::vector<PetscInt>& y);
PetscErrorCode AugmentSpanningTree(Graph& g,const PetscInt root,PetscScalar& maxCong);
PetscErrorCode DecomposeSpanningTree(Graph& g,const PetscInt root,
				     const PetscScalar maxInternalStretch,
				     const PetscScalar maxExternalWeight,
				     std::vector<Component*>& componentList,
				     std::vector<ComponentPair>& edgeComponentMap);
PetscErrorCode DecomposeSubTree(Graph& g,const PetscInt root,
				const PetscScalar maxInternalStretch,
				const PetscScalar maxExternalWeight,
				std::vector<Component*>& componentList,
				std::vector<ComponentPair>& edgeComponentMap,
				Component*& currComponent,
				PetscScalar& currInternalStretch,
				PetscScalar& currExternalWeight);
PetscErrorCode AddBridges(Graph& g,
			  std::vector<Component*>& componentList,
			  std::vector<ComponentPair>& edgeComponentMap,
			  PetscScalar& maxCong);


/* -------------------------------------------------------------------------- */
/*
   LowStretchSpanningTree - Applies EEST algorithm to construct 
                            low-stretch spanning tree preconditioner
                            

   Input Parameters:
.  mat - input matrix

   Output Parameter:
.  pre - preconditioner matrix with cholesky factorization precomputed in place
 */
#undef __FUNCT__  
#define __FUNCT__ "LowStretchSpanningTree"
PetscErrorCode LowStretchSpanningTree(Mat mat,Mat *prefact,
				      PetscReal tol,PetscReal& maxCong)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;

  ierr = AugmentedLowStretchSpanningTree(mat,prefact,PETSC_FALSE,tol,maxCong);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   AugmentedLowStretchSpanningTree - Applies EEST algorithm to construct 
                                     low-stretch spanning tree preconditioner;
				     then augments tree with additional edges
                            

   Input Parameters:
.  mat - input matrix
.  augment - augmenting options

   Output Parameter:
.  pre - preconditioner matrix with cholesky factorization precomputed in place
 */
#undef __FUNCT__  
#define __FUNCT__ "AugmentedLowStretchSpanningTree"
PetscErrorCode AugmentedLowStretchSpanningTree(Mat mat,Mat *prefact,PetscTruth augment,
					       PetscReal tol,PetscReal& maxCong)
{
#ifndef PETSC_USE_COMPLEX
  PetscInt          *idx;
  PetscInt          start,end,ncols,i,j,k;
  MatFactorInfo     info;
  // IS                perm, iperm;
  const PetscInt    *cols_c;
  const PetscScalar *vals_c;
  PetscInt          *rows, *cols, *dnz, *onz;
  PetscScalar       *vals, *diag, absval;
  Mat               *pre;
  graph_traits<Graph>::out_edge_iterator e, e_end;
  const PetscInt    root = 0;
#endif
  PetscInt          n;
  PetscErrorCode    ierr;
  PetscFunctionBegin;

  ierr = MatGetSize(mat,PETSC_NULL,&n);CHKERRQ(ierr);

  Graph g(n);

  EdgeKeep edge_keep_g = get(edge_keep_t(),g);

#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_ERR_SUP, "Complex numbers not supported for support graph PC");
#else
  ierr = PetscMalloc3(n,PetscScalar,&diag,n,PetscInt,&dnz,n,PetscInt,&onz);CHKERRQ(ierr);
  ierr = PetscMemzero(dnz, n * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(onz, n * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat, &start, &end);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = MatGetRow(mat,i,&ncols,&cols_c,&vals_c);CHKERRQ(ierr);
    diag[i] = 0;
    for (k=0; k<ncols; k++) {
      if (cols_c[k] == i) {
        diag[i] += vals_c[k];
      } else if (vals_c[k] != 0) {
	absval = vals_c[k]>0?vals_c[k]:-vals_c[k];
	diag[i] -= absval;
	if (cols_c[k] > i && absval > tol) { 
	  // we set edge_weight = absval; edge_length = 1.0/absval; edge_keep = PETSC_FALSE
	  add_edge(i,cols_c[k],EDGE_PROPERTY(absval,1.0/absval,PETSC_FALSE),g);
	}
      }
    }
    ierr = MatRestoreRow(mat,i,&ncols,&cols_c,&vals_c);CHKERRQ(ierr);
  }

  ierr = PetscMalloc(n*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  put(get(vertex_depth_t(),g),root,0);
  ierr = LowStretchSpanningTreeHelper(g,root,log(4.0/3.0)/(2.0*log((double)n)),idx);CHKERRQ(ierr);

  if (augment) {
    ierr = AugmentSpanningTree(g,root,maxCong);CHKERRQ(ierr);
  }

  pre = prefact;
  ierr = MatCreate(PETSC_COMM_WORLD,pre);CHKERRQ(ierr);
  ierr = MatSetSizes(*pre,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetType(*pre,MATAIJ);CHKERRQ(ierr);

  ierr = PetscMalloc3(1,PetscInt,&rows,n,PetscInt,&cols,n,PetscScalar,&vals);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    for (tie(e, e_end) = out_edges(i,g); e != e_end; e++) {
      if (get(edge_keep_g,*e)) {
        const PetscInt col =  target(*e,g);

        if (col >= start && col < end) {
          dnz[i]++;
        } else {
          onz[i]++;
        }
      }
    }
  }
  ierr = MatSeqAIJSetPreallocation(*pre, 0, dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*pre, 0, dnz, 0, onz);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    rows[0] = i;
    k = 0;
    for (tie(e, e_end) = out_edges(i,g); e != e_end; e++) {
      if (get(edge_keep_g,*e)) {
        cols[k++] = target(*e,g);
      }
    }
    MatGetValues(mat,1,rows,k,cols,vals);
    for (j=0; j<k; j++) {
      absval = vals[j]>0?vals[j]:-vals[j];
      diag[i] += absval;
    }
    cols[k] = i;
    vals[k] = diag[i];
    MatSetValues(*pre,1,rows,k+1,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree3(rows,cols,vals);CHKERRQ(ierr);
  ierr = PetscFree3(diag,dnz,onz);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*pre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*pre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,idx,&perm);CHKERRQ(ierr);
    ierr = ISSetPermutation(perm);CHKERRQ(ierr);
    ierr = ISInvertPermutation(perm,PETSC_DECIDE,&iperm);CHKERRQ(ierr);
    ierr = PetscFree(idx);CHKERRQ(ierr);
    ierr = ISView(perm,PETSC_VIEWER_STDOUT_SELF);
  */
  IS rperm, cperm;
  ierr = MatGetOrdering(*pre,MATORDERING_QMD,&rperm,&cperm);CHKERRQ(ierr);

  ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
  /*
  ierr = MatCholeskyFactorSymbolic(*pre,iperm,&info,prefact);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(*pre,&info,prefact);CHKERRQ(ierr);
  ierr = MatDestroy(*pre);CHKERRQ(ierr);
  ierr = ISDestroy(perm);CHKERRQ(ierr);
  ierr = ISDestroy(iperm);CHKERRQ(ierr);
  */
  ierr = MatLUFactor(*pre,rperm,cperm,&info);CHKERRQ(ierr);
  ierr = ISDestroy(rperm);CHKERRQ(ierr);
  ierr = ISDestroy(cperm);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   LowStretchSpanningTreeHelper

   Input Parameters:
.  g - input graph; all edges have edge_keep = PETSC_FALSE;
                    root has vertex_depth set to distance from global root
.  root - root vertex
.  alpha - parameter
.  perm - preallocated array of size num_vertices(g) in which to store vertex ordering

   Output Parameter:
.  g - edges in low-stretch spanning tree are marked with edge_keep = PETSC_TRUE;
       also vertex_parent and vertex_children are set (vertex_parent of global root is undefined)
       and vertex_depth is set to be distance from global root (where weight on edge is inverse distance)
.  perm - list of vertices in which a vertex precedes its parent (with vertices referred to by global index)
 */
#undef __FUNCT__  
#define __FUNCT__ "LowStretchSpanningTreeHelper"
PetscErrorCode LowStretchSpanningTreeHelper(Graph& g,const PetscInt root,const PetscScalar alpha,PetscInt perm[])
{
  PetscInt n,i,j,k;
  std::vector<PetscInt> size,x,y;
  std::vector<std::vector<PetscInt> > idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  EdgeLength edge_length_g = get(edge_length_t(),g);
  EdgeKeep edge_keep_g = get(edge_keep_t(),g);
  VertexParent vertex_parent_g = get(vertex_parent_t(),g);
  VertexChildren vertex_children_g = get(vertex_children_t(),g);
  VertexDepth vertex_depth_g = get(vertex_depth_t(),g);
  n = num_vertices(g);

  if (n > 2) {
    ierr = StarDecomp(g,root,1.0/3,alpha,k,size,idx,x,y);CHKERRQ(ierr);
    j = n - size[0];
    Graph& g1 = g.create_subgraph(idx[0].begin(),idx[0].end());
    ierr = LowStretchSpanningTreeHelper(g1,g1.global_to_local(g.local_to_global(root)),alpha,perm+j);CHKERRQ(ierr);
    for (i=1;i<=k;i++) {
      Edge e = edge(x[i-1],y[i-1],g).first;
      put(edge_keep_g,e,PETSC_TRUE);
      put(vertex_parent_g,x[i-1],g.local_to_global(y[i-1]));
      get(vertex_children_g,y[i-1]).push_back(g.local_to_global(x[i-1]));
      put(vertex_depth_g,x[i-1],get(vertex_depth_g,y[i-1])+get(edge_length_g,e));

      j -= size[i];
      Graph& g1 = g.create_subgraph(idx[i].begin(),idx[i].end());
      ierr = LowStretchSpanningTreeHelper(g1,g1.global_to_local(g.local_to_global(x[i-1])),alpha,perm+j);CHKERRQ(ierr);
    }
  } else if (n == 2) {
    Edge e = *(out_edges(root,g).first);
    PetscInt t = target(e,g);

    put(edge_keep_g,e,PETSC_TRUE);
    put(vertex_parent_g,t,g.local_to_global(root));
    get(vertex_children_g,root).push_back(g.local_to_global(t));
    put(vertex_depth_g,t,get(vertex_depth_g,root)+get(edge_length_g,e));

    perm[0] = g.local_to_global(t);
    perm[1] = g.local_to_global(root);
  } else /* n == 1 */ {
    perm[0] = g.local_to_global(root);
  }

  PetscFunctionReturn(0);
}



/* -------------------------------------------------------------------------- */
/*
   StarDecomp - calculate a star decomposition of the graph

   Input Parameters:
.  g - input graph
.  root - center of star decomposition
.  delta, epsilon - parameters of star decomposition

   Output Parameter:
.  k - number of partitions, not-including central one.
.  size[i] - number of vertices in partition i
.  idx[i] - list of vertices in partition i; partition 0 contains root
.  x[i-1] - "anchor" vertex of non-central partition i
.  y[i-i] - vertex in partition 0 forming "bridge" with x[i-1]
 */
#undef __FUNCT__  
#define __FUNCT__ "LowStretchSpanningTreeHelper"
PetscErrorCode StarDecomp(Graph g,const PetscInt root,const PetscScalar delta,const PetscScalar epsilon,
			  PetscInt& k,std::vector<PetscInt>& size,std::vector<std::vector<PetscInt> >& idx,
			  std::vector<PetscInt>& x,std::vector<PetscInt>& y)
{
#ifndef PETSC_USE_COMPLEX
  PetscInt n,m,edgesLeft;
  //PetscErrorCode ierr;
  ShortestPathPriorityQueue pq;
  PetscScalar radius;
  PetscInt centerSize;
  std::vector<PetscInt> centerIdx;
  PQNode node;
#endif

  PetscFunctionBegin;
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_ERR_SUP, "Complex numbers not supported for support graph PC");
#else
  EdgeWeight edge_weight_g = get(edge_weight_t(),g);
  EdgeLength edge_length_g = get(edge_length_t(),g);
  n = num_vertices(g);
  m = num_edges(g);
  edgesLeft = m;

  std::vector<PetscInt> pred(n,-1);
  std::vector<PetscInt> *succ = new std::vector<PetscInt>[n];
  std::vector<PetscInt>::iterator i;
  PetscScalar *dist = new PetscScalar[n];
  std::vector<PetscTruth> taken(n,PETSC_FALSE);

  /** form tree of shortest paths to root **/
  graph_traits<Graph>::out_edge_iterator e, e_end;  
  for (tie(e,e_end)=out_edges(root,g); e!=e_end; e++) {
    PetscInt t = target(*e,g);
    pq.push(PQNode(t,root,get(edge_length_g,*e)));
  }
  pred[root] = root;
  while (!pq.empty()) {
    node = pq.top();pq.pop();
    if (pred[node.vertex] == -1) {
      succ[node.pred].push_back(node.vertex);
      pred[node.vertex] = node.pred;
      dist[node.vertex] = node.dist;
      for (tie(e,e_end)=out_edges(node.vertex,g); e!=e_end; e++) {
	PetscInt t = target(*e,g);
	if (pred[t] == -1) {
	  pq.push(PQNode(t,node.vertex,node.dist+get(edge_length_g,*e)));
	}
      }
      radius = node.dist;
    }
  }

  /** BALL CUT **/
  for (i=succ[root].begin();i!=succ[root].end();i++) {
    pq.push(PQNode(*i,dist[*i]));
  }
  PetscScalar boundary = 0;
  PetscInt edgeCount = 0;
  centerIdx.push_back(g.local_to_global(root));
  taken[root] = PETSC_TRUE;
  centerSize = 1;
  for (tie(e,e_end)=out_edges(root,g); e!=e_end; e++) {
    boundary += get(edge_weight_g,*e);
    edgeCount++;
  }
  const PetscScalar minRadius = delta*radius;
  while (dist[pq.top().vertex] < minRadius) {
    assert(!pq.empty());
    node = pq.top();pq.pop();
    centerIdx.push_back(g.local_to_global(node.vertex));
    taken[node.vertex] = PETSC_TRUE;
    centerSize++;
    for (tie(e,e_end)=out_edges(node.vertex,g); e!=e_end; e++) {
      if (taken[target(*e,g)]) {
	boundary -= get(edge_weight_g,*e);
      } else {
	boundary += get(edge_weight_g,*e);
	edgeCount++;
      }
    }
    for (i=succ[node.vertex].begin();i!=succ[node.vertex].end();i++) {
      pq.push(PQNode(*i,dist[*i]));
    }
  }
  while (boundary > (edgeCount+1)*log((double)m)/(log(2.0)*(1.0-2.0*delta)*radius)) {
    assert(!pq.empty());
    node = pq.top();pq.pop();
    centerIdx.push_back(g.local_to_global(node.vertex));
    taken[node.vertex] = PETSC_TRUE;
    centerSize++;
    for (tie(e,e_end)=out_edges(node.vertex,g); e!=e_end; e++) {
      if (taken[target(*e,g)]) {
	boundary -= get(edge_weight_g,*e);
      } else {
	boundary += get(edge_weight_g,*e);
	edgeCount++;
      }
    }
    for (i=succ[node.vertex].begin();i!=succ[node.vertex].end();i++) {
      pq.push(PQNode(*i,dist[*i]));
    }
  }
  size.push_back(centerSize);
  idx.push_back(centerIdx);
  edgesLeft -= edgeCount;

  k = 0;
  assert(!pq.empty());
  std::queue<PetscInt> anchor_q;
  ShortestPathPriorityQueue cone_pq;
  std::vector<PetscInt> *cone_succ = new std::vector<PetscInt>[n];
  std::vector<PetscTruth> cone_found(n,PETSC_FALSE);

  /** form tree of shortest paths to an anchor **/
  while (!pq.empty()) {
    node = pq.top();pq.pop();
    cone_found[node.vertex] = PETSC_TRUE;
    anchor_q.push(node.vertex);
    for (tie(e,e_end)=out_edges(node.vertex,g); e!=e_end; e++) {
      PetscInt t = target(*e,g);
      if (!taken[t]) {
	cone_pq.push(PQNode(t,node.vertex,get(edge_length_g,*e)));
      }
    }
  }
  while (!cone_pq.empty()) {
    node = cone_pq.top();cone_pq.pop();
    if (!cone_found[node.vertex]) {
      cone_succ[node.pred].push_back(node.vertex);
      cone_found[node.vertex] = PETSC_TRUE;
      for (tie(e,e_end)=out_edges(node.vertex,g); e!=e_end; e++) {
	PetscInt t = target(*e,g);
	if (!taken[t] && !cone_found[t]) {
	  cone_pq.push(PQNode(t,node.vertex,node.dist+get(edge_length_g,*e)));
	}
      }
    }
  }

  while (!anchor_q.empty()) {
    /** CONE CUT **/
    PetscInt anchor = anchor_q.front();anchor_q.pop();
    if (!taken[anchor]) {
      PetscInt v;
      PetscInt thisSize = 0;
      std::vector<PetscInt> thisIdx;
      std::queue<PetscInt> q;
      ShortestPathPriorityQueue mycone_pq;
      std::vector<PetscTruth> mycone_taken(n,PETSC_FALSE);
      PetscInt initialInternalConeEdges = 0;

      boundary = 0;
      edgeCount = 0;
      q.push(anchor);
      while (!q.empty()) {
	v = q.front();q.pop();
	taken[v] = PETSC_TRUE;
	mycone_taken[v] = PETSC_TRUE;
	thisIdx.push_back(g.local_to_global(v));
	thisSize++;
	for (i=cone_succ[v].begin();i!=cone_succ[v].end();i++) {
	  q.push(*i);
	}
	for (tie(e,e_end)=out_edges(v,g); e!=e_end; e++) {
	  PetscInt t = target(*e,g);
	  if (!taken[t]) {
	    mycone_pq.push(PQNode(t,v,get(edge_length_g,*e)));
	    boundary += get(edge_weight_g,*e);
	    edgeCount++;
	  } else if (mycone_taken[t]) {
	    boundary -= get(edge_weight_g,*e);
	    initialInternalConeEdges++;
	  }
	}
      }
      if (initialInternalConeEdges < edgesLeft) {
	while (initialInternalConeEdges == 0 ?
	       boundary > (edgeCount+1)*log((double)(edgesLeft+1))*2.0/(log(2.0)*epsilon*radius) : 
	       boundary > (edgeCount)*log(edgesLeft*1.0/initialInternalConeEdges)*2.0/(log(2.0)*epsilon*radius))
	  {
	    assert(!mycone_pq.empty());
	    node = mycone_pq.top();mycone_pq.pop();
	    if (!mycone_taken[node.vertex]) {
	      q.push(node.vertex);
	      while (!q.empty()) {
		v = q.front();q.pop();
		taken[v] = PETSC_TRUE;
		mycone_taken[v] = PETSC_TRUE;
		thisIdx.push_back(g.local_to_global(v));
		thisSize++;
		for (i=cone_succ[v].begin();i!=cone_succ[v].end();i++) {
		  q.push(*i);
		}
		for (tie(e,e_end)=out_edges(v,g); e!=e_end; e++) {
		  PetscInt t = target(*e,g);
		  if (!taken[t]) {
		    mycone_pq.push(PQNode(t,v,node.dist+get(edge_length_g,*e)));
		    boundary += get(edge_weight_g,*e);
		    edgeCount++;
		  } else if (mycone_taken[t]) {
		    boundary -= get(edge_weight_g,*e);
		  }
		}
	      }
	    }
	  }
      }
      edgesLeft -= edgeCount;
      size.push_back(thisSize);
      idx.push_back(thisIdx);
      x.push_back(anchor);
      y.push_back(pred[anchor]);
      k++;
    }
  }
#endif    
  

  /*
  // pseudo cone cut
  while (!pq.empty()) {
    node = pq.top();pq.pop();

    PetscInt thisSize = 1;
    std::vector<PetscInt> thisIdx;
    std::queue<PetscInt> q;

    thisIdx.push_back(g.local_to_global(node.vertex));
    for (i=succ[node.vertex].begin();i!=succ[node.vertex].end();i++) {
      q.push(*i);
    }

    PetscInt v;
    while (!q.empty()) {
      v = q.front();q.pop();
      thisSize++;
      thisIdx.push_back(g.local_to_global(v));
      for (i=succ[v].begin();i!=succ[v].end();i++) {
	q.push(*i);
      }
    }
    size.push_back(thisSize);
    idx.push_back(thisIdx);
    x.push_back(node.vertex);
    y.push_back(pred[node.vertex]);
    k++;
  }
  */
#ifndef PETSC_USE_COMPLEX
  delete [] succ;
  delete [] dist;
  delete [] cone_succ;
#endif
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   AugmentSpanningTree

   Input Parameters:
.  g - input graph with spanning tree defined by vertex_parent and vertex_children;
       vertex_depth gives distance in tree from root
.  maxCong - target upper bound on congestion

   Output Parameter:
.  g - edges in augmented spanning tree are marked with edge_keep = PETSC_TRUE
.  maxCong - an actual upper bound on congestion
 */
#undef __FUNCT__  
#define __FUNCT__ "AugmentSpanningTree"
PetscErrorCode AugmentSpanningTree(Graph& g,const PetscInt root,PetscScalar& maxCong)
{
  //  const PetscInt n = num_vertices(g);
  const PetscInt m = num_edges(g);
  //PetscInt i;
  PetscErrorCode ierr;
  //PetscInt *component;  // maps each vertex to a vertex component
  //std::vector<PetscScalar>
  //  maxCongestion;       /* maps each edge component to an upper bound on the
  //			    congestion through any of its edges */

  const EdgeIndex edge_index_g = get(edge_index_t(),g);

  PetscFunctionBegin;

  std::vector<Component*> componentList;
  std::vector<ComponentPair> edgeComponentMap(m);

  ierr = DecomposeSpanningTree(g,root,maxCong,maxCong,
			       componentList,edgeComponentMap);CHKERRQ(ierr);
  /*
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"COMPONENTS:\n");
  for (int i=0; i<componentList.size(); i++) {
    for (int j=0; j<componentList[i]->vertices.size(); j++) {
      ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"%d ",componentList[i]->vertices[j]);
    }
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"EDGES: ");

    graph_traits<Graph>::edge_iterator e, e_end;  
    for (tie(e,e_end)=edges(g); e!=e_end; e++) {
      if (edgeComponentMap[get(edge_index_g,*e)].getIndex(componentList[i]) != -1) {
	ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"(%d,%d) ",source(*e,g),target(*e,g));
      }
    }
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"\n");
  }
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"END COMPONENTS\n");
  */
  ierr = AddBridges(g,componentList,edgeComponentMap,maxCong);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



/* -------------------------------------------------------------------------- */
/*
   DecomposeSpanningTree

   Input Parameters:
.  g - input graph with spanning tree defined by vertex_parent and vertex_children;
       vertex_depth gives distance in tree from root
.  component - a preallocated array of length num_vertices(g)
.  edgeComponent - a preallocated vector of length num_edges(g)

   Output Parameter:
.  component - a vector mapping each vertex to a component
 */
#undef __FUNCT__  
#define __FUNCT__ "DecomposeSpanningTree"
PetscErrorCode DecomposeSpanningTree(Graph& g,const PetscInt root,
				     const PetscScalar maxInternalStretch,
				     const PetscScalar maxExternalWeight,
				     std::vector<Component*>& componentList,
				     std::vector<ComponentPair>& edgeComponentMap)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  Component* currComponent;
  PetscScalar currInternalStretch, currExternalWeight;

  //Component::next_id = 0;
  //Component::max_id = num_edges(g);
  ierr = DecomposeSubTree(g,root,
			  maxInternalStretch,maxExternalWeight,
			  componentList,edgeComponentMap,
			  currComponent,
			  currInternalStretch,
			  currExternalWeight);CHKERRQ(ierr);
  
  componentList.push_back(currComponent);

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DecomposeSubTree"
PetscErrorCode DecomposeSubTree(Graph& g,const PetscInt root,
				const PetscScalar maxInternalStretch,
				const PetscScalar maxExternalWeight,
				std::vector<Component*>& componentList,
				std::vector<ComponentPair>& edgeComponentMap,
				Component*& currComponent,
				PetscScalar& currInternalStretch,
				PetscScalar& currExternalWeight)
{
#ifndef PETSC_USE_COMPLEX
  const EdgeWeight edge_weight_g = get(edge_weight_t(),g);
  const EdgeIndex edge_index_g = get(edge_index_t(),g);
  const EdgeKeep edge_keep_g = get(edge_keep_t(),g);
  const VertexParent vertex_parent_g = get(vertex_parent_t(),g);
  const VertexChildren vertex_children_g = get(vertex_children_t(),g);
  const VertexDepth vertex_depth_g = get(vertex_depth_t(),g);
  const PetscScalar rootDepth = get(vertex_depth_g,root);
  std::vector<PetscInt>::const_iterator i,j;
  graph_traits<Graph>::out_edge_iterator e, e_end;  
  PetscErrorCode ierr;
  PetscScalar newInternalStretch, newExternalWeight;
  PetscInt v,w,edgeIndex,compIndex;
  PetscScalar weight;
#endif

  PetscFunctionBegin;
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_ERR_SUP, "Complex numbers not supported for support graph PC");
#else
  currComponent = new Component();
  currComponent->vertices.push_back(root);
  currInternalStretch = 0;
  currExternalWeight = 0;
  // temporarily add all external edges to component, but may remove some at end
  for (tie(e,e_end)=out_edges(root,g); e!=e_end; e++) {
    if (!get(edge_keep_g,*e)) {
      edgeIndex = get(edge_index_g,*e);
      
      if (edgeComponentMap[edgeIndex].get(0) == PETSC_NULL) {
	edgeComponentMap[edgeIndex].put(0,currComponent);
      } else {
	assert(edgeComponentMap[edgeIndex].get(1) == PETSC_NULL);
	edgeComponentMap[edgeIndex].put(1,currComponent);
      }
    }
  }

  std::vector<PetscInt> children = get(vertex_children_g,root);

  Component *childComponent;
  PetscScalar childInternalStretch, childExternalWeight;
  for (i=children.begin();i!=children.end();i++) {
    PetscInt child = *i;
    ierr = DecomposeSubTree(g,child,maxInternalStretch,maxExternalWeight,
			    componentList,edgeComponentMap,
			    childComponent,
			    childInternalStretch,childExternalWeight);CHKERRQ(ierr);

    newInternalStretch = currInternalStretch + childInternalStretch;
    newExternalWeight = currExternalWeight;

    for (j = childComponent->vertices.begin(), v = *j; 
	 j != childComponent->vertices.end() && (newInternalStretch <= maxInternalStretch); 
	 v = *(++j)) {
      for (tie(e,e_end)=out_edges(v,g); 
	   e!=e_end && (newInternalStretch <= maxInternalStretch); 
	   e++) {
	if (!get(edge_keep_g,*e)) {
	  w = target(*e,g);
	  edgeIndex = get(edge_index_g,*e);
	  compIndex = edgeComponentMap[edgeIndex].getIndex(childComponent);
	  
	  if (compIndex != -1) {
	    weight = get(edge_weight_g,*e);
	    
	    if (edgeComponentMap[edgeIndex].get(1-compIndex) == currComponent) {
	      newExternalWeight -= weight;
	      newInternalStretch += 
		(get(vertex_depth_g,v) + get(vertex_depth_g,w) - 2*rootDepth) * weight;
	    } else {
	      newExternalWeight += weight;
	    }
	  }
	}
      }
    }

    if (newInternalStretch <= maxInternalStretch && newExternalWeight <= maxExternalWeight) {
      // merge the components

      currInternalStretch = newInternalStretch;
      currExternalWeight = newExternalWeight;

      for (j = childComponent->vertices.begin(), v = *j; 
	   j != childComponent->vertices.end(); 
	   v = *(++j)) {
	currComponent->vertices.push_back(v);
	for (tie(e,e_end)=out_edges(v,g); e!=e_end; e++) {
	  if (!get(edge_keep_g,*e)) {
	    edgeIndex = get(edge_index_g,*e);
	    if (edgeComponentMap[edgeIndex].get(0) == childComponent) {
	      edgeComponentMap[edgeIndex].put(0,currComponent);
	    }
	    if (edgeComponentMap[edgeIndex].get(1) == childComponent) {
	      edgeComponentMap[edgeIndex].put(1,currComponent);
	    }
	  }
	}
      }
      delete childComponent;
    } else {
      componentList.push_back(childComponent);
    }
  }

  const Component *origCurrComponent = currComponent;
  for (tie(e,e_end)=out_edges(root,g); e!=e_end; e++) {
    edgeIndex = get(edge_index_g,*e);
    if (!get(edge_keep_g,*e)) {
      if (edgeComponentMap[edgeIndex].get(0) == origCurrComponent) {
	compIndex = 0;
      } else {
	assert(edgeComponentMap[edgeIndex].get(1) == origCurrComponent);
	compIndex = 1;
      }
      
      if (edgeComponentMap[edgeIndex].get(1-compIndex) != origCurrComponent) {
	weight = get(edge_weight_g,*e);
	if (currExternalWeight + weight <= maxExternalWeight) {
	  currExternalWeight += weight;
	} else {
	  componentList.push_back(currComponent);
	  currComponent = new Component();
	  currComponent->vertices.push_back(root);
	  currInternalStretch = 0;
	  currExternalWeight = 0;
	}
	edgeComponentMap[edgeIndex].put(compIndex,currComponent);
      }
    }
  }
#endif
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "AddBridges"
PetscErrorCode AddBridges(Graph& g,
			  std::vector<Component*>& componentList,
			  std::vector<ComponentPair>& edgeComponentMap,
			  PetscScalar& maxCong) {

#ifndef PETSC_USE_COMPLEX
  const PetscInt m = num_edges(g);
  std::vector<PetscTruth> edgeSupported(m); // edgeSupported[i] if edge i's component pair has been bridged
  const EdgeLength edge_length_g = get(edge_length_t(),g);
  const EdgeWeight edge_weight_g = get(edge_weight_t(),g);
  const EdgeIndex edge_index_g = get(edge_index_t(),g);
  const EdgeKeep edge_keep_g = get(edge_keep_t(),g);
  const VertexParent vertex_parent_g = get(vertex_parent_t(),g);
  const VertexChildren vertex_children_g = get(vertex_children_t(),g);
  const VertexDepth vertex_depth_g = get(vertex_depth_t(),g);
  PetscInt edgeIndex, eeIndex;
  Component *comp1, *comp2;
  PetscInt comp1size, comp2size, i, v, w, parent;
  graph_traits<Graph>::edge_iterator e, e_end;
  graph_traits<Graph>::out_edge_iterator ee, ee_end;  
  PetscScalar realMaxCong;
#endif

  PetscFunctionBegin;
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_ERR_SUP, "Complex numbers not supported for support graph PC");
#else
  realMaxCong = 0;

  for (tie(e,e_end)=edges(g); e!=e_end; e++) {
    if (!get(edge_keep_g,*e)) {
      edgeIndex = get(edge_index_g,*e);
      comp1 = edgeComponentMap[edgeIndex].get(0);
      comp2 = edgeComponentMap[edgeIndex].get(1);
      if ((comp1 != comp2) && !edgeSupported[edgeIndex]) {
	comp1size = comp1->vertices.size();
	comp2size = comp2->vertices.size();
	std::map<PetscInt,PetscScalar> congestionBelow1,weightBelow1;
	std::map<PetscInt,PetscScalar> congestionBelow2,weightBelow2;
	for (i=0; i<comp1size; i++) {
	  congestionBelow1[comp1->vertices[i]] = 0;
	  weightBelow1[comp1->vertices[i]] = 0;
	}
	for (i=0; i<comp2size; i++) {
	  congestionBelow2[comp2->vertices[i]] = 0;
	  weightBelow2[comp2->vertices[i]] = 0;
	}

	for (i=comp1size-1; i>=0; i--) {
	  v = comp1->vertices[i];
	  for (tie(ee,ee_end)=out_edges(v,g); ee!=ee_end; ee++) {
	    if (!get(edge_keep_g,*ee)) {
	      eeIndex = get(edge_index_g,*ee);
	      if (edgeComponentMap[eeIndex].match(comp1,comp2) &&
		  weightBelow2.count(target(*ee,g)) > 0) {
		edgeSupported[eeIndex] = PETSC_TRUE;
		weightBelow1[v] += get(edge_weight_g,*ee);
	      }
	    }
	  }
	  if (i>0) {
	    parent = get(vertex_parent_g,v);
	    weightBelow1[parent] += weightBelow1[v];
	    congestionBelow1[parent] += 
	      weightBelow1[v]*(get(vertex_depth_g,v)-get(vertex_depth_g,parent));
	  }
	}
	for (i=1; i<comp1size; i++) {
	  v = comp1->vertices[i];
	  parent = get(vertex_parent_g,v);
	  congestionBelow1[v] = congestionBelow1[parent] -
	    (weightBelow1[comp1->vertices[0]] - 2*weightBelow1[v])*(get(vertex_depth_g,v)-get(vertex_depth_g,parent));
	}
	
	for (i=comp2size-1; i>=0; i--) {
	  v = comp2->vertices[i];
	  for (tie(ee,ee_end)=out_edges(v,g); ee!=ee_end; ee++) {
	    if (!get(edge_keep_g,*ee)) {
	      eeIndex = get(edge_index_g,*ee);
	      if (edgeComponentMap[eeIndex].match(comp1,comp2) &&
		  weightBelow1.count(target(*ee,g)) > 0) {
		assert(edgeSupported[eeIndex] == PETSC_TRUE);
		weightBelow2[v] += get(edge_weight_g,*ee);
	      }
	    }
	  }
	  if (i>0) {
	    parent = get(vertex_parent_g,v);
	    weightBelow2[parent] += weightBelow2[v];
	    congestionBelow2[parent] += 
	      weightBelow2[v]*(get(vertex_depth_g,v)-get(vertex_depth_g,parent));
	  }
	}
	for (i=1; i<comp2size; i++) {
	  v = comp2->vertices[i];
	  parent = get(vertex_parent_g,v);
	  congestionBelow2[v] = congestionBelow2[parent] -
	    (weightBelow2[comp2->vertices[0]] - 2*weightBelow2[v])*(get(vertex_depth_g,v)-get(vertex_depth_g,parent));
	}
	/*
	for (std::map<PetscInt,PetscScalar>::iterator it = congestionBelow1.begin();
	     it != congestionBelow1.end();
	     it++) {
	  PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"congestionBelow1[%d]=%f\n",(*it).first,(*it).second);
	}
	for (std::map<PetscInt,PetscScalar>::iterator it = weightBelow1.begin();
	     it != weightBelow1.end();
	     it++) {
	  PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"weightBelow1[%d]=%f\n",(*it).first,(*it).second);
	}
	for (std::map<PetscInt,PetscScalar>::iterator it = congestionBelow2.begin();
	     it != congestionBelow2.end();
	     it++) {
	  PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"congestionBelow2[%d]=%f\n",(*it).first,(*it).second);
	}
	for (std::map<PetscInt,PetscScalar>::iterator it = weightBelow2.begin();
	     it != weightBelow2.end();
	     it++) {
	  PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"weightBelow2[%d]=%f\n",(*it).first,(*it).second);
	}
	*/
	assert(weightBelow1[comp1->vertices[0]] - weightBelow2[comp2->vertices[0]] < 1e-9 &&
	       weightBelow1[comp1->vertices[0]] - weightBelow2[comp2->vertices[0]] > -1e-9);

	Edge bestEdge;
	PetscScalar bestCongestion = -1;
	for (i=0; i<comp1size; i++) {
	  v = comp1->vertices[i];
	  for (tie(ee,ee_end)=out_edges(v,g); ee!=ee_end; ee++) {
	    if (!get(edge_keep_g,*ee)) {
	      eeIndex = get(edge_index_g,*ee);
	      if (edgeComponentMap[eeIndex].match(comp1,comp2)) {
		w = target(*ee,g);
		PetscScalar newCongestion = 
		  weightBelow1[comp1->vertices[0]] * get(edge_length_g,*ee) +
		  congestionBelow1[v] + congestionBelow2[w];
		if (bestCongestion < 0 || newCongestion < bestCongestion) {
		  bestEdge = *ee;
		  bestCongestion = newCongestion;
		}
	      }
	    }
	  }
	}
	put(edge_keep_g,bestEdge,PETSC_TRUE);
	if (bestCongestion > realMaxCong)
	  realMaxCong = bestCongestion;
      }
    }
  }
  maxCong = realMaxCong;
#endif
  PetscFunctionReturn(0);
}

