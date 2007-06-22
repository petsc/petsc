#define PETSCKSP_DLL

#include <queue>
#include "private/pcimpl.h"   /*I "petscpc.h" I*/
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/subgraph.hpp"

using namespace boost;

/*
  Boost graph definitions
*/
struct vertex_weight_t {
  typedef vertex_property_tag kind;
};
typedef subgraph<adjacency_list<vecS, vecS, undirectedS, 
				property<vertex_weight_t, PetscScalar, property<vertex_index_t, PetscInt> >, 
				property<edge_weight_t, PetscScalar, property<edge_index_t, PetscInt> > > > Graph;
typedef property_map<Graph, vertex_weight_t>::type VertexWeight;
typedef property_map<Graph, edge_weight_t>::type EdgeWeight;
typedef std::pair<PetscInt,PetscInt> Edge;

struct PQNode {
  PetscInt vertex;
  PetscInt pred;
  PetscScalar dist;

  PQNode() {}

  PQNode(const PetscInt v,const PetscScalar d) {
    vertex = v;
    dist = d;
  }

  PQNode(const PetscInt v,const PetscInt p,const PetscScalar d) {
    vertex = v;
    pred = p;
    dist = d;
  }

  bool operator<( const PQNode &a ) const {
    return dist > a.dist;
  }
};

/*
  Function headers
*/
PetscErrorCode LowStretchSpanningTree(Mat mat,Mat *pre);
PetscErrorCode LowStretchSpanningTreeHelper(Graph& g,const PetscInt root,const PetscScalar alpha,
					    Graph& h,PetscInt perm[]);
PetscErrorCode StarDecomp(const Graph g,const PetscInt root,const PetscScalar delta,const PetscScalar epsilon,
			  PetscInt& k,std::vector<PetscInt>& size,std::vector<std::vector<PetscInt> >& idx,
			  std::vector<PetscInt>& x,std::vector<PetscInt>& y);
PetscErrorCode BallCut(const Graph g,const PetscInt root,const PetscScalar minRadius,const PetscScalar maxBoundary,PetscInt *idx[]);


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
PetscErrorCode LowStretchSpanningTree(Mat mat,Mat *prefact)
{
  PetscErrorCode    ierr;
  PetscInt          *idx;
  PetscInt          n,ncols,i,k;
  MatFactorInfo     info;
  IS                perm;
  const PetscInt    *cols;
  const PetscScalar *vals;
  Mat               pre;

  PetscFunctionBegin;

  ierr = MatGetSize(mat,PETSC_NULL,&n);CHKERRQ(ierr);

  Graph g(n),h(n);
  VertexWeight vertex_weight_h = get(vertex_weight_t(), h);
  EdgeWeight edge_weight_h = get(edge_weight_t(), h);

  for (i=0; i<n; i++) {
    PetscScalar sum = 0;
    ierr = MatGetRow(mat,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    for (k=0; k<ncols; k++) {
      if (!(-.000000001 < vals[k] && vals[k] < .000000001)) { /****** fix this ******/
	sum += vals[k];
	if (cols[k] > i) {
	  add_edge(i,cols[k],-vals[k],g);
	}
      }
    }
    put(vertex_weight_h,i,sum);
    ierr = MatRestoreRow(mat,i,&ncols,&cols,&vals);CHKERRQ(ierr);
  }

  ierr = PetscMalloc(n*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  ierr = LowStretchSpanningTreeHelper(g,0,log(4.0/3)/(2*log(n)),h,idx);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,idx,&perm);CHKERRQ(ierr);
  ierr = ISSetPermutation(perm);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);

  /*
  printf ("\nperm:\n");
  ISView(perm,PETSC_VIEWER_STDOUT_SELF);
  */

  ierr = MatCreate(PETSC_COMM_WORLD,&pre);CHKERRQ(ierr);
  ierr = MatSetSizes(pre,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetType(pre,MATAIJ);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    ierr = MatSetValue(pre,i,i,get(vertex_weight_h,i),INSERT_VALUES);CHKERRQ(ierr);
  }
  
  graph_traits<Graph>::edge_iterator e, e_end;
  for (tie(e, e_end) = edges(h); e != e_end; e++) {
    ierr = MatSetValue(pre,source(*e,h),target(*e,h),-get(edge_weight_h,*e),INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(pre,target(*e,h),source(*e,h),-get(edge_weight_h,*e),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(pre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  

  //ierr = MatDuplicate(mat,MAT_COPY_VALUES,&pre);CHKERRQ(ierr);
  

  /*

  ierr = VecCreate(&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(x,VECSEQ);CHKERRQ(ierr);
  ierr = VecCreate(&y1);CHKERRQ(ierr);
  ierr = VecSetSizes(y1,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(y1,VECSEQ);CHKERRQ(ierr);
  ierr = VecCreate(&y2);CHKERRQ(ierr);
  ierr = VecSetSizes(y2,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(y2,VECSEQ);CHKERRQ(ierr);
  ierr = VecSetRandom(x,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMult(mat,x,y1);CHKERRQ(ierr);
  ierr = MatMult(*pre,x,y2);CHKERRQ(ierr);
  */

  /*
  printf("\n----------\nOriginal matrix:\n"); 
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  printf("\n----------\nPreconditioner:\n\n"); 
  ierr = MatView(pre,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  printf("\n----------\n"); 
  */

  /*
  ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = VecView(y1,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = VecView(y2,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  */

  ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
  ierr = MatCholeskyFactorSymbolic(pre,perm,&info,prefact);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(pre,&info,prefact);CHKERRQ(ierr);
  ierr = MatDestroy(pre);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   LowStretchSpanningTreeHelper

   Input Parameters:
.  g - input graph
.  alpha - parameter
.  h - empty graph with same number of vertices as g, and vertex weights equal row sums
.  perm - preallocated array in which to store vertex ordering

   Output Parameter:
.  h - low-stretch spanning tree of input graph
.  perm - vertex ordering
 */
#undef __FUNCT__  
#define __FUNCT__ "LowStretchSpanningTreeHelper"
PetscErrorCode LowStretchSpanningTreeHelper(Graph& g,const PetscInt root,const PetscScalar alpha,
					    Graph& h,PetscInt perm[])
{
  PetscInt n,i,j,k;
  std::vector<PetscInt> size,x,y;
  std::vector<std::vector<PetscInt> > idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  EdgeWeight edge_weight_g = get(edge_weight_t(),g);
  VertexWeight vertex_weight_h = get(vertex_weight_t(),h);
  n = num_vertices(g);

  if (n > 2) {
    ierr = StarDecomp(g,root,1.0/3,alpha,k,size,idx,x,y);
    j = 0;
    for (i=1;i<=k;i++) {
      Graph& g1 = g.create_subgraph(idx[i].begin(),idx[i].end());
      Graph& h1 = h.create_subgraph(idx[i].begin(),idx[i].end());
      LowStretchSpanningTreeHelper(g1,g1.global_to_local(g.local_to_global(x[i-1])),alpha,h1,perm+j);
      j += size[i];
    }
    Graph& g1 = g.create_subgraph(idx[0].begin(),idx[0].end());
    Graph& h1 = h.create_subgraph(idx[0].begin(),idx[0].end());
    LowStretchSpanningTreeHelper(g1,g1.global_to_local(g.local_to_global(root)),alpha,h1,perm+j);
    for (i=0;i<k;i++) {
      PetscScalar w = get(edge_weight_g,edge(x[i],y[i],g).first);
      add_edge(x[i],y[i],w,h);
      put(vertex_weight_h,x[i],get(vertex_weight_h,x[i])+w);
      put(vertex_weight_h,y[i],get(vertex_weight_h,y[i])+w);
    }
  } else if (n == 2) {
    graph_traits<Graph>::edge_descriptor e = *(out_edges(root,g).first);
    PetscInt t = target(e,g);

    PetscScalar w = get(edge_weight_g,e);
    add_edge(root,t,w,h);
    put(vertex_weight_h,root,get(vertex_weight_h,root)+w);
    put(vertex_weight_h,t,get(vertex_weight_h,t)+w);
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
  PetscInt n,m;
  //PetscErrorCode ierr;
  std::priority_queue<PQNode> pq;
  PetscScalar radius;
  PetscInt centerSize;
  std::vector<PetscInt> centerIdx;
  PQNode node;

  PetscFunctionBegin;

  EdgeWeight edge_weight_g = get(edge_weight_t(),g);
  n = num_vertices(g);
  m = num_edges(g);

  // don't declare like this...
  std::vector<PetscInt> pred(n,-1);
  std::vector<PetscInt> succ[n]; 
  std::vector<PetscInt>::iterator i;
  PetscScalar dist[n];
  PetscTruth taken[n];

  // form tree of shortest paths to root
  graph_traits<Graph>::out_edge_iterator e, e_end;  
  for (tie(e,e_end)=out_edges(root,g); e!=e_end; e++) {
    PetscInt t = target(*e,g);
    pq.push(PQNode(t,root,get(edge_weight_g,*e)));
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
	  pq.push(PQNode(t,node.vertex,node.dist+get(edge_weight_g,*e)));
	}
      }
    }
  }
  radius = node.dist;
  /*
  for (int w=0; w<n; w++) {
    printf("pred[%d] = %d \n",w,pred[w]);
    printf("dist[%d] = %f \n",w,dist[w]);
  }
  */
  // do ball cut
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
    node = pq.top();pq.pop();
    centerIdx.push_back(g.local_to_global(node.vertex));
    taken[node.vertex] = PETSC_TRUE;
    centerSize++;
    for (tie(e,e_end)=out_edges(node.vertex,g); e!=e_end; e++) {
      if (taken[target(*e,g)]) {
	boundary -= get(edge_weight_g,*e);
	edgeCount++;
      } else {
	boundary += get(edge_weight_g,*e);
      }
    }
    for (i=succ[node.vertex].begin();i!=succ[node.vertex].end();i++) {
      pq.push(PQNode(*i,dist[*i]));
    }
    if (pq.empty()) {
    }
    assert(!pq.empty());
  }
  while (boundary > (edgeCount+1)*log(m)/(log(2)*(1-2*delta)*radius)) {
    node = pq.top();pq.pop();
    centerIdx.push_back(g.local_to_global(node.vertex));
    taken[node.vertex] = PETSC_TRUE;
    centerSize++;
    for (tie(e,e_end)=out_edges(node.vertex,g); e!=e_end; e++) {
      if (taken[target(*e,g)]) {
	boundary -= get(edge_weight_g,*e);
	edgeCount++;
      } else {
	boundary += get(edge_weight_g,*e);
      }
    }
    for (i=succ[node.vertex].begin();i!=succ[node.vertex].end();i++) {
      pq.push(PQNode(*i,dist[*i]));
    }
  }
  size.push_back(centerSize);
  idx.push_back(centerIdx);
  
  // pseudo cone cut
  k = 0;
  assert(!pq.empty());
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


  


  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
/*
   BallCut

   Input Parameters:
.  g - input graph
.  root - center vertex of cut
.  minRadius - cut must include all vertices closer than this to root
.  maxBoundary - weight of boundary of cut may be at most this

   Output Parameter:
.  idx - list of vertices in smallest cut satisfying given conditions
 */
#undef __FUNCT__  
#define __FUNCT__ "LowStretchSpanningTreeHelper"
PetscErrorCode BallCut(const Graph g,const PetscInt root,const PetscScalar minRadius,const PetscScalar maxBoundary,PetscInt *idx[])
{
  PetscFunctionBegin;

  

  PetscFunctionReturn(0);
}

