
/* Red-Black Tree Code */
/*  Part of this code is taken from
**  http://ciips.ee.uwa.edu.au/~morris/PLDS210/Notes/rbtc.htm
*/
/* modify these lines to establish data type */
typedef unsigned long atom;

typedef struct{
  atom key,value;
}T;

#define Key(a) (((a)->Data).key)
#define Value(a) (((a)->Data).value)

/* red-black tree description */
typedef enum { Black, Red } NodeColor;

typedef struct Node_ {
    struct Node_ *Left;         /* left child */
    struct Node_ *Right;        /* right child */
    struct Node_ *Parent;       /* parent */
    NodeColor Color;            /* node color (black, red) */
    T Data;                     /* data stored in node */
} Node,*Tree;

#define NIL &Sentinel           /* all leaves are sentinels */
#define NUL ((Tree)0);
/*
Node Sentinel = { NIL, NIL, 0, Black, 0};
Node *Root = NIL;
*/

extern Node Sentinel;

/*
 *  Comparison functions.
 */
int CompLT(T a,T b);
int CompEQ(T a,T b);
int AtomEQ(atom a,atom b);
int AtomLT(atom a,atom b);

/*
 * Constructor for key-value pairs.
 */
T MakePair(atom k,atom v);
/*
 * Destructor for trees.
 */
void FreeTree(Tree PRoot);
/*
 * Print routines for debugging.
 */
void PrintNode(Tree node);
void PrintTree(Tree PRoot);
void PrintTree0(Tree PRoot,int nodeId);
/*
 * Utility functions never called by the user.
 */
void InsertFixup(Tree X,Tree *PRoot);
void DeleteFixup(Tree X,Tree *PRoot);
void RotateLeft(Tree X,Tree *PRoot);
void RotateRight(Tree X,Tree *PRoot);
Tree FindOne(atom key,Tree ix);
Tree FindLower(Tree ix);
Tree FindUpper(Tree ix);

/*
 * Node insertion, deletion, and location functions.
 */
Tree InsertNode(T Data,Tree *PRoot);
void DeleteNode(Tree Z,Tree *PRoot);
Tree FindNode(T Data,Tree PRoot);
Tree FindMin(Tree PRoot);
Tree FindMax(Tree PRoot);

/*
 * Find next or previous node in the ordering.
 */
Tree Successor(Tree theTree);
Tree Predecessor(Tree theTree);
/*
 *  Find the interval of nodes matching a specific key.
 *  tb points one past the last node in the interval.
 *  This makes stl-like for loops possible:
 *  for(t=ta;t!=tb;t=Successor(t)) f(t);
 *  Compare to
 *  for(i=ia;i!=ib;i++)v[i]=f(i);
 */
void FindInterval(atom key,Tree ix,Tree *ta,Tree *tb);
/*
 *  Count the number of values matching a key.
 */
atom CountValues(atom key,Tree a);
/*
 *  Compose and transpose trees as relations.
 *  Unions of relations are even easier to implement.
 */
void Compose(Tree a,Tree b,Tree *c);
void Transpose(Tree a,Tree *b);
/*
** Pack a coloring
*/
void Pack(Tree coloring,Tree *packing,Tree *densecoloring);

