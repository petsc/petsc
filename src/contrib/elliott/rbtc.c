
/* Red-Black Tree Code */

#include <stdlib.h>
#include <stdio.h>
#include "rbtc.h"

Node Sentinel = { NIL, NIL, 0, Black, 0};

/*
Node *Root = NIL;
*/
               /* root of red-black tree */

int CompLT(T a,T b)
{
  if(a.key<b.key)return 1;
  if(a.key==b.key&&a.value<b.value)return 1;
  return 0;
}

int CompEQ(T a,T b)
{
  if(a.key==b.key&&a.value==b.value)return 1;
  return 0;
}

int AtomLT(atom a,atom b)
{
  if(a<b)return 1;
  return 0;
}

int AtomEQ(atom a,atom b)
{
  if(a==b)return 1;
  return 0;
}



T MakePair(unsigned long k,unsigned long v)
{
  T theT;
  theT.key=k;
  theT.value=v;
  return theT;
}

void FreeTree(Node *PRoot)
{
  if(PRoot!=NIL){
    FreeTree(PRoot->Left);
    FreeTree(PRoot->Right);
    free(PRoot);
  };
}

void PrintNode(Tree node)
{
  if(node==NIL)printf("NIL\n");
  printf("%d->%d",(node->Data).key,(node->Data).value);
}

void PrintTree(Node *PRoot)
{
  PrintTree0(PRoot,1);
}

void PrintTree0(Node *PRoot,int nodeId)
{
  if(PRoot!=NIL){
    PrintTree0(PRoot->Left,2*nodeId);
    PrintNode(PRoot);
    printf("(%4d)\n",nodeId);
    /*
    printf("%4d->%4d(%d,%s)\n",(PRoot->Data).key,(PRoot->Data).value,
	   nodeId,(PRoot->Color==Red?"R":"B"));
    */
    PrintTree0(PRoot->Right,2*nodeId+1);
  };
}

Node *InsertNode(T Data,Node **PRoot) {
    Node *Current, *Parent, *X;

   /***********************************************
    *  allocate node for Data and insert in tree  *
    ***********************************************/

    /* find where node belongs */
    Current = *PRoot;
    Parent = NUL;
    while (Current != NIL) {
        if (CompEQ(Data, Current->Data)) return (Current);
        Parent = Current;
        Current = CompLT(Data, Current->Data) ?
            Current->Left : Current->Right;
    }

    /* setup new node */
    if ((X = malloc (sizeof(*X))) == 0) {
        printf ("insufficient memory (InsertNode)\n");
        exit(1);
    }
    X->Data = Data;
    X->Parent = Parent;
    X->Left = NIL;
    X->Right = NIL;
    X->Color = Red;

    /* insert node in tree */
    if(Parent) {
        if(CompLT(Data, Parent->Data))
            Parent->Left = X;
        else
            Parent->Right = X;
    } else {
        *PRoot = X;
    }

    InsertFixup(X,PRoot);
    return(X);
}

void InsertFixup(Node *X,Node **PRoot) {

   /*************************************
    *  maintain red-black tree balance  *
    *  after inserting node X           *
    *************************************/

    /* check red-black properties */
    while (X != *PRoot && X->Parent->Color == Red) {
        /* we have a violation */
        if (X->Parent == X->Parent->Parent->Left) {
            Node *Y = X->Parent->Parent->Right;
            if (Y->Color == Red) {

                /* uncle is red */
                X->Parent->Color = Black;
                Y->Color = Black;
                X->Parent->Parent->Color = Red;
                X = X->Parent->Parent;
            } else {

                /* uncle is black */
                if (X == X->Parent->Right) {
                    /* make X a left child */
                    X = X->Parent;
                    RotateLeft(X,PRoot);
                }

                /* recolor and rotate */
                X->Parent->Color = Black;
                X->Parent->Parent->Color = Red;
                RotateRight(X->Parent->Parent,PRoot);
            }
        } else {

            /* mirror image of above code */
            Node *Y = X->Parent->Parent->Left;
            if (Y->Color == Red) {

                /* uncle is red */
                X->Parent->Color = Black;
                Y->Color = Black;
                X->Parent->Parent->Color = Red;
                X = X->Parent->Parent;
            } else {

                /* uncle is black */
                if (X == X->Parent->Left) {
                    X = X->Parent;
                    RotateRight(X,PRoot);
                }
                X->Parent->Color = Black;
                X->Parent->Parent->Color = Red;
                RotateLeft(X->Parent->Parent,PRoot);
            }
        }
    }
    (*PRoot)->Color = Black;
}

void RotateLeft(Node *X,Node **PRoot) {

   /**************************
    *  rotate Node X to left *
    **************************/

    Node *Y = X->Right;

    /* establish X->Right link */
    X->Right = Y->Left;
    if (Y->Left != NIL) Y->Left->Parent = X;

    /* establish Y->Parent link */
    if (Y != NIL) Y->Parent = X->Parent;
    if (X->Parent) {
        if (X == X->Parent->Left)
            X->Parent->Left = Y;
        else
            X->Parent->Right = Y;
    } else {
        *PRoot = Y;
    }

    /* link X and Y */
    Y->Left = X;
    if (X != NIL) X->Parent = Y;
}

void RotateRight(Node *X,Node **PRoot) {

   /****************************
    *  rotate Node X to right  *
    ****************************/

    Node *Y = X->Left;

    /* establish X->Left link */
    X->Left = Y->Right;
    if (Y->Right != NIL) Y->Right->Parent = X;

    /* establish Y->Parent link */
    if (Y != NIL) Y->Parent = X->Parent;
    if (X->Parent) {
        if (X == X->Parent->Right)
            X->Parent->Right = Y;
        else
            X->Parent->Left = Y;
    } else {
        *PRoot = Y;
    }

    /* link X and Y */
    Y->Right = X;
    if (X != NIL) X->Parent = Y;
}

void DeleteNode(Node *Z,Node **PRoot) {
    Node *X, *Y;

   /*****************************
    *  delete node Z from tree  *
    *****************************/

    if (!Z || Z == NIL) return;

    if (Z->Left == NIL || Z->Right == NIL) {
        /* Y has a NIL node as a child */
        Y = Z;
    } else {
        /* find tree successor with a NIL node as a child */
        Y = Z->Right;
        while (Y->Left != NIL) Y = Y->Left;
    }

    /* X is Y's only child */
    if (Y->Left != NIL)
        X = Y->Left;
    else
        X = Y->Right;

    /* remove Y from the parent chain */
    X->Parent = Y->Parent;
    if (Y->Parent)
        if (Y == Y->Parent->Left)
            Y->Parent->Left = X;
        else
            Y->Parent->Right = X;
    else
        *PRoot = X;

    if (Y != Z) Z->Data = Y->Data;
    if (Y->Color == Black)
        DeleteFixup (X,PRoot);
    free (Y);
}

void DeleteFixup(Node *X,Node **PRoot) {

   /*************************************
    *  maintain red-black tree balance  *
    *  after deleting node X            *
    *************************************/

    while (X != *PRoot && X->Color == Black) {
        if (X == X->Parent->Left) {
            Node *W = X->Parent->Right;
            if (W->Color == Red) {
                W->Color = Black;
                X->Parent->Color = Red;
                RotateLeft (X->Parent,PRoot);
                W = X->Parent->Right;
            }
            if (W->Left->Color == Black && W->Right->Color == Black) {
                W->Color = Red;
                X = X->Parent;
            } else {
                if (W->Right->Color == Black) {
                    W->Left->Color = Black;
                    W->Color = Red;
                    RotateRight (W,PRoot);
                    W = X->Parent->Right;
                }
                W->Color = X->Parent->Color;
                X->Parent->Color = Black;
                W->Right->Color = Black;
                RotateLeft (X->Parent,PRoot);
                X = *PRoot;
            }
        } else {
            Node *W = X->Parent->Left;
            if (W->Color == Red) {
                W->Color = Black;
                X->Parent->Color = Red;
                RotateRight (X->Parent,PRoot);
                W = X->Parent->Left;
            }
            if (W->Right->Color == Black && W->Left->Color == Black) {
                W->Color = Red;
                X = X->Parent;
            } else {
                if (W->Left->Color == Black) {
                    W->Right->Color = Black;
                    W->Color = Red;
                    RotateLeft (W,PRoot);
                    W = X->Parent->Left;
                }
                W->Color = X->Parent->Color;
                X->Parent->Color = Black;
                W->Left->Color = Black;
                RotateRight (X->Parent,PRoot);
                X = *PRoot;
            }
        }
    }
    X->Color = Black;
}

/**************************
  Produce Modified Version to Find Intervals by Key
***************************/

Node *FindNode(T Data,Node *PRoot) {

   /*******************************
    *  find node containing Data  *
    *******************************/

    Node *Current = PRoot;
    while(Current != NIL)
        if(CompEQ(Data, Current->Data))
            return (Current);
        else
            Current = CompLT (Data, Current->Data) ?
                Current->Left : Current->Right;
    return(NIL);
}

Tree FindMin(Tree PRoot)
{
  T d;
  if(PRoot==NIL)return NIL;
  d=PRoot->Data;
  if(PRoot->Left==NIL)return PRoot;
  return FindMin(PRoot->Left);
}

Tree FindMax(Tree PRoot)
{
  T d;
  if(PRoot==NIL)return NIL;
  d=PRoot->Data;
  if(PRoot->Right==NIL)return PRoot;
  return FindMax(PRoot->Right);
}

Tree Successor(Tree ix)
{
  T d;
  Tree x,y;
  x=ix;
  if(x==NIL)return NIL;
  if(x->Right!=NIL){
    return FindMin(x->Right);
  };
  y=x->Parent;
  while(y!=NIL&&(x==(y->Right)))
    {
      x=y;
      y=(y->Parent==(Tree)0?NIL:y->Parent);
    };
  return y;
}

Tree Predecessor(Tree ix)
{
  T d;
  Tree x,y;
  x=ix;
  if(x==NIL)return NIL;
  if(x->Left!=NIL){
    return FindMax(x->Left);
  };
  y=x->Parent;
  while(y!=(Tree)0&&(x==(y->Left)))
    {
      x=y;
      y=y->Parent;
    };
  if(y==(Tree)0)return NIL;
  return y;
}

Tree FindOne(atom key,Tree ix)
{
  if(ix==NIL)return NIL;
  if(AtomEQ(key,Key(ix)))return ix;
  if(AtomLT(key,Key(ix)))return FindOne(key,ix->Left);
  if(AtomLT(Key(ix),key))return FindOne(key,ix->Right);
  return NIL;
}

Tree FindLower(Tree ix)
{
  atom key;
  Tree x,y,z;
  if(ix==NIL)return NIL;
  x=ix;
  key=Key(x);
  while(((x->Left)!=NIL)&&AtomEQ(key,Key(x->Left)))
  {
    x=x->Left;
  };
  if(x->Left->Right==NIL)
    {
      return x;
    };
  y=x->Left->Right;
  z=FindOne(key,y);
  if(z==NIL)return x;
  else return FindLower(z);
}

Tree FindUpper(Tree ix)
{
  atom key;
  Tree x,y,z;
  if(ix==NIL)return NIL;
  x=ix;
  key=Key(x);
  while(x->Right!=NIL&&AtomEQ(key,Key(x->Right))){
    x=x->Right;
  };
  if(x->Right->Left==NIL)
    {
      return x;
    };
  y=x->Right->Left;
  z=FindOne(key,y);
  if(z==NIL)return x;
  else return FindUpper(z);
}

void FindInterval(atom key,Tree ix,Tree *ta,Tree *tb)
{
  Tree x,y,a,b;
  x=FindOne(key,ix);
  if(x==NIL)
    {
      a=NIL;
      b=NIL;
    }
  else{
    a=FindLower(x);
    b=FindUpper(x);
    b=Successor(b);
  };
  *ta=a;
  *tb=b;
}

atom CountValues(atom key,Tree a)
{
  Tree it,start,stop;
  atom count=0;
  FindInterval(key,a,&start,&stop);
  for(it=start;it!=stop;it=Successor(it),count++);
  return count;
}

void Compose(Tree a,Tree b,Tree *c)
{
  Tree it,jt,start,stop;
  *c=NIL;
  for(it=FindMin(a);it!=NIL;it=Successor(it))
    {
      FindInterval(Value(it),b,&start,&stop);
      for(jt=start;jt!=stop;jt=Successor(jt))
	{
	  InsertNode(MakePair(Key(it),Value(jt)),c);
	};
    };
}

void Transpose(Tree a,Tree *b)
{
  Tree it;
  *b=NIL;
  for(it=FindMin(a);it!=NIL;it=Successor(it))
    InsertNode(MakePair(Value(it),Key(it)),b);
}

/*
 *  Packing works for node colorings.  Some of the nodes
 *  in a discretization are assigned colors in a sparse
 *  way.  This sparse coloring is contained in the tree
 *  coloring.  This coloring is then packed into a tree
 *  called dense coloring.
 */

void Pack(Tree coloring,Tree *packing,Tree *densecoloring)
{
  Tree it;
  atom count;
  *packing=NIL;
  *densecoloring=NIL;
  for(it=FindMin(coloring),count=0;it!=NIL;it=Successor(it),count++)
    {
      InsertNode(MakePair(Value(it),count),packing);
      InsertNode(MakePair(Key(it),count),densecoloring);
    };
}

