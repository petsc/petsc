#include <stdlib.h>
#include <stdio.h>
#include "rbtc.h"

main(int argc,char *argv[])
{
  Tree theTree=NIL,ttree,vtree,ctree;
  /*
   *  The first entry in the pair is the node number.
   *  The second entry is the "color".
   */
  InsertNode(MakePair(2,0),&theTree);
  InsertNode(MakePair(3,0),&theTree);
  InsertNode(MakePair(3,0),&theTree);
  InsertNode(MakePair(5,0),&theTree);
  InsertNode(MakePair(15,0),&theTree);
  InsertNode(MakePair(16,0),&theTree);
  InsertNode(MakePair(15,1),&theTree);
  InsertNode(MakePair(16,1),&theTree);
  InsertNode(MakePair(17,1),&theTree);
  InsertNode(MakePair(18,1),&theTree);
  printf("Sparsely defined function ...\n");
  PrintTree(theTree);
  /*
   *  This relation from nodes to colors is transposed to obtain
   *  a "coloring."  This coloring is sparse in the sense that
   *  the node numbers are larger than they need to be to keep
   *  the node coloring distinct.  The function "pack" removes
   *  this redundancy from the coloring and computes the map
   *  which goes from the sparse to the dense coloring.  This
   *  is one of the gather operations needed to compute local
   *  problems for Schwarz methods.
   */
  printf("Sparse Coloring...\n");
  Transpose(theTree,&ttree);
  PrintTree(ttree);
  printf("Packing...\n");
  Pack(ttree,&vtree,&ctree);
  PrintTree(vtree);
  printf("Packed Coloring...\n");
  PrintTree(ctree);
  FreeTree(theTree);
  FreeTree(ttree);
  FreeTree(vtree);
  FreeTree(ctree);
}

