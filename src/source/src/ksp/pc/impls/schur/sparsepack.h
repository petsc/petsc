/*****************************************************************/
/*********     ROOTLS ..... ROOTED LEVEL STRUCTURE      **********/
/*****************************************************************/
/*    PURPOSE - ROOTLS GENERATES THE LEVEL STRUCTURE ROOTED */
/*       AT THE INPUT NODE CALLED ROOT. ONLY THOSE NODES FOR*/
/*       WHICH MASK IS NONZERO WILL BE CONSIDERED.*/
/*                                                */
/*    INPUT PARAMETERS -                          */
/*       ROOT - THE NODE AT WHICH THE LEVEL STRUCTURE IS TO*/
/*              BE ROOTED.*/
/*       (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR FOR THE*/
/*              GIVEN GRAPH.*/
/*       MASK - IS USED TO SPECIFY A SECTION SUBGRAPH. NODES*/
/*              WITH MASK(I)=0 ARE IGNORED.*/
/*    OUTPUT PARAMETERS -*/
/*       NLVL - IS THE NUMBER OF LEVELS IN THE LEVEL STRUCTURE.*/
/*       (XLS, LS) - ARRAY PAIR FOR THE ROOTED LEVEL STRUCTURE.*/
/*****************************************************************/

EXTERN PetscErrorCode SPARSEPACKrootls(PetscInt *root, PetscInt *xadj, PetscInt *adjncy, PetscInt *mask, 
				       PetscInt *nlvl, PetscInt *xls, PetscInt *ls);


/*****************************************************************/
/*************     FNDSEP ..... FIND SEPARATOR       *************/
/*****************************************************************/
/*    PURPOSE - THIS ROUTINE IS USED TO FIND A SMALL             */
/*              SEPARATOR FOR A CONNECTED COMPONENT SPECIFIED    */
/*              BY MASK IN THE GIVEN GRAPH.                      */
/*                                                               */
/*    INPUT PARAMETERS -                                         */
/*       ROOT - IS THE NODE THAT DETERMINES THE MASKED           */
/*              COMPONENT.                                       */
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE PAIR.          */
/*                                                               */
/*    OUTPUT PARAMETERS -                                        */
/*       NSEP - NUMBER OF VARIABLES IN THE SEPARATOR.            */
/*       SEP - VECTOR CONTAINING THE SEPARATOR NODES.            */
/*                                                               */
/*    UPDATED PARAMETER -                                        */
/*       MASK - NODES IN THE SEPARATOR HAVE THEIR MASK           */
/*              VALUES SET TO ZERO.                              */
/*                                                               */
/*    WORKING PARAMETERS -                                       */
/*       (XLS, LS) - LEVEL STRUCTURE PAIR FOR LEVEL STRUCTURE    */
/*              FOUND BY FNROOT.                                 */
/*                                                               */
/*    PROGRAM SUBROUTINES -                                      */
/*       FNROOT.                                                 */
/*                                                               */
/*****************************************************************/

EXTERN PetscErrorCode SPARSEPACKfndsep(PetscInt *root, PetscInt *xadj, PetscInt *adjncy, PetscInt *mask,
				       PetscInt *nsep, PetscInt *sep,
				       PetscInt *xls, PetscInt *ls);

/*****************************************************************/
/********     FN1WD ..... FIND ONE-WAY DISSECTORS        *********/
/*****************************************************************/
/*    PURPOSE - THIS SUBROUTINE FINDS ONE-WAY DISSECTORS OF      */
/*       A CONNECTED COMPONENT SPECIFIED BY MASK AND ROOT.       */
/*                                                               */
/*    INPUT PARAMETERS -                                         */
/*       ROOT - A NODE THAT DEFINES (ALONG WITH MASK) THE        */
/*              COMPONENT TO BE PROCESSED.                       */
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE.               */
/*                                                               */
/*    OUTPUT PARAMETERS -                                        */
/*       NSEP - NUMBER OF NODES IN THE ONE-WAY DISSECTORS.       */
/*       SEP - VECTOR CONTAINING THE DISSECTOR NODES.            */
/*                                                               */
/*    UPDATED PARAMETER -                                        */
/*       MASK - NODES IN THE DISSECTOR HAVE THEIR MASK VALUES    */
/*              SET TO ZERO.                                     */
/*                                                               */
/*    WORKING PARAMETERS-                                        */
/*       (XLS, LS) - LEVEL STRUCTURE USED BY THE ROUTINE FNROOT. */
/*                                                               */
/*    PROGRAM SUBROUTINE -                                       */
/*       FNROOT.                                                 */
/*****************************************************************/
EXTERN PetscErrorCode SPARSEPACKfn1wd(PetscInt *root, PetscInt *xadj, PetscInt *adjncy, PetscInt *mask,
				      PetscInt *nsep, PetscInt *sep,
				      PetscInt *nlvl, PetscInt *xls, PetscInt *ls);
