function closeport(port)
% $Id: closeport.c,v 1.14 1999/10/24 14:01:02 bsmith Exp $
%
%   closeport(port)
%   Closes a PETSc port opened with openport()
%see openport, receive
disp('You must build the closeport mex file by doing: cd $PETSC_DIR; make BOPT=g matlabcodes')