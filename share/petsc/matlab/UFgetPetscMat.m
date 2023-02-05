% UFgetPetscMat.m  
% modified from UFget_example.m 
%   This script 
%     (1) gets the selected index file of the UF sparse matrix collection,
%     (2) loads in matrices in matlab format in increasing order of
%         number of rows in the selected matrices,
%     (3) writes into PETSc binary format in the given directory with
%         each matrix named as A_{id}
%
%   See also UFget_example.m 
%   Copyright 2009, PETSc Team.

index = UFget;

% sets selection here
f = find (index.nrows == index.ncols & index.nrows > 940000 & index.isReal) ;
[y, j] = sort (index.nrows (f)) ;
f = f (j) ;

for i = f
    %loads in matrix in matlab format 
    %---------------------------------
    fprintf ('Loading %s%s%s, please wait ...\n', ...
        index.Group {i}, filesep, index.Name {i}) ;
    Problem = UFget (i,index) ;
    disp (Problem) ;
    title (sprintf ('%s:%s', Problem.name, Problem.title')) ;

    % converts to PETSc binary format and writes into ~mat/A_{id}
    %-----------------------------------------------------------
    fname = ['mat/A',num2str(i)];
    fprintf ('write matrix into petsc binary file %s ...\n',fname);
    PetscBinaryWrite(fname,Problem.A);
    %input ('hit enter to continue:') ;
end

