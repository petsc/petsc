function PetscBinaryWrite(filename,varargin)
%
%  Writes in PETSc binary file sparse matrices and vectors
%  if the array is multidimensional and dense it is saved
%  as a one dimensional array
%
fd = fopen(filename,'w','ieee-be');
for l=1:nargin-1
  A = varargin{l};
  if issparse(A)
    % save sparse matrix in special Matlab format
    [m,n] = size(A);
    majic = 1.2345678910e-30;
    for i=1:min(m,n)
      if A(i,i) == 0
        A(i,i) = majic;
      end
    end

    nz   = nnz(A);
    header = fwrite(fd,[1211216,m,n,nz],'int32');
    n_nz = full(sum(A' ~= 0));

    sum_nz = sum(n_nz);
    if(sum_nz ~=nz)
      str = sprintf('No-Nonzeros sum-rowlengths do not match %d %d',nz,sum_nz);
      error(str);
    end

    fwrite(fd,n_nz,'int32');  %nonzeros per row
    [i,j,s] = find(A');
    fwrite(fd,i-1,'int32');
    for i=1:nz
      if s(i) == majic
        s(i) = 0;
      end
    end
    fwrite(fd,s,'double');
  else
    [m,n] = size(A);
    fwrite(fd,[1211214,m*n],'int32');
    fwrite(fd,A,'double');
  end
end
fclose(fd);
