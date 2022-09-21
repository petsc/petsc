function [varargout] = PetscBinaryRead(inarg,varargin)
%
%   [varargout] = PetscBinaryRead(inarg,['complex',false or true],['indices','int32' or 'int64'],['cell',cnt],['precision','float64' or 'float32'])
%
%  Reads in PETSc binary file matrices or vectors
%  emits as MATLAB sparse matrice or vectors.
%
%  [] indices optional arguments
%  There are no [] in the arguments
%
%  Examples: A = PetscBinaryRead('myfile'); read from file
%            b = PetscBinaryRead(1024);   read from socket
%            c = PetscBinaryRead();       read from default socket PETSc uses
%
%  Argument may be file name (string), socket number (integer)
%  or any Matlab class that provides the read() and close() methods
%  [We provide PetscOpenFile() and PetscOpenSocket() for binary files and sockets]
%  For example: fd = PetscOpenFile('filename');
%                a = PetscBinaryRead(fd);
%                b = PetscBinaryRead(fd);
%
%  'complex', true indicates the numbers in the file are complex, that is PETSc was built with --with-scalar-type=complex
%  'indices','int64' indicates the PETSc program was built with --with-64-bit-indices
%  'cell',cnt  means return a Matlab cell array containing the first cnt objects in the file, use 10,000 to read in all objects
%  'precision','float32' indicates the PETSc program was built with --with-precision=single
%
%  Examples:  A = PetscBinaryRead('myfile','cell',10000);  read all objects in file
%             A = PetscBinaryRead(1024,'cell',2);  read two objects from socket
%
if nargin == 0
  fd = PetscOpenSocket();
else if ischar(inarg)
  fd = PetscOpenFile(inarg);
else if isnumeric(inarg)
  fd = PetscOpenSocket(inarg);
else % assume it is a PetscOpenFile or PetscOpenSocket object and handles read()
  fd = inarg;
end
end
end

indices = 'int32';
precision = 'float64';
arecell = 0;
arecomplex = false;

tnargin = nargin;
for l=1:nargin-2
  if ischar(varargin{l}) && strcmpi(varargin{l},'indices')
    tnargin = min(l,tnargin-1);
    indices = varargin{l+1};
  end
  if ischar(varargin{l}) && strcmpi(varargin{l},'precision')
    tnargin = min(l,tnargin-1);
    precision = varargin{l+1};
  end
  if ischar(varargin{l}) && strcmpi(varargin{l},'cell')
    tnargin = min(l,tnargin-1);
    arecell = varargin{l+1};
  end
  if ischar(varargin{l}) && strcmpi(varargin{l},'complex')
    tnargin = min(l,tnargin-1);
    arecomplex = varargin{l+1};
  end
end

if strcmp(precision,'float128')
  precision = 'float64';
  system(['./convert -f ' inarg]);
  fd = PetscOpenFile([inarg '_double']);
end

if arecell
  narg = arecell;
  rsult = cell(1);
else
  narg = nargout;
end

for l=1:narg
  header = double(read(fd,1,indices));
  if isempty(header)
    if arecell
      varargout(1) = {result};
      return
    else
      disp('File/Socket does not have that many items')
    end
    return
  end
  if header == 1211216 % Petsc Mat Object

    header = double(read(fd,3,indices));
    m      = header(1);
    n      = header(2);
    nz     = header(3);
    if (nz == -1)
      if arecomplex
        s     = read(fd,2*m*n,precision);
        iReal = 1:2:n*m*2-1;
        iImag = iReal +1 ;
        A     = complex(reshape(s(iReal),n,m)',reshape(s(iImag),n,m)') ;
      else
        s   = read(fd,m*n,precision);
        A   = reshape(s,n,m)';
      end
    else
      nnz = double(read(fd,m,indices));  %nonzeros per row
      sum_nz = sum(nnz);
      if(sum_nz ~=nz)
        str = sprintf('No-Nonzeros sum-rowlengths do not match %d %d',nz,sum_nz);
        error(str);
      end
      j   = double(read(fd,nz,indices)) + 1;
      if arecomplex
        s   = read(fd,2*nz,precision);
      else
        s   = read(fd,nz,precision);
      end
      i   = ones(nz,1);
      cnt = 1;
      for k=1:m
        next = cnt+nnz(k)-1;
        i(cnt:next,1) = (double(k))*ones(nnz(k),1);
        cnt = next+1;
      end
      if arecomplex
        A = sparse(i,j,complex(s(1:2:2*nz),s(2:2:2*nz)),m,n,nz);
      else
        A = sparse(i,j,s,m,n,nz);
      end
    end
    if arecell
      result{l} = A;
    else
      varargout(l) = {A};
    end
  elseif  header == 1211214 % Petsc Vec Object
    m = double(read(fd,1,indices));
    if arecomplex
      v = read(fd,2*m,precision);
      v = complex(v(1:2:2*m),v(2:2:2*m));
    else
      v = read(fd,m,precision);
    end
    if arecell
      result{l} = v;
    else
      varargout(l) = {v};
    end

  elseif  header == 1211213 % single real number
    v = read(fd,1,precision);

    if arecell
      result{l} = v;
    else
      varargout(l) = {v};
    end

  elseif  header == 1211218 % Petsc IS Object
    m = double(read(fd,1,indices));
    v = read(fd,m,'int') + 1; % Indexing in Matlab starts at 1, 0 in PETSc
    if arecell
      result{l} = v;
    else
      varargout(l) = {v};
    end

  elseif header == 1211219 % Petsc Bag Object
    b = PetscBagRead(fd);
    if arecell
      result{l} = b;
    else
      varargout(l) = {b};
    end

  elseif header == 1211221 % Petsc DM Object
    m  = double(read(fd,7,indices));
    me = double(read(fd,5,indices));
    b = [' dm ' int2str(m(3)) ' by ' int2str(m(4)) ' by ' int2str(m(5))];
    if arecell
      result{l} = b;
    else
      varargout(l) = {b};
    end

  else
    disp(['Found unrecognized header ' int2str(header) ' in file. If your file contains complex numbers'])
    disp(' then call PetscBinaryRead() with "complex",true as two additional arguments')
    return
  end

end

if arecell
  varargout(1) = {result};
end

% close the reader if we opened it

if nargin > 0
  if (ischar(inarg) || isinteger(inarg)) close(fd); end;
end
