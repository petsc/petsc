function [lambda, V, A] = laplacian(varargin)

% LAPLACIAN   Sparse Negative Laplacian in 1D, 2D, or 3D
%
%    [~,~,A]=LAPLACIAN(N) generates a sparse negative 3D Laplacian matrix
%    with Dirichlet boundary conditions, from a rectangular cuboid regular
%    grid with j x k x l interior grid points if N = [j k l], using the
%    standard 7-point finite-difference scheme,  The grid size is always
%    one in all directions.
%
%    [~,~,A]=LAPLACIAN(N,B) specifies boundary conditions with a cell array
%    B. For example, B = {'DD' 'DN' 'P'} will Dirichlet boundary conditions
%    ('DD') in the x-direction, Dirichlet-Neumann conditions ('DN') in the
%    y-direction and period conditions ('P') in the z-direction. Possible
%    values for the elements of B are 'DD', 'DN', 'ND', 'NN' and 'P'.
%
%    LAMBDA = LAPLACIAN(N,B,M) or LAPLACIAN(N,M) outputs the m smallest
%    eigenvalues of the matrix, computed by an exact known formula, see
%    http://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors_of_the_second_derivative
%    It will produce a warning if the mth eigenvalue is equal to the
%    (m+1)th eigenvalue. If m is absebt or zero, lambda will be empty.
%
%    [LAMBDA,V] = LAPLACIAN(N,B,M) also outputs orthonormal eigenvectors
%    associated with the corresponding m smallest eigenvalues.
%
%    [LAMBDA,V,A] = LAPLACIAN(N,B,M) produces a 2D or 1D negative
%    Laplacian matrix if the length of N and B are 2 or 1 respectively.
%    It uses the standard 5-point scheme for 2D, and 3-point scheme for 1D.
%
%    % Examples:
%    [lambda,V,A] = laplacian([100,45,55],{'DD' 'NN' 'P'}, 20); 
%    % Everything for 3D negative Laplacian with mixed boundary conditions.
%    laplacian([100,45,55],{'DD' 'NN' 'P'}, 20);
%    % or
%    lambda = laplacian([100,45,55],{'DD' 'NN' 'P'}, 20);
%    % computes the eigenvalues only
%
%    [~,V,~] = laplacian([200 200],{'DD' 'DN'},30);
%    % Eigenvectors of 2D negative Laplacian with mixed boundary conditions.
%
%    [~,~,A] = laplacian(200,{'DN'},30);
%    % 1D negative Laplacian matrix A with mixed boundary conditions.
%
%    % Example to test if outputs correct eigenvalues and vectors:
%    [lambda,V,A] = laplacian([13,10,6],{'DD' 'DN' 'P'},30);
%    [Veig D] = eig(full(A)); lambdaeig = diag(D(1:30,1:30));
%    max(abs(lambda-lambdaeig))  %checking eigenvalues
%    subspace(V,Veig(:,1:30))    %checking the invariant subspace
%    subspace(V(:,1),Veig(:,1))  %checking selected eigenvectors
%    subspace(V(:,29:30),Veig(:,29:30)) %a multiple eigenvalue 
%    
%    % Example showing equivalence between laplacian.m and built-in MATLAB
%    % DELSQ for the 2D case. The output of the last command shall be 0.
%    A1 = delsq(numgrid('S',32)); % input 'S' specifies square grid.
%    [~,~,A2] = laplacian([30,30]);
%    norm(A1-A2,inf)
%    
%    Class support for inputs:
%    N - row vector float double  
%    B - cell array
%    M - scalar float double 
%
%    Class support for outputs:
%    lambda and V  - full float double, A - sparse float double.
%
%    Note: the actual numerical entries of A fit int8 format, but only
%    double data class is currently (2010) supported for sparse matrices. 
%
%    This program is designed to efficiently compute eigenvalues,
%    eigenvectors, and the sparse matrix of the (1-3)D negative Laplacian
%    on a rectangular grid for Dirichlet, Neumann, and Periodic boundary
%    conditions using tensor sums of 1D Laplacians. For more information on
%    tensor products, see
%    http://en.wikipedia.org/wiki/Kronecker_sum_of_discrete_Laplacians
%    For 2D case in MATLAB, see 
%    http://www.mathworks.com/access/helpdesk/help/techdoc/ref/kron.html.
%
%    This code is a part of the BLOPEX package: 
%    http://en.wikipedia.org/wiki/BLOPEX or directly 
%    http://code.google.com/p/blopex/

%    Revision 1.1 changes: rearranged the output variables, always compute 
%    the eigenvalues, compute eigenvectors and/or the matrix on demand only.

%    License: BSD
%    Copyright 2010-2011 Bryan C. Smith, Andrew V. Knyazev
%    $Revision: 1.1 $ $Date: 1-Aug-2011
%    Tested in MATLAB 7.11.0 (R2010b) and Octave 3.4.0.

tic

% Input/Output handling.
if nargin > 3
    error('BLOPEX:laplacian:TooManyInputs',...
        '%s','Too many input arguments.');
elseif nargin == 0
    error('BLOPEX:laplacian:NoInputArguments',...
        '%s','Must have at least one input argument.');
end

if nargout > 3
    error('BLOPEX:laplacian:TooManyOutputs',...
        '%s','Maximum number of outputs is 3.');
end

u = varargin{1};
dim2 = size(u);
if dim2(1) ~= 1
    error('BLOPEX:laplacian:WrongVectorOfGridPoints',...
        '%s','Number of grid points must be in a row vector.')
end
if dim2(2) > 3
    error('BLOPEX:laplacian:WrongNumberOfGridPoints',...
        '%s','Number of grid points must be a 1, 2, or 3')
end
dim=dim2(2); clear dim2;

uint = round(u);
if max(uint~=u)
    warning('BLOPEX:laplacian:NonIntegerGridSize',...
        '%s','Grid sizes must be integers. Rounding...');
    u = uint; clear uint
end
if max(u<=0 )
    error('BLOPEX:laplacian:NonIntegerGridSize',...
        '%s','Grid sizes must be positive.');
end

if nargin == 3
    m = varargin{3};
    B = varargin{2};
elseif nargin == 2
    f = varargin{2};
    a = whos('regep','f');
    if sum(a.class(1:4)=='cell') == 4
        B = f;
        m = 0;
    elseif sum(a.class(1:4)=='doub') == 4
        if dim == 1
            B = {'DD'};
        elseif dim == 2
            B = {'DD' 'DD'};
        else
            B = {'DD' 'DD' 'DD'};
        end
        m = f;
    else
        error('BLOPEX:laplacian:InvalidClass',...
            '%s','Second input must be either class double or a cell array.');
    end
else
    if dim == 1
        B = {'DD'};
    elseif dim == 2
        B = {'DD' 'DD'};
    else
        B = {'DD' 'DD' 'DD'};
    end
    m = 0;
end

if max(size(m) - [1 1]) ~= 0
    error('BLOPEX:laplacian:WrongNumberOfEigenvalues',...
        '%s','The requested number of eigenvalues must be a scalar.');
end

maxeigs = prod(u);
mint = round(m);
if mint ~= m || mint > maxeigs
    error('BLOPEX:laplacian:InvalidNumberOfEigs',...
        '%s','Number of eigenvalues output must be a nonnegative ',...
        'integer no bigger than number of grid points.');
end
m = mint;

bdryerr = 0;
a = whos('regep','B');
if sum(a.class(1:4)=='cell') ~= 4 || sum(a.size == [1 dim]) ~= 2
    bdryerr = 1;
else
    BB = zeros(1, 2*dim);
    for i = 1:dim
        if (length(B{i}) == 1)
            if B{i} == 'P'
                BB(i) = 3;
                BB(i + dim) = 3;
            else
                bdryerr = 1;
            end
        elseif (length(B{i}) == 2)
            if B{i}(1) == 'D'
                BB(i) = 1;
            elseif B{i}(1) == 'N'
                BB(i) = 2;
            else
                bdryerr = 1;
            end
            if B{i}(2) == 'D'
                BB(i + dim) = 1;
            elseif B{i}(2) == 'N'
                BB(i + dim) = 2;
            else
                bdryerr = 1;
            end
        else
            bdryerr = 1;
        end
    end
end

if bdryerr == 1
    error('BLOPEX:laplacian:InvalidBdryConds',...
        '%s','Boundary conditions must be a cell of length 3 for 3D, 2',...
        ' for 2D, 1 for 1D, with values ''DD'', ''DN'', ''ND'', ''NN''',...
        ', or ''P''.');
end

% Set the component matrices. SPDIAGS converts int8 into double anyway.
e1 = ones(u(1),1); %e1 = ones(u(1),1,'int8');
if dim > 1
    e2 = ones(u(2),1);
end
if dim > 2
    e3 = ones(u(3),1);
end

% Calculate m smallest exact eigenvalues.
if m > 0
    if (BB(1) == 1) && (BB(1+dim) == 1)
        a1 = pi/2/(u(1)+1);
        N = (1:u(1))';
    elseif (BB(1) == 2) && (BB(1+dim) == 2)
        a1 = pi/2/u(1);
        N = (0:(u(1)-1))';
    elseif ((BB(1) == 1) && (BB(1+dim) == 2)) || ((BB(1) == 2)...
            && (BB(1+dim) == 1))
        a1 = pi/4/(u(1)+0.5);
        N = 2*(1:u(1))' - 1;
    else
        a1 = pi/u(1);
        N = floor((1:u(1))/2)';
    end
    
    lambda1 = 4*sin(a1*N).^2;
    
    if dim > 1
        if (BB(2) == 1) && (BB(2+dim) == 1)
            a2 = pi/2/(u(2)+1);
            N = (1:u(2))';
        elseif (BB(2) == 2) && (BB(2+dim) == 2)
            a2 = pi/2/u(2);
            N = (0:(u(2)-1))';
        elseif ((BB(2) == 1) && (BB(2+dim) == 2)) || ((BB(2) == 2)...
                && (BB(2+dim) == 1))
            a2 = pi/4/(u(2)+0.5);
            N = 2*(1:u(2))' - 1;
        else
            a2 = pi/u(2);
            N = floor((1:u(2))/2)';
        end
        lambda2 = 4*sin(a2*N).^2;
    else
        lambda2 = 0;
    end
    
    if dim > 2
        if (BB(3) == 1) && (BB(6) == 1)
            a3 = pi/2/(u(3)+1);
            N = (1:u(3))';
        elseif (BB(3) == 2) && (BB(6) == 2)
            a3 = pi/2/u(3);
            N = (0:(u(3)-1))';
        elseif ((BB(3) == 1) && (BB(6) == 2)) || ((BB(3) == 2)...
                && (BB(6) == 1))
            a3 = pi/4/(u(3)+0.5);
            N = 2*(1:u(3))' - 1;
        else
            a3 = pi/u(3);
            N = floor((1:u(3))/2)';
        end
        lambda3 = 4*sin(a3*N).^2;
    else
        lambda3 = 0;
    end
    
    if dim == 1
        lambda = lambda1;
    elseif dim == 2
        lambda = kron(e2,lambda1) + kron(lambda2, e1);
    else
        lambda = kron(e3,kron(e2,lambda1)) + kron(e3,kron(lambda2,e1))...
            + kron(lambda3,kron(e2,e1));
    end
    [lambda, p] = sort(lambda);
    if m < maxeigs - 0.1
        w = lambda(m+1);
    else
        w = inf;
    end
    lambda = lambda(1:m);
    p = p(1:m)';
else
    lambda = [];
end

V = []; 
if nargout > 1 && m > 0 % Calculate eigenvectors if specified in output.
    
    p1 = mod(p-1,u(1))+1;
    
    if (BB(1) == 1) && (BB(1+dim) == 1)
        V1 = sin(kron((1:u(1))'*(pi/(u(1)+1)),p1))*(2/(u(1)+1))^0.5;
    elseif (BB(1) == 2) && (BB(1+dim) == 2)
        V1 = cos(kron((0.5:1:u(1)-0.5)'*(pi/u(1)),p1-1))*(2/u(1))^0.5;
        V1(:,p1==1) = 1/u(1)^0.5;
    elseif ((BB(1) == 1) && (BB(1+dim) == 2))
        V1 = sin(kron((1:u(1))'*(pi/2/(u(1)+0.5)),2*p1 - 1))...
            *(2/(u(1)+0.5))^0.5;
    elseif ((BB(1) == 2) && (BB(1+dim) == 1))
        V1 = cos(kron((0.5:1:u(1)-0.5)'*(pi/2/(u(1)+0.5)),2*p1 - 1))...
            *(2/(u(1)+0.5))^0.5;
    else
        V1 = zeros(u(1),m);
        a = (0.5:1:u(1)-0.5)';
        V1(:,mod(p1,2)==1) = cos(a*(pi/u(1)*(p1(mod(p1,2)==1)-1)))...
            *(2/u(1))^0.5;
        pp = p1(mod(p1,2)==0);
        if ~isempty(pp)
            V1(:,mod(p1,2)==0) = sin(a*(pi/u(1)*p1(mod(p1,2)==0)))...
                *(2/u(1))^0.5;
        end
        V1(:,p1==1) = 1/u(1)^0.5;
        if mod(u(1),2) == 0
            V1(:,p1==u(1)) = V1(:,p1==u(1))/2^0.5;
        end
    end
    
    
    if dim > 1
        p2 = mod(p-p1,u(1)*u(2));
        p3 = (p - p2 - p1)/(u(1)*u(2)) + 1;
        p2 = p2/u(1) + 1;
        if (BB(2) == 1) && (BB(2+dim) == 1)
            V2 = sin(kron((1:u(2))'*(pi/(u(2)+1)),p2))*(2/(u(2)+1))^0.5;
        elseif (BB(2) == 2) && (BB(2+dim) == 2)
            V2 = cos(kron((0.5:1:u(2)-0.5)'*(pi/u(2)),p2-1))*(2/u(2))^0.5;
            V2(:,p2==1) = 1/u(2)^0.5;
        elseif ((BB(2) == 1) && (BB(2+dim) == 2))
            V2 = sin(kron((1:u(2))'*(pi/2/(u(2)+0.5)),2*p2 - 1))...
                *(2/(u(2)+0.5))^0.5;
        elseif ((BB(2) == 2) && (BB(2+dim) == 1))
            V2 = cos(kron((0.5:1:u(2)-0.5)'*(pi/2/(u(2)+0.5)),2*p2 - 1))...
                *(2/(u(2)+0.5))^0.5;
        else
            V2 = zeros(u(2),m);
            a = (0.5:1:u(2)-0.5)';
            V2(:,mod(p2,2)==1) = cos(a*(pi/u(2)*(p2(mod(p2,2)==1)-1)))...
                *(2/u(2))^0.5;
            pp = p2(mod(p2,2)==0);
            if ~isempty(pp)
                V2(:,mod(p2,2)==0) = sin(a*(pi/u(2)*p2(mod(p2,2)==0)))...
                    *(2/u(2))^0.5;
            end
            V2(:,p2==1) = 1/u(2)^0.5;
            if mod(u(2),2) == 0
                V2(:,p2==u(2)) = V2(:,p2==u(2))/2^0.5;
            end
        end
    else
        V2 = ones(1,m);
    end
    
    if dim > 2
        if (BB(3) == 1) && (BB(3+dim) == 1)
            V3 = sin(kron((1:u(3))'*(pi/(u(3)+1)),p3))*(2/(u(3)+1))^0.5;
        elseif (BB(3) == 2) && (BB(3+dim) == 2)
            V3 = cos(kron((0.5:1:u(3)-0.5)'*(pi/u(3)),p3-1))*(2/u(3))^0.5;
            V3(:,p3==1) = 1/u(3)^0.5;
        elseif ((BB(3) == 1) && (BB(3+dim) == 2))
            V3 = sin(kron((1:u(3))'*(pi/2/(u(3)+0.5)),2*p3 - 1))...
                *(2/(u(3)+0.5))^0.5;
        elseif ((BB(3) == 2) && (BB(3+dim) == 1))
            V3 = cos(kron((0.5:1:u(3)-0.5)'*(pi/2/(u(3)+0.5)),2*p3 - 1))...
                *(2/(u(3)+0.5))^0.5;
        else
            V3 = zeros(u(3),m);
            a = (0.5:1:u(3)-0.5)';
            V3(:,mod(p3,2)==1) = cos(a*(pi/u(3)*(p3(mod(p3,2)==1)-1)))...
                *(2/u(3))^0.5;
            pp = p1(mod(p3,2)==0);
            if ~isempty(pp)
                V3(:,mod(p3,2)==0) = sin(a*(pi/u(3)*p3(mod(p3,2)==0)))...
                    *(2/u(3))^0.5;
            end
            V3(:,p3==1) = 1/u(3)^0.5;
            if mod(u(3),2) == 0
                V3(:,p3==u(3)) = V3(:,p3==u(3))/2^0.5;
            end
            
        end
    else
        V3 = ones(1,m);
    end
    
    if dim == 1
        V = V1;
    elseif dim == 2
        V = kron(e2,V1).*kron(V2,e1);
    else
        V = kron(e3, kron(e2, V1)).*kron(e3, kron(V2, e1))...
            .*kron(kron(V3,e2),e1);
    end
    
    if m ~= 0
        if abs(lambda(m) - w) < maxeigs*eps('double')
            sprintf('\n%s','Warning: (m+1)th eigenvalue is  nearly equal',...
                ' to mth.')
            
        end
    end
    
end

A = [];
if nargout > 2 %also calulate the matrix if specified in the output
    
    % Set the component matrices. SPDIAGS converts int8 into double anyway.
    %    e1 = ones(u(1),1); %e1 = ones(u(1),1,'int8');
    D1x = spdiags([-e1 2*e1 -e1], [-1 0 1], u(1),u(1));
    if dim > 1
        %        e2 = ones(u(2),1);
        D1y = spdiags([-e2 2*e2 -e2], [-1 0 1], u(2),u(2));
    end
    if dim > 2
        %        e3 = ones(u(3),1);
        D1z = spdiags([-e3 2*e3 -e3], [-1 0 1], u(3),u(3));
    end
    
    
    % Set boundary conditions if other than Dirichlet.
    for i = 1:dim
        if BB(i) == 2
            eval(['D1' char(119 + i) '(1,1) = 1;'])
        elseif BB(i) == 3
            eval(['D1' char(119 + i) '(1,' num2str(u(i)) ') = D1'...
                char(119 + i) '(1,' num2str(u(i)) ') -1;']);
            eval(['D1' char(119 + i) '(' num2str(u(i)) ',1) = D1'...
                char(119 + i) '(' num2str(u(i)) ',1) -1;']);
        end
        
        if BB(i+dim) == 2
            eval(['D1' char(119 + i)...
                '(',num2str(u(i)),',',num2str(u(i)),') = 1;'])
        end
    end
    
    % Form A using tensor products of lower dimensional Laplacians
    Ix = speye(u(1));
    if dim == 1
        A = D1x;
    elseif dim == 2
        Iy = speye(u(2));
        A = kron(Iy,D1x) + kron(D1y,Ix);
    elseif dim == 3
        Iy = speye(u(2));
        Iz = speye(u(3));
        A = kron(Iz, kron(Iy, D1x)) + kron(Iz, kron(D1y, Ix))...
            + kron(kron(D1z,Iy),Ix);
    end
end

disp('  ')
toc
if ~isempty(V)
    a = whos('regep','V');
    disp(['The eigenvectors take ' num2str(a.bytes) ' bytes'])
end
if  ~isempty(A)
    a = whos('regexp','A');
    disp(['The Laplacian matrix takes ' num2str(a.bytes) ' bytes'])
end
disp('  ')

