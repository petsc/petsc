function err = snesdvi_monitor(snes,it,fnorm,x,user)
%% Monitor function for exSNES_DVI
%
err = 0;
% Plot function norm
figure(1),plot(it,fnorm,'x','linewidth',2);
hold on
h = gca;
set(h,'FontSize',12,'FontWeight','bold');
xlabel('SNES Iteration number');
ylabel('|| f ||_2');

% Create surface area plot of solution
figure(2),
x_sol = reshape(x(:),user.mx,user.my);
surf(user.ledge+user.hx:user.hx:user.redge-user.hx,user.bedge+user.hy:user.hy:user.tedge-user.hy,x_sol');

pause(.1);
