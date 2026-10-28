function fractional_thermal_transport
% FRACTIONAL_THERMAL_TRANSPORT
% Reproducible 1-D and 2-D solver for
%   D_C^alpha theta + kappa A^s theta + mu theta
%       = f + b(x)u(t) + a theta - eta theta^3,
% with spectral Dirichlet fractional diffusion and Caputo L1 time stepping.
% Includes a feedback-threshold check, projected-gradient temperature control,
% and temporal-refinement verification. MATLAB R2021a compatible; no toolboxes.
%
% IMPORTANT: this is a numerical implementation of a clearly specified test
% instance of the manuscript's abstract model. Parameters below are illustrative,
% not calibrated physical data. The control objective is temperature tracking,
% not boundary heat-flux optimization.

clc; close all;
rng(7);

% Shared model parameters
p.alpha = 0.80;       % Caputo order in (0,1]
p.s = 0.80;           % spectral fractional order in (0,1]
p.kappa = 0.08;       % transport coefficient
p.mu = 0.20;          % linear relaxation
p.a = 0.25;           % scalar thermal feedback gain
p.eta = 0.10;         % stabilizing cubic response: N(theta)=-eta*theta^3
p.Tf = 1.0;
p.Nt = 80;
p.controlBound = 1.5;
p.controlBlocks = 6;
p.controlIterations = 8;
p.controlStep = 0.08;
p.controlReg = 2.0e-2;
p.trackWeight = 1.0;
p.terminalWeight = 2.0;

fprintf('\nFRACTIONAL THERMAL TRANSPORT: MATLAB TEST RUN\n');
fprintf('alpha = %.3f, s = %.3f, kappa = %.3g, mu = %.3g, feedback a = %.3g\n', ...
    p.alpha,p.s,p.kappa,p.mu,p.a);

% ----------------------------- 1-D experiment -----------------------------
q1.Lx = 1.0; q1.Nx = 48; q1.dim = 1; q1.name = '1-D';
r1 = run_case(q1,p,true);

% ----------------------------- 2-D experiment -----------------------------
q2.Lx = 1.0; q2.Ly = 1.0; q2.Nx = 24; q2.Ny = 24; q2.dim = 2; q2.name = '2-D';
r2 = run_case(q2,p,true);

% ------------------------- Temporal refinement test ------------------------
% Same 1-D spatial grid and data, with controls disabled and zero forcing.
% Compare final states against a finer-time reference.
qv = q1; pv = p; pv.a = 0.10; pv.eta = 0; pv.Nt = 40;
rv40 = run_case(qv,pv,false);
pv.Nt = 80; rv80 = run_case(qv,pv,false);
pv.Nt = 160; rv160 = run_case(qv,pv,false);
e40 = norm(rv40.final(:)-rv160.final(:))/max(norm(rv160.final(:)),eps);
e80 = norm(rv80.final(:)-rv160.final(:))/max(norm(rv160.final(:)),eps);
fprintf('\nTEMPORAL REFINEMENT (1-D, fixed spatial grid)\n');
fprintf('Relative final-state error, Nt=40 versus Nt=160:  %.6e\n',e40);
fprintf('Relative final-state error, Nt=80 versus Nt=160:  %.6e\n',e80);
if e80 > 0
    fprintf('Observed refinement ratio e40/e80: %.4f (not a proof of an asymptotic order).\n',e40/e80);
end

% Summary table
Dim = {'1-D';'2-D'};
Nspace = [q1.Nx; q2.Nx*q2.Ny];
Lambda1 = [r1.lambda1; r2.lambda1];
CriticalGain = [r1.aCrit; r2.aCrit];
FinalL2 = [r1.finalL2; r2.finalL2];
ControlledCost = [r1.controlCost; r2.controlCost];
Summary = table(Dim,Nspace,Lambda1,CriticalGain,FinalL2,ControlledCost);
disp(Summary);

fprintf('\nInterpretation notes:\n');
fprintf('1) Scalar commuting-feedback threshold: a_crit = kappa*lambda_1^s + mu.\n');
fprintf('2) alpha changes the transient memory response, not this threshold formula.\n');
fprintf('3) The reported control is a projected-gradient minimizer of a discrete\n');
fprintf('   temperature-tracking objective; it is not a boundary heat-flux optimizer.\n');
fprintf('4) Refinement results are numerical checks for this test problem only.\n\n');
end

function out = run_case(g,p,doControl)
% Construct interior grids and orthonormal discrete sine transforms.
x = (1:g.Nx)'*g.Lx/(g.Nx+1);
Qx = sqrt(2/(g.Nx+1))*sin(pi*(1:g.Nx)'*(1:g.Nx)/(g.Nx+1));
% The expression above produces Nx-by-Nx sine basis.
if g.dim == 1
    [X] = ndgrid(x);
    Qy = 1; y = 0; %#ok<NASGU>
    lam = (pi*(1:g.Nx)'/g.Lx).^2;
    Lam = lam.^p.s;
    toCoeff = @(U) Qx'*U;
    fromCoeff = @(C) Qx*C;
    shape = sin(pi*X/g.Lx);
    theta0 = 0.8*sin(pi*X/g.Lx) + 0.15*sin(3*pi*X/g.Lx);
    plotGrid = x;
else
    y = (1:g.Ny)'*g.Ly/(g.Ny+1);
    Qy = sqrt(2/(g.Ny+1))*sin(pi*(1:g.Ny)'*(1:g.Ny)/(g.Ny+1));
    [X,Y] = ndgrid(x,y);
    lamx = (pi*(1:g.Nx)'/g.Lx).^2;
    lamy = (pi*(1:g.Ny)'/g.Ly).^2;
    lam = lamx + lamy';
    Lam = lam.^p.s;
    toCoeff = @(U) Qx'*U*Qy;
    fromCoeff = @(C) Qx*C*Qy';
    shape = sin(pi*X/g.Lx).*sin(pi*Y/g.Ly);
    theta0 = 0.8*shape + 0.15*sin(2*pi*X/g.Lx).*sin(pi*Y/g.Ly);
    plotGrid = x;
end

% Forcing and control act through a fixed spatial heating/cooling profile.
Bshape = shape/max(abs(shape(:)));
F = zeros(size(theta0));
C0 = toCoeff(theta0);
nt = p.Nt; dt = p.Tf/nt; t = (0:nt)*dt;
cap = dt^(-p.alpha)/gamma(2-p.alpha);
bw = (1:nt).^(1-p.alpha) - (0:nt-1).^(1-p.alpha);
% First Dirichlet eigenvalue of -Delta in the selected geometry.
if g.dim == 1
    lambda1 = (pi/g.Lx)^2;
else
    lambda1 = (pi/g.Lx)^2 + (pi/g.Ly)^2;
end
aCrit = p.kappa*lambda1^p.s + p.mu;
fprintf('\n%s: grid %d',g.name,g.Nx);
if g.dim==2, fprintf(' x %d',g.Ny); end
fprintf(', Nt=%d, lambda1=%.6g, a_crit=%.6g, a/a_crit=%.4f\n',nt,lambda1,aCrit,p.a/aCrit);

% Target is zero temperature deviation; optimize a bounded scalar control
% represented by piecewise-constant temporal blocks.
if doControl
    nblock = p.controlBlocks;
    v = zeros(nblock,1);
    opts.p = p; opts.g = g; opts.Lam = Lam; opts.toCoeff = toCoeff;
    opts.fromCoeff = fromCoeff; opts.Bshape = Bshape; opts.F = F;
    opts.C0 = C0; opts.cap = cap; opts.bw = bw; opts.nt = nt;
    opts.dt = dt; opts.t = t; opts.theta0 = theta0;
    opts.trackWeight = p.trackWeight; opts.terminalWeight = p.terminalWeight;
    opts.controlReg = p.controlReg; opts.bound = p.controlBound;
    Jhist = zeros(p.controlIterations,1);
    for it=1:p.controlIterations
        [J,~,u] = objective_blocks(v,opts);
        Jhist(it)=J;
        grad = zeros(size(v)); epsFD = 2e-5;
        for j=1:numel(v)
            vp=v; vm=v; vp(j)=min(opts.bound,v(j)+epsFD); vm(j)=max(-opts.bound,v(j)-epsFD);
            Jp=objective_blocks(vp,opts); Jm=objective_blocks(vm,opts);
            grad(j)=(Jp-Jm)/(vp(j)-vm(j)+eps);
        end
        v = min(opts.bound,max(-opts.bound,v-p.controlStep*grad));
    end
    [controlCost,Uhist,u] = objective_blocks(v,opts);
else
    u = zeros(nt+1,1);
    [Uhist,~] = solve_state(u,p,Lam,toCoeff,fromCoeff,Bshape,F,C0,cap,bw,nt,dt,theta0);
    controlCost = NaN;
end

% Baseline no-control solution for visual comparison.
u0 = zeros(nt+1,1);
[Ubase,~] = solve_state(u0,p,Lam,toCoeff,fromCoeff,Bshape,F,C0,cap,bw,nt,dt,theta0);
final = Uhist(:,:,end);
finalL2 = sqrt(sum(final(:).^2)*g.Lx/(g.Nx+1));
if g.dim==2, finalL2=sqrt(sum(final(:).^2)*g.Lx*g.Ly/((g.Nx+1)*(g.Ny+1))); end

% Plot the principal 1-D and 2-D experiments (skip refinement-only runs).
if doControl
    figure('Name',[g.name ' fractional thermal transport'],'Color','w');
    if g.dim==1
        subplot(2,1,1); plot(x,theta0,'k-','LineWidth',1.6); hold on;
        plot(x,Ubase(:,1,end),'r--','LineWidth',1.5); plot(x,final(:,1),'b-','LineWidth',1.5);
        xlabel('x'); ylabel('\theta(x,t)'); grid on;
        legend('Initial','Final, no control','Final, controlled','Location','best');
        subplot(2,1,2); plot(t,u,'LineWidth',1.5); xlabel('t'); ylabel('u(t)'); grid on; title('Bounded optimized control');
    else
        subplot(2,2,1); surf(y,x,theta0); axis xy image; colorbar; title('Initial temperature deviation'); view(2);shading interp;
        subplot(2,2,2); surf(y,x,Ubase(:,:,end)); axis xy image; colorbar; title('Final, no control'); view(2);shading interp; 
        subplot(2,2,3); surf(y,x,final); axis xy image; colorbar; title('Final, controlled'); colormap hot; view(2);shading interp;
        subplot(2,2,4); plot(t,u,'LineWidth',1.5); xlabel('t'); ylabel('u(t)'); grid on; title('Bounded optimized control');
    end
end

% Report whether the selected scalar feedback is below the modal threshold.
if p.a < aCrit
    stabilityText='below threshold (linearly damped modes)';
elseif abs(p.a-aCrit) <= 1e-10*max(1,aCrit)
    stabilityText='at threshold (critical first mode)';
else
    stabilityText='above threshold (first mode linearly unstable)';
end
fprintf('%s feedback assessment: %s.\n',g.name,stabilityText);
if doControl
    fprintf('%s control objective: %.6e; max |u| = %.6g.\n',g.name,controlCost,max(abs(u)));
end
out.final=final; out.finalL2=finalL2; out.lambda1=lambda1; out.aCrit=aCrit;
out.controlCost=controlCost; out.control=u; out.t=t; out.base=Ubase; out.history=Uhist;
end

function [J,U,u] = objective_blocks(v,o)
% Expand block controls to time nodes, then evaluate discrete objective.
nt=o.nt; nb=numel(v);
idx=min(nb,max(1,ceil((1:nt+1)*nb/(nt+1))));
u=v(idx(:));
[U,~]=solve_state(u,o.p,o.Lam,o.toCoeff,o.fromCoeff,o.Bshape,o.F,o.C0, ...
    o.cap,o.bw,nt,o.dt,o.theta0);
Jstate=0;
for n=1:nt+1
    th=U(:,:,n);
    Jstate=Jstate+o.dt*o.trackWeight*mean(th(:).^2);
end
Jterminal=o.terminalWeight*mean(U(:,:,end).^2,'all');
Jcontrol=o.controlReg*o.dt*sum(u(1:nt).^2);
J=Jstate+Jterminal+Jcontrol;
end

function [U,C] = solve_state(u,p,Lam,toCoeff,fromCoeff,Bshape,F,C0,cap,bw,nt,dt,theta0)
% L1 Caputo method. Nonlinear cubic term is explicit at the previous time.
% Linear scalar feedback is included in the diagonal solve.
sp=size(theta0); U=zeros([sp,nt+1]); U(:,:,1)=theta0;
C=zeros([size(C0),nt+1]); C(:,:,1)=C0;
% Modal diagonal of kappa*A^s + mu*I - a*I.
D=p.kappa*Lam+p.mu-p.a;
den=cap+D;
for n=1:nt
    thPrev=U(:,:,n);
    rhsPhysical=F + Bshape*u(n+1) - p.eta*(thPrev.^3);
    rhs=toCoeff(rhsPhysical);
    hist=zeros(size(C0));
    for k=1:n-1
        hist=hist + bw(k)*(C(:,:,n-k+1)-C(:,:,n-k));
    end
    rhs=rhs + cap*C(:,:,n) - cap*hist;
    C(:,:,n+1)=rhs./den;
    U(:,:,n+1)=fromCoeff(C(:,:,n+1));
    if any(~isfinite(U(:,:,n+1)),'all')
        error('Non-finite state encountered at step %d. Reduce time step or feedback/nonlinearity.',n);
    end
end
end
