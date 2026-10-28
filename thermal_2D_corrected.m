
%% ================================================================
% EXPERIMENT B: TWO-DIMENSIONAL FRACTIONAL THERMAL TRANSPORT
% Complete corrected MATLAB script -- MATLAB R2021a+
%
% Model:
%   C D_t^alpha(theta) + kappa*A^s(theta) + mu*theta
%       = f + b(x,y)*u(t) + a*theta - eta*theta^3
%
% Methods:
%   1. Sine-spectral spatial discretization
%   2. Caputo L1 fractional time stepping
%   3. Bounded projected-gradient control
%   4. Temporal refinement comparison
%
% Figures:
%   experiment_B_2D_evolution.png
%   experiment_B_2D_control.png
%   experiment_B_2D_refinement.png
%
% Data:
%   experiment_B_2D_results.mat
%
% The objective is temperature tracking, NOT boundary heat-flux
% optimization.
%% ================================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf(' EXPERIMENT B: TWO-DIMENSIONAL FRACTIONAL THERMAL TRANSPORT\n');
fprintf('============================================================\n\n');

%% 1. PARAMETERS

p.alpha = 0.80;
p.s     = 0.80;
p.kappa = 0.08;
p.mu    = 0.20;
p.a     = 0.25;
p.eta   = 0.10;

p.Lx = 1.0;
p.Ly = 1.0;
p.Tf = 1.0;

p.Nx = 24;
p.Ny = 24;
p.Nt = 80;

p.controlBlocks     = 6;
p.controlIterations = 5;
p.controlStep       = 0.08;
p.controlReg        = 0.02;
p.controlBound      = 1.5;

p.trackWeight    = 1.0;
p.terminalWeight = 2.0;

p.forcingAmplitude = 0.0;
p.fdStep = 2.0e-5;

p.initialAmplitude1 = 0.80;
p.initialAmplitude2 = 0.15;

fprintf('Model parameters:\n');
fprintf('  alpha = %.3f\n',p.alpha);
fprintf('  s     = %.3f\n',p.s);
fprintf('  kappa = %.3f\n',p.kappa);
fprintf('  mu    = %.3f\n',p.mu);
fprintf('  a     = %.3f\n',p.a);
fprintf('  eta   = %.3f\n',p.eta);
fprintf('  Grid  = %d x %d\n',p.Nx,p.Ny);
fprintf('  Nt    = %d\n\n',p.Nt);

%% 2. LINEAR FEEDBACK THRESHOLD

lambda1 = (pi/p.Lx)^2 + (pi/p.Ly)^2;
aCrit = p.kappa*lambda1^p.s + p.mu;
feedbackRatio = p.a/aCrit;

fprintf('Linear feedback diagnostic:\n');
fprintf('  lambda_1 = %.8f\n',lambda1);
fprintf('  a_crit   = %.8f\n',aCrit);
fprintf('  a/a_crit = %.8f\n',feedbackRatio);

if p.a < aCrit
    fprintf('  Assessment: below the first-mode linear threshold.\n\n');
else
    fprintf('  Assessment: at or above the first-mode linear threshold.\n\n');
end

%% 3. UNCONTROLLED SIMULATION

fprintf('Running uncontrolled 2-D simulation...\n');

uZero = zeros(p.controlBlocks,1);
outFree = solve_thermal_2D(p,uZero);

fprintf('Uncontrolled simulation completed.\n');
fprintf('  Final spatial L2 norm = %.8e\n',outFree.finalL2);
fprintf('  Maximum temperature   = %.8e\n',max(outFree.thetaFinal(:)));
fprintf('  Minimum temperature   = %.8e\n\n',min(outFree.thetaFinal(:)));

%% 4. TARGET TEMPERATURE

x = outFree.x;
y = outFree.y;
[X,Y] = meshgrid(x,y);

% Store target in Nx-by-Ny orientation.
targetShape = 0.35*sin(pi*X/p.Lx).*sin(pi*Y/p.Ly);
targetShape = targetShape';

%% 5. BOUNDED PROJECTED-GRADIENT CONTROL

fprintf('Optimizing the bounded control...\n');

nBlocks = p.controlBlocks;
uBlocks = zeros(nBlocks,1);

costHistory = zeros(p.controlIterations+1,1);
costHistory(1) = thermal_objective(p,uBlocks,targetShape);

for iter = 1:p.controlIterations

    grad = zeros(nBlocks,1);

    % Central finite-difference gradient
    for j = 1:nBlocks

        du = zeros(nBlocks,1);
        du(j) = p.fdStep;

        Jplus = thermal_objective(p,uBlocks+du,targetShape);
        Jminus = thermal_objective(p,uBlocks-du,targetShape);

        grad(j) = (Jplus-Jminus)/(2*p.fdStep);
    end

    Jold = thermal_objective(p,uBlocks,targetShape);

    % Projected-gradient step with backtracking
    localStep = p.controlStep;
    accepted = false;

    for back = 1:7

        uTrial = uBlocks-localStep*grad;
        uTrial = min(max(uTrial,-p.controlBound),p.controlBound);

        Jtrial = thermal_objective(p,uTrial,targetShape);

        if isfinite(Jtrial) && Jtrial <= Jold
            uBlocks = uTrial;
            accepted = true;
            break;
        end

        localStep = 0.5*localStep;
    end

    if ~accepted
        fprintf(['  Iteration %2d: no improving step found; ', ...
                 'retaining current control.\n'],iter);
    end

    costHistory(iter+1) = ...
        thermal_objective(p,uBlocks,targetShape);

    fprintf('  Iteration %2d: objective = %.8e\n', ...
        iter,costHistory(iter+1));
end

%% 6. CONTROLLED SIMULATION

fprintf('\nRunning controlled 2-D simulation...\n');

outControl = solve_thermal_2D(p,uBlocks);

fprintf('Controlled simulation completed.\n');
fprintf('  Control objective     = %.8e\n',costHistory(end));
fprintf('  Maximum |u|           = %.8e\n',max(abs(outControl.u)));
fprintf('  Final spatial L2 norm = %.8e\n\n',outControl.finalL2);

%% 7. NUMERICAL DIAGNOSTICS

freeEuclideanNorm = norm(outFree.thetaFinal(:),2);
controlledEuclideanNorm = norm(outControl.thetaFinal(:),2);

relativeFinalDifference = ...
    norm(outControl.thetaFinal(:)-outFree.thetaFinal(:),2) ...
    / max(freeEuclideanNorm,eps);

fprintf('Numerical diagnostics:\n');
fprintf('  Uncontrolled final Euclidean norm = %.8e\n',freeEuclideanNorm);
fprintf('  Controlled final Euclidean norm   = %.8e\n',controlledEuclideanNorm);
fprintf('  Relative state change             = %.8e\n', ...
    relativeFinalDifference);

if any(~isfinite(outFree.theta(:)))
    error('Uncontrolled solution contains NaN or Inf.');
end

if any(~isfinite(outControl.theta(:)))
    error('Controlled solution contains NaN or Inf.');
end

if any(~isfinite(uBlocks))
    error('Optimized control contains NaN or Inf.');
end

%% 8. FIGURE 1: TEMPERATURE FIELDS

figure('Color','w','Position',[80 100 1300 440]);

subplot(1,3,1);
%imagesc(outFree.x,outFree.y,outFree.theta(:,:,1)'); axis xy equal tight;
surf(outFree.x,outFree.y,outFree.theta(:,:,1)'); shading interp;
colorbar;
xlabel('x'); ylabel('y'); zlabel(' \theta(x,y,t) ')
title('Initial temperature');

subplot(1,3,2);
%imagesc(outFree.x,outFree.y,outFree.thetaFinal'); axis xy equal tight;
surf(outFree.x,outFree.y,outFree.thetaFinal'); shading interp;
colorbar;
xlabel('x'); ylabel('y'); zlabel(' \theta(x,y,t) ')
title('Uncontrolled final temperature');

subplot(1,3,3);
surf(outControl.x,outControl.y,outControl.thetaFinal');
%imagesc(outControl.x,outControl.y,outControl.thetaFinal'); axis xy equal tight;
colorbar; shading interp;
xlabel('x');
ylabel('y');
zlabel(' \theta(x,y,t) ');
title('Controlled final temperature');

sgtitle('Two-dimensional fractional thermal transport');

print(gcf,'experiment_B_2D_evolution.png','-dpng','-r300');

%% 9. FIGURE 2: CONTROL AND OBJECTIVE HISTORY

figure('Color','w','Position',[100 100 1100 440]);

subplot(1,2,1);

% There are Nt control values, one for each time interval, but Nt+1
% time nodes. Use left endpoints so the x and y vectors have equal size.
stairs(outControl.t(1:end-1),outControl.u,'LineWidth',1.8);

xlabel('Time t');
ylabel('Control u(t)');
title('Optimized bounded control');
grid on;
xlim([0 p.Tf]);

subplot(1,2,2);

semilogy(0:p.controlIterations,costHistory,'o-', ...
    'LineWidth',1.8,'MarkerSize',6);

xlabel('Projected-gradient iteration');
ylabel('Objective value');
title('Control objective history');
grid on;

print(gcf,'experiment_B_2D_control.png','-dpng','-r300');

%% 10. TEMPORAL REFINEMENT STUDY

fprintf('\nRunning temporal refinement test...\n');

NtList = [40 80 160];
refineResults = struct([]);

for k = 1:numel(NtList)

    pr = p;
    pr.Nt = NtList(k);

    % Linear-feedback test case
    pr.a = 0.10;
    pr.eta = 0.0;
    pr.forcingAmplitude = 0.0;

    r = solve_thermal_2D(pr,zeros(pr.controlBlocks,1));

    refineResults(k).Nt = NtList(k);
    refineResults(k).t = r.t;
    refineResults(k).thetaFinal = r.thetaFinal;
    refineResults(k).finalL2 = r.finalL2;

    fprintf('  Nt = %3d: final L2 norm = %.8e\n', ...
        NtList(k),r.finalL2);
end

theta40  = refineResults(1).thetaFinal;
theta80  = refineResults(2).thetaFinal;
theta160 = refineResults(3).thetaFinal;

err40 = norm(theta40(:)-theta160(:),2) ...
    / max(norm(theta160(:),2),eps);

err80 = norm(theta80(:)-theta160(:),2) ...
    / max(norm(theta160(:),2),eps);

refinementRatio = err40/max(err80,eps);

fprintf('\nTemporal refinement results:\n');
fprintf('  Relative error Nt=40 versus Nt=160: %.8e\n',err40);
fprintf('  Relative error Nt=80 versus Nt=160: %.8e\n',err80);
fprintf('  Error ratio e40/e80:                %.8f\n',refinementRatio);
fprintf(['  These are reference-solution comparisons, not a proof ', ...
         'of asymptotic convergence order.\n']);

%figure('Color','w','Position',[100 100 800 500]);
figure;
loglog(NtList(1:2),[err40 err80],'o-', ...
    'LineWidth',1.8,'MarkerSize',7);

xlabel('Number of time steps N_t');
ylabel('Relative final-state difference');
title('Temporal refinement at fixed spatial resolution');
grid on;

print(gcf,'experiment_B_2D_refinement.png','-dpng','-r300');

%% 11. SAVE RESULTS

results.parameters = p;
results.lambda1 = lambda1;
results.criticalGain = aCrit;
results.feedbackRatio = feedbackRatio;

results.uncontrolled = outFree;
results.controlled = outControl;

results.controlBlocks = uBlocks;
results.controlCostHistory = costHistory;
results.finalObjective = costHistory(end);

results.refinement = refineResults;
results.refinementError40 = err40;
results.refinementError80 = err80;
results.refinementRatio = refinementRatio;

save('experiment_B_2D_results.mat','results','-v7.3');

fprintf('\n============================================================\n');
fprintf('EXPERIMENT B COMPLETED\n');
fprintf('============================================================\n');
fprintf('Saved files:\n');
fprintf('  experiment_B_2D_evolution.png\n');
fprintf('  experiment_B_2D_control.png\n');
fprintf('  experiment_B_2D_refinement.png\n');
fprintf('  experiment_B_2D_results.mat\n\n');

%% ================================================================
% LOCAL FUNCTION 1: TWO-DIMENSIONAL FRACTIONAL PDE SOLVER
%% ================================================================

function out = solve_thermal_2D(p,uBlocks)

    Nx = p.Nx;
    Ny = p.Ny;
    Nt = p.Nt;

    x = (1:Nx)'*p.Lx/(Nx+1);
    y = (1:Ny)'*p.Ly/(Ny+1);

    [X,Y] = meshgrid(x,y);

    dx = p.Lx/(Nx+1);
    dy = p.Ly/(Ny+1);

    t = linspace(0,p.Tf,Nt+1);
    dt = p.Tf/Nt;

    % Orthonormal discrete sine transform matrices
    Qx = sqrt(2/(Nx+1))*sin( ...
        (pi/(Nx+1))*((1:Nx)'*(1:Nx)));

    Qy = sqrt(2/(Ny+1))*sin( ...
        (pi/(Ny+1))*((1:Ny)'*(1:Ny)));

    % Continuous Laplacian eigenvalue approximation
    jx = (1:Nx)';
    jy = (1:Ny)';

    lamX = (pi*jx/p.Lx).^2;
    lamY = (pi*jy/p.Ly).^2;

    lam = lamX+lamY';
    Lam = lam.^p.s;

    % Control profile, stored as Nx-by-Ny
    bShape = sin(pi*X/p.Lx).*sin(pi*Y/p.Ly);
    bShape = bShape';

    bNorm = sqrt(sum(bShape(:).^2)*dx*dy);
    if bNorm > 0
        bShape = bShape/bNorm;
    end

    % Initial temperature
    theta0 = p.initialAmplitude1* ...
        sin(pi*X/p.Lx).*sin(pi*Y/p.Ly) ...
        + p.initialAmplitude2* ...
        sin(2*pi*X/p.Lx).*sin(pi*Y/p.Ly);

    theta0 = theta0';

    % Modal coefficients C(:,:,1) correspond to time t=0.
    C = zeros(Nx,Ny,Nt+1);
    C(:,:,1) = Qx'*theta0*Qy;

    % Caputo L1 coefficient
    cap = dt^(-p.alpha)/gamma(2-p.alpha);

    % L1 weights b_k = (k+1)^(1-alpha)-k^(1-alpha),
    % stored at MATLAB index k+1 for k=0,...,Nt.
    weights = zeros(Nt+1,1);

    for k = 0:Nt
        weights(k+1) = (k+1)^(1-p.alpha)-k^(1-p.alpha);
    end

    % Piecewise-constant control on Nt time intervals
    blockIndex = min(p.controlBlocks, ...
        floor((0:Nt-1)*p.controlBlocks/Nt)+1);

    uTime = zeros(Nt,1);

    for n = 1:Nt
        uTime(n) = uBlocks(blockIndex(n));
    end

    theta = zeros(Nx,Ny,Nt+1);
    theta(:,:,1) = theta0;

    linearDenominator = cap+kappa_safe(p)*Lam+p.mu;

    % Time stepping:
    % At loop index n, compute the physical state at time t_n,
    % stored in C(:,:,n+1).
    for n = 1:Nt

        thetaN = Qx*C(:,:,n)*Qy';

        % Explicit nonlinear thermal feedback
        reaction = p.a*thetaN-p.eta*thetaN.^3;

        forcing = p.forcingAmplitude*ones(Nx,Ny);
        forcing = forcing+bShape*uTime(n);

        totalForce = forcing+reaction;
        forceCoeff = Qx'*totalForce*Qy;

        % Correct L1 history:
        % Sum previous increments C_j-C_{j-1}, j=2,...,n,
        % with weights b_{n+1-j}. For n=1, the history is empty.
        history = zeros(Nx,Ny);

        for j = 2:n
            k = n+1-j;
            history = history+weights(k+1)* ...
                (C(:,:,j)-C(:,:,j-1));
        end

        rhs = cap*C(:,:,n)-cap*history+forceCoeff;

        C(:,:,n+1) = rhs./linearDenominator;

        thetaNext = Qx*C(:,:,n+1)*Qy';
        theta(:,:,n+1) = thetaNext;

        if any(~isfinite(thetaNext(:)))
            error(['Non-finite temperature at time step ', ...
                num2str(n),'. Reduce the time step or adjust parameters.']);
        end
    end

    out.x = x;
    out.y = y;
    out.t = t;

    out.theta = theta;
    out.thetaInitial = theta(:,:,1);
    out.thetaFinal = theta(:,:,end);

    out.u = uTime;
    out.uBlocks = uBlocks;
    out.bShape = bShape;

    out.finalL2 = sqrt(sum(out.thetaFinal(:).^2)*dx*dy);

    out.Nx = Nx;
    out.Ny = Ny;
    out.Nt = Nt;
    out.dx = dx;
    out.dy = dy;
    out.dt = dt;
end

%% ================================================================
% LOCAL FUNCTION 2: TEMPERATURE-TRACKING OBJECTIVE
%% ================================================================

function J = thermal_objective(p,uBlocks,targetShape)

    out = solve_thermal_2D(p,uBlocks);

    Nt = p.Nt;
    dt = p.Tf/Nt;

    % Trapezoidal approximation to the time-integrated tracking error
    trackingCost = 0;

    for n = 1:Nt+1

        target = targetShape*exp(-0.5*out.t(n));
        errorField = out.theta(:,:,n)-target;

        meanSquareError = sum(errorField(:).^2)/(p.Nx*p.Ny);

        if n == 1 || n == Nt+1
            weight = 0.5;
        else
            weight = 1.0;
        end

        trackingCost = trackingCost+weight*meanSquareError;
    end

    trackingCost = dt*trackingCost;

    % Terminal tracking penalty
    targetFinal = targetShape*exp(-0.5*p.Tf);
    terminalError = out.thetaFinal-targetFinal;

    terminalCost = p.terminalWeight* ...
        sum(terminalError(:).^2)/(p.Nx*p.Ny);

    % Control regularization
    controlCost = p.controlReg*sum(uBlocks(:).^2);

    J = p.trackWeight*trackingCost+terminalCost+controlCost;

    if ~isfinite(J)
        J = realmax;
    end
end

%% ================================================================
% LOCAL FUNCTION 3: SAFE SCALAR PARAMETER
%% ================================================================

function value = kappa_safe(p)
    value = p.kappa;
end
