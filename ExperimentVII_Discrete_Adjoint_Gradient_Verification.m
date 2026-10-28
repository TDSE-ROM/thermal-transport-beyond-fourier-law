function ExperimentVII_Discrete_Adjoint_Gradient_Verification()
% EXPERIMENTVII_DISCRETE_ADJOINT_GRADIENT_VERIFICATION
% MATLAB R2021a-compatible verification of a discrete adjoint gradient for
% Caputo L1 time stepping and P1 FEM fractional nonlocal diffusion.
%
% Outputs:
%   ExperimentVII_Taylor_Remainder.csv
%   ExperimentVII_Adjoint_Gradient.csv
%   ExperimentVII_Adjoint_Diagnostics.csv
%   ExperimentVII_Taylor_Remainder.png
%
% This is a gradient-verification experiment, not an optimization routine.
% The discrete adjoint is derived from the same state recurrence used below.

clc;
close all;
fprintf('\nEXPERIMENT VII: DISCRETE ADJOINT AND GRADIENT VERIFICATION\n');
fprintf('=========================================================\n');

%% 1. Parameters
N       = 30;      % Interior finite-element nodes
alpha   = 0.8;     % Caputo order
s       = 0.8;     % Fractional spatial order
kappa   = 0.15;
mu      = 0.20;
Tf      = 4.0;
Nt      = 200;
dt      = Tf/Nt;
nBlocks = 10;
gammaU  = 0.08;
omegaT  = 1.0;
uMin    = -2.0;
uMax    =  2.0;

fprintf('Interior nodes N               = %d\n',N);
fprintf('Time steps Nt                  = %d\n',Nt);
fprintf('Control blocks                 = %d\n',nBlocks);
fprintf('alpha                          = %.3f\n',alpha);
fprintf('s                              = %.3f\n',s);
fprintf('Final time Tf                  = %.3f\n',Tf);
fprintf('dt                             = %.4e\n',dt);

%% 2. P1 finite-element matrices on (0,1), homogeneous Dirichlet data
xAll = linspace(0,1,N+2)';
x = xAll(2:end-1);
h = 1/(N+1);
M = sparse(N,N);
K = sparse(N,N);
for i = 1:N
    M(i,i) = 2*h/3;
    K(i,i) = 2/h;
    if i < N
        M(i,i+1) = h/6;
        M(i+1,i) = h/6;
        K(i,i+1) = -1/h;
        K(i+1,i) = -1/h;
    end
end

%% 3. Spectral fractional stiffness matrix
% Generalized eigenvectors satisfy V'*M*V = I.
[V,D] = eig(full(K),full(M));
lam = real(diag(D));
[lam,idx] = sort(lam,'ascend');
V = real(V(:,idx));
for j = 1:N
    V(:,j) = V(:,j)/sqrt(V(:,j)'*M*V(:,j));
end
Ks = M*V*diag(lam.^s)*V'*M;
Ks = sparse((Ks+Ks')/2);
L = kappa*Ks + mu*M;

%% 4. Caputo L1 coefficients and fixed matrix
% a_k = k^(1-alpha) - (k-1)^(1-alpha), k=1,...,Nt.
a = zeros(Nt,1);
for k = 1:Nt
    a(k) = k^(1-alpha)-(k-1)^(1-alpha);
end
cL1 = 1/(gamma(2-alpha)*dt^alpha);
A = cL1*a(1)*M + L;
dA = decomposition(A,'chol');

%% 5. Initial state, target, forcing and control profile
thetaInitial = 0.10*sin(pi*x);
thetaTarget = zeros(N,Nt);
forcing = zeros(N,Nt);
for n = 1:Nt
    t = n*dt;
    thetaTarget(:,n) = ...
        (0.55 + 0.20*sin(pi*t/Tf))*sin(pi*x) ...
        + 0.15*sin(2*pi*t/Tf)*sin(2*pi*x);
    forcing(:,n) = ...
        0.10*(1 + 0.25*cos(2*pi*t/Tf))*sin(pi*x);
end
b = sin(pi*x);
b = b/sqrt(b'*M*b);
Mb = M*b;

%% 6. Block control, direction, and perturbation sizes
blockOfStep = min(nBlocks,ceil((1:Nt)'*nBlocks/Nt));
U = 0.15*cos((1:nBlocks)'*pi/(nBlocks+1));
d = sin((1:nBlocks)'*sqrt(2)) ...
  + 0.3*cos((1:nBlocks)'*sqrt(3));
d = d/norm(d);
epsList = 2.^(-(2:9))';

if any(U < uMin) || any(U > uMax)
    error('Base control is outside the admissible bounds.');
end
if any(U + max(epsList)*d < uMin) || ...
   any(U + max(epsList)*d > uMax) || ...
   any(U - max(epsList)*d < uMin) || ...
   any(U - max(epsList)*d > uMax)
    error('Taylor perturbation leaves the admissible control set.');
end

%% 7. Base objective, state, adjoint and gradient
[J0,gU,theta,lambda,diag0] = reducedObjectiveGradient(U);
Jcheck = reducedObjectiveOnly(U);
gDotD = real(gU(:)'*d(:));

fprintf('\nBASE CONTROL AND RESIDUAL AUDIT\n');
fprintf('Objective J(U)                  = %.15e\n',J0);
fprintf('Independent objective check     = %.15e\n',Jcheck);
fprintf('Objective consistency error     = %.3e\n',abs(J0-Jcheck));
fprintf('Maximum normalized state res.   = %.3e\n',diag0.stateResidual);
fprintf('Maximum normalized adjoint res. = %.3e\n',diag0.adjointResidual);
fprintf('Adjoint gradient norm           = %.6e\n',norm(gU));

%% 8. Independent directional-derivative check
fdStep = 1.0e-5;
fdDirectional = (reducedObjectiveOnly(U+fdStep*d) ...
                -reducedObjectiveOnly(U-fdStep*d))/(2*fdStep);
derivativeDifference = abs(gDotD-fdDirectional);
relativeDerivativeDifference = derivativeDifference / ...
    max([1,abs(gDotD),abs(fdDirectional)]);

fprintf('\nDIRECTIONAL DERIVATIVE CHECK\n');
fprintf('Adjoint derivative              = %.12e\n',gDotD);
fprintf('Central finite-difference value = %.12e\n',fdDirectional);
fprintf('Absolute difference             = %.3e\n',derivativeDifference);
fprintf('Relative difference             = %.3e\n',relativeDerivativeDifference);

%% 9. Taylor remainder test
% R(eps) = |J(U+eps*d)-J(U)-eps*gradJ(U)'*d|.
% All vectors are explicitly allocated with the same nEps-by-1 shape.
epsCol = epsList(:);
nEps = numel(epsCol);
RCol = zeros(nEps,1);
JplusCol = zeros(nEps,1);
JminusCol = zeros(nEps,1);
for k = 1:nEps
    epsk = epsCol(k);
    JplusCol(k) = reducedObjectiveOnly(U + epsk*d);
    JminusCol(k) = reducedObjectiveOnly(U - epsk*d);
    RCol(k) = abs(JplusCol(k)-J0-epsk*gDotD);
end
ObservedOrder = NaN(nEps,1);
for k = 1:nEps-1
    if RCol(k) > 0 && RCol(k+1) > 0
        ObservedOrder(k) = log(RCol(k)/RCol(k+1)) / ...
                           log(epsCol(k)/epsCol(k+1));
    end
end

% Defensive dimension checks (all are built from epsCol in this function).
assert(numel(RCol)==nEps && numel(JplusCol)==nEps && ...
       numel(JminusCol)==nEps && numel(ObservedOrder)==nEps, ...
       'Internal Taylor-test arrays have inconsistent lengths.');
TaylorTable = table(epsCol,RCol,ObservedOrder,JplusCol,JminusCol, ...
    'VariableNames',{'Epsilon','TaylorRemainder','ObservedOrder', ...
                     'Jplus','Jminus'});
fprintf('\nTAYLOR REMAINDER TABLE\n');
disp(TaylorTable);
writetable(TaylorTable,'ExperimentVII_Taylor_Remainder.csv');

%% 10. Taylor remainder plot and slope-two reference
valid = isfinite(epsCol) & epsCol>0 & isfinite(RCol) & RCol>0;
if any(valid)
    epsValid = epsCol(valid);
    RValid = RCol(valid);
    [~,iAnchor] = max(epsValid);
    epsAnchor = epsValid(iAnchor);
    RAnchor = RValid(iAnchor);
    epsLine = sort(epsValid);
    RLine = RAnchor*(epsLine/epsAnchor).^2;

    fig = figure('Color','w','Name','Experiment VII: Taylor Remainder');
    loglog(epsValid,RValid,'o-','LineWidth',1.6,'MarkerSize',6);
    hold on;
    loglog(epsLine,RLine,'--','LineWidth',1.5);
    grid on; box on;
    xlabel('\epsilon','Interpreter','tex');
    ylabel('Taylor remainder R(\epsilon)','Interpreter','tex');
    title('Discrete-adjoint Taylor remainder test');
    legend('Computed remainder','Reference slope 2','Location','best');
    set(gca,'FontSize',11,'LineWidth',1);
    set(fig,'ToolBar','none','MenuBar','none');
    print(fig,'ExperimentVII_Taylor_Remainder.png','-dpng','-r300');
else
    warning('No positive finite Taylor remainders; log-log plot was not generated.');
end

%% 11. Save gradient and diagnostic tables
GradientTable = table((1:nBlocks)',U(:),d(:),gU(:), ...
    'VariableNames',{'Block','Control','Direction','AdjointGradient'});
writetable(GradientTable,'ExperimentVII_Adjoint_Gradient.csv');

DiagnosticTable = table(J0,Jcheck,diag0.stateResidual, ...
    diag0.adjointResidual,norm(gU),gDotD,fdDirectional, ...
    derivativeDifference,relativeDerivativeDifference, ...
    'VariableNames',{'Objective','ObjectiveCheck','StateResidual', ...
    'AdjointResidual','GradientNorm','AdjointDirectionalDerivative', ...
    'FiniteDifferenceDirectionalDerivative','DerivativeDifference', ...
    'RelativeDerivativeDifference'});
writetable(DiagnosticTable,'ExperimentVII_Adjoint_Diagnostics.csv');

fprintf('\nFiles written:\n');
fprintf('  ExperimentVII_Taylor_Remainder.csv\n');
fprintf('  ExperimentVII_Adjoint_Gradient.csv\n');
fprintf('  ExperimentVII_Adjoint_Diagnostics.csv\n');
fprintf('  ExperimentVII_Taylor_Remainder.png\n');
fprintf('\nInterpretation:\n');
fprintf(['For a consistent discrete adjoint, the Taylor remainder should ' ...
         'show an approximately quadratic regime.\n']);
fprintf(['A slope near two is expected only before roundoff and other ' ...
         'numerical errors dominate.\n']);
fprintf(['This test verifies gradient consistency; it does not establish ' ...
         'global optimality.\n']);

%% Nested functions
    function [J,gBlock,thetaState,lambdaState,diagOut] = ...
            reducedObjectiveGradient(Ublock)
        u = Ublock(blockOfStep);
        thetaState = zeros(N,Nt);
        stateResidual = 0;

        % Forward Caputo-L1 state solve.
        for nn = 1:Nt
            rhs = M*forcing(:,nn) + Mb*u(nn) ...
                + cL1*a(nn)*M*thetaInitial;
            for kk = 1:nn-1
                qhist = a(nn-kk+1)-a(nn-kk);
                rhs = rhs-cL1*qhist*M*thetaState(:,kk);
            end
            thetaState(:,nn) = dA\rhs;

            % Direct residual evaluation of the discrete state equation.
            caputoTerm = zeros(N,1);
            for kk = 1:nn
                if kk == 1
                    thetaPrevious = thetaInitial;
                else
                    thetaPrevious = thetaState(:,kk-1);
                end
                caputoTerm = caputoTerm + a(nn-kk+1) ...
                    *(thetaState(:,kk)-thetaPrevious);
            end
            residualVector = cL1*M*caputoTerm + L*thetaState(:,nn) ...
                - M*forcing(:,nn)-Mb*u(nn);
            scale = max(1,norm(cL1*M*caputoTerm) ...
                + norm(L*thetaState(:,nn)) ...
                + norm(M*forcing(:,nn))+norm(Mb*u(nn)));
            stateResidual = max(stateResidual,norm(residualVector)/scale);
        end

        % Fully discrete objective.
        tracking = thetaState-thetaTarget;
        Jtrack = 0;
        for nn = 1:Nt
            Jtrack = Jtrack + 0.5*dt* ...
                (tracking(:,nn)'*M*tracking(:,nn));
        end
        terminalError = tracking(:,Nt);
        Jterminal = 0.5*omegaT*(terminalError'*M*terminalError);
        Jcontrol = 0.5*gammaU*dt*sum(u.^2);
        J = Jtrack+Jterminal+Jcontrol;

        % Derivative with respect to each time-level state.
        stateGradient = dt*M*tracking;
        stateGradient(:,Nt) = stateGradient(:,Nt) ...
            + omegaT*M*terminalError;

        % Backward discrete adjoint (transpose of the state Jacobian).
        lambdaState = zeros(N,Nt);
        adjointResidual = 0;
        for nn = Nt:-1:1
            rhsAdj = stateGradient(:,nn);
            for mm = nn+1:Nt
                qhist = a(mm-nn+1)-a(mm-nn);
                rhsAdj = rhsAdj-cL1*qhist*M'*lambdaState(:,mm);
            end
            lambdaState(:,nn) = dA'\rhsAdj;

            lhsAdj = A'*lambdaState(:,nn);
            for mm = nn+1:Nt
                qhist = a(mm-nn+1)-a(mm-nn);
                lhsAdj = lhsAdj+cL1*qhist*M'*lambdaState(:,mm);
            end
            adjResVec = lhsAdj-stateGradient(:,nn);
            scaleAdj = max(1,norm(lhsAdj)+norm(stateGradient(:,nn)));
            adjointResidual = max(adjointResidual,norm(adjResVec)/scaleAdj);
        end

        % Gradient with respect to the block controls.
        gTime = zeros(Nt,1);
        for nn = 1:Nt
            gTime(nn) = Mb'*lambdaState(:,nn)+gammaU*dt*u(nn);
        end
        gBlock = zeros(nBlocks,1);
        for qq = 1:nBlocks
            gBlock(qq) = sum(gTime(blockOfStep==qq));
        end
        diagOut.stateResidual = stateResidual;
        diagOut.adjointResidual = adjointResidual;
    end

    function J = reducedObjectiveOnly(Ublock)
        % Objective-only evaluation using the identical state scheme.
        u = Ublock(blockOfStep);
        thetaState = zeros(N,Nt);
        for nn = 1:Nt
            rhs = M*forcing(:,nn)+Mb*u(nn) ...
                + cL1*a(nn)*M*thetaInitial;
            for kk = 1:nn-1
                qhist = a(nn-kk+1)-a(nn-kk);
                rhs = rhs-cL1*qhist*M*thetaState(:,kk);
            end
            thetaState(:,nn) = dA\rhs;
        end
        tracking = thetaState-thetaTarget;
        Jtrack = 0;
        for nn = 1:Nt
            Jtrack = Jtrack+0.5*dt* ...
                (tracking(:,nn)'*M*tracking(:,nn));
        end
        terminalError = tracking(:,Nt);
        Jterminal = 0.5*omegaT*(terminalError'*M*terminalError);
        Jcontrol = 0.5*gammaU*dt*sum(u.^2);
        J = Jtrack+Jterminal+Jcontrol;
    end
end
