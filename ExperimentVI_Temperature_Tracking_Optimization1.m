
function ExperimentVI_Temperature_Tracking_Optimization1
% EXPERIMENT VI: CONSTRAINED TEMPERATURE-TRACKING OPTIMIZATION
% MATLAB R2021a compatible; no Optimization Toolbox required.
%
% State:
%   D_C^alpha theta + kappa*K_s*theta + mu*M*theta
%       = F(t) + M*b*u(t).
%
% Time discretization: Caputo L1.
% Space discretization: P1 finite elements.
% Control: scalar, bounded, piecewise constant for optimization.
%
% Comparisons:
%   (1) No control.
%   (2) Saturated proportional feedback.
%   (3) Best feasible candidate among the baseline and multi-start
%       block-control optimization runs.
%
% Outputs:
%   ExperimentVI_Optimization_Results.csv
%   ExperimentVI_Control_Histories.png
%   ExperimentVI_Temperature_Profiles.png
%   ExperimentVI_Optimization_Diagnostics.csv
%
% Important: fminsearch is a local optimizer. This code does not
% certify global optimality. The selected candidate is compared
% against the admissible zero and feedback controls.

    clc;
    close all;

    %% 1. Parameters
    N       = 30;
    alpha   = 0.80;
    s       = 0.80;
    kappa   = 0.15;
    mu      = 0.20;

    Tf      = 4.0;
    Nt      = 400;
    dt      = Tf/Nt;

    gammaU  = 0.08;
    omegaF  = 2.0;

    uMin    = -2.0;
    uMax    =  2.0;
    epsSat  = 1.0e-6;
    epsDen  = 1.0e-12;

    nBlocks = 10;
    Kp      = 3.0;

    if alpha <= 0 || alpha > 1 || s <= 0
        error('Require 0 < alpha <= 1 and s > 0.');
    end
    if mod(Nt,nBlocks) ~= 0
        error('Nt must be divisible by nBlocks.');
    end
    if gammaU <= 0 || omegaF < 0 || uMin >= uMax
        error('Invalid objective weights or control bounds.');
    end

    blockLength = Nt/nBlocks;
    time = (0:Nt)'*dt;

    %% 2. Finite-element matrices
    h = 1/(N+1);
    x = (1:N)'*h;

    M = spdiags([(h/6)*ones(N,1), ...
                 (2*h/3)*ones(N,1), ...
                 (h/6)*ones(N,1)],[-1 0 1],N,N);

    K = spdiags([(-1/h)*ones(N,1), ...
                 (2/h)*ones(N,1), ...
                 (-1/h)*ones(N,1)],[-1 0 1],N,N);

    M = (M+M')/2;
    K = (K+K')/2;

    %% 3. Fractional spatial stiffness K_s
    [V,D] = eig(full(K),full(M));
    lambda = real(diag(D));
    [lambda,idx] = sort(lambda,'ascend');
    V = real(V(:,idx));

    for j = 1:N
        q = sqrt(real(V(:,j)'*M*V(:,j)));
        if ~isfinite(q) || q <= 0
            error('Invalid generalized eigenvector normalization.');
        end
        V(:,j) = V(:,j)/q;
    end

    if any(lambda <= 0)
        error('The generalized eigenvalues must be positive.');
    end

    Ks = M*V*diag(lambda.^s)*V'*M;
    Ks = (Ks+Ks')/2;

    %% 4. Initial data, target and forcing
    theta0 = sin(pi*x);

    % Control acts through a spatial profile aligned with the
    % dominant target mode.
    b = sin(pi*x);
    b = b/sqrt(real(b'*M*b));

    target = zeros(N,Nt+1);
    forcing = zeros(N,Nt);

    for n = 1:Nt+1
        tn = time(n);
        target(:,n) = ...
            (0.55+0.20*sin(pi*tn/Tf))*sin(pi*x) ...
            + 0.15*sin(2*pi*tn/Tf)*sin(2*pi*x);
    end

    for n = 1:Nt
        tn = time(n+1);
        forcing(:,n) = ...
            0.10*(1+0.25*cos(2*pi*tn/Tf))*sin(pi*x);
    end

    thetaF = target(:,end);

    %% 5. L1 coefficients and state matrix
    % a_j = (j+1)^(1-alpha)-j^(1-alpha), j=0,...,Nt-1.
    a = zeros(Nt,1);
    a(1) = 1;
    for j = 1:Nt-1
        a(j+1) = (j+1)^(1-alpha)-j^(1-alpha);
    end

    cL1 = 1/(gamma(2-alpha)*dt^alpha);
    A = cL1*a(1)*M + kappa*Ks + mu*M;

    %% 6. Uncontrolled case
    uZero = zeros(Nt,1);
    [thetaZero,resZero] = simulateState( ...
        uZero,theta0,forcing,A,M,b,a,cL1);

    %% 7. Explicitly defined saturated feedback baseline
    % u_n = sat(Kp * <b, theta_d(t_n)-theta_{n-1}>_M).
    thetaBase = zeros(N,Nt+1);
    thetaBase(:,1) = theta0;
    uBase = zeros(Nt,1);

    for n = 1:Nt
        e = target(:,n+1)-thetaBase(:,n);
        rawControl = Kp*real(b'*M*e);
        uBase(n) = min(max(rawControl,uMin),uMax);

        thetaBase(:,n+1) = advanceState( ...
            n,thetaBase,forcing,uBase,A,M,b,a,cL1);
    end

    [thetaBase,resBase] = simulateState( ...
        uBase,theta0,forcing,A,M,b,a,cL1);

    %% 8. Baseline and zero-control objectives
    JZero = objectiveValue(thetaZero,uZero,target,thetaF, ...
        M,gammaU,omegaF,dt);

    JBase = objectiveValue(thetaBase,uBase,target,thetaF, ...
        M,gammaU,omegaF,dt);

    %% 9. Multi-start optimization over blockwise controls
    % The transformed variables enforce the box bounds.
    % Start from zero and from block averages of the baseline.
    midpoint = (uMin+uMax)/2;
    halfwidth = (uMax-uMin)/2;

    uBaseBlock = zeros(nBlocks,1);
    for j = 1:nBlocks
        ids = (j-1)*blockLength+(1:blockLength);
        uBaseBlock(j) = mean(uBase(ids));
    end

    zZero = zeros(nBlocks,1);
    zBase = controlToZ(uBaseBlock,midpoint,halfwidth);

    % Deterministic additional starting guesses.
    zStarts = [zZero,zBase, ...
        0.5*sin((1:nBlocks)'*2*pi/nBlocks), ...
        0.5*cos((1:nBlocks)'*2*pi/nBlocks)];
    
    %zStarts = [zZero,zBase];

%     options = optimset( ...
%         'Display','off', ...
%         'MaxIter',5000, ...
%         'MaxFunEvals',30000, ...
%         'TolX',1e-8, ...
%         'TolFun',1e-10);

    options = optimset( ...
    'Display','iter', ...
    'MaxIter',300, ...
    'MaxFunEvals',1500, ...
    'TolX',1e-5, ...
    'TolFun',1e-7);

    bestBlockControl = uBaseBlock;
    bestBlockObjective = inf;
    bestExitFlag = NaN;
    totalIterations = 0;
    successfulRuns = 0;

    for run = 1:size(zStarts,2)
        zStart = zStarts(:,run);

        fun = @(z) blockObjective(z,midpoint,halfwidth, ...
            nBlocks,blockLength,theta0,forcing,A,M,b,a,cL1, ...
            target,thetaF,gammaU,omegaF,dt);

        [zCandidate,Jcandidate,exitflag,output] = ...
            fminsearch(fun,zStart,options);

        totalIterations = totalIterations+output.iterations;

        if exitflag > 0
            successfulRuns = successfulRuns+1;
        end

        uCandidateBlock = midpoint+halfwidth*tanh(zCandidate(:));
        uCandidateBlock = min(max(uCandidateBlock,uMin),uMax);

        if isfinite(Jcandidate) && Jcandidate < bestBlockObjective
            bestBlockObjective = Jcandidate;
            bestBlockControl = uCandidateBlock;
            bestExitFlag = exitflag;
        end
    end

    uBlockOpt = bestBlockControl;
    uBlockExpanded = expandBlocks(uBlockOpt,Nt,blockLength);

    [thetaBlockOpt,resBlockOpt] = simulateState( ...
        uBlockExpanded,theta0,forcing,A,M,b,a,cL1);

    JBlockOpt = objectiveValue(thetaBlockOpt,uBlockExpanded, ...
        target,thetaF,M,gammaU,omegaF,dt);

    %% 10. Select best admissible candidate
    % The feedback baseline is admissible and may outperform the
    % local optimizer. Include it explicitly in the candidate set.
    candidateObjectives = [JZero,JBase,JBlockOpt];
    [bestJ,bestIndex] = min(candidateObjectives);

    if bestIndex == 1
        caseOptimal = 'No control candidate';
        uOpt = uZero;
        thetaOpt = thetaZero;
        residualOpt = resZero;
    elseif bestIndex == 2
        caseOptimal = 'Baseline feedback candidate';
        uOpt = uBase;
        thetaOpt = thetaBase;
        residualOpt = resBase;
    else
        caseOptimal = 'Optimized block-control candidate';
        uOpt = uBlockExpanded;
        thetaOpt = thetaBlockOpt;
        residualOpt = resBlockOpt;
    end

    JOpt = objectiveValue(thetaOpt,uOpt,target,thetaF, ...
        M,gammaU,omegaF,dt);

    %% 11. Diagnostics for the three requested cases
    D0 = computeDiagnostics(thetaZero,uZero,target,thetaF, ...
        M,gammaU,omegaF,dt,uMin,uMax,epsSat);

    DB = computeDiagnostics(thetaBase,uBase,target,thetaF, ...
        M,gammaU,omegaF,dt,uMin,uMax,epsSat);

    DO = computeDiagnostics(thetaOpt,uOpt,target,thetaF, ...
        M,gammaU,omegaF,dt,uMin,uMax,epsSat);

    trackingGain = 100*(D0.Etrack-DO.Etrack) ...
        /max(D0.Etrack,epsDen);

    objectiveReduction = 100*(JZero-JOpt)/max(JZero,epsDen);

    %% 12. Report
    fprintf('\nEXPERIMENT VI: TEMPERATURE-TRACKING OPTIMIZATION\n');
    fprintf('====================================================\n');
    fprintf('Interior nodes N                  = %d\n',N);
    fprintf('Time steps Nt                     = %d\n',Nt);
    fprintf('alpha                             = %.4f\n',alpha);
    fprintf('s                                 = %.4f\n',s);
    fprintf('Final time Tf                     = %.4f\n',Tf);
    fprintf('dt                                = %.6e\n',dt);
    fprintf('Number of control blocks          = %d\n',nBlocks);
    fprintf('Control bounds                    = [%.3f, %.3f]\n',uMin,uMax);
    fprintf('Successful fminsearch runs        = %d / %d\n', ...
        successfulRuns,size(zStarts,2));
    fprintf('Total optimizer iterations        = %d\n',totalIterations);
    fprintf('Best block-run exit flag          = %g\n',bestExitFlag);
    fprintf('No-control objective              = %.12e\n',JZero);
    fprintf('Baseline objective                = %.12e\n',JBase);
    fprintf('Best block-control objective      = %.12e\n',JBlockOpt);
    fprintf('Selected feasible candidate       = %s\n',caseOptimal);
    fprintf('Selected objective                = %.12e\n',JOpt);
    fprintf('Re-evaluated objective            = %.12e\n', ...
        objectiveValue(thetaOpt,uOpt,target,thetaF,M, ...
        gammaU,omegaF,dt));
    fprintf('Tracking improvement vs no control = %.6f %%\n',trackingGain);
    fprintf('Objective reduction vs no control = %.6f %%\n',objectiveReduction);
    fprintf('Maximum normalized state residual = %.6e\n\n', ...
        max([resZero,resBase,residualOpt]));

    %% 13. Results table
    CaseName = {'No control';'Baseline feedback';'Selected best feasible'};

    TrackingError = [D0.Etrack;DB.Etrack;DO.Etrack];
    TerminalError = [D0.Eterminal;DB.Eterminal;DO.Eterminal];
    ControlNorm = [D0.Econtrol;DB.Econtrol;DO.Econtrol];
    TrackingContribution = [D0.Jtrack;DB.Jtrack;DO.Jtrack];
    TerminalContribution = [D0.Jterminal;DB.Jterminal;DO.Jterminal];
    ControlPenalty = [D0.Jcontrol;DB.Jcontrol;DO.Jcontrol];
    TotalObjective = [D0.J;DB.J;DO.J];
    SaturationFraction = [D0.saturation;DB.saturation;DO.saturation];

    Results = table(CaseName,TrackingError,TerminalError, ...
        ControlNorm,TrackingContribution,TerminalContribution, ...
        ControlPenalty,TotalObjective,SaturationFraction);

    disp(Results);
    writetable(Results,'ExperimentVI_Optimization_Results.csv');

    %% 14. Residual and optimization audit
    AuditName = { ...
        'Maximum normalized state residual, no control'; ...
        'Maximum normalized state residual, baseline'; ...
        'Maximum normalized state residual, selected candidate'; ...
        'Objective: no control'; ...
        'Objective: baseline'; ...
        'Objective: block optimization'; ...
        'Objective: selected candidate'; ...
        'Best block-run exit flag'; ...
        'Successful optimization runs'};

    AuditValue = [resZero;resBase;residualOpt; ...
        JZero;JBase;JBlockOpt;JOpt;bestExitFlag;successfulRuns];

    Audit = table(AuditName,AuditValue);
    writetable(Audit,'ExperimentVI_Optimization_Diagnostics.csv');

    %% 15. Figure: control histories
    fig1 = figure('Color','w','Position',[100 100 950 520]);

    stairs(time(2:end),uBase,'--','LineWidth',1.6);
    hold on;
    stairs(time(2:end),uBlockExpanded,'-','LineWidth',1.6);
    stairs(time(2:end),uOpt,':','LineWidth',1.8);
    yline(uMin,'k:');
    yline(uMax,'k:');
    hold off;

    xlabel('Time, t');
    ylabel('Control amplitude, u(t)');
    title('Feedback and optimized admissible controls');
    legend('Baseline feedback','Block optimizer candidate', ...
        'Selected best feasible candidate','Lower bound','Upper bound', ...
        'Location','best');
    xlim([0,Tf]);
    ylim([uMin-0.1,uMax+0.1]);
    grid on;
    box on;
    set(gca,'FontSize',11,'LineWidth',1.0);
    removeToolbar(fig1);
    exportFigure(fig1,'ExperimentVI_Control_Histories.png');

    %% 16. Figure: terminal profiles and tracking errors
    fig2 = figure('Color','w','Position',[100 100 1000 620]);

    subplot(2,1,1);
    plot(x,thetaZero(:,end),'LineWidth',1.6);
    hold on;
    plot(x,thetaBase(:,end),'--','LineWidth',1.6);
    plot(x,thetaOpt(:,end),'-','LineWidth',1.8);
    plot(x,thetaF,':','LineWidth',1.8);
    hold off;
    xlabel('Position, x');
    ylabel('Temperature');
    title('Terminal temperature profiles');
    legend('No control','Baseline feedback','Selected candidate', ...
        'Terminal target','Location','best');
    grid on;
    box on;
    set(gca,'FontSize',10,'LineWidth',1.0);

    subplot(2,1,2);
    errZero = zeros(Nt+1,1);
    errBase = zeros(Nt+1,1);
    errOpt = zeros(Nt+1,1);

    for n = 1:Nt+1
        e0 = thetaZero(:,n)-target(:,n);
        eb = thetaBase(:,n)-target(:,n);
        eo = thetaOpt(:,n)-target(:,n);
        errZero(n) = sqrt(max(0,real(e0'*M*e0)));
        errBase(n) = sqrt(max(0,real(eb'*M*eb)));
        errOpt(n) = sqrt(max(0,real(eo'*M*eo)));
    end

    plot(time,errZero,'LineWidth',1.5);
    hold on;
    plot(time,errBase,'--','LineWidth',1.5);
    plot(time,errOpt,'-','LineWidth',1.7);
    hold off;
    xlabel('Time, t');
    ylabel('Mass-norm tracking error');
    title('Tracking error over time');
    legend('No control','Baseline feedback','Selected candidate', ...
        'Location','best');
    grid on;
    box on;
    set(gca,'FontSize',10,'LineWidth',1.0);

    removeToolbar(fig2);
    exportFigure(fig2,'ExperimentVI_Temperature_Profiles.png');

    fprintf('\nFiles written:\n');
    fprintf('  ExperimentVI_Optimization_Results.csv\n');
    fprintf('  ExperimentVI_Optimization_Diagnostics.csv\n');
    fprintf('  ExperimentVI_Control_Histories.png\n');
    fprintf('  ExperimentVI_Temperature_Profiles.png\n');

    fprintf('\nInterpretation:\n');
    fprintf(['The selected feasible candidate has objective no larger ', ...
        'than the zero-control or baseline candidate.\n']);
    fprintf(['The multi-start local search does not certify a global ', ...
        'minimum; report it as a numerical candidate.\n']);
    fprintf(['Inspect the optimizer exit flag, residual audit, ', ...
        'and saturation fraction before publication.\n']);

end

%% ============================================================
function z = controlToZ(u,midpoint,halfwidth)
    q = (u(:)-midpoint)/halfwidth;
    q = min(max(q,-0.999999),0.999999);
    z = atanh(q);
end

%% ============================================================
function u = expandBlocks(uBlock,Nt,blockLength)
    u = zeros(Nt,1);
    for j = 1:numel(uBlock)
        ids = (j-1)*blockLength+(1:blockLength);
        u(ids) = uBlock(j);
    end
end

%% ============================================================
function J = blockObjective(z,midpoint,halfwidth,nBlocks, ...
    blockLength,theta0,forcing,A,M,b,a,cL1,target,thetaF, ...
    gammaU,omegaF,dt)

    uBlock = midpoint+halfwidth*tanh(z(:));
    uBlock = min(max(uBlock,midpoint-halfwidth),midpoint+halfwidth);
    u = expandBlocks(uBlock,size(forcing,2),blockLength);

    theta = simulateState(u,theta0,forcing,A,M,b,a,cL1);

    J = objectiveValue(theta,u,target,thetaF,M, ...
        gammaU,omegaF,dt);

    if ~isfinite(J) || any(~isfinite(theta(:)))
        J = realmax/1e100;
    end
end

%% ============================================================
function [theta,maxResidual] = simulateState( ...
    u,theta0,forcing,A,M,b,a,cL1)

    N = numel(theta0);
    Nt = numel(u);
    theta = zeros(N,Nt+1);
    theta(:,1) = theta0;
    maxResidual = 0;

    for n = 1:Nt
        theta(:,n+1) = advanceState( ...
            n,theta,forcing,u,A,M,b,a,cL1);

        % Reconstruct the full discrete equation residual.
        history = zeros(N,1);
        for j = 1:n-1
            history = history ...
                + a(n-j+1)*(theta(:,j+1)-theta(:,j));
        end

        lhs = A*theta(:,n+1);
        rhs = forcing(:,n)+M*b*u(n) ...
            + cL1*M*(a(1)*theta(:,n)-history);

        res = norm(lhs-rhs,2)/max(1,norm(rhs,2));
        maxResidual = max(maxResidual,res);
    end
end

%% ============================================================
function yNext = advanceState(n,theta,forcing,u,A,M,b,a,cL1)

    % At time t_n, sum all known L1 increments:
    % sum_{j=1}^{n-1} a_{n-j} (theta_j-theta_{j-1}).
    history = zeros(size(theta,1),1);

    for j = 1:n-1
        history = history ...
            + a(n-j+1)*(theta(:,j+1)-theta(:,j));
    end

    rhs = forcing(:,n)+M*b*u(n) ...
        + cL1*M*(a(1)*theta(:,n)-history);

    yNext = A\rhs;
end

%% ============================================================
function J = objectiveValue(theta,u,target,thetaF,M, ...
    gammaU,omegaF,dt)

    Nt = numel(u);
    Jtrack = 0;

    for n = 1:Nt
        e = theta(:,n+1)-target(:,n+1);
        Jtrack = Jtrack+0.5*dt*real(e'*M*e);
    end

    eF = theta(:,end)-thetaF;
    Jterminal = 0.5*omegaF*real(eF'*M*eF);
    Jcontrol = 0.5*gammaU*dt*sum(u.^2);

    J = Jtrack+Jterminal+Jcontrol;
end

%% ============================================================
function D = computeDiagnostics(theta,u,target,thetaF,M, ...
    gammaU,omegaF,dt,uMin,uMax,epsSat)

    Nt = numel(u);
    trackSquared = 0;

    for n = 1:Nt
        e = theta(:,n+1)-target(:,n+1);
        trackSquared = trackSquared+dt*real(e'*M*e);
    end

    eF = theta(:,end)-thetaF;
    terminalSquared = max(0,real(eF'*M*eF));

    D.Etrack = sqrt(max(0,trackSquared));
    D.Eterminal = sqrt(terminalSquared);
    D.Econtrol = sqrt(dt*sum(u.^2));

    D.Jtrack = 0.5*trackSquared;
    D.Jterminal = 0.5*omegaF*terminalSquared;
    D.Jcontrol = 0.5*gammaU*dt*sum(u.^2);
    D.J = D.Jtrack+D.Jterminal+D.Jcontrol;

    active = abs(u-uMin)<=epsSat | abs(u-uMax)<=epsSat;
    D.saturation = sum(active)/numel(u);
end

%% ============================================================
function removeToolbar(fig)
    try
        set(fig,'ToolBar','none','MenuBar','none');
    catch
        % Older MATLAB graphics may not expose both properties.
    end
end

%% ============================================================
function exportFigure(fig,filename)
    set(fig,'PaperPositionMode','auto');
    try
        exportgraphics(fig,filename,'Resolution',300);
    catch
        print(fig,filename,'-dpng','-r300');
    end
end
