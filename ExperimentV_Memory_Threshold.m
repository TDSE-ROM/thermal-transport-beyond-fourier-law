
function ExperimentV_Memory_Threshold
% EXPERIMENT V: HEREDITARY FEEDBACK AND MEMORY THRESHOLD
%
% Model:
%   D_C^alpha y(t) + r*y(t)
%       = integral_0^t km*exp(-omega*(t-s))*y(s) ds,
%   y(0) = 1.
%
% Analytical threshold:
%   km_crit = omega*r.
%
% Numerical methods:
%   1. Caputo derivative: uniform-grid L1 approximation.
%   2. Memory integral: piecewise-linear product integration.
%   3. Characteristic roots: positive-real-axis bracketing.
%
% Outputs:
%   ExperimentV_Memory_Threshold_Results.csv
%   ExperimentV_Memory_Threshold.png
%
% Compatible with MATLAB R2021a.
% No Optimization Toolbox or Symbolic Math Toolbox required.
%
% IMPORTANT:
% Failure to find a positive real root is not proof that no complex
% right-half-plane roots exist. The analytical threshold statement
% remains the primary theoretical benchmark.

    clc;
    close all;

    %% 1. Parameters
    alpha = 0.7;
    r     = 1.0;
    omega = 0.8;
    delta = 0.1;

    Tf = 20;
    Nt = 2000;
    dt = Tf/Nt;

    y0 = 1.0;

    if alpha <= 0 || alpha > 1
        error('This implementation requires 0 < alpha <= 1.');
    end

    if r <= 0 || omega <= 0
        error('Require r > 0 and omega > 0.');
    end

    kcrit = omega*r;

    if delta <= 0 || delta >= kcrit
        error('Require 0 < delta < omega*r.');
    end

    kmValues = [kcrit-delta, kcrit, kcrit+delta];

    regimeNames = { ...
        'Below threshold', ...
        'At threshold', ...
        'Above threshold'};

    time = (0:Nt)'*dt;

    fprintf('\nEXPERIMENT V: HEREDITARY-FEEDBACK THRESHOLD\n');
    fprintf('====================================================\n');
    fprintf('Fractional order alpha       = %.6f\n',alpha);
    fprintf('Damping parameter r          = %.6f\n',r);
    fprintf('Memory decay omega            = %.6f\n',omega);
    fprintf('Threshold km_crit = omega*r   = %.8f\n',kcrit);
    fprintf('Threshold offset delta        = %.6f\n',delta);
    fprintf('Final time Tf                 = %.6f\n',Tf);
    fprintf('Time steps Nt                 = %d\n',Nt);
    fprintf('Time step dt                  = %.6e\n\n',dt);

    %% 2. Precompute Caputo L1 coefficients
    % a_j = (j+1)^(1-alpha) - j^(1-alpha), j >= 0.
    %
    % For alpha = 1, the L1 approximation reduces to the
    % backward-Euler first difference.

    a = zeros(Nt+1,1);

    for j = 0:Nt
        a(j+1) = (j+1)^(1-alpha) - j^(1-alpha);
    end

    cL1 = 1/(gamma(2-alpha)*dt^alpha);

    %% 3. Product-integration weights for the memory kernel
    % On each interval, approximate y(s) by linear interpolation.
    %
    % For v in [0,dt]:
    %   y(t_j-v) is represented using the two endpoint values.
    %
    % A = integral_0^dt exp(-omega*v) dv
    % B = integral_0^dt v*exp(-omega*v) dv
    %
    % wLeft  = B/dt
    % wRight = A-B/dt
    %
    % The weights are subsequently multiplied by the appropriate
    % exponential lag factor.

    q = omega*dt;

    Aweight = -expm1(-q)/omega;

    Bweight = (-expm1(-q)-q*exp(-q))/omega^2;

    wLeft  = Bweight/dt;
    wRight = Aweight-wLeft;

    %% 4. Solve the three memory regimes
    Y = zeros(Nt+1,3);
    rootP = nan(3,1);
    rootResidual = nan(3,1);
    maxLinearResidual = zeros(3,1);
    finalAmplitude = zeros(3,1);
    maxAmplitude = zeros(3,1);
    minAmplitude = zeros(3,1);
    rootDiagnostic = cell(3,1);

    for icase = 1:3

        km = kmValues(icase);

        fprintf('Solving regime: %s, km = %.8f\n', ...
            regimeNames{icase},km);

        y = zeros(Nt+1,1);
        y(1) = y0;

        maxRes = 0;

        % The current memory contribution includes
        % km*wRight*y_n, which is moved to the left-hand side.
        denominator = cL1*a(1) + r - km*wRight;

        if denominator <= 0
            error(['Nonpositive time-step denominator. ', ...
                   'Reduce dt or re-examine the discretization.']);
        end

        for n = 1:Nt

            % Caputo L1 history excluding the current increment.
            historyCaputo = 0;

            for j = 1:n-1
                historyCaputo = historyCaputo ...
                    + a(n-j+1)*(y(j+1)-y(j));
            end

            % Memory history from completed intervals.
            historyMemory = 0;

            for j = 1:n-1
                lagFactor = exp(-omega*(n-j)*dt);

                historyMemory = historyMemory ...
                    + lagFactor*(wLeft*y(j) ...
                    + wRight*y(j+1));
            end

            % Current memory interval contributes
            % wLeft*y_{n-1} + wRight*y_n.
            %
            % Equation:
            % cL1*[a0*(y_n-y_{n-1}) + historyCaputo]
            % + r*y_n
            % = km*[historyMemory + wLeft*y_{n-1}
            %       + wRight*y_n].

            rhs = cL1*a(1)*y(n) ...
                - cL1*historyCaputo ...
                + km*(historyMemory+wLeft*y(n));

            y(n+1) = rhs/denominator;

            % Residual of the fully assembled scalar equation.
            caputoValue = cL1*( ...
                a(1)*(y(n+1)-y(n)) + historyCaputo);

            memoryValue = historyMemory ...
                + wLeft*y(n) + wRight*y(n+1);

            residual = abs(caputoValue+r*y(n+1) ...
                - km*memoryValue);

            residualScale = max(1, ...
                abs(caputoValue)+abs(r*y(n+1)) ...
                + abs(km*memoryValue));

            normalizedResidual = residual/residualScale;

            maxRes = max(maxRes,normalizedResidual);
        end

        Y(:,icase) = y;

        maxLinearResidual(icase) = maxRes;
        finalAmplitude(icase) = y(end);
        maxAmplitude(icase) = max(y);
        minAmplitude(icase) = min(y);

        %% 5. Characteristic-root diagnostic
        % Delta(p) = (p+omega)*(p^alpha+r)-km.
        %
        % We search only the nonnegative real axis. This does not
        % constitute a search for all complex characteristic roots.

        Delta = @(p) (p+omega)*(p.^alpha+r)-km;

        switch icase

            case 1
                % Below threshold:
                % Delta(0) = omega*r-km > 0.
                % Since Delta(p) is increasing for real p >= 0,
                % there is no positive real root.
                rootDiagnostic{icase} = ...
                    'No positive real root; complex roots not searched';

            case 2
                % At threshold, p=0 is an exact characteristic root.
                rootP(icase) = 0;

                rootResidual(icase) = ...
                    abs(Delta(0))/max(1, ...
                    abs((0+omega)*(0^alpha+r))+km);

                rootDiagnostic{icase} = ...
                    'Threshold root p = 0';

            case 3
                % Above threshold, Delta(0)<0 and Delta(p) becomes
                % positive for sufficiently large positive p.
                pUpper = 1;

                while Delta(pUpper) <= 0
                    pUpper = 2*pUpper;

                    if pUpper > 1e8
                        error('Could not bracket the positive real root.');
                    end
                end

                pRoot = fzero(Delta,[0,pUpper]);

                rootP(icase) = pRoot;

                rootResidual(icase) = ...
                    abs(Delta(pRoot))/max(1, ...
                    abs((pRoot+omega)*(pRoot^alpha+r))+km);

                rootDiagnostic{icase} = ...
                    'Positive real root detected';
        end
    end

    %% 6. Assemble numerical results
    Regime = regimeNames(:);
    MemoryStrength = kmValues(:);

    AnalyticalPrediction = { ...
        'No RHP roots under stated analytical assumptions'; ...
        'Characteristic root at p = 0'; ...
        'Positive real characteristic root'};

    RootDiagnostic = rootDiagnostic;
    ComputedRoot = rootP;
    NormalizedRootResidual = rootResidual;
    FinalAmplitude = finalAmplitude;
    MaximumAmplitude = maxAmplitude;
    MinimumAmplitude = minAmplitude;
    MaximumNormalizedEquationResidual = maxLinearResidual;

    Results = table(Regime,MemoryStrength, ...
        AnalyticalPrediction,RootDiagnostic,ComputedRoot, ...
        NormalizedRootResidual,FinalAmplitude,MaximumAmplitude, ...
        MinimumAmplitude,MaximumNormalizedEquationResidual);

    fprintf('\nTHRESHOLD AND TIME-DOMAIN RESULTS\n');
    fprintf('====================================================\n');

    for icase = 1:3
        fprintf('\n%s\n',regimeNames{icase});
        fprintf('  km                       = %.8f\n',kmValues(icase));
        fprintf('  Final state y(Tf)        = %.12e\n', ...
            finalAmplitude(icase));
        fprintf('  Maximum state             = %.12e\n', ...
            maxAmplitude(icase));
        fprintf('  Minimum state             = %.12e\n', ...
            minAmplitude(icase));
        fprintf('  Max normalized equation residual = %.12e\n', ...
            maxLinearResidual(icase));

        if isfinite(rootP(icase))
            fprintf('  Computed real root p      = %.12e\n',rootP(icase));
            fprintf('  Normalized root residual  = %.12e\n', ...
                rootResidual(icase));
        else
            fprintf('  Root diagnostic           = %s\n', ...
                rootDiagnostic{icase});
            fprintf('  Root residual             = N/A (no root reported)\n');
        end
    end

    disp(Results);

    writetable(Results, ...
        'ExperimentV_Memory_Threshold_Results.csv');

    %% 7. Plot the three time histories
%     figure('Color','w','Units','pixels', ...
%         'Position',[100 100 1000 620]);
    figure;
    plot(time,Y(:,1),'b-','LineWidth',1.8);
    hold on;
    plot(time,Y(:,2),'k--','LineWidth',1.8);
    plot(time,Y(:,3),'r-','LineWidth',1.8);
    hold off;

    xlabel('Time, $t$','Interpreter','latex','FontSize',12);
    ylabel('State, $y(t)$','Interpreter','latex','FontSize',12);

    title(sprintf(['Hereditary-feedback threshold: ', ...
        '$\\alpha=%.2f$, $r=%.2f$, $\\omega=%.2f$'], ...
        alpha,r,omega), ...
        'Interpreter','latex','FontSize',13);

    legend({ ...
        sprintf('Below: k_m = %.3f',kmValues(1)), ...
        sprintf('Critical: k_m = %.3f',kmValues(2)), ...
        sprintf('Above: k_m = %.3f',kmValues(3))}, ...
        'Location','best');

    grid on;
    box on;
    set(gca,'FontSize',11,'LineWidth',1.0, ...
        'TickDir','out','Layer','top');

    xlim([0,Tf]);

    % Display the theoretical threshold in the plotting area.
    thresholdText = sprintf( ...
        'Theoretical threshold: k_{m,crit} = \\omega r = %.3f', ...
        kcrit);

    annotation('textbox',[0.16 0.82 0.38 0.07], ...
        'String',thresholdText, ...
        'Interpreter','tex', ...
        'FitBoxToText','on', ...
        'BackgroundColor','white', ...
        'EdgeColor',[0.4 0.4 0.4], ...
        'FontSize',10);

    set(gcf,'PaperPositionMode','auto');

    try
        exportgraphics(gcf, ...
            'ExperimentV_Memory_Threshold.png', ...
            'Resolution',300);
    catch
        print(gcf,'ExperimentV_Memory_Threshold.png', ...
            '-dpng','-r300');
    end

    %% 8. Characteristic function plot on the positive real axis
    % This separate diagnostic makes the threshold crossing explicit.
    % It does not search the complex plane.

    pMax = max(2,2*max(rootP(isfinite(rootP))));
    if isempty(pMax) || ~isfinite(pMax)
        pMax = 2;
    end

    pGrid = linspace(0,pMax,1000);

%     figure('Color','w','Units','pixels', ...
%         'Position',[120 120 950 560]);
    figure;
    plot(pGrid, ...
        (pGrid+omega).*(pGrid.^alpha+r)-kmValues(1), ...
        'b-','LineWidth',1.7);
    hold on;

    plot(pGrid, ...
        (pGrid+omega).*(pGrid.^alpha+r)-kmValues(2), ...
        'k--','LineWidth',1.7);

    plot(pGrid, ...
        (pGrid+omega).*(pGrid.^alpha+r)-kmValues(3), ...
        'r-','LineWidth',1.7);

    yline(0,'k:','LineWidth',1.1);

    if isfinite(rootP(3))
        plot(rootP(3),0,'ro','MarkerFaceColor','r', ...
            'MarkerSize',7);
    end

    hold off;

    xlabel('Real $p\geq 0$','Interpreter','latex','FontSize',12);
    ylabel('$\Delta(p)$','Interpreter','latex','FontSize',12);

    title('Characteristic function on the nonnegative real axis', ...
        'Interpreter','none','FontSize',12);

    legend({ ...
        sprintf('Below threshold: k_m = %.3f',kmValues(1)), ...
        sprintf('At threshold: k_m = %.3f',kmValues(2)), ...
        sprintf('Above threshold: k_m = %.3f',kmValues(3)), ...
        '$\Delta(p)=0$', ...
        'Positive real root'}, ...
        'Interpreter','latex','Location','best');

    grid on;
    box on;
    set(gca,'FontSize',11,'LineWidth',1.0, ...
        'TickDir','out','Layer','top');

    xlim([0,pMax]);

    try
        exportgraphics(gcf, ...
            'ExperimentV_Characteristic_Function.png', ...
            'Resolution',300);
    catch
        print(gcf,'ExperimentV_Characteristic_Function.png', ...
            '-dpng','-r300');
    end

    fprintf('\nFiles written:\n');
    fprintf('  ExperimentV_Memory_Threshold_Results.csv\n');
    fprintf('  ExperimentV_Memory_Threshold.png\n');
    fprintf('  ExperimentV_Characteristic_Function.png\n');

    fprintf('\nInterpretation notes:\n');
    fprintf(['  The root search is restricted to the nonnegative ', ...
        'real axis.\n']);
    fprintf(['  A missing positive real root is not proof of ', ...
        'absence of complex RHP roots.\n']);
    fprintf(['  Compare the time histories with the analytical ', ...
        'threshold, not as a substitute for it.\n']);

end
