
function ExperimentIII_Modal_Stability1
% EXPERIMENT III: Modal stability and feedback threshold
% MATLAB R2021a compatible.
%
% D_C^alpha y + d(a)y = 0,  y(0)=1
% y_exact(t) = E_alpha(-d(a)t^alpha)
%
% Outputs:
%   ExperimentIII_Modal_Results.csv
%   ExperimentIII_Time_Refinement.csv
%   ExperimentIII_Modal_Amplitudes.png
%   ExperimentIII_Modal_Errors.png
%   ExperimentIII_Time_Refinement.png

    clc;
    close all;

    %% 1. Parameters
    alpha   = 0.6;
    kappa   = 0.4;
    s       = 0.8;
    mu      = 0.3;
    lambda1 = 1.0;

    % Replace with your computed first discrete eigenvalue.
    lambda1h = 0.9998053484;

    delta = 0.2;
    Tf    = 8.0;

    % Refinement sequence; the finest mesh is used for the main plots.
    NtList = [400 800 1600 3200];
    NtFine = NtList(end);

    %% 2. Analytical and discrete thresholds
    acrit  = kappa*lambda1^s  + mu;
    acrith = kappa*lambda1h^s + mu;

    feedback = [acrit-delta; acrit; acrit+delta];
    names = {'Stable'; 'Critical'; 'Unstable'};

    dContinuous = kappa*lambda1^s + mu - feedback;

    fprintf('\nEXPERIMENT III: MODAL STABILITY\n');
    fprintf('====================================================\n');
    fprintf('alpha                  = %.8f\n', alpha);
    fprintf('Final time             = %.8f\n', Tf);
    fprintf('Continuous lambda_1    = %.10f\n', lambda1);
    fprintf('Discrete lambda_1,h    = %.10f\n', lambda1h);
    fprintf('Analytical threshold   = %.10f\n', acrit);
    fprintf('Discrete threshold     = %.10f\n', acrith);
    fprintf('Threshold difference   = %.12e\n', acrith-acrit);
    fprintf('Feedback offset delta  = %.8f\n\n', delta);

    fprintf('Continuous-threshold feedback values:\n');
    fprintf('  a_minus = %.10f\n', feedback(1));
    fprintf('  a_zero  = %.10f\n', feedback(2));
    fprintf('  a_plus  = %.10f\n\n', feedback(3));

    fprintf('Discrete-threshold feedback values:\n');
    fprintf('  a_minus,h = %.10f\n', acrith-delta);
    fprintf('  a_zero,h  = %.10f\n', acrith);
    fprintf('  a_plus,h  = %.10f\n\n', acrith+delta);

    % At the continuous threshold, the spatially discrete coefficient
    % is generally not zero.
    dhAtContinuousThreshold = acrith-acrit;
    fprintf('d_h(a_crit) = %.12e\n', dhAtContinuousThreshold);
    fprintf('d_h(a_crit,h) = %.12e\n\n', acrith-acrith);

    %% 3. Main L1 calculation on the finest mesh
    Nt = NtFine;
    dt = Tf/Nt;
    t  = (0:Nt)'*dt;

    Ynum = zeros(Nt+1,3);
    Yex  = zeros(Nt+1,3);
    Err  = zeros(Nt+1,3);

    finalAmplitude = zeros(3,1);
    exactFinalAmplitude = zeros(3,1);
    finalRelativeError = zeros(3,1);
    maximumRelativeError = zeros(3,1);

    for j = 1:3
        d = dContinuous(j);

        Ynum(:,j) = l1_modal_solver(alpha,d,dt,Nt);

        for n = 1:Nt+1
            z = -d*t(n)^alpha;
            Yex(n,j) = mittag_leffler_real(alpha,z);
        end

        Err(:,j) = abs(Ynum(:,j)-Yex(:,j)) ...
                   ./ max(1,abs(Yex(:,j)));

        finalAmplitude(j) = abs(Ynum(end,j));
        exactFinalAmplitude(j) = abs(Yex(end,j));
        finalRelativeError(j) = Err(end,j);
        maximumRelativeError(j) = max(Err(:,j));
    end

    Results = table(names,feedback,dContinuous, ...
        exactFinalAmplitude,finalAmplitude, ...
        finalRelativeError,maximumRelativeError, ...
        'VariableNames',{'Regime','Feedback','ModalCoefficient', ...
        'ExactFinalAmplitude','ComputedFinalAmplitude', ...
        'FinalRelativeError','MaximumRelativeError'});

    fprintf('MAIN MODAL RESULTS: Nt = %d, dt = %.6e\n',Nt,dt);
    disp(Results);
    writetable(Results,'ExperimentIII_Modal_Results.csv');

    %% 4. Time-step refinement
    % Errors are measured against the continuous exact solution.
    nRef = numel(NtList);
    nCases = 3;

    rows = nRef*nCases;
    AlphaCol = zeros(rows,1);
    NtCol = zeros(rows,1);
    DtCol = zeros(rows,1);
    RegimeCol = cell(rows,1);
    FinalAmplitudeCol = zeros(rows,1);
    ExactAmplitudeCol = zeros(rows,1);
    FinalErrorCol = zeros(rows,1);
    MaxErrorCol = zeros(rows,1);
    ObservedOrderCol = nan(rows,1);

    row = 0;
    previousFinalError = nan(3,1);
    previousNt = nan;

    for ir = 1:nRef
        Nr = NtList(ir);
        dtr = Tf/Nr;
        tr = (0:Nr)'*dtr;

        thisFinalError = zeros(3,1);

        for j = 1:3
            row = row+1;
            d = dContinuous(j);

            yn = l1_modal_solver(alpha,d,dtr,Nr);
            ye = zeros(Nr+1,1);

            for n = 1:Nr+1
                ye(n) = mittag_leffler_real( ...
                    alpha,-d*tr(n)^alpha);
            end

            er = abs(yn-ye)./max(1,abs(ye));
            thisFinalError(j) = er(end);

            AlphaCol(row) = alpha;
            NtCol(row) = Nr;
            DtCol(row) = dtr;
            RegimeCol{row} = names{j};
            FinalAmplitudeCol(row) = abs(yn(end));
            ExactAmplitudeCol(row) = abs(ye(end));
            FinalErrorCol(row) = er(end);
            MaxErrorCol(row) = max(er);

            if ir > 1 && thisFinalError(j) > 0 ...
                    && previousFinalError(j) > 0
                ObservedOrderCol(row) = ...
                    log(previousFinalError(j)/thisFinalError(j)) ...
                    /log(Nr/previousNt);
            end
        end

        previousFinalError = thisFinalError;
        previousNt = Nr;
    end

    Refinement = table(AlphaCol,NtCol,DtCol,RegimeCol, ...
        ExactAmplitudeCol,FinalAmplitudeCol,FinalErrorCol, ...
        MaxErrorCol,ObservedOrderCol, ...
        'VariableNames',{'Alpha','Nt','Dt','Regime', ...
        'ExactFinalAmplitude','ComputedFinalAmplitude', ...
        'FinalRelativeError','MaximumRelativeError', ...
        'ObservedOrder'});

    fprintf('\nTIME-STEP REFINEMENT RESULTS\n');
    disp(Refinement);
    writetable(Refinement,'ExperimentIII_Time_Refinement.csv');

    %% 5. Plot computed amplitudes and exact reference curves
    colors = [0.0000 0.4470 0.7410; ...
              0.8500 0.3250 0.0980; ...
              0.4660 0.6740 0.1880];

    styles = {'-','--',':'};

    %fig1 = figure('Color','w','Position',[100 100 1000 650]);
    figure;
    hold on;

    for j = 1:3
        plot(t,Ynum(:,j),'Color',colors(j,:), ...
            'LineStyle',styles{j},'LineWidth',2.0, ...
            'DisplayName',[names{j} ' (L1)']);

        % Exact reference sampled at selected points.
        idx = unique([1:100:Nt+1,Nt+1]);
        plot(t(idx),Yex(idx,j),'o', ...
            'Color',colors(j,:),'MarkerSize',4, ...
            'HandleVisibility','off');
    end

    plot([0 Tf],[1 1],'k-.','LineWidth',1.0, ...
        'DisplayName','Initial amplitude');

    xlabel('Time, t');
    ylabel('Modal amplitude, y(t)');
    title('Modal stability across the feedback threshold');
    legend('Location','best');
    grid on;
    box on;
    set(gca,'FontSize',11,'LineWidth',1.5);

%     exportgraphics(fig1,'ExperimentIII_Modal_Amplitudes.png', ...
%         'Resolution',300);

    %% 6. Plot relative modal errors
    %fig2 = figure('Color','w','Position',[120 120 1000 650]);
    figure;
    hold on;

    for j = 1:3
        semilogy(t,max(Err(:,j),realmin), ...
            'Color',colors(j,:),'LineStyle',styles{j}, ...
            'LineWidth',1.8,'DisplayName',names{j});
    end

    xlabel('Time, t');
    ylabel('Relative modal error');
    title('L1 error against the exact modal solution');
    legend('Location','best');
    grid on;
    box on;
    set(gca,'FontSize',11,'LineWidth',1.5);

%     exportgraphics(fig2,'ExperimentIII_Modal_Errors.png', ...
%         'Resolution',300);

    %% 7. Plot final-time error refinement
    %fig3 = figure('Color','w','Position',[140 140 900 600]);
    figure;
    hold on;

    for j = 1:3
        mask = strcmp(Refinement.Regime,names{j});
        loglog(Refinement.Nt(mask), ...
            Refinement.FinalRelativeError(mask),'o-', ...
            'Color',colors(j,:),'LineWidth',1.8, ...
            'MarkerSize',6,'DisplayName',names{j});
    end

    set(gca,'XDir','reverse');
    xlabel('Number of time steps, N_t');
    ylabel('Final-time relative error');
    title('Temporal refinement of the L1 modal solution');
    legend('Location','best');
    grid on;
    box on;
    set(gca,'FontSize',11,'LineWidth',1.5);

%     exportgraphics(fig3,'ExperimentIII_Time_Refinement.png', ...
%         'Resolution',300);

    fprintf('\nFiles written successfully if no MATLAB error occurred:\n');
    fprintf('  ExperimentIII_Modal_Results.csv\n');
    fprintf('  ExperimentIII_Time_Refinement.csv\n');
    fprintf('  ExperimentIII_Modal_Amplitudes.png\n');
    fprintf('  ExperimentIII_Modal_Errors.png\n');
    fprintf('  ExperimentIII_Time_Refinement.png\n');

end


% =================================================================
function y = l1_modal_solver(alpha,d,dt,Nt)
% Uniform-mesh L1 method for
% D_C^alpha y + d*y = 0, y(0)=1.
%
% At time t_n:
% c*sum_{k=1}^n a_{n-k}(y^k-y^(k-1)) + d*y^n = 0,
% c = 1/(Gamma(2-alpha)*dt^alpha).

    if alpha <= 0 || alpha >= 1
        error('Require 0 < alpha < 1.');
    end
    if dt <= 0 || Nt < 1
        error('Require dt > 0 and Nt >= 1.');
    end

    j = (0:Nt)';
    a = (j+1).^(1-alpha)-j.^(1-alpha);

    c = 1/(gamma(2-alpha)*dt^alpha);

    y = zeros(Nt+1,1);
    y(1) = 1;

    for n = 1:Nt
        history = 0;

        for k = 1:n-1
            % MATLAB y(k+1) corresponds to physical y^k.
            % a(n-k+1) corresponds to a_{n-k}.
            history = history + ...
                a(n-k+1)*(y(k+1)-y(k));
        end

        % Isolate the current increment with a_0 = 1.
        y(n+1) = (c*y(n)-c*history)/(c+d);
    end
end


% =================================================================
function value = mittag_leffler_real(alpha,z)
% Evaluate E_alpha(z) for real z using its power series and
% compensated summation.
%
% The series is:
% E_alpha(z) = sum_{k=0}^\infty z^k/Gamma(alpha*k+1).
%
% Stopping criterion: after the absolute term sequence has entered
% its decreasing regime, bound the remaining tail geometrically.
% Intended for moderate real arguments in this benchmark.

    if alpha <= 0 || alpha >= 1
        error('Require 0 < alpha < 1.');
    end

    if abs(z) < 1e-15
        value = 1;
        return;
    end

    maxTerms = 20000;
    tol = 5e-14;
    logAbsZ = log(abs(z));
    negativeZ = (z < 0);

    % Kahan compensated summation.
    value = 1;
    compensation = 0;

    for k = 1:maxTerms
        logTerm = k*logAbsZ-gammaln(alpha*k+1);
        absTerm = exp(logTerm);

        if negativeZ && mod(k,2)==1
            term = -absTerm;
        else
            term = absTerm;
        end

        correctedTerm = term-compensation;
        newValue = value+correctedTerm;
        compensation = (newValue-value)-correctedTerm;
        value = newValue;

        % Ratio |term_(k+1)| / |term_k|.
        logRatio = logAbsZ + gammaln(alpha*k+1) ...
                   - gammaln(alpha*(k+1)+1);
        ratio = exp(logRatio);

        % For this power series, subsequent absolute-term ratios
        % decrease. Once ratio < 1, use a geometric tail bound.
        if ratio < 1
            logNextTerm = (k+1)*logAbsZ ...
                          -gammaln(alpha*(k+1)+1);
            nextAbsTerm = exp(logNextTerm);
            tailBound = nextAbsTerm/(1-ratio);

            if tailBound <= tol*max(1,abs(value))
                return;
            end
        end
    end

    error(['Mittag-Leffler series did not meet the stopping ' ...
           'criterion. Reduce the argument or use a validated ' ...
           'special-function implementation.']);
end
