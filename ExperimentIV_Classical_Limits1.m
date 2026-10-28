
function ExperimentIV_Classical_Limits
% EXPERIMENT IV: RECOVERY OF CLASSICAL DIFFUSION LIMITS
%
% Test 1: s = 1 recovers the original FEM stiffness matrix K.
% Test 2: alpha = 1 agrees with an independently assembled
%         backward-Euler reference solution.
%
% Spatial discretization:
%   P1 finite elements on (0,1), homogeneous Dirichlet conditions.
%
% Outputs:
%   ExperimentIV_Classical_Limit_Results.csv
%   ExperimentIV_Classical_Limit_Discrepancy.png
%
% Compatible with MATLAB R2021a.
% No Symbolic Math Toolbox required.

    clc;
    close all;

    %% 1. Model and discretization parameters
    N       = 40;
    kappa   = 0.4;
    mu      = 0.3;
    alpha   = 1.0;
    s       = 1.0;
    Tf      = 1.0;
    Nt      = 200;
    dt      = Tf/Nt;

    %% 2. Assemble P1 finite-element matrices
    h = 1/(N+1);
    x = (1:N)'*h;

    % Consistent mass matrix
    M = spdiags([ ...
        (h/6)*ones(N,1), ...
        (2*h/3)*ones(N,1), ...
        (h/6)*ones(N,1)], [-1 0 1], N, N);

    % Stiffness matrix
    K = spdiags([ ...
        (-1/h)*ones(N,1), ...
        (2/h)*ones(N,1), ...
        (-1/h)*ones(N,1)], [-1 0 1], N, N);

    M = (M+M')/2;
    K = (K+K')/2;

    %% 3. Construct the fractional stiffness matrix K_s
    % Generalized eigenproblem:
    %       K*V = M*V*D,   V'*M*V = I.
    [V,D] = eig(full(K),full(M));
    lambda = real(diag(D));

    [lambda,idx] = sort(lambda,'ascend');
    V = real(V(:,idx));

    % Normalize eigenvectors in the M-inner product.
    for j = 1:N
        normMj = sqrt(real(V(:,j)'*M*V(:,j)));

        if normMj <= 0 || ~isfinite(normMj)
            error('Invalid generalized eigenvector normalization.');
        end

        V(:,j) = V(:,j)/normMj;
    end

    if any(lambda <= 0)
        error('The generalized stiffness eigenvalues must be positive.');
    end

    % Spectral fractional stiffness:
    %       K_s = M*V*diag(lambda.^s)*V'*M.
    Ks = M*V*diag(lambda.^s)*V'*M;
    Ks = (Ks+Ks')/2;

    % Relative spectral-norm error for K_{1,h} = K_h.
    Eoperator = norm(full(Ks-K),2) ...
                / max(1,norm(full(K),2));

    %% 4. Initial data and common input
    theta0 = sin(pi*x);
    B = M;

    % The two implementations use identical initial data,
    % forcing, and prescribed input.

    %% 5. alpha = 1 implementation under test
    thetaTest = zeros(N,Nt+1);
    thetaTest(:,1) = theta0;

    % Backward-Euler system in divided-by-dt form.
    Atest = (1/dt)*M + kappa*Ks + mu*M;

    maxResidualTest = 0;

    for n = 1:Nt
        tn = n*dt;

        fvec = (1+tn)*sin(pi*x);
        Fh = M*fvec;
        U = 0.15*cos(tn)*sin(2*pi*x);

        rhs = Fh + B*U + (1/dt)*M*thetaTest(:,n);

        thetaTest(:,n+1) = Atest\rhs;

        residual = norm(Atest*thetaTest(:,n+1)-rhs,2) ...
                   / max(1,norm(rhs,2));

        maxResidualTest = max(maxResidualTest,residual);
    end

    %% 6. Independently assembled backward-Euler reference
    % Reference equation:
    % [M + dt*(kappa*K + mu*M)]*theta^n
    %       = M*theta^(n-1) + dt*(Fh + B*U).
    %
    % The reference uses K directly, not the reconstructed Ks.

    Aref = M + dt*(kappa*K + mu*M);

    thetaRef = zeros(N,Nt+1);
    thetaRef(:,1) = theta0;

    maxResidualRef = 0;

    for n = 1:Nt
        tn = n*dt;

        fvec = (1+tn)*sin(pi*x);
        Fh = M*fvec;
        U = 0.15*cos(tn)*sin(2*pi*x);

        rhsRef = dt*(Fh+B*U) + M*thetaRef(:,n);

        thetaRef(:,n+1) = Aref\rhsRef;

        residualRef = norm( ...
            Aref*thetaRef(:,n+1)-rhsRef,2) ...
            / max(1,norm(rhsRef,2));

        maxResidualRef = max(maxResidualRef,residualRef);
    end

    %% 7. Calculate discrepancies over time
    time = (0:Nt)'*dt;

    discrepancy = zeros(Nt+1,1);
    absoluteDiscrepancy = zeros(Nt+1,1);

    for n = 1:Nt+1
        difference = thetaTest(:,n)-thetaRef(:,n);

        absoluteDiscrepancy(n) = norm(difference,2);

        discrepancy(n) = absoluteDiscrepancy(n) ...
                         / max(1,norm(thetaRef(:,n),2));
    end

    Emax = max(discrepancy);
    Efinal = discrepancy(end);
    maxLinearResidual = max(maxResidualTest,maxResidualRef);

    %% 8. Report numerical diagnostics
    Diagnostic = { ...
        'Relative operator identity error, s=1'; ...
        'Maximum discrepancy for alpha=1'; ...
        'Final-time discrepancy for alpha=1'; ...
        'Maximum normalized linear-system residual'};

    ComputedValue = [ ...
        Eoperator; ...
        Emax; ...
        Efinal; ...
        maxLinearResidual];

    Results = table(Diagnostic,ComputedValue);

    fprintf('\nEXPERIMENT IV: CLASSICAL-LIMIT VERIFICATION\n');
    fprintf('====================================================\n');
    fprintf('Interior nodes N             = %d\n',N);
    fprintf('Time steps Nt                = %d\n',Nt);
    fprintf('Time step dt                 = %.6e\n',dt);
    fprintf('alpha                        = %.6f\n',alpha);
    fprintf('s                            = %.6f\n',s);
    fprintf('Relative operator error      = %.12e\n',Eoperator);
    fprintf('Maximum temporal discrepancy = %.12e\n',Emax);
    fprintf('Final-time discrepancy       = %.12e\n',Efinal);
    fprintf('Maximum normalized residual  = %.12e\n\n', ...
        maxLinearResidual);

    disp(Results);

    writetable(Results, ...
        'ExperimentIV_Classical_Limit_Results.csv');

    %% 9. Publication-quality discrepancy figure
    % Panel (a): relative discrepancy on a logarithmic scale.
    % Panel (b): absolute discrepancy on a linear scale.
    %
    % The relative discrepancy is dimensionless. The absolute
    % discrepancy is the Euclidean norm of thetaTest-thetaRef.

    figure('Color','w','Units','pixels', ...
           'Position',[100 100 1050 700]);

    % Roundoff-scale guide for interpreting the logarithmic plot.
    roundoffLevel = eps;

    % Panel (a): relative discrepancy
    subplot(2,1,1);

    semilogy(time, max(discrepancy,realmin), ...
        'b-', 'LineWidth',2.0);
    hold on;

    semilogy([time(1),time(end)], ...
        [roundoffLevel,roundoffLevel], ...
        'k--','LineWidth',1.2);

    hold off;
    xlim([time(1),time(end)]);

    positiveRelative = discrepancy(discrepancy > 0);

    if ~isempty(positiveRelative)
        lowerRelative = min(positiveRelative);
        upperRelative = max(discrepancy);

        lowerRelative = min(lowerRelative,roundoffLevel);
        upperRelative = max(upperRelative,10*roundoffLevel);

        if upperRelative > lowerRelative
            ylim([max(realmin,lowerRelative/2), ...
                  max(upperRelative*2, ...
                      10*lowerRelative)]);
        end
    end

    xlabel('Time, $t$','Interpreter','latex','FontSize',12);
    ylabel('Relative discrepancy','Interpreter','none','FontSize',12);

    title('(a) Relative difference between the two implementations', ...
        'Interpreter','none','FontSize',12);

    legend('Relative discrepancy','Machine-epsilon reference', ...
        'Location','best');

    grid on;
    box on;
    set(gca,'FontSize',11,'LineWidth',1.0, ...
        'TickDir','out','Layer','top');

    % Panel (b): absolute discrepancy
    subplot(2,1,2);

    plot(time,absoluteDiscrepancy, ...
        'r-','LineWidth',2.0);
    hold on;

    % Show the absolute-error range without forcing a zero-width axis.
    maxAbs = max(absoluteDiscrepancy);

    if maxAbs > 0
        ylim([0,1.15*maxAbs]);
    else
        ylim([0,1]);
    end

    hold off;
    xlim([time(1),time(end)]);

    xlabel('Time, $t$','Interpreter','latex','FontSize',12);
    ylabel('Absolute discrepancy, $\|\theta_{\rm test}-\theta_{\rm ref}\|_2$', ...
        'Interpreter','latex','FontSize',12);

    title('(b) Absolute Euclidean discrepancy', ...
        'Interpreter','none','FontSize',12);

    grid on;
    box on;
    set(gca,'FontSize',11,'LineWidth',1.0, ...
        'TickDir','out','Layer','top');

    % Export the figure.
    set(gcf,'PaperPositionMode','auto');

    try
        exportgraphics(gcf, ...
            'ExperimentIV_Classical_Limit_Discrepancy.png', ...
            'Resolution',300);
    catch
        % Fallback for MATLAB installations without exportgraphics.
        print(gcf, ...
            'ExperimentIV_Classical_Limit_Discrepancy.png', ...
            '-dpng','-r300');
    end

    fprintf('\nFiles written:\n');
    fprintf('  ExperimentIV_Classical_Limit_Results.csv\n');
    fprintf('  ExperimentIV_Classical_Limit_Discrepancy.png\n');

end
