function [modelParameters] = positionEstimatorTraining(training_data)
    % Trajectory Calculation
    numDirections = size(training_data, 2);    % e.g., 8
    numTrials = size(training_data, 1);        % e.g., 50
    numNeurons = size(training_data(1,1).spikes, 1); % e.g., 98

    % Find the shortest trial length for each direction
    maxTimePerAngle = zeros(numDirections, 1);
    for k = 1:numDirections
        allLengths = zeros(numTrials, 1);
        for n = 1:numTrials
            allLengths(n) = size(training_data(n,k).spikes, 2);
        end
        maxTimePerAngle(k) = min(allLengths);
    end

    % Compute average trajectory per direction
    avgTrajectory = cell(numDirections, 1);
    for k = 1:numDirections
        T = maxTimePerAngle(k);
        sumPos = zeros(2, T);
        for n = 1:numTrials
            sumPos = sumPos + training_data(n,k).handPos(1:2, 1:T);
        end
        avgTrajectory{k} = sumPos / numTrials; % 2 x T
    end

    % Process Data for Classification
    [X_train, Y_train] = process_trials(training_data);

    % Scale Data
    mu = mean(X_train);
    sigma = std(X_train);
    sigma = max(sigma, 1e-6); % Prevent division by zero
    X_train_scaled = (X_train - mu) ./ sigma;

    % Binary Classification Groups (labels 1-8 in MATLAB)
    group_labels = {[2,3,4,5,6], 7, [1,8]};

    % Create binary labels
    Y_train_binary = create_binary_labels(Y_train, group_labels);

    % Train binary classifier using manual LDA
    binary_model = manual_lda_train(X_train_scaled, Y_train_binary);

    % Step 1: PCA + LDA for (2-6) group
    mask_train_1to5 = (Y_train_binary == 1);
    X_train_1to5 = X_train(mask_train_1to5, :);
    Y_train_1to5 = Y_train(mask_train_1to5) - 1; % Convert to 1-5

    % Scale data for (2-6) group
    mu_1to5 = mean(X_train_1to5);
    sigma_1to5 = std(X_train_1to5);
    sigma_1to5 = max(sigma_1to5, 1e-6);
    X_train_1to5_scaled = (X_train_1to5 - mu_1to5) ./ sigma_1to5;

    % PCA for (2-6) group
    coeff_1to5 = manual_pca(X_train_1to5_scaled, 15);
    X_train_1to5_pca = X_train_1to5_scaled * coeff_1to5;

    % LDA model for (2-6) group
    lda_1to5 = manual_lda_train(X_train_1to5_pca, Y_train_1to5);

    % Step 2: PCA + KNN for (1,8) group
    mask_train_07 = (Y_train_binary == 3);
    X_train_07 = X_train(mask_train_07, :);
    Y_train_07 = Y_train(mask_train_07);

    % Scale data for (1,8) group
    mu_07 = mean(X_train_07);
    sigma_07 = std(X_train_07);
    sigma_07 = max(sigma_07, 1e-6);
    X_train_07_scaled = (X_train_07 - mu_07) ./ sigma_07;

    % PCA for (1,8) group
    available_components = size(X_train_07_scaled, 2);
    num_components_07 = min(2, available_components);
    coeff_07 = manual_pca(X_train_07_scaled, num_components_07);
    if num_components_07 > 0
        X_train_07_pca = X_train_07_scaled * coeff_07;
    else
        X_train_07_pca = X_train_07_scaled;
        coeff_07 = eye(size(X_train_07_scaled, 2));
    end

    % KNN model for (1,8) group
    knn_07 = manual_knn_train(X_train_07_pca, Y_train_07, 7);

    % Store model parameters
    modelParameters.avgTrajectory = avgTrajectory;
    modelParameters.numAngles = numDirections;
    modelParameters.numNeurons = numNeurons;
    modelParameters.maxTimePerAngle = maxTimePerAngle;
    modelParameters.binary_model = binary_model;
    modelParameters.lda_1to5 = lda_1to5;
    modelParameters.knn_07 = knn_07;
    modelParameters.coeff_1to5 = coeff_1to5;
    modelParameters.coeff_07 = coeff_07;
    modelParameters.mu = mu;
    modelParameters.sigma = sigma;
    modelParameters.mu_1to5 = mu_1to5;
    modelParameters.sigma_1to5 = sigma_1to5;
    modelParameters.mu_07 = mu_07;
    modelParameters.sigma_07 = sigma_07;

    % Helper Functions
    function [X, Y] = process_trials(trial_data)
        X = [];
        Y = [];
        for i = 1:size(trial_data, 1)
            for j = 1:size(trial_data, 2)
                trial = trial_data(i,j);
                spikes = trial.spikes(:, 1:320);
                spikes_mean = mean(spikes, 2)';
                X = [X; spikes_mean];
                Y = [Y; j]; % Directions 1-8
            end
        end
    end

    function binary_labels = create_binary_labels(Y, groups)
        binary_labels = zeros(size(Y));
        for i = 1:numel(Y)
            if ismember(Y(i), groups{1})
                binary_labels(i) = 1; % Group [2,3,4,5,6]
            elseif ismember(Y(i), groups{2})
                binary_labels(i) = 2; % Group [7]
            else
                binary_labels(i) = 3; % Group [1,8]
            end
        end
    end

    function coeff = manual_pca(X, num_components)
        % Manual PCA: Assumes X is scaled externally
        cov_matrix = (X' * X) / (size(X,1) - 1);
        [V, D] = eig(cov_matrix);
        [~, idx] = sort(diag(D), 'descend');
        V = V(:, idx);
        num_components = min(num_components, size(V,2));
        coeff = V(:, 1:num_components);
    end

    function lda_model = manual_lda_train(X, Y)
        classes = unique(Y);
        k = length(classes);
        n_features = size(X,2);
        class_means = zeros(k, n_features);
        for i = 1:k
            class_means(i,:) = mean(X(Y==classes(i),:), 1);
        end
        Sw = zeros(n_features, n_features);
        for i = 1:k
            X_class = X(Y==classes(i),:);
            X_class_centered = X_class - class_means(i,:);
            Sw = Sw + (X_class_centered' * X_class_centered);
        end
        Sw = Sw / (size(X,1) - k);
        inv_Sw = inv(Sw);
        priors = zeros(k,1);
        for i = 1:k
            priors(i) = sum(Y==classes(i)) / length(Y);
        end
        lda_model.class_means = class_means;
        lda_model.inv_Sw = inv_Sw;
        lda_model.priors = priors;
        lda_model.classes = classes;
    end

    function knn_model = manual_knn_train(X, Y, k)
        knn_model.X_train = X;
        knn_model.Y_train = Y;
        knn_model.k = k;
    end
end