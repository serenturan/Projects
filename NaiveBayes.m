%clc
%clear all

%Please note that the code in this file is inspired from and contains pieces
%from the codes created during the Machine Learning lectures and during one-to-one drop in session
%conducted with the teaching assistants

%Importing the undersampled data 

%initialData = csvread('Train_balanced10.csv');

%Getting the size of the dataset and creating variables to prepare datasate
%for random splitting. Here, we will select 70% of the dataset as training
%and 30% of the dataset as test set so we define P as 0.70

%[m,n] = size(initialData);
%P=0.70;
%idx = randperm(m);

%Creating the training set:

% we randomly split the 70% of the data. Then, we identify the non
% categorical and categorical variable. We split and define them in two
% variables as seen below. We normalise the non-categorical variables, and
% finally merge the normalised variables with the categorical variables.
% Also, we are spliiting the training labels.

%trainingSplit = initialData(idx(1:round(P*m)),:); 
%trainingData_non_cat = trainingSplit(:,10); 
%trainingData_cat = trainingSplit(:,2:9); 
%trainingData_norm = normalize(trainingData_non_cat); 
%trainingData = [trainingData_cat trainingData_norm]; 
%trainingLabel = trainingSplit(:,1); 

%Creating the testing set:

%we repeat the process in creating the training set for the test set:

%testingSplit = initialData(idx(round(P*m)+1:end),:); 
%testingData_non_cat = testingSplit(:,10); 
%testingData_cat = testingSplit(:,2:9); 
%testingData_norm = normalize(testingData_non_cat); 
%testingData= [testingData_norm testingData_cat]; 
%testingLabel = testingSplit(:,1); 


%We identify the variable names and distribution types:			
%predictor_names = {'Light_Conditions','Weather_Conditions','Urban_or_Rural_Area','Junction_Detail','Junction_Control','Sex_of_Driver','Road_Type','Age_Band_of_Driver','Speed_Limit'}; distro_types1 = {'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn', 'normal'}; distro_types2 = {'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn','kernel'}; categorical_predictors = {'Light_Conditions','Weather_Conditions','Urban_or_Rural_Area','Junction_Detail','Junction_Control','Sex_of_Driver','Road_Type','Age_Band_of_Driver'};


%TRAINING NAIVE BAYES

%Firstly, we will identify the best distribution type to use for the prior
%For this reason, we store the distribution types in dis_list and prior.
%We run the model for each combination of them within a loop to see which
%combination results in the lowest loss. 

%kfLosses_table= zeros(6,3);
%dis_list={'Normal', 'kernel', 'mvmn'};
%prior = {'empirical', 'uniform'};

%tic

%for i = length(dis_list);
    %for j = 1:2;
        %rng default
        %Mdl_1 = fitcnb(trainingData, trainingLabel, 'CrossVal', 'on', 'Prior', char(prior(j)), 'DistributionNames', char(dis_list(i)), 'PredictorNames', predictor_names);
        %Loss_1(i, j) = kfoldLoss(Mdl_1);
    %end
%end
%toc



%tic
%Here, we are creating 3 fold cross validation and training our naive bayes model
%based on the distribution type selected as a result of above loop (kernel).
%we store the loss for every model trained and select the model with the lowest
%error as the best model.

%width = linspace(0,1,1);  
%err=1;
%n=1;

%for i = 1:3;
    %rng default
    %Mdl_kernel = fitcnb(trainingData, trainingLabel,'CrossVal','on','Prior','empirical','Distribution','kernel','PredictorNames', predictor_names, 'Width',width);
    %Loss_kernel(n)=kfoldLoss(Mdl_kernel);
    %if Loss_kernel(n)<err;
        %err=Loss_kernel(n);
        %Best_Mdl = Mdl_kernel
    %end
    %n=n+1
%end
%Loss_kernel
%toc


%In this step, using the best model we selected above, we train the 3 folds
%and select the model with the lowest error and save it as the final best
%model.
%tic
%kfLoss_2=1;
%for f=1:3;
    %err(f)=loss(Best_Mdl.Trained{f},trainingData,trainingLabel);
    %if err(f)<kfLoss_2
        %kfLoss_2=err(f);
        %final_best_model = Best_Mdl.Trained{f};
    %end
%end
%toc


%TESTING:

%Having trained the dataset and chosen a best model, now we will predict
%the classifications and test the model on the test data.
%Finally we obtain the test error

%tic
%[label,score,cost] = predict(final_best_model, testingData) %posterior probabilities
%Results = confusionmat(testingLabel, label) %confusion matrix
%Test_Err = loss(final_best_model, testingData, testingLabel) %test errors
%toc

%tic
%save the best model:
%and plot the confusionmatrix, get the precision, recall and f1 score
%kfLoss_2 = 1;
%for f = 1:3;
    %err(f) = loss(Best_Mdl.Trained{f},trainingData,trainingLabel);
    %if err(f) < kfLoss_2
        %kfLoss_2 = err(f);
        %final_best_model = Best_Mdl.Trained{f};
    %end
%end
%err 

%toc


%cm = confusionchart(testingLabel, label)

%save('optNB3.mat', 'final_best_model')


%TESTING THE NB MODEL ON THE UNSEEN DATA:

%Please run the piece below to test the model that was developed above on
%the unseen data that was hold out for testing purposes

load('optNB3.mat', 'final_best_model')

test_data = csvread('Test_balanced10.csv');

labels_test = test_data(:,1);

[label,score,cost] = predict(final_best_model, test_data(:,2:end)) %posterior probabilities
Results = confusionmat(labels_test, label) %confusion matrix
Test_Err = loss(final_best_model, test_data(:,2:end), labels_test) %test errors

cm = confusionchart(labels_test, label)










