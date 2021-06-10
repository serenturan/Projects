%clc
%clear all

%Please note that the code in this file is inspired from and contains pieces
%from the codes created during the Machine Learning lectures and during one-to-one drop in session
%conducted with the teaching assistants


%Importing the undersampled data 

%initialData = readtable('Train_balanced10.csv'); 

%Getting the size of the dataset and creating variables to prepare datasate
%for random splitting. Here, we will select 70% of the dataset as training
%and 30% of the dataset as test set so we define P as 0.70

%[m,n] = size(initialData);
%P=0.70;
%idx = randperm(m);

%rng default



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
%Training = [trainingData_cat trainingData_norm]; 



%Creating the testing set:

%we repeat the process in creating the training set for the test set:

%testingSplit = initialData(idx(round(P*m)+1:end),:); 
%testingData_non_cat = testingSplit(:,10); 
%testingData_cat = testingSplit(:,2:9); 
%testingData_norm = normalize(testingData_non_cat); 
%Testing= [testingData_norm testingData_cat]; 

%predictor_names = {'Light_Conditions','Weather_Conditions','Urban_or_Rural_Area','Junction_Detail','Junction_Control','Sex_of_Driver','Road_Type','Age_Band_of_Driver','Speed_Limit'}; distro_types1 = {'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn', 'normal'}; distro_types2 = {'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn', 'mvmn','kernel'}; categorical_predictors = {'Light_Conditions','Weather_Conditions','Urban_or_Rural_Area','Junction_Detail','Junction_Control','Sex_of_Driver','Road_Type','Age_Band_of_Driver'};


%dataMerged = [trainingSplit; testingSplit];

%In this step we are defining the arrays for our hyperparameters that we
%will optimise in the next steps (number of trees, minimum number of 
%observations per tree leaf and number of predictors)

%numTree = linspace(10,100,10);
%minLeaf = 2;
%numPredictors = linspace(2,9,8);
%folds=3;

%Similarly, here we are defining the arrays for storing the parameters,
%errors, final results and the accuracy as will be seen in the upcoming
%steps

%Parameters = [];
%Errors=[];
%Final= table;
%Accuracy=[];
%ErrorAllTable=[];

%In the following lines, we identify the training and testing data
%labels by splitting the relevant poriton of the training and testing data


%labels_training = table2cell(trainingSplit(:,1));
%labels_testing = testingSplit(:,1);
%labels = [labels_training; labels_testing];



%Before training the model, we are creating the cross validation indices in
%this step
%cvp = crossvalind('Kfold', size(initialData,1),folds);



%TRAINING THE RANDOM FORESTS MODEL

%for i = 1:length(numTree)
    
    %for j = 1:length(minLeaf)
        
        %for k= 1:length(numPredictors)
            
            %for m = 1:folds
            
                %rng default
                
                %validation = cvp ==m;
                %train = ~validation;
                
                %data_train = dataMerged(train,:); %creating the training data based on the cross validation fold created above
                %data_validation = dataMerged(validation,:);  %creating the validation data based on the cross validation fold created above
                
                %label_train = labels(train,:); %creating the training labels data based on the cross validation fold created above
                %label_validation = labels(validation,:); %creating the validation labels data based on the cross validation fold created above
            
                %In the next line we are creating our model
                %Model = TreeBagger(numTree(i), data_train, label_train, 'Method', 'classification', 'OOBPrediction', 'on', 'minLeafSize', minLeaf(j), 'NumPredictorsToSample', numPredictors(k));
            
                %We are getting the parameters of the model and storing
                %them in an array
                %Parameters = [Parameters; numTree(i), minLeaf(j), numPredictors(k)];
            
                %In the following 3 lines, we are computing the mean error for training, validation and out-of-bag in CV: 
                %oob_error(m) = mean(oobError(Model));  
                
                %train_error(m) = mean(error(Model,data_train, label_train));
                
                %valid_error(m) = mean(error(Model, data_validation, label_validation));
                
                %model_error = oobError(Model, 'Mode', 'Ensemble');
                
                %Errors = [Errors; model_error];
                
                %Error_all = oobError(Model);
            
                %ErrorAllTable = [ErrorAllTable ; Error_all];
                
            
                            
            
            %In the next line we are computing the predicted labels for
            %out of bag observations in the training data
                %predicted_labels= str2double(oobPredict(Model));
            
            
            %Having computed the predicted labels, we can now evaluate the
            %models for the relevant parameters by using confusion matrix
            %and accuracy metrices.
                %label_train2 = table2array(label_train);
                    
                %CM_model = confusionmat(label_train2, predicted_labels);
            
                %Accuracy_model = 100*sum(diag(CM_model))./sum(CM_model(:));
            
                %Accuracy = [Accuracy; Accuracy_model];
            
            %Finally, we store the parameters, error and the accuracy of
            %the model for the relevant parameter
                %Final = [Parameters Errors Accuracy];
            
            %end
            
            %Keeping record on the mean training validation and oob error for relevant hyperparameters:
            
            %Mean_err_train_RF(find(numTree==i), find(minLeaf==j), find(numPredictors==k))=mean(train_error);
            %Mean_err_valid_RF(find(numTree==i), find(minLeaf==j), find(numPredictors==k))=mean(valid_error);
            %Mean_err_oob_RF(find(numTree==i), find(minLeaf==j), find(numPredictors==k))=mean(oob_error);
        %end
    %end
%end


%TESTING THE RF MODEL:

%In the next lines, we are naming the columns of the Final array we have
%created above
%Final = array2table(Final);

%Final.Properties.VariableNames = {'NumTrees', 'NumLeaves', 'NumSamples', 'oobErrorValue', 'AccuracyValue'};

%in the next lines we are extracting the minimum error and highest accuracy model and storing it in best_model_f 
%min_error = min(Final{:,4});

%highestAccuracy = max(Final{:,5});

%best_model = Final(Final.AccuracyValue == highestAccuracy, :);

%best_model_f = best_model(1,:)

%In the next lines we are extracting the hyperparameters of the best model
%and storing them seperately
%numtrees = best_model_f{:,1};
%numLeaves = best_model_f{:,2};
%numSamples = best_model_f{:,3};

%we are training the model based on the hyperparameters extracted above
%Model_Final = TreeBagger(numtrees, Training, cell2mat(labels_training),'Method', 'classification','OObPrediction', 'on', 'MinLeafSize', numLeaves, 'NumPredictorsToSample', numSamples);

%and predicting the labels using the final model created on the testing
%data
%predicted_labels2= predict(Model_Final, Testing(:,1:end));

%labels_testing2 = table2array(labels_testing);

%finally, we calculate the accuracy and confusion matrix of the model
%CM_model_final = confusionmat(labels_testing2, str2double(predicted_labels2));
%figure
%confusionchart(CM_model_final)

%Accuracy_Final = 100*sum(diag(CM_model_final(:)));

%save('optRF1.mat', 'Model_Final')



%TESTING THE RF MODEL ON THE UNSEEN DATA:

%Please run the piece below to test the model that was developed above on
%the unseen data that was hold out for testing purposes

load('optRF1.mat', 'Model_Final')

test_data = readtable('Test_balanced10.csv');

labels_test = test_data(:,1);

predicted_test= predict(Model_Final, test_data(:,2:end));

test_label = table2array(labels_test);

CM_model_test = confusionmat(test_label, str2double(predicted_test));
figure
confusionchart(CM_model_test)

Accuracy_Final = 100*sum(diag(CM_model_test(:)));

test_error = mean(error(Model_Final, test_data(:,2:end), test_label))





