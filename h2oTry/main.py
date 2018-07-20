
import h2o
from h2o.automl import H2OAutoML
h2o.init()
train = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
test = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")

# Identify predictors and response
x = train.columns
y = "response"
x.remove(y)

# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Run AutoML for 30 seconds
aml = H2OAutoML(max_runtime_secs = 30)
aml.train(x = x, y = y,
          training_frame = train,
          leaderboard_frame = test)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb

# model_id                                            auc       logloss
# --------------------------------------------------  --------  ---------
#           StackedEnsemble_model_1494643945817_1709  0.780384  0.561501
# GBM_grid__95ebce3d26cd9d3997a3149454984550_model_0  0.764791  0.664823
# GBM_grid__95ebce3d26cd9d3997a3149454984550_model_2  0.758109  0.593887
#                          DRF_model_1494643945817_3  0.736786  0.614430
#                        XRT_model_1494643945817_461  0.735946  0.602142
# GBM_grid__95ebce3d26cd9d3997a3149454984550_model_3  0.729492  0.667036
# GBM_grid__95ebce3d26cd9d3997a3149454984550_model_1  0.727456  0.675624
# GLM_grid__95ebce3d26cd9d3997a3149454984550_model_1  0.685216  0.635137
# GLM_grid__95ebce3d26cd9d3997a3149454984550_model_0  0.685216  0.635137


# The leader model is stored here
aml.leader


# If you need to generate predictions on a test set, you can make
# predictions directly on the `"H2OAutoML"` object, or on the leader
# model object directly

preds = aml.predict(test)

# or:
preds = aml.leader.predict(test)