# Copyright 2019 RBC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# evaluate.py is used to create the synthetic data generation and evaluation pipeline.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingRegressor, AdaBoostClassifier, \
    BaggingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error, average_precision_score
from scipy.special import expit

from models import dp_wgan, pate_gan
import argparse
import numpy as np
import pandas as pd
import collections
import os

parser = argparse.ArgumentParser()
parser.add_argument('--categorical', action='store_true', help='All attributes of the data are categorical with small domains')
parser.add_argument('--target-variable', help='Required if data has a target class')
parser.add_argument('--train-data-path', required=True)
parser.add_argument('--test-data-path', required=True)
parser.add_argument('--normalize-data', action='store_true', help='Apply sigmoid function to each value in the data')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--downstream-task', default="classification", help='classification | regression')
parser.add_argument('--test-mode', default="tstr", help='tstr | tsts')

privacy_parser = argparse.ArgumentParser(add_help=False)

privacy_parser.add_argument('--enable-privacy', action='store_true', help='Enable private data generation')
privacy_parser.add_argument('--target-epsilon', type=float, default=8, help='Epsilon differential privacy parameter')
privacy_parser.add_argument('--target-delta', type=float, default=1e-5, help='Delta differential privacy parameter')
privacy_parser.add_argument('--save-synthetic', action='store_true', help='Save the synthetic data into csv')
privacy_parser.add_argument('--output-data-path', help='Required if synthetic data needs to be saved')

noisy_sgd_parser = argparse.ArgumentParser(add_help=False)

noisy_sgd_parser.add_argument('--sigma', type=float,
                              default=2, help='Gaussian noise variance multiplier. A larger sigma will make the model '
                                              'train for longer epochs for the same privacy budget')
noisy_sgd_parser.add_argument('--clip-coeff', type=float,
                              default=0.1, help='The coefficient to clip the gradients before adding noise for private '
                                                'SGD training')
noisy_sgd_parser.add_argument('--micro-batch-size',
                              type=int, default=8,
                              help='Parameter to tradeoff speed vs efficiency. Gradients are averaged for a microbatch '
                                   'and then clipped before adding noise')

noisy_sgd_parser.add_argument('--num-epochs', type=int, default=500)
noisy_sgd_parser.add_argument('--batch-size', type=int, default=64)

subparsers = parser.add_subparsers(help="generative model type", dest="model")

parser_pate_gan = subparsers.add_parser('pate-gan', parents=[privacy_parser])
parser_pate_gan.add_argument('--lap-scale', type=float,
                             default=0.0001, help='Inverse laplace noise scale multiplier. A larger lap_scale will '
                                                  'reduce the noise that is added per iteration of training.')
parser_pate_gan.add_argument('--batch-size', type=int, default=64)
parser_pate_gan.add_argument('--num-teachers', type=int, default=10, help="Number of teacher disciminators in the pate-gan model")
parser_pate_gan.add_argument('--teacher-iters', type=int, default=5, help="Teacher iterations during training per generator iteration")
parser_pate_gan.add_argument('--student-iters', type=int, default=5, help="Student iterations during training per generator iteration")
parser_pate_gan.add_argument('--num-moments', type=int, default=100, help="Number of higher moments to use for epsilon calculation for pate-gan")

parser_real_data = subparsers.add_parser('real-data')

parser_dp_wgan = subparsers.add_parser('dp-wgan', parents=[privacy_parser, noisy_sgd_parser])
parser_dp_wgan.add_argument('--clamp-lower', type=float, default=-0.01, help="Clamp parameter for wasserstein GAN")
parser_dp_wgan.add_argument('--clamp-upper', type=float, default=0.01, help="Clamp parameter for wasserstein GAN")

opt = parser.parse_args()

# Loading the data
train = pd.read_csv(opt.train_data_path)
test = pd.read_csv(opt.test_data_path)

data_columns = [col for col in train.columns if col != opt.target_variable]
if opt.categorical:
    combined = train.append(test)
    config = {}
    for col in combined.columns:
        col_count = len(combined[col].unique())
        config[col] = col_count

class_ratios = None

if opt.downstream_task == "classification":
    class_ratios = train[opt.target_variable].sort_values().groupby(train[opt.target_variable]).size().values/train.shape[0]


X_train = np.nan_to_num(train.drop([opt.target_variable], axis=1).values)
y_train = np.nan_to_num(train[opt.target_variable].values)
X_test = np.nan_to_num(test.drop([opt.target_variable], axis=1).values)
y_test = np.nan_to_num(test[opt.target_variable].values)

if opt.normalize_data:
    X_train = expit(X_train)
    X_test = expit(X_test)

input_dim = X_train.shape[1]
z_dim = int(input_dim / 4 + 1) if input_dim % 4 == 0 else int(input_dim / 4)

conditional = (opt.downstream_task == "classification")

# Training the generative model
if opt.model == 'pate-gan':
    Hyperparams = collections.namedtuple(
        'Hyperarams',
        'batch_size num_teacher_iters num_student_iters num_moments lap_scale class_ratios lr')
    Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None)

    model = pate_gan.PATE_GAN(input_dim, z_dim, opt.num_teachers, opt.target_epsilon, opt.target_delta, conditional)
    model.train(X_train, y_train, Hyperparams(batch_size=opt.batch_size, num_teacher_iters=opt.teacher_iters,
                                              num_student_iters=opt.student_iters, num_moments=opt.num_moments,
                                              lap_scale=opt.lap_scale, class_ratios=class_ratios, lr=1e-4))

elif opt.model == 'dp-wgan':
    Hyperparams = collections.namedtuple(
        'Hyperarams',
        'batch_size micro_batch_size clamp_lower clamp_upper clip_coeff sigma class_ratios lr num_epochs')
    Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None, None, None)

    model = dp_wgan.DP_WGAN(input_dim, z_dim, opt.target_epsilon, opt.target_delta, conditional)
    model.train(X_train, y_train, Hyperparams(batch_size=opt.batch_size, micro_batch_size=opt.micro_batch_size,
                                              clamp_lower=opt.clamp_lower, clamp_upper=opt.clamp_upper,
                                              clip_coeff=opt.clip_coeff, sigma=opt.sigma, class_ratios=class_ratios, lr=
                                              5e-5, num_epochs=opt.num_epochs), private=opt.enable_privacy)

elif opt.model == 'ct-gan':
    pass

# Generating synthetic data from the trained model
if opt.model == 'real-data':
    X_syn = X_train
    y_syn = y_train

elif opt.model == 'dp-wgan' or opt.model == 'pate-gan' or opt.model == 'ct-gan':
    syn_data = model.generate(X_train.shape[0], class_ratios)
    X_syn, y_syn = syn_data[:, :-1], syn_data[:, -1]

# train on synthetic test on synthetic
if opt.test_mode == 'tsts':
    X_syn, X_test, y_syn, y_test = train_test_split(syn_data[:, :-1], syn_data[:, -1], test_size=0.2, random_state=42)

# Train on synthetic, test on real (TSTR)
# Testing the quality of synthetic data by training and testing the downstream learners

# Creating downstream learners
learners = []
roc_avg = 0
prc_avg = 0
if opt.downstream_task == "classification":
    learners.append((LogisticRegression(max_iter=1000)))
    learners.append((RandomForestClassifier()))
    learners.append((GaussianNB()))
    learners.append((BernoulliNB()))
    learners.append((DecisionTreeClassifier()))
    learners.append((LinearDiscriminantAnalysis()))
    learners.append((AdaBoostClassifier(n_estimators=100)))
    learners.append((BaggingClassifier()))
    learners.append((GradientBoostingClassifier()))
    learners.append((MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300, activation='relu', \
                                   solver='adam', random_state=42, early_stopping=True)))

    print(f"\nEvaluate classifiers with testing mode {str(opt.test_mode)}:")
    for i in range(0, len(learners)):
        score = learners[i].fit(X_syn, y_syn)
        pred_probs = learners[i].predict_proba(X_test)
        auc_score = roc_auc_score(y_test, pred_probs[:, 1])
        auprc = average_precision_score(y_test, pred_probs[:, 1])
        roc_avg += auc_score
        prc_avg += auprc
        print('-' * 60)
        print(f'{str(type(learners[i]).__name__):<30} auc {round(auc_score, 4):>5}\t auprc {round(auprc, 4):>5}')

    for model in [LinearSVC(max_iter=10000), GradientBoostingRegressor()]:
        model.fit(X_syn, y_syn)
        preds = model.predict(X_test)
        auc_score = roc_auc_score(y_test, preds)
        auprc = average_precision_score(y_test, preds)
        roc_avg += auc_score
        prc_avg += auprc
        print('-' * 60)
        print(f'{type(model).__name__:<30} auc {round(auc_score, 4):>5}\t auprc {round(auprc, 4):>5}')

    print('-' * 60)
    print(f'{"Average ":<30} auc {round(roc_avg / 12, 4)}\t auprc {round(prc_avg / 12, 4):>5}')

else:
    names = ['Ridge', 'Lasso', 'ElasticNet', 'Bagging', 'MLP']

    learners.append((Ridge()))
    learners.append((Lasso()))
    learners.append((ElasticNet()))
    learners.append((BaggingRegressor()))
    learners.append((MLPRegressor()))

    print("RMSE scores of downstream regressors on test data : ")
    for i in range(0, len(learners)):
        score = learners[i].fit(X_syn, y_syn)
        pred_vals = learners[i].predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_vals))
        print('-' * 40)
        print('{0}: {1}'.format(names[i], rmse))

if opt.model != 'real-data':
    if opt.save_synthetic:

        if not os.path.isdir(opt.output_data_path):
            raise Exception('Output directory does not exist')

        X_syn_df = pd.DataFrame(data=X_syn, columns=data_columns)
        y_syn_df = pd.DataFrame(data=y_syn, columns=[opt.target_variable])

        syn_df = pd.concat([X_syn_df, y_syn_df], axis=1)
        syn_df.to_csv(opt.output_data_path + "/synthetic_data.csv")
        print("Saved synthetic data at : ", opt.output_data_path)