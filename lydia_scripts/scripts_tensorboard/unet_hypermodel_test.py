"""
Unit tests for U-net hypermodel for tuning the U-net
hyperparameters.

execute:
>> python -m unittest 
"""

import sys, unittest, argparse
import unet_hypermodel as uh

from keras_tuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from keras_tuner.engine import tuner as tuner_module
from keras_tuner.engine import oracle as oracle_module

class TestUNetHyperModel(unittest.TestCase):

    def setUp(self):
        '''
        Set up performed for every test method
        '''
        self.cmdline = {}
        self.cmdline['t00_rand'] = [
                        '--in_dir', '',
                        '--in_dir_val', '', 
                        '--out_dir', '', 
                        #[--objective OBJECTIVE] 
                        #[--project_name_prefix PROJECT_NAME_PREFIX]
                        #[--max_trials MAX_TRIALS] 
                        #[--overwrite] 
                        #[--max_retries_per_trial MAX_RETRIES_PER_TRIAL]
                        #[--max_consecutive_failed_trials MAX_CONSECUTIVE_FAILED_TRIALS] 
                        #[--executions_per_trial EXECUTIONS_PER_TRIAL]
                        #[--n_labels N_LABELS]
                        #[--x_shape X_SHAPE] 
                        #[--y_shape Y_SHAPE] 
                        #[--epochs EPOCHS] 
                        #[--batch_size BATCH_SIZE] 
                        #[--lrate LRATE] 
                        #[--patience PATIENCE]
                        #[--min_delta MIN_DELTA] 
                        #[--dataset DATASET] 
                        #[--gpu] 
                        '--dry_run',
                        'rand'
                        #{random,rand,bayesian,bayes,hyperband,hyper,custom}
                    ]
        self.cmdline['t00_hyper'] = [
                        '--in_dir', '',
                        '--in_dir_val', '', 
                        '--out_dir', '', 
                        '--dry_run',
                        'hyper'
                    ]
        self.cmdline['t00_bayes'] = [
                        '--in_dir', '',
                        '--in_dir_val', '', 
                        '--out_dir', '', 
                        '--dry_run',
                        'bayes'
                    ]

        sys.argv[1:] = ['--in_dir', '../../../test_data/3D_light/training_int_tor/training_ZH_only.tf', 
                        #f'/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_{path}/training_ZH_only.tf'
                        '--in_dir_val', '../../../test_data/3D_light/validation_int_tor/validation_ZH_only.tf', 
                        #f'/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/validation_{path}/validation_ZH_only.tf'
                        '--out_dir', '../../../test_data/tmp',
                        '--dry_run'] 
        self.args = uh.parse_args()

    def test_args2string(self):
        self.maxDiff = None 

        # T00
        argstr = uh.args2string(self.args)
        self.assertEqual(argstr, 'objective=val_loss_max_trials=0005_max_retries_per_trial=03_max_consecutive_failed_trials=03_executions_per_trial=01_tuner=None_n_labels=01_x_shape=032_032_012_y_shape=032_032_001_epochs=0005_batch_size=0128_lrate=0.001000_patience=010_min_delta=0.000100_dataset=tor_gpu=0_dry_run=0',
        msg='')

        # T01
        sys.argv[1:] = self.cmdline['t00_rand']
        args = uh.parse_args()
        argstr = uh.args2string(args)
        self.assertEqual(argstr, 'objective=val_loss_max_trials=0005_max_retries_per_trial=03_max_consecutive_failed_trials=03_executions_per_trial=01_tuner=rand_n_labels=01_x_shape=032_032_012_y_shape=032_032_001_epochs=0005_batch_size=0128_lrate=0.001000_patience=010_min_delta=0.000100_dataset=tor_gpu=0_dry_run=1',
        msg='')

    def test_create_tuner(self):
        tuner, hypermodel = uh.create_tuner(self.args)

        isBaseTuner = isinstance(tuner, tuner_module.Tuner)
        self.assertTrue(isBaseTuner, msg='not an instance of the Base Tuner class')

        isUNetHyperModel = isinstance(hypermodel, uh.UNetHyperModel)
        self.assertTrue(isUNetHyperModel, msg='not an instance of the UNetHyperModel class')

        # T01 - Random Search
        sys.argv[1:] = self.cmdline['t00_rand']
        args = uh.parse_args()
        argstr = uh.args2string(args)
        tuner, hypermodel = uh.create_tuner(args)
        print(tuner)
        isRandomSearch = isinstance(tuner, RandomSearch)
        self.assertTrue(isRandomSearch, msg=f'not an instance of RandomSearch. type is {type(tuner)}. args.tuner is {args.tuner}')
        isUNetHyperModel = isinstance(hypermodel, uh.UNetHyperModel)
        self.assertTrue(isUNetHyperModel, msg='not an instance of the UNetHyperModel class')

        # T02 - Hyperband Search
        sys.argv[1:] = self.cmdline['t00_hyper']
        args = uh.parse_args()
        argstr = uh.args2string(args)
        tuner, hypermodel = uh.create_tuner(args)
        isHyperband = isinstance(tuner, Hyperband)
        self.assertTrue(isHyperband, msg=f'not an instance of Hyperband. type is {type(tuner)}. args.tuner is {args.tuner}')
        isUNetHyperModel = isinstance(hypermodel, uh.UNetHyperModel)
        self.assertTrue(isUNetHyperModel, msg='not an instance of the UNetHyperModel class')

        # T03 - BayesOpt
        sys.argv[1:] = self.cmdline['t00_bayes']
        args = uh.parse_args()
        argstr = uh.args2string(args)
        tuner, hypermodel = uh.create_tuner(args)
        isBayesOpt = isinstance(tuner, BayesianOptimization)
        self.assertTrue(isBayesOpt, msg=f'not an instance of BayesianOptimization. type is {type(tuner)}. args.tuner is {args.tuner}')
        isUNetHyperModel = isinstance(hypermodel, uh.UNetHyperModel)
        self.assertTrue(isUNetHyperModel, msg='not an instance of the UNetHyperModel class')

        # TODO: T03 - Custom Tuner

        #with self.assertRaises(TypeError):
        #    s.split(2)

    def test_prep_data(self):
        tuner, hypermodel = uh.create_tuner(self.args)
        ds_train, ds_val = uh.prep_data(self.args, n_labels=hypermodel.n_labels)
        self.assertEqual(ds_train.element_spec[0].shape, (None, 32, 32, 12), 
                         msg='Training Dataset input shape was not (None, 32, 32, 12)')
        self.assertEqual(ds_train.element_spec[1].shape, (None, 32, 32, 1), 
                         msg='Training Dataset output shape was not (None, 32, 32, 1)')
        self.assertEqual(ds_val.element_spec[0].shape, (None, 32, 32, 12), 
                         msg='Validation Dataset input shape was not (None, 32, 32, 12)')
        self.assertEqual(ds_val.element_spec[1].shape, (None, 32, 32, 1), 
                         msg='Validation Dataset output shape was not (None, 32, 32, 1)')

    def test_search(self):
        args = self.args

        tuner, hypermodel = uh.create_tuner(args, DB=args.dry_run)
        ds_train, ds_val = uh.prep_data(args, n_labels=hypermodel.n_labels)

        PROJ_NAME_PREFIX = args.project_name_prefix
        PROJ_NAME = f'{PROJ_NAME_PREFIX}_{args.tuner}'

        # If a tuner is specified, run the hyperparameter search
        if not args.tuner is None:
            uh.execute_search(args, tuner, ds_train, X_val=ds_val, 
                              callbacks=None, DB=args.dry_run)
        #self.assertTrue('FOO'.isupper())
        #self.assertFalse('Foo'.isupper())

def create_args():
    parser = uh.create_argsparser()
    ns, args = parser.parse_known_args(namespace=unittest)
    print(ns, args)
    return ns, sys.argv[:1] + args

def parse_args(DB=0):
    args, unittest_args = create_args()
    if DB: print(args, unittest_args)
    # create cleans argv for main()
    sys.argv[:] = unittest_args
    return args, unittest_args, sys.argv


if __name__ == '__main__':
    #parse_args(DB=0)
    unittest.main()
