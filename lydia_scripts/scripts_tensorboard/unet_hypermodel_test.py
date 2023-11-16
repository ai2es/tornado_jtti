"""
Unit tests for U-net hypermodel for tuning the U-net
hyperparameters.

execute:
>> python -m unittest -v lydia_scripts/scripts_tensorboard/unet_hypermodel_test.py [METHOD]
"""

import os, sys, unittest, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("lydia_scripts/scripts_tensorboard")
import unet_hypermodel as uh

#from tensorflow import distribute #import MirroredStrategy
from tensorflow.keras.callbacks import EarlyStopping #, TensorBoard, ModelCheckpoint 

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
        self.cmdline['oscer_t00'] = ['--in_dir', '../../../test_data/3D_light/training_int_tor/training_ZH_only.tf', 
                        #f'/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_{path}/training_ZH_only.tf'
                        '--in_dir_val', '../../../test_data/3D_light/validation_int_tor/validation_ZH_only.tf', 
                        #f'/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/validation_{path}/validation_ZH_only.tf'
                        '--out_dir', '../../../test_data/tmp',
                        '--dry_run'] 

        sys.argv[1:] = ['--in_dir', '../test_data/3D_light/training_ZH_only.tf', 
                        '--in_dir_val', '../test_data/3D_light/validation1_ZH_only.tf', 
                        '--out_dir', '../test_data/tmp',
                        '--out_dir_tuning', '../test_data/tn',
                        #'--overwrite',
                        '--epochs', '6',
                        '--batch_size', '320',
                        '--dry_run',
                        '--gpu',
                        'hyper',
                        '--max_epochs', '6', #slightly higher than expected epochs to convergence for your largest Model
                        '--factor', '3'] 
        self.args = uh.parse_args()

        # Grab select GPU(s)
        from py3nvml import grab_gpus
        #if self.args.gpu: grab_gpus(num_gpus=1, gpu_select=[0])

        from tensorflow.debugging import set_log_device_placement
        if self.args.dry_run: set_log_device_placement(True)

        from tensorflow.config import get_visible_devices
        physical_devices = get_visible_devices('GPU')
        n_physical_devices = len(physical_devices)

        from tensorflow.config.experimental import set_memory_growth
        for physical_device in physical_devices:
            set_memory_growth(physical_device, False)
        print(f'We have {n_physical_devices} GPUs\n')

    def test_args2string(self):
        self.maxDiff = None 

        # T00
        _, argstr = uh.args2string(self.args)
        self.assertEqual(argstr, 'objective=val_loss_max_trials=0005_max_retries_per_trial=00_max_consecutive_failed_trials=01_executions_per_trial=01_tuner=hyper_n_labels=01_x_shape=032_032_012_y_shape=032_032_001_epochs=0006_batch_size=0320_lrate=0.001000_patience=010_min_delta=0.000100_dataset=tor_number_of_summary_trials=02_gpu=1_max_epochs=0006_factor=03_hyperband_iterations=0001',
        msg='args main. ') #gpu=1_dry_run=1_

        # OSCER T00
        sys.argv[1:] = self.cmdline['oscer_t00']
        args = uh.parse_args()
        _, argstr = uh.args2string(args)
        self.assertEqual(argstr, 'objective=val_loss_max_trials=0005_max_retries_per_trial=00_max_consecutive_failed_trials=01_executions_per_trial=01_tuner=None_n_labels=01_x_shape=032_032_012_y_shape=032_032_001_epochs=0005_batch_size=0128_lrate=0.001000_patience=010_min_delta=0.000100_dataset=tor_number_of_summary_trials=02_gpu=0',
        msg='args oscer_t00. ') #gpu=0_dry_run=1

        # T01
        sys.argv[1:] = self.cmdline['t00_rand']
        args = uh.parse_args()
        _, argstr = uh.args2string(args)
        self.assertEqual(argstr, 'objective=val_loss_max_trials=0005_max_retries_per_trial=00_max_consecutive_failed_trials=01_executions_per_trial=01_tuner=rand_n_labels=01_x_shape=032_032_012_y_shape=032_032_001_epochs=0005_batch_size=0128_lrate=0.001000_patience=010_min_delta=0.000100_dataset=tor_number_of_summary_trials=02_gpu=0',
        msg='args t00_rand. ') #gpu=0_dry_run=1

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
        #inspect.signature() or inspect.getfullargspec()

    def test_search(self):
        import time
        t0 = time.time()
        args = self.args

        tuner, hypermodel = uh.create_tuner(args, DB=args.dry_run) #, strategy=distribute.MirroredStrategy()
        ds_train, ds_val = uh.prep_data(args, n_labels=hypermodel.n_labels)

        PROJ_NAME_PREFIX = args.project_name_prefix
        PROJ_NAME = f'{PROJ_NAME_PREFIX}_{args.tuner}'

        # If a tuner is specified, run the hyperparameter search
        if not args.tuner is None:
            #tuner.tuner_id = 'tid'
            uh.execute_search(args, tuner, ds_train, X_val=ds_val, 
                              callbacks=None, DB=args.dry_run)

            print(" ")
            tuner.results_summary(2)

            # HPS
            print('-------------------------------------------')
            print("Training with best hyperparameters")
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(best_hps)
            es = EarlyStopping(monitor=args.objective, patience=args.patience,  
                               min_delta=args.min_delta, restore_best_weights=True)
            H = model.fit(ds_train, validation_data=ds_val, callbacks=[es],
                          batch_size=args.batch_size, epochs=args.epochs)#, steps_per_epoch=80)
            fname = os.path.join(args.out_dir, f"hp_model00_learning_plot.png")
            uh.plot_learning_loss(H, fname, save=True)

            #del tuner

            # Predictions
            print("predicting...")
            train_preds = model.predict(ds_train)
            val_preds = model.predict(ds_val)
            print(train_preds.shape)
            #print(val_preds.shape)
            fname = os.path.join(args.out_dir, f"hp_model00_preds_distr.png")
            uh.plot_predictions(train_preds.ravel(), val_preds.ravel(), fname, save=True)

            # CSI Curve
            y_train = np.concatenate([y for x, y in ds_train])
            y_val = np.concatenate([y for x, y in ds_val])
            fname = os.path.join(args.out_dir, f"hp_model00_csi_train_val.png")
            fig, ax = uh.plot_csi(y_train.ravel(), train_preds.ravel(), fname, label='Train', show_cb=False)
            uh.plot_csi(y_val.ravel(), val_preds.ravel(), fname, label='Val', color='orange', save=True, fig_ax=(fig, ax))
            plt.close(fig)
            del fig, ax

            # Confusion Matrix
            threshs = np.linspace(0, 1, 51).tolist()
            tps, fps, fns, tns = uh.contingency_curves(y_val, val_preds, threshs)
            csis = uh.compute_csi(tps, fns, fps) #tps / (tps + fns + fps)
            xi = np.argmax(csis)
            cutoff_probab = threshs[xi] #.12 #cutoff with heightest CSI
            print(f"Max CSI: {csis[xi]}  Thres: {cutoff_probab}  Index: {xi}")
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs = axs.ravel()
            plt.subplots_adjust(wspace=.1)
            fname = os.path.join(args.out_dir, f"hp_model00_confusion_matrix_train_val.png")
            uh.plot_confusion_matrix(y_train.ravel(), train_preds.ravel(), fname, 
                            p=cutoff_probab, fig_ax=(fig, axs[0]), save=False)  
            #fname = os.path.join(args.out_dir, f"hp_model00_confusion_matrix_train_val.png")      
            uh.plot_confusion_matrix(y_val.ravel(), val_preds.ravel(), fname, 
                            p=cutoff_probab, fig_ax=(fig, axs[1]), save=True)
            plt.close(fig)
            del fig, axs
            '''fname = os.path.join(args.out_dir, f"hp_model00_confusion_matrix_train.png")
            uh.plot_confusion_matrix(y_train.ravel(), train_preds.ravel(), fname, p=0.12, save=True)
            fname = os.path.join(args.out_dir, f"hp_model00_confusion_matrix_val.png")
            uh.plot_confusion_matrix(y_val.ravel(), val_preds.ravel(), fname, p=0.12, save=True)'''

            # TODO: Reliability Curve
            print("train > 1", np.any(train_preds.ravel() > 1))
            print("val > 1", np.any(val_preds.ravel() > 1))
            print("train < 0", np.any(train_preds.ravel() < 0))
            print("val < 0", np.any(val_preds.ravel() < 0))
            to_one = np.where(train_preds.ravel() > 1)[0]
            to_zero = np.where(train_preds.ravel() < 0)[0]
            tt = train_preds.ravel().copy()
            tt[to_one] = 1
            tt[to_zero] = 0
            to_one = np.where(val_preds.ravel() > 1)[0]
            to_zero = np.where(val_preds.ravel() < 0)[0]
            vv = val_preds.ravel().copy()
            vv[to_one] = 1
            vv[to_zero] = 0
            fname = os.path.join(args.out_dir, f"hp_model00_reliability_train_val.png")
            fig, ax = uh.plot_reliabilty_curve(y_train.ravel(), tt, #train_preds.ravel(),  
                                     fname, save=False, label='Train')
            uh.plot_reliabilty_curve(y_val.ravel(), vv, #val_preds.ravel(), 
                                     fname, fig_ax=(fig, ax), save=True, label='Val', c='orange')
            plt.close(fig)
            del fig, ax

            # PRC
            fname = os.path.join(args.out_dir, f"hp_model00_prc_train_val.png")
            fig, ax = uh.plot_prc(y_train.ravel(), train_preds.ravel(), fname, save=False, label='Train')
            #fname = os.path.join(args.out_dir, f"hp_model00_roc_val.png")
            uh.plot_prc(y_val.ravel(), val_preds.ravel(), fname, fig_ax=(fig, ax), save=True, label='Val', c='orange')
            plt.close(fig)
            del fig, ax

            # ROC
            fname = os.path.join(args.out_dir, f"hp_model00_roc_train_val.png")
            fig, ax = uh.plot_roc(y_train.ravel(), train_preds.ravel(), fname, save=False, label='Train')
            #fname = os.path.join(args.out_dir, f"hp_model00_roc_val.png")
            uh.plot_roc(y_val.ravel(), val_preds.ravel(), fname, fig_ax=(fig, ax), save=True, label='Val', c='orange')
            plt.close(fig)
            del fig, ax

            # Evaluate
            print("evaluating...")
            train_eval = model.evaluate(ds_train)
            val_eval = model.evaluate(ds_val)
            metrics = H.history.keys()
            evals = [ {k: v for k, v in zip(metrics, train_eval)} ]
            evals.append( {k: v for k, v in zip(metrics, val_eval)} )
            df_eval = pd.DataFrame(evals, index=['train', 'val'])
            print(df_eval)
            fname = os.path.join(args.out_dir, f"hp_model00_eval.csv")
            df_eval.to_csv(fname)

            t1 = time.time()
            print(f"Elapsed {(t1-t0)/60:02f}min")

            # MODEL
            '''
            print('-------------------------------------------')
            best_model = tuner.get_best_models(num_models=1)[0]
            best_model.summary()
            best_model.build(input_shape=ds_train.element_spec[0].shape) #(None, 28, 28))
            Hb = best_model.fit(ds_train, validation_data=ds_val, callbacks=[es],
                                batch_size=args.batch_size, epochs=args.epochs)

            train_preds = best_model.predict(ds_train)
            val_preds = best_model.predict(ds_val)

            train_eval = best_model.evaluate(ds_train)
            val_eval = best_model.evaluate(ds_val)
            print("M T EVAL", train_eval)
            print("M V EVAL", val_eval)
            '''

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
