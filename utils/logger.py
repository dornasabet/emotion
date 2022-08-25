from matplotlib import pyplot as plt
import os


class Logger:
    def __init__(self):
        self.loss_train = []
        self.loss_val = []

        self.acc_train = []
        self.acc_val = []

        self.vrmse_train = []
        self.vrmse_val = []

        self.vsagr_train = []
        self.vsagr_val = []

        self.vpcc_train = []
        self.vpcc_val = []

        self.vccc_train = []
        self.vccc_val = []

        self.armse_train = []
        self.armse_val = []

        self.asagr_train = []
        self.asagr_val = []

        self.apcc_train = []
        self.apcc_val = []

        self.accc_train = []
        self.accc_val = []

    def get_logs(self):
        return self.loss_train, self.loss_val, self.acc_train, self.acc_val, self.vrmse_train, self.vrmse_val, self.vsagr_train, self.vsagr_val, self.vpcc_train, self.vpcc_val, self.vccc_train, self.vccc_val, self.armse_train, self.armse_val, self.asagr_train, self.asagr_val, self.apcc_train, self.apcc_val, self.accc_train, self.accc_val

    def restore_logs(self, logs):
        self.loss_train, self.loss_val, self.acc_train, self.acc_val, self.vrmse_train, self.vrmse_val, self.vsagr_train, self.vsagr_val, self.vpcc_train, self.vpcc_val, self.vccc_train, self.vccc_val, self.armse_train, self.armse_val, self.asagr_train, self.asagr_val, self.apcc_train, self.apcc_val, self.accc_train, self.accc_val = logs

    def save_plt(self, hps):
        loss_path = os.path.join(hps['model_save_dir'], 'loss.jpg')
        acc_path = os.path.join(hps['model_save_dir'], 'acc.jpg')
        vrmse_path = os.path.join(hps['model_save_dir'], 'vrmse.jpg')
        vsagr_path = os.path.join(hps['model_save_dir'], 'vsagr.jpg')
        vpcc_path = os.path.join(hps['model_save_dir'], 'vpcc.jpg')
        vccc_path = os.path.join(hps['model_save_dir'], 'vccc.jpg')
        armse_path = os.path.join(hps['model_save_dir'], 'armse.jpg')
        asagr_path = os.path.join(hps['model_save_dir'], 'asagr.jpg')
        apcc_path = os.path.join(hps['model_save_dir'], 'apcc.jpg')
        accc_path = os.path.join(hps['model_save_dir'], 'accc.jpg')

        plt.figure()
        plt.plot(self.acc_train, 'g', label='Training Acc')
        plt.plot(self.acc_val, 'b', label='Validation Acc')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend()
        plt.grid()
        plt.savefig(acc_path)

        plt.figure()
        plt.plot(self.loss_train, 'g', label='Training Loss')
        plt.plot(self.loss_val, 'b', label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(loss_path)

        plt.figure()
        plt.plot(self.vrmse_train, 'g', label='Training valence RMSE')
        plt.plot(self.vrmse_val, 'b', label='Validation valence RMSE')
        plt.title('valence RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        plt.savefig(vrmse_path)

        plt.figure()
        plt.plot(self.vsagr_train, 'g', label='Training valence SAGR')
        plt.plot(self.vsagr_val, 'b', label='Validation valence SAGR')
        plt.title('valence SAGR')
        plt.xlabel('Epoch')
        plt.ylabel('SAGR')
        plt.legend()
        plt.grid()
        plt.savefig(vsagr_path)

        plt.figure()
        plt.plot(self.vpcc_train, 'g', label='Training valence PCC')
        plt.plot(self.vpcc_val, 'b', label='Validation valence PCC')
        plt.title('valence PCC')
        plt.xlabel('Epoch')
        plt.ylabel('PCC')
        plt.legend()
        plt.grid()
        plt.savefig(vpcc_path)

        plt.figure()
        plt.plot(self.vccc_train, 'g', label='Training valence CCC')
        plt.plot(self.vccc_val, 'b', label='Validation valence CCC')
        plt.title('valence CCC')
        plt.xlabel('Epoch')
        plt.ylabel('CCC')
        plt.legend()
        plt.grid()
        plt.savefig(vccc_path)

        plt.figure()
        plt.plot(self.armse_train, 'g', label='Training arousal RMSE')
        plt.plot(self.armse_val, 'b', label='Validation arousal RMSE')
        plt.title('arousal RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        plt.savefig(armse_path)

        plt.figure()
        plt.plot(self.asagr_train, 'g', label='Training arousal SAGR')
        plt.plot(self.asagr_val, 'b', label='Validation arousal SAGR')
        plt.title('arousal SAGR')
        plt.xlabel('Epoch')
        plt.ylabel('SAGR')
        plt.legend()
        plt.grid()
        plt.savefig(asagr_path)

        plt.figure()
        plt.plot(self.apcc_train, 'g', label='Training arousal PCC')
        plt.plot(self.apcc_val, 'b', label='Validation arousal PCC')
        plt.title('arousal PCC')
        plt.xlabel('Epoch')
        plt.ylabel('PCC')
        plt.legend()
        plt.grid()
        plt.savefig(apcc_path)

        plt.figure()
        plt.plot(self.accc_train, 'g', label='Training arousal CCC')
        plt.plot(self.accc_val, 'b', label='Validation arousal CCC')
        plt.title('arousal CCC')
        plt.xlabel('Epoch')
        plt.ylabel('CCC')
        plt.legend()
        plt.grid()
        plt.savefig(accc_path)
