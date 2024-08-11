import os

if __name__ == "__main__":

    for ckpt in range(0,3):
        os.system("python scripts/evaluate_metrics_left.py --task=%s --load_run %s --headless" % ('cyber2_stand_dance', "seed{}".format(ckpt+1)))

        os.system("python scripts/evaluate_metrics_left.py --task=%s --load_run %s --headless" % ('cyber2_stand_dance_aug', "seed{}".format(ckpt+1)))

        os.system("python scripts/evaluate_metrics_left.py --task=%s --load_run %s --headless" % ('cyber2_stand_dance_emlp', "seed{}".format(ckpt+1)))
