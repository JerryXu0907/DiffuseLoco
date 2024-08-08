import os
import sys

if __name__ == "__main__":
    # os.system("python scripts/evaluate_metrics_left.py --task=%s --load_run %s --headless" % ('cyber2_stand_dance_aug', "seed{}".format(0+1)))
    # os.system("python scripts/evaluate_metrics_right.py --task=%s --load_run %s --headless" % ('cyber2_stand_dance_aug', "seed{}".format(0+1)))
    # os.system("python scripts/evaluate_metrics_left.py --task=%s --load_run %s --headless" % ('cyber2_stand_dance_emlp', "seed{}".format(0+1)))
    # os.system("python scripts/evaluate_metrics_right.py --task=%s --load_run %s --headless" % ('cyber2_stand_dance_emlp', "seed{}".format(0+1)))
#     task_name = sys.argv[1]
#     left_or_right = sys.argv[2]
#     if left_or_right == "left":
#         print("eval left")
#         for ckpt in range(0,3):
#             os.system("python scripts/evaluate_metrics_left.py --task=%s --load_run %s --headless" % (task_name, "seed{}".format(ckpt+1)))
#     else:
#         print("eval right")
#         for ckpt in range(0,3):
#             os.system("python scripts/evaluate_metrics_right.py --task=%s --load_run %s --headless" % (task_name, "seed{}".format(ckpt+1)))


    for ckpt in range(0,3):
        # os.system("python scripts/evaluate_metrics_forward.py --task=%s --load_run %s --headless" % ('cyber2_walk_slope_aug', "seed{}".format(ckpt+1)))

        # os.system("python scripts/evaluate_metrics_forward.py --task=%s --load_run %s --headless" % ('cyber2_walk_slope_emlp', "seed{}".format(ckpt+1)))

        # os.system("python scripts/evaluate_metrics_forward.py --task=%s --load_run %s --headless" % ('cyber2_walk_slope', "seed{}".format(ckpt+1)))

        os.system("python scripts/evaluate_metrics_left.py --task=%s --load_run %s --headless" % ('cyber2_stand_dance', "seed{}".format(ckpt+1)))

        os.system("python scripts/evaluate_metrics_left.py --task=%s --load_run %s --headless" % ('cyber2_stand_dance_aug', "seed{}".format(ckpt+1)))

        os.system("python scripts/evaluate_metrics_left.py --task=%s --load_run %s --headless" % ('cyber2_stand_dance_emlp', "seed{}".format(ckpt+1)))

        # os.system("python scripts/evaluate_metrics_right.py --task=%s --load_run %s --headless" % ('cyber2_stand_dance', "seed{}".format(ckpt+1)))

        # os.system("python scripts/evaluate_metrics_right.py --task=%s --load_run %s --headless" % ('cyber2_stand_dance_aug', "seed{}".format(ckpt+1)))

        # os.system("python scripts/evaluate_metrics_right.py --task=%s --load_run %s --headless" % ('cyber2_stand_dance_emlp', "seed{}".format(ckpt+1)))