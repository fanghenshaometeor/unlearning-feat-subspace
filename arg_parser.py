import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MU in low-dimensional feature subspace")

    ##################################### Dataset and model #################################################
    parser.add_argument("--dataset", type=str, default="tinyimagenet", help="dataset")
    parser.add_argument("--data_dir", type=str, default="./tiny-imagenet-200", help="dir to tiny-imagenet")
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--arch", type=str, default="swin_t", help="model architecture")

    ##################################### General setting ############################################
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--save_dir", type=str, default=None, help="dir to save the trained models")
    parser.add_argument("--model_path", type=str, default=None, help="the path of original model")

    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=None, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=None, help="weight decay")
    parser.add_argument("--epochs", type=int, default=None, help="number of total epochs to run")
    parser.add_argument("--optim", type=str, default=None, help="optimizer")

    ##################################### Unlearn setting #################################################
    parser.add_argument("--unlearn", type=str, default=None, help="method to unlearn")
    parser.add_argument("--unlearn_lr", type=float, default=1, help="initial learning rate for unlearning")
    parser.add_argument("--unlearn_epochs", type=int, default=10, help="number of total epochs for unlearn to run")

    parser.add_argument("--class_forget", action="store_true", help="Specific class to forget")
    parser.add_argument("--label_to_forget", type=str, default=None, nargs="*", help="class to forget")
    parser.add_argument("--num_label_to_forget", type=int, default=None, help="number of classes to forget, for extreme unlearning")
    parser.add_argument("--forget_only", action="store_true", help="only forget data involved into training")

    parser.add_argument("--continual_unlearn", action="store_true", help="activate continual unlearning")
    parser.add_argument("--continual_round", type=int, default=None, help="rounds of continual unlearning")

    parser.add_argument("--random_forget", action="store_true", help="Specific random data to forget")
    parser.add_argument("--num_to_forget", type=int, default=None, help="Number of data to forget")

    # bad teaching
    parser.add_argument("--BT_KL_temp", type=float, default=1, help="KL temperature in bad teaching")

    # learn to unlearn
    parser.add_argument("--L2UL_Nadv", type=int, default=1, help="number of generated adv. examples per input")
    parser.add_argument("--L2UL_attack_eps", type=float, default=1, help="bound of generated adv. examples")
    parser.add_argument("--L2UL_attack_steps", type=int, default=1, help="attack steps")
    parser.add_argument("--L2UL_attack_alpha", type=float, default=1, help="attack step length")
    parser.add_argument("--L2UL_reg_lamb", type=float, default=1, help="loss coefficient")

    # subspace machine unlearing
    parser.add_argument("--subspace_s", type=int, default=None, help="subspace dimension")
    parser.add_argument("--subspace_steps", type=int, default=10, help="number of steps for subspace to run")

    return parser.parse_args()
