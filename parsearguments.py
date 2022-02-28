import sys
import argparse

def getarguments():
    parser = argparse.ArgumentParser(description='GRAPHITE')
    # Key parameters
    parser.add_argument('--victim_id', '-v', default='14', help='The victim class id.')
    parser.add_argument('--target_id', '-t', default='1', help='The target class id.')
    parser.add_argument('--tr_lo', type=float, default=0.25, help='The threshold for coarse grained reduction.')
    parser.add_argument('--tr_hi', type=float, default=0.5, help='The threshold for fine grained reduction.')
    parser.add_argument('--scorefile', '-s', default='score.py', help='The file for scoring in mask generation.')
    parser.add_argument('--network', '-n', default='GTSRB', help='The dataset / type of network to attack.')
    parser.add_argument('--heatmap', default='Target', help='The type of heatmap to use.')
    parser.add_argument('--boost_transforms', '-b', type=int, default=100, help='The number of transforms to use in boosting.')
    parser.add_argument('--mask_transforms', '-m', type=int, default=100, help='The number of transforms to use in mask generation.')
    parser.add_argument('--coarse_mode', default='binary', help='The type of mode to perform coarse reduction: binary or linear.')
    parser.add_argument('--heatmap_file', help='Start heatmap file, if pre-saved already')
    parser.add_argument('--num_test_xforms', type=int, default=-1, help='Set to > 0 if you want to evaluate the result on x transforms, \
                                                                         where x != the value of -b.')
    parser.add_argument('--max_mask_size', type=int, default=-1, help='Set to > 0 to enable m_max.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--joint_iters', type=int, default=1, help='Number of times to altnerate mask generation and boosting')
    parser.add_argument('--image_id', default='', help='An image id to tag to outputs. Useful for CIFAR-10.')
    parser.add_argument('--img_v', help='File to the victim image if it is not in the default location.')
    parser.add_argument('--img_t', help='File to the target image if it is not in the default location.')
    parser.add_argument('--hull', help='File to specify the mask of the object if it is not in default location.')
    parser.add_argument('--pt_file', help='File to specify the corners of the object for perspective transform if it is not in default location.')
    parser.add_argument('--out_path', default='outputs/', help='Output directory base.')

    # Extra ALPR parameters
    parser.add_argument('--vic_license_plate', help='The victim license plate, if ALPR attack.')
    parser.add_argument('--tar_license_plate', help='The target license plate, if ALPR attack.')
    parser.add_argument('--border_outer', help='Mask for the outer border for ALPR.')
    parser.add_argument('--border_inner', help='Mask for the inner border for ALPR.')
    parser.add_argument('--tag', default='', help='Tag to optionally add to ALPR outputs.')

    # Misc. extra knobs for additional tweaking of settings, if desired. Can mostly be left alone.
    parser.add_argument('--bt', action='store_true', help='Whether or not to apply backtracking line search in RGF.')
    parser.add_argument('--square_x', help='x coordinate of square, if specifying a square mask.')
    parser.add_argument('--square_y', help='y coordinate of square, if specifying a square mask.')
    parser.add_argument('--square_size', help='Size square, if specifying a square mask.')
    parser.add_argument('--early_boost_exit', action='store_true', help='If testing a square mask and you want to exit boosting \
                                                                         early if the result is clearly not transform-robust.')
    args = parser.parse_args()

    args.coarse_error = 1 - args.tr_hi
    args.reduce_error = 1 - args.tr_lo


    assert args.coarse_error <= args.reduce_error
    assert args.coarse_error >= 0  # upon coarse_reduction, we want transform_robustness to be >= 1-coarseerror
    assert args.reduce_error <= 1   # upon reduction, we want transform_robustness to be > 1-reduceerror
    assert args.heatmap in ['Target', 'Victim', 'Random']

    baseimagedir = 'inputs/' + args.network + "/"
    if args.img_v is None:
        args.img_v = baseimagedir + 'images/' + args.victim_id + ".png"
    if args.img_t is None:
        args.img_t = baseimagedir + 'images/' + args.target_id + ".png"
    if args.hull is None:
        try:
            args.mask = baseimagedir + 'Hulls/' + args.victim_id  + ".png"
        except:
            args.mask = "trivial_hull.png"
    else:
        args.mask = args.hull
    args.lbl_v = int(args.victim_id)
    args.lbl_t = int(args.target_id)
    if args.pt_file is None:
        try:
            pt_file = baseimagedir + 'Points/' + args.victim_id + ".csv"
            open(pt_file)
            args.pt_file = pt_file
        except:
            args.pt_file = None
    return args
