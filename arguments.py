import argparse

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--working-dir', default='./',
                        help='working directory')

    parser.add_argument('--exp-prefix', default='debug_',
                        help='exp prefix')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 3e-4)')

    parser.add_argument('--lr-decay', action='store_true', default=False,
                        help='decay of learning rate.')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')

    parser.add_argument('--test-batch-size', type=int, default=1,
                        help='test batch size (default: 1)')

    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')

    parser.add_argument('--train-step', type=int, default=1000000,
                        help='number of interactions in total (default: 1000000)')

    parser.add_argument('--snapshot', type=int, default=100000,
                        help='snapshot (default: 100000)')

    parser.add_argument('--image-width', type=int, default=224,
                        help='image width (default: 640)')

    parser.add_argument('--image-height', type=int, default=224,
                        help='image height (default: 320)')

    parser.add_argument('--hsv-rand', type=float, default=0.0,
                        help='augment rand-hsv by adding different hsv to a set of images (default: 0.0)')

    parser.add_argument('--rand-blur', type=float, default=0.0,
                        help='randomly load blur image for training (default: 0.0)')

    parser.add_argument('--data-file', default='',
                        help='txt file specify the training data (default: "")')

    parser.add_argument('--val-file', default='',
                        help='txt file specify the validation data (default: "")')

    parser.add_argument('--load-model', action='store_true', default=False,
                        help='load pretrained model (default: False)')

    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')

    parser.add_argument('--test', action='store_true', default=False,
                        help='test (default: False)')

    parser.add_argument('--test-num', type=int, default=10,
                        help='test (default: 10)')

    parser.add_argument('--test-interval', type=int, default=100,
                        help='The test interval.')

    parser.add_argument('--print-interval', type=int, default=1,
                        help='The plot interval for updating the figures.')

    parser.add_argument('--plot-interval', type=int, default=100,
                        help='The plot interval for updating the figures.')

    parser.add_argument('--network', type=int, default=1,
                        help='network structure')

    parser.add_argument('--no-data-augment', action='store_true', default=False,
                        help='no data augmentation (default: False)')

    parser.add_argument('--multi-gpu', type=int, default=1,
                        help='multiple gpus numbers (default: False)')

    parser.add_argument('--platform', default='local',
                        help='deal with different data root directory in dataloader, could be one of local, cluster, azure (default: "local")')

    parser.add_argument('--stride', type=int, default=2,
                        help='The stride of the dataloader.')

    parser.add_argument('--skip', type=int, default=0,
                        help='The skip of the dataloader.')

    parser.add_argument('--crop-num', type=int, default=0,
                        help='The number of patches of the dataloader.')

    parser.add_argument('--data-root', default='/project/learningphysics',
                        help='root folder of the data')

    args = parser.parse_args()

    return args
