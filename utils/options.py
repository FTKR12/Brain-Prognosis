import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Brain Segmentation")
    ########## base options ##########
    parser.add_argument('--name', required=True)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--dataset_dir', default='dir/to/dataset/dir')
    parser.add_argument('--ans_path', default='dir/to/ans/path')
    parser.add_argument('--synthesize_model', default='resvit')
    parser.add_argument('--input', default='real_mri')

    ########## model options ##########
    parser.add_argument('--model_name', default='segresnet')
    # TODO: add mdoel args

    ########## training options ##########
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    ########## eval options ##########
    parser.add_argument('--eval_batch_size', default=1, type=int)
    parser.add_argument('--test_batch_size', default=1, type=int)

    args = parser.parse_args()
    return args