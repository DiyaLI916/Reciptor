import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='tri-joint parameters')
    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')

    # data
    parser.add_argument('--img_path', default='data/images/')
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--workers', default=30, type=int)

    # model
    parser.add_argument('--batch_size', default=300, type=int) #160
    parser.add_argument('--snapshots', default='snapshots/', type=str)

    # im2recipe model
    parser.add_argument('--embDim', default=600, type=int)  #1024

    parser.add_argument('--nRNNs', default=1, type=int)
    parser.add_argument('--srnnDim', default=1024, type=int)
    parser.add_argument('--irnnDim', default=300, type=int)

    parser.add_argument('--imfeatDim', default=2048, type=int)
    parser.add_argument('--stDim', default=1024, type=int)
    parser.add_argument('--ingrW2VDim', default=300, type=int)
    parser.add_argument('--maxSeqlen', default=20, type=int)
    parser.add_argument('--maxIngrs', default=20, type=int)
    parser.add_argument('--numClasses', default=9, type=int)

    ##
    parser.add_argument('--triplet_loss', default=True, type=bool)
    parser.add_argument('--triplet_path', default='data/recipe1M/triplet_sample', type=str)

    ## train towards label
    parser.add_argument('--semantic_reg', default=True, type=bool)
    # parser.add_argument('--semantic_reg', default=False, type=bool)

    # training 
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=20, type=int) # 720
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--ingrW2V', default='data/text/vocab.bin', type=str)
    parser.add_argument('--valfreq', default=2, type=int) # 10
    parser.add_argument('--patience', default=1, type=int)
    parser.add_argument('--freeVision', default=False, type=bool)
    parser.add_argument('--freeRecipe', default=True, type=bool)

    parser.add_argument('--cos_weight', default=0.80, type=float)
    parser.add_argument('--cls_weight', default=0.10, type=float) # 0.01
    parser.add_argument('--tri_weight', default=0.90, type=float)  # 0.01

    # resume
    parser.add_argument('--model_state_dict', default='', type=str)
    parser.add_argument('--resume', default='', type=str)

    # test
    parser.add_argument('--test', default=False, type=bool)

    parser.add_argument('--path_results', default='results/', type=str)
    parser.add_argument('--model_path', default='', type=str)

    # dataset
    parser.add_argument('--maxlen', default=167, type=int) # default 167
    parser.add_argument('--vocab', default='../data/text/vocab.txt', type=str)
    parser.add_argument('--dataset', default='../data/recipe1M/', type=str)
    parser.add_argument('--sthdir', default='../data/', type=str)

    ## for new classification task
    parser.add_argument("--output_dir", default='predictionResult', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--output_dir", default=None, type=str,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_labels", default='1048', type=int)
    parser.add_argument("--hidden_dropout_prob", default='0.1', type=float)
    parser.add_argument("--hidden_size", default='600', type=int)
    parser.add_argument("--do_train", default=False, type=bool)
    parser.add_argument("--do_eval", default=True, type=bool)

    parser.add_argument("--eval_model", default='./predictionResult/checkpoint-29000/training_state.pth', type=str)
    parser.add_argument("--eval_output_dir", default='labelPredResults', type=str)
    parser.add_argument("--output_mode", default='classification')

    return parser
