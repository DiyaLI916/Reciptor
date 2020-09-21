import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='reciptor parameters')

    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')

    # data
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--workers', default=30, type=int)
    parser.add_argument('--full_data_path', default='foodcom_sample', help=['foodcom_sample', 'foodcom'])

    # model
    # must be N*3 for triplet
    parser.add_argument('--batch_size', default=120, type=int) #160
    parser.add_argument('--snapshots', default='snapshots/', type=str)

    # im2recipe model
    parser.add_argument('--embDim', default=600, type=int)  #1024
    parser.add_argument('--nRNNs', default=1, type=int)
    parser.add_argument('--srnnDim', default=1024, type=int)
    parser.add_argument('--irnnDim', default=300, type=int)
    parser.add_argument('--stDim', default=1024, type=int)
    parser.add_argument('--ingrW2VDim', default=300, type=int)
    parser.add_argument('--maxSeqlen', default=20, type=int)
    parser.add_argument('--maxIngrs', default=20, type=int)
    parser.add_argument('--numClasses', default=9, type=int)
    parser.add_argument('--model_type', type=str, default='sjm', help='[reciptor|jm|sjm]')

    ##
    parser.add_argument('--triplet_loss', action="store_true", default=False)
    parser.add_argument('--triplet_path', default='data/triplet_sample.txt', type=str)
    parser.add_argument('--semantic_reg', default=False, type=bool)
    parser.add_argument('--category_cla', default=False, type=bool)

    # training 
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=100, type=int) # 720
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--ingrW2V', default='data/text/vocab.bin', type=str)
    parser.add_argument('--valfreq', default=4, type=int) # 10
    parser.add_argument('--patience', default=1, type=int)
    parser.add_argument('--freeRecipe', default=True, type=bool)

    parser.add_argument('--cos_weight', default=0.80, type=float)
    parser.add_argument('--cls_weight', default=0.10, type=float) # 0.01
    parser.add_argument('--tri_weight', default=0.20, type=float)

    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--save_tuned_embed', default=False, action="store_true")

    # dataset
    parser.add_argument('--maxlen', default=167, type=int)
    parser.add_argument('--vocab', default='data/text/vocab.txt', type=str)
    parser.add_argument('--dataset', default='data/', type=str)
    parser.add_argument('--sthdir', default='data/', type=str)

    ## for classification task
    parser.add_argument("--pretrained_embed_path", default='', type=str)
    parser.add_argument("--hidden_dropout_prob", default='0.1', type=float)
    parser.add_argument("--do_test", default=False, type=bool)
    parser.add_argument("--hidden_size", default='600', type=int)

    return parser