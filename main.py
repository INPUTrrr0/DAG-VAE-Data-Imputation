from train import *



parser = argparse.ArgumentParser()

parser.add_argument('--missing_percentage', type = float, default = 0.5,
                    help ='missing percentage/100 data has missing value')
parser.add_argument('--data_variable_size', type=int, default=30,
                    help='the number of variables in synthetic generated data')
parser.add_argument('--graph_degree', type=int, default=3,
                    help='the number of degree in generated DAG graph')

parser.add_argument('--batch-size', type=int, default = 100, # note: should be divisible by sample size, otherwise throw an error
                    help='Number of samples per batch.')
parser.add_argument('--k_max_iter', type = int, default = 100,
                    help ='the max iteration number for searching lambda and c')
parser.add_argument('--data_sample_size', type=int, default=500,
                    help='the number of samples of data')
parser.add_argument('--epochs', type=int, default= 200,
                    help='Number of epochs to train.')

# -----------data parameters ------
# configurations
parser.add_argument('--data_type', type=str, default= 'synthetic',
                    choices=['synthetic', 'discrete', 'real'],
                    help='choosing which experiment to do.')
parser.add_argument('--data_filename', type=str, default= 'alarm',
                    help='data file name containing the discrete files.')
parser.add_argument('--data_dir', type=str, default= 'data/',
                    help='data file name containing the discrete files.')

parser.add_argument('--graph_type', type=str, default='erdos-renyi',
                    help='the type of DAG graph by generation method')

parser.add_argument('--graph_sem_type', type=str, default='linear-gauss',
                    help='the structure equation model (SEM) parameter type')
parser.add_argument('--graph_linear_type', type=str, default='linear',
                    help='the synthetic data type: linear -> linear SEM, nonlinear_1 -> x=Acos(x+1)+z, nonlinear_2 -> x=2sin(A(x+0.5))+A(x+0.5)+z')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--x_dims', type=int, default=1, #changed here
                    help='The number of input dimensions: default 1.')
parser.add_argument('--z_dims', type=int, default=1,
                    help='The number of latent variable dimensions: default the same as variable size.')

# -----------training hyperparameters
parser.add_argument('--optimizer', type = str, default = 'Adam',
                    help = 'the choice of optimizer used')
parser.add_argument('--graph_threshold', type=  float, default = 0.3,  # 0.3 is good, 0.2 is error prune
                    help = 'threshold for learned adjacency matrix binarization')
parser.add_argument('--tau_A', type = float, default=0.0,
                    help='coefficient for L-1 norm of A.')
parser.add_argument('--lambda_A',  type = float, default= 0.,
                    help='coefficient for DAG constraint h(A).')
parser.add_argument('--c_A',  type = float, default= 1,
                    help='coefficient for absolute value h(A).')
parser.add_argument('--use_A_connect_loss',  type = int, default= 0,
                    help='flag to use A connect loss')
parser.add_argument('--use_A_positiver_loss', type = int, default = 0,
                    help = 'flag to enforce A must have positive values')


parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')


parser.add_argument('--lr', type=float, default=3e-3,  # basline rate = 1e-3
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')


parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp, or sem).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='_springs5',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default=r"C:\Users\space\DAG-GNN\src\logs",
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')


parser.add_argument('--h_tol', type=float, default = 1e-8,
                    help='the tolerance of error of h(A) to zero')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default= 1.0,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")

# Save model and meta-data. Always saves in a new sub-folder.
#if args.save_folder:
    #exp_counter = 0
    #now = datetime.datetime.now()
    #timestamp = now.isoformat()
    #save_folder = r'{}\exp{}'.format(args.save_folder, timestamp)
    # safe_name = save_folder.text.replace('/', '_')
    #os.makedirs(save_folder)
    #os.mkdir(r"C:\Users\space\DAG-GNN\src\logs\exp3")
    #save_folder=r"C:\Users\space\DAG-GNN\src\logs\exp3"
    #meta_file = os.path.join(save_folder, 'metadata.pkl')
    #encoder_file = os.path.join(save_folder, 'encoder.pt')
    #decoder_file = os.path.join(save_folder, 'decoder.pt')

    #log_file = os.path.join(save_folder, 'log.txt')
    #log = open(log_file, 'w')

    #pickle.dump({'args': args}, open(meta_file, "wb"))
#else:
    #print("WARNING: No save_folder provided!" +
     #     "Testing (within this script) will throw an error.")







count=0
while (count<12):
    print("trial{}".format(count))
    try:
        experiment(args)
        count=count+1
    except:
        print("error!")