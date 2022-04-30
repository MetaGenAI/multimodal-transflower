from pathlib import Path
from itertools import tee
import numpy as np
import torch
from .base_dataset import BaseDataset
import torch.nn.functional as F

def find_example_idx(n, cum_sums, idx = 0):
    N = len(cum_sums)
    search_i = N//2 - 1
    if N > 1:
        if n < cum_sums[search_i]:
            return find_example_idx(n, cum_sums[:search_i+1], idx=idx)
        else:
            return find_example_idx(n, cum_sums[search_i+1:], idx=idx+search_i+1)
    else:
        if n < cum_sums[0]:
            return idx
        else:
            return idx + 1


class MultimodalDataset(BaseDataset):

    def __init__(self, opt, split="train"):
        super().__init__()
        self.opt = opt
        data_path = Path(opt.data_dir)
        if not data_path.is_dir():
            raise ValueError('Invalid directory:'+opt.data_dir)

        print(opt.base_filenames_file)
        if split == "train":
            temp_base_filenames = [x[:-1] for x in open(data_path.joinpath(opt.base_filenames_file), "r").readlines()]
        else:
            temp_base_filenames = [x[:-1] for x in open(data_path.joinpath("base_filenames_"+split+".txt"), "r").readlines()]
        if opt.num_train_samples > 0:
            temp_base_filenames = np.random.choice(temp_base_filenames, size=opt.num_train_samples, replace=False)
        self.base_filenames = []

        input_mods = self.opt.input_modalities.split(",")
        output_mods = self.opt.output_modalities.split(",")
        self.input_lengths = input_lengths = [int(x) for x in str(self.opt.input_lengths).split(",")]
        self.output_lengths = output_lengths = [int(x) for x in str(self.opt.output_lengths).split(",")]
        self.output_time_offsets = output_time_offsets = [int(x) for x in str(self.opt.output_time_offsets).split(",")]
        self.input_time_offsets = input_time_offsets = [int(x) for x in str(self.opt.input_time_offsets).split(",")]
        self.input_dropouts = input_dropouts = [float(x) for x in str(self.opt.input_dropouts).split(",")]

        if self.opt.input_types is None:
            input_types = ["c" for inp in input_mods]
        else:
            input_types = self.opt.input_types.split(",")

        if self.opt.input_proc_types is None:
            input_proc_types = ["none" for inp in input_mods]
        else:
            input_proc_types = self.opt.input_proc_types.split(",")

        if self.opt.output_proc_types is None:
            output_proc_types = ["none" for out in output_mods]
        else:
            output_proc_types = self.opt.output_proc_types.split(",")

        assert len(input_types) == len(input_mods)
        assert len(input_proc_types) == len(input_mods)
        assert len(output_proc_types) == len(output_mods)
        self.input_types = input_types
        self.input_proc_types = input_proc_types
        self.output_proc_types = output_proc_types

        if self.opt.input_num_tokens is None:
            self.input_num_tokens = [0 for inp in input_mods]
        else:
            self.input_num_tokens  = [int(x) for x in self.opt.input_num_tokens.split(",")]
            # for i,mod in enumerate(input_mods):
            #     if self.input_types[i] == "d" and self.opt.use_one_hot:
            #         assert self.input_num_tokens[i] == self.dins[i]

        if self.opt.output_num_tokens is None:
            self.output_num_tokens = [0 for inp in output_mods]
        else:
            self.output_num_tokens  = [int(x) for x in self.opt.output_num_tokens.split(",")]

        if len(output_time_offsets) < len(output_mods):
            if len(output_time_offsets) == 1:
                self.output_time_offsets = output_time_offsets = output_time_offsets*len(output_mods)
            else:
                raise Exception("number of output_time_offsets doesnt match number of output_mods")

        if len(input_time_offsets) < len(input_mods):
            if len(input_time_offsets) == 1:
                self.input_time_offsets = input_time_offsets = input_time_offsets*len(input_mods)
            else:
                raise Exception("number of input_time_offsets doesnt match number of input_mods")

        if len(input_dropouts) < len(input_mods):
            if len(input_dropouts) == 1:
                self.input_dropouts = input_dropouts = input_dropouts*len(input_mods)
            else:
                raise Exception("number of input_dropouts doesnt match number of input_mods")

        self.input_features = {input_mod:{} for input_mod in input_mods}
        self.output_features = {output_mod:{} for output_mod in output_mods}

        min_length = max(max(np.array(input_lengths) + np.array(input_time_offsets)), max(np.array(output_time_offsets) + np.array(output_lengths)) ) - min(0,min(min(output_time_offsets),min(input_time_offsets)))
        padding_length = max(np.array(input_lengths) + np.array(input_time_offsets))
        print(min_length)

        self.total_frames = 0
        self.frame_cum_sums = []

        #Get the list of files containing features (in numpy format for now), and populate the dictionaries of input and output features (separated by modality)
        for base_filename in temp_base_filenames:
            length_0 = 1
            file_too_short = False
            first_length=True
            for i, mod in enumerate(input_mods):
                feature_file = data_path.joinpath(base_filename+"."+mod+".npy")
                if self.input_proc_types[i] != "none": continue
                # print(feature_file)
                try:
                    features = np.load(feature_file)
                    length = features.shape[0]
                    if opt.zero_padding:
                        length += padding_length
                    if first_length:
                        length_0 = length
                        first_length=False
                    else:
                        assert length == length_0
                    if length < min_length:
                        # print("Smol sequence "+base_filename+"."+mod+"; ignoring..")
                        file_too_short = True
                        break
                except FileNotFoundError:
                    raise Exception("An unprocessed input feature found "+base_filename+"."+mod+"; need to run preprocessing script before starting to train with them")

            if file_too_short: continue

            #length_0 = 1
            #first_length=True
            for i, mod in enumerate(output_mods):
                feature_file = data_path.joinpath(base_filename+"."+mod+".npy")
                if self.output_proc_types[i] != "none": continue
                try:
                    features = np.load(feature_file)
                    length = features.shape[0]
                    if opt.zero_padding:
                        length += padding_length
                    #if first_length:
                    #    length_0 = length
                    #    first_length=False
                    #else:
                    assert length == length_0
                    #if length < min_length:
                    #    # print("Smol sequence "+base_filename+"."+mod+"; ignoring..")
                    #    file_too_short = True
                    #    break
                except FileNotFoundError:
                    raise Exception("An unprocessed output feature found "+base_filename+"."+mod+"; need to run preprocessing script before starting to train with them")

            if file_too_short: continue

            for i, mod in enumerate(input_mods):
                feature_file = data_path.joinpath(base_filename+"."+mod+".npy")
                features = np.load(feature_file)
                features = self.preprocess_inputs(i,features,tile_length=length_0,padding_length=padding_length) #assumes the sequence length of things to be tiled is 1
                self.input_features[mod][base_filename] = features

            for i, mod in enumerate(output_mods):
                feature_file = data_path.joinpath(base_filename+"."+mod+".npy")
                features = np.load(feature_file)
                features = self.preprocess_outputs(i,features,tile_length=length_0,padding_length=padding_length) #assumes the sequence length of things to be tiled is 1
                self.output_features[mod][base_filename] = features

            found_full_seq = False
            for i,mod in enumerate(input_mods):
                if self.input_proc_types[i] == "none":
                    sequence_length = self.input_features[mod][base_filename].shape[0]
                    found_full_seq = True
                if found_full_seq: break
            for i,mod in enumerate(output_mods):
                if found_full_seq: break
                if self.output_proc_types[i] == "none":
                    sequence_length = self.output_features[mod][base_filename].shape[0]
                    found_full_seq = True
                if found_full_seq: break
            if not found_full_seq:
                sequence_length = 1

            possible_init_frames = sequence_length-max(max(input_lengths)+max(input_time_offsets),max(output_time_offsets)+max(output_lengths))+1
            self.total_frames += possible_init_frames
            self.frame_cum_sums.append(self.total_frames)

            self.base_filenames.append(base_filename)

        print("sequences added: "+str(len(self.base_filenames)))
        assert len(self.base_filenames)>0, "List of files for training cannot be empty"
        for mod in input_mods:
            assert len(self.input_features[mod].values()) == len(self.base_filenames)
        for mod in output_mods:
            assert len(self.output_features[mod].values()) == len(self.base_filenames)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--sampling_rate', default=44100, type=float)
        parser.add_argument('--dins', default=None, help="input dimension for continuous inputs. Embedding dimension for discrete inputs")
        parser.add_argument('--douts', default=None)
        parser.add_argument('--input_modalities', default='mp3_mel_100')
        parser.add_argument('--output_modalities', default='mp3_mel_100')
        parser.add_argument('--input_lengths', help='input sequence length')
        parser.add_argument('--input_num_tokens', help='num_tokens. use 0 for continuous inputs')
        parser.add_argument('--output_num_tokens', help='num_tokens. use 0 for continuous inputs')
        parser.add_argument('--input_types', default=None, help='Comma-separated list of input types: d for discrete, c for continuous. E.g. d,c,c. Assumes continuous if not specified')
        parser.add_argument('--input_proc_types', default=None, help='Comma-separated list of approaches to process input: tile for tiling, single for non-timedsynnced, e.g. Assumes none if not specified')
        parser.add_argument('--output_proc_types', default=None, help='Comma-separated list of approaches to process outputs: tile for tiling, single for non-timedsynnced, e.g.. Assumes none if not specified')
        # parser.add_argument('--predict_deltas', default="false", help='Comma-separated list of true or false, specifying whether to predict deltas (changes) for each the output modalities')
        parser.add_argument('--output_lengths', help='output sequence length')
        parser.add_argument('--output_time_offsets', default="1", help='time shift between the last read input, and the output predicted. The default value of 1 corresponds to predicting the next output')
        parser.add_argument('--input_time_offsets', default="0", help='time shift between the beginning of each modality and the first modality')
        parser.add_argument('--input_dropouts', default="0", help='dropout applied to the different input modalities')
        parser.add_argument('--max_token_seq_len', type=int, default=1024)
        parser.add_argument('--use_one_hot', action='store_true', help='whether to use one hot representation for discrete inputs')
        parser.add_argument('--not_shuffle', action='store_true', help='whether to not shuffle the dataset')
        parser.add_argument('--num_train_samples', type=int, default=0, help='if 0 then use all of them')
        parser.add_argument('--zero_padding', action='store_true', help='whether to not to left pad the examples with zeros')

        return parser

    def name(self):
        return "MultiModalDataset"

    def preprocess_inputs(self,i,features, tile_length = 1, padding_length = 0):
        if self.input_types[i] == "d" and self.opt.use_one_hot:
            features = F.one_hot(torch.tensor(features).long(),num_classes=self.input_num_tokens[i]).numpy()
        if self.input_proc_types[i] == "tile":
            assert features.shape[0] == 1 # havent implemented other cases..
            reps = np.ones(len(features.shape)).astype(np.int32)
            reps[0] = tile_length
            features = np.tile(features,reps)
            # import pdb;pdb.set_trace()
        elif self.input_proc_types[i] == "none":
            if self.opt.zero_padding:
                ## we pad the song features with zeros to imitate during training what happens during generation
                features = np.concatenate([np.zeros((padding_length, features.shape[1])), features])
        return features

    def preprocess_outputs(self,i,features, tile_length = 1, padding_length = 0):
        #not implemented yet:
        #TODO: implement
        # if self.output_types[i] == "d" and self.opt.use_one_hot:
        #     features = F.one_hot(torch.tensor(features),num_classes=self.output_num_tokens[i]).numpy()
        if self.output_proc_types[i] == "tile":
            assert features.shape[0] == 1 # havent implemented other cases..
            reps = np.ones(len(features.shape)).astype(np.int32)
            reps[0] = tile_length
            features = np.tile(features,reps)
        elif self.output_proc_types[i] == "none":
            if self.opt.zero_padding:
                ## we pad the song features with zeros to imitate during training what happens during generation
                features = np.concatenate([np.zeros((padding_length, features.shape[1])), features])
        return features

    def process_input(self,j,xx,index):
        input_lengths = self.input_lengths
        input_time_offsets = self.input_time_offsets
        if self.input_proc_types[j]!= "single":
            return_tensor = torch.tensor(xx[index+input_time_offsets[j]:index+input_time_offsets[j]+input_lengths[j]]).float()
        else:
            if len(xx.shape) == 3:
                idx = np.random.randint(xx.shape[0]) #chose random from the set
                xx = xx[idx]
                # print(xx.shape)
            # if self.opt.use_one_hot:
            #     return_tensor = torch.tensor(xx)
            # else:
            #     return_tensor = torch.tensor(xx).long().unsqueeze(1)
            if len(xx.shape) == 1:
                return_tensor = torch.tensor(xx).long().unsqueeze(1)
            else:
                #print(xx)
                return_tensor = torch.tensor(xx).long()

        if self.input_dropouts[j]>0:
            mask = torch.rand(return_tensor.shape[0])<(1-self.input_dropouts[j])
            mask = mask.unsqueeze(1)
            return_tensor = return_tensor*mask

        return return_tensor

    def process_output(self,j,yy,index):
        output_lengths = self.output_lengths
        output_time_offsets = self.output_time_offsets
        if self.output_proc_types[j]!="single":
            return torch.tensor(yy[index+output_time_offsets[j]:index+output_time_offsets[j]+output_lengths[j]]).float()
        else:
            if self.opt.use_one_hot:
                return torch.tensor(xx)
            else:
                return torch.tensor(xx).long().unsqueeze(1)

    def __getitem__(self, item):
        idx = find_example_idx(item, self.frame_cum_sums)
        base_filename = self.base_filenames[idx]
        #print(base_filename)

        input_mods = self.opt.input_modalities.split(",")
        #print(input_mods)
        output_mods = self.opt.output_modalities.split(",")

        x = [self.input_features[mod][base_filename] for mod in input_mods]
        y = [self.output_features[mod][base_filename] for mod in output_mods]

        # normalization of individual features for the sequence
        # not doing this any more as we are normalizing over all examples now
        #x = [(xx-np.mean(xx,0,keepdims=True))/(np.std(xx,0,keepdims=True)+1e-5) for xx in x]
        #y = [(yy-np.mean(yy,0,keepdims=True))/(np.std(yy,0,keepdims=True)+1e-5) for yy in y]

        if idx > 0: index = item - self.frame_cum_sums[idx-1]
        else: index = item

        ## CONSTRUCT TENSOR OF INPUT FEATURES ##
        input_windows = [self.process_input(j,xx,index) for j,xx in enumerate(x)]

        ## CONSTRUCT TENSOR OF OUTPUT FEATURES ##
        output_windows = [self.process_output(j,yy,index) for j,yy in enumerate(y)]

        # print(input_windows[i])
        return_dict = {}
        for i,mod in enumerate(input_mods):
            return_dict["in_"+mod] = input_windows[i]
        for i,mod in enumerate(output_mods):
            return_dict["out_"+mod] = output_windows[i]

        return return_dict

    def __len__(self):
        # return len(self.base_filenames)
        return self.total_frames
        # return 2


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
