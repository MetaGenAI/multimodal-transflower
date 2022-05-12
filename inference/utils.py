import torch
import pickle, json, yaml
from pathlib import Path
from training.utils import get_latest_checkpoint
from models import create_model
from training.options.train_options import TrainOptions

def load_model_from_logs_path(logs_path, no_grad=True, version_index=-1):
    latest_checkpoint = get_latest_checkpoint(logs_path, index=version_index)
    print(latest_checkpoint)
    checkpoint_dir = Path(latest_checkpoint).parent.parent.absolute()
    # exp_opt = json.loads(open("training/experiments/"+args.experiment_name+"/opt.json","r").read())
    exp_opt = yaml.load(open(str(checkpoint_dir)+"/hparams.yaml","r").read())
    opt = vars(TrainOptions().parse(parse_args=["--model", exp_opt["model"]]))
    print(opt)
    opt.update(exp_opt)
    # opt["cond_concat_dims"] = True
    # opt["bn_momentum"] = 0.0
    opt["batch_size"] = 1
    opt["phase"] = "inference"
    opt["tpu_cores"] = 0
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    print(opt)
    opt = Struct(**opt)

    # Load latest trained checkpoint from experiment
    model = create_model(opt)
    #model = model.load_from_checkpoint(latest_checkpoint, opt=opt)
    #model = model.load_from_checkpoint(latest_checkpoint, opt=opt, strict=False)
    #for name,param in model.named_parameters():
    #    print(name)
    model = model.load_from_checkpoint(latest_checkpoint, opt=opt)
    if torch.cuda.is_available():
        model.cuda()
    if no_grad:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        for name,param in model.named_parameters():
            param.requires_grad = False
        if hasattr(model,"module_names"):
            for module in model.module_names:
                for name,param in getattr(model, "net"+module).named_parameters():
                    param.requires_grad = False
    return model, opt
