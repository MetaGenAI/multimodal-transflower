import torch
import torch.nn.functional as F
import time
import numpy as np

def autoregressive_generation_multimodal(features, model, autoreg_mods=[], teacher_forcing=False, 
                                         ground_truth=False, keep_latents=False, seed_lengths=None, sequence_length=None, 
                                         concat_autoreg_mods = True,
                                         use_temperature=False, temperature=1.0, save_jit=False, save_jit_path=None,
                                         apply_input_dropouts=False, verbose=False):
    # PREPARE INPUTS
    inputs_ = []
    for i,mod in enumerate(model.input_mods):
        input_ = features["in_"+mod]
        if model.input_types[i] == "c":
            input_ = torch.from_numpy(input_).float().to(model.device)
        else:
            input_ = torch.from_numpy(input_).long().to(model.device)
        if model.input_types[i] == "d" and model.opt.use_one_hot:
            input_ = F.one_hot(input_,num_classes=model.input_num_tokens[i]).squeeze(2)
        inputs_.append(input_)
    #NOTE: we are currently assuming the last modality is the one determining the sequence length
    if sequence_length is None:
        sequence_length = inputs_[-1].shape[0]
        #TODO: make this less ad-hoc
    for i,mod in enumerate(model.input_mods):
        input_ = inputs_[i]
        if model.input_proc_types[i] == "tile":
            print("tile")
            assert input_.shape[0] == 1 # havent implemented other cases..
            reps = [1 for a in input_.shape]
            reps[0] = sequence_length
            input_ = torch.tile(input_,tuple(reps))
        inputs_[i] = input_
    output_time_offsets = model.output_time_offsets
    input_time_offsets = model.input_time_offsets
    input_lengths = model.input_lengths
    output_lengths = model.output_lengths
    input_mods = model.input_mods
    output_mods = model.output_mods
    douts = model.douts

    for mod in autoreg_mods:
        assert mod in output_mods

    input_tmp = []
    if seed_lengths is None:
        seed_lengths = []
        for i,mod in enumerate(input_mods):
            seed_lengths.append(input_lengths[i])

    for i,mod in enumerate(input_mods):
        input_tmp.append(inputs_[i].clone()[input_time_offsets[i]:input_time_offsets[i]+seed_lengths[i]])

    if keep_latents:
        latents = None

    output_seq = []
    for i, mod in enumerate(output_mods):
        if concat_autoreg_mods:
            assert mod in autoreg_mods
            j = input_mods.index(mod)
            output_seq.append(input_tmp[j].detach().clone())
        else:
            output = torch.zeros((0,1,douts[i])).to(model.device)
            output_seq.append(output)

    print(sequence_length)
    with torch.no_grad():
        start_time = time.time()
        for t in range(sequence_length-max(input_lengths)+1):
            start_time_inner = time.time()
            print(t)
            inputs = [x.clone().to(model.device) for x in input_tmp]
            if apply_input_dropouts:
                for i,mod in enumerate(model.input_mods):
                    input_dropouts = float(model.opt.input_dropouts.split(",")[i])
                    if input_dropouts == 0: continue
                    mask = torch.rand(inputs[i].shape[0])<(1-input_dropouts)
                    mask = mask.unsqueeze(1).unsqueeze(1).to(model.device)
                    inputs[i] = inputs[i]*mask

            # PRODUCE OUTPUTS
            if not ground_truth:
                if keep_latents:
                    #TODO: do the keep latents stuff on the model side
                    if use_temperature:
                        outputs, latents = model.forward(inputs, zss=latents, temp=temperature)
                        if save_jit and t==0:
                            with torch.no_grad():
                                trace = torch.jit.trace(lambda x,y: model(x,zss=y,temp=temperature), ((inputs,latents),))
                    else:
                        outputs, latents = model.forward(inputs, zss=latents)
                        if save_jit and t==0:
                            with torch.no_grad():
                                trace = torch.jit.trace(lambda x,y: model(x,zss=y), ((inputs,latents),))
                else:
                    if use_temperature:
                        outputs = model.forward(inputs, temp=temperature)
                        if type(outputs) is tuple:
                            outputs=outputs[0]
                        if save_jit and t==0:
                            with torch.no_grad():
                                trace = torch.jit.trace(model, (inputs,), check_trace=False)
                    else:
                        outputs = model.forward(inputs)
                        if type(outputs) is tuple:
                            outputs=outputs[0]
                        if save_jit and t==0:
                            with torch.no_grad():
                                trace = torch.jit.trace(model, (inputs,), check_trace=False)

                if save_jit and t==0:
                    torch.jit.save(trace, save_jit_path)

            # APPEND OUTPUTS TO OUTPUT SEQUENCE
            for i, mod in enumerate(output_mods):
                if not ground_truth:
                    output = outputs[i]
                else:
                    j = input_mods.index(mod)
                    output = inputs_[j][t+output_time_offsets[i]+output_lengths[i]:t+output_time_offsets[i]+output_lengths[i]+1]
                    
                output_seq[i] = torch.cat([output_seq[i], output[:1].detach()])

            # AUTOREGRESSIVELY PRODUCE NEXT INPUTS
            for i, mod in enumerate(input_mods):
                if mod in autoreg_mods:
                    j = output_mods.index(mod)
                    if not ground_truth:
                        output = outputs[j]
                    else:
                        output = inputs_[i][t+input_time_offsets[j]+input_lengths[j]:t+input_time_offsets[j]+input_lengths[j]+1]
                    
                    if teacher_forcing:
                        input_tmp[i] = torch.cat([input_tmp[i][1:],inputs_[i][t+input_time_offsets[i]+input_lengths[i]:t+input_time_offsets[i]+input_lengths[i]+1]],0)
                    else:
                        # import pdb;pdb.set_trace()
                        if input_lengths[i]>1:
                            input_tmp[i] = torch.cat([input_tmp[i][-(input_lengths[i]-1):],output[:1].detach()],0)
                        else:
                            input_tmp[i] = output[:1].detach()

                    if verbose and not ground_truth:
                        targets = inputs_[i][t+output_time_offsets[j]:t+output_time_offsets[j]+1].permute(1,0,2).unsqueeze(1)
                        print(torch.mean((targets-outputs[j][:1].detach())**2))
                else:
                    if model.input_proc_types[i] == "single":
                        pass
                    else:
                        if input_lengths[i]>1:
                            input_tmp[i] = torch.cat([input_tmp[i][-(input_lengths[i]-1):],inputs_[i][input_time_offsets[i]+seed_lengths[i]+t:input_time_offsets[i]+seed_lengths[i]+t+1]],0)
                        else:
                            input_tmp[i] = inputs_[i][input_time_offsets[i]+seed_lengths[i]+t:input_time_offsets[i]+seed_lengths[i]+t+1]

            print("--- %s seconds ---" % (time.time() - start_time_inner))
    print("--- %s seconds ---" % (time.time() - start_time))
    return output_seq

