
import torch
import torch.nn.functional as F
import time

#TODO: implement option to include the conditioning bit of input in the output
def autoregressive_generation_multimodal(features, model, autoreg_mods=[], teacher_forcing=False, ground_truth=False, keep_latents=False, seed_lengths=None, sequence_length=None, use_temperature=False, temperature=1.0, save_jit=False, save_jit_path=None):
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
    # predicted_inputs = model.predicted_inputs
    for mod in autoreg_mods:
        assert mod in output_mods

    input_tmp = []
    if seed_lengths is None:
        seed_lengths = []
        for i,mod in enumerate(input_mods):
            seed_lengths.append(input_lengths[i])
    #print(seed_lengths)
    for i,mod in enumerate(input_mods):
        input_tmp.append(inputs_[i].clone()[input_time_offsets[i]:input_time_offsets[i]+seed_lengths[i]])

    if keep_latents:
        latents = None

    #TODO: append the initial conditioning bit to the output too

    output_seq = []
    print(sequence_length)
    #import pdb;pdb.set_trace()
    with torch.no_grad():
        # for t in range(min(512, sequence_length-max(input_lengths)-1)):
        import time
        start_time = time.time()
        for t in range(sequence_length-max(input_lengths)+1):
        #for t in range(512):
            start_time_inner = time.time()
            print(t)
            inputs = [x.clone().to(model.device) for x in input_tmp]
            # import pdb;pdb.set_trace()

            if not ground_truth:
                if keep_latents:
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
                        if save_jit and t==0:
                            outputs = model.forward(inputs, temp=temperature)
                        elif save_jit:
                            outputs = trace(inputs)
                        if type(outputs) is tuple:
                            outputs=outputs[0]
                        #outputs = model.forward(inputs)#, temp=temperature)
                        #outputs = model.forward(inputs)[0]#, temp=temperature)
                        if save_jit and t==0:
                            with torch.no_grad():
                                #eps_std=1.0
                                #z_shape = (1, 8, 1)
                                #noise = torch.normal(mean=torch.zeros(z_shape), std=torch.ones(z_shape)*eps_std).cuda()
                                #noise = noise.unsqueeze(-1)
                                #noises = [noise]
                                #with torch.jit.optimized_execution(True):
                                #    trace = torch.jit.trace(lambda x,y: model(x, temp=temperature, noises=y)[0], (inputs,noises))
                                #trace = torch.jit.trace(lambda x: model(x,temp=temperature), (inputs,), check_trace=False)
                                trace = torch.jit.trace(model, (inputs,), check_trace=False)
                                #trace = torch.jit.trace(lambda x: model.forward_internal(x), (inputs,), check_trace=False)
                                #trace = torch.jit.trace_module(model, {"forward_internal":inputs}, check_trace=False)
                                #trace = torch.jit.trace(model, (inputs,noise))
                                #trace = torch.jit.trace(lambda x: model.get_latent(x,temp=temperature), (inputs,))
                                #latent = model.get_latent(inputs,temp=temperature)
                                #print(latent.shape)
                                #trace = torch.jit.trace(lambda x: model.run_norm_flow(x,temp=temperature), (latent,))
                                #l_shape = (1,1,800)
                                #l_shape = (1,800,1,1)
                                #latent = torch.normal(mean=torch.zeros(l_shape), std=torch.ones(l_shape)*eps_std).cuda()
                                #sldj = torch.zeros(noise.size(0), device=noise.device)
                                #trace = torch.jit.trace(model.output_mod_glows[0].flows, (noise,latent,sldj))
                                #trace = torch.jit.trace(lambda x,y: model.run_norm_flow(x,y), (noise,latent))
                                #trace = torch.jit.trace(model.output_mod_glows[0], (noise,latent))
                                #trace = torch.jit.trace(lambda x,y: model.output_mod_glows[0](x,y), (noise,latent))
                                #trace = torch.jit.trace(lambda x,y: model.output_mod_glows[0](x,y), (noise,latent), check_trace=False)
                                #trace = torch.jit.script(model.forward, example_inputs=(inputs,1.0))
                    else:
                        outputs = model.forward(inputs)
                        if save_jit and t==0:
                            with torch.no_grad():
                                trace = torch.jit.trace(model, (inputs,), check_trace=False)

                if save_jit and t==0:
                    torch.jit.save(trace, save_jit_path)

            #outputs[0][:,0,-4] = 0.0
            #outputs[0][:,0,-6] = 0.0
            if t == 0:
                for i, mod in enumerate(output_mods):
                    # output[:,0,:-3] = torch.clamp(output[:,0,:-3],-3,3)

                    if not ground_truth:
                        # import pdb;pdb.set_trace()
                        output = outputs[i]
                    else:
                        j = input_mods.index(mod)
                        output = inputs_[j][t+output_time_offsets[i]+output_lengths[i]:t+output_time_offsets[i]+output_lengths[i]+1]
                    output_seq.append(output[:1].detach().clone())

                    #output_seq.append(inputs_[i][t+input_time_offsets[i]+input_lengths[i]:t+input_time_offsets[i]+input_lengths[i]+1]+0.15*torch.randn(1,219).to(model.device))
            else:
                for i, mod in enumerate(output_mods):
                    #output_seq[i] = torch.cat([output_seq[i], inputs_[i][t+input_time_offsets[i]+input_lengths[i]:t+input_time_offsets[i]+input_lengths[i]+1]+0.15*torch.randn(1,219).to(model.device)])

                    if not ground_truth:
                        output = outputs[i]
                    else:
                        j = input_mods.index(mod)
                        output = inputs_[j][t+output_time_offsets[i]+output_lengths[i]:t+output_time_offsets[i]+output_lengths[i]+1]
                    output_seq[i] = torch.cat([output_seq[i], output[:1].detach().clone()])

                    # output[:,0,:-3] = torch.clamp(output[:,0,:-3],-3,3)
                    # print(outputs[i][:1])
            if t < sequence_length-1: #hmm dont really need this conditional i think now
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
                                input_tmp[i] = torch.cat([input_tmp[i][-(input_lengths[i]-1):],output[:1].detach().clone()],0)
                            else:
                                input_tmp[i] = output[:1].detach().clone()
                        # print(torch.mean((inputs_[i][t+input_time_offsets[i]+input_lengths[i]+1:t+input_time_offsets[i]+input_lengths[i]+1+1]-outputs[j][:1].detach().clone())**2))

                        if not ground_truth:
                            print(torch.mean((inputs_[i][t+output_time_offsets[j]:t+output_time_offsets[j]+1]-outputs[j][:1].detach().clone())**2))
                            print(outputs[j][:1].shape)
                            #print(outputs[j][:1][0,0,9])
                            #print(outputs[j][:1][0,0,16])
                    else:
                        if model.input_proc_types[i] == "single":
                            #input_tmp[i] = torch.cat([input_tmp[i][1:],inputs_[i][input_time_offsets[i]+input_lengths[i]+t:input_time_offsets[i]+input_lengths[i]+t+1]],0)
                            pass
                        else:
                            if input_lengths[i]>1:
                                input_tmp[i] = torch.cat([input_tmp[i][-(input_lengths[i]-1):],inputs_[i][input_time_offsets[i]+seed_lengths[i]+t:input_time_offsets[i]+seed_lengths[i]+t+1]],0)
                            else:
                                input_tmp[i] = inputs_[i][input_time_offsets[i]+seed_lengths[i]+t:input_time_offsets[i]+seed_lengths[i]+t+1]

            print("--- %s seconds ---" % (time.time() - start_time_inner))
    print("--- %s seconds ---" % (time.time() - start_time))
    return output_seq

