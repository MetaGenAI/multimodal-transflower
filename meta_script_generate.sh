#!/bin/bash

#exp=transglower_moglow_pos
#exp=transglower_residual_moglow_pos
#exp=transflower_residual_moglow_pos
#exp=transflower_moglow_pos
#exp=residualflower2_transflower_moglow_pos
#exp=moglow_moglow_pos

#exp=transglower_aistpp_expmap
#exp=transglower_residual_aistpp_expmap
#exp=transflower_residual_aistpp_expmap
#exp=transflower_aistpp_expmap
#exp=residualflower2_transflower_aistpp_expmap
#exp=moglow_aistpp_expmap


#for exp in transglower_moglow_pos transglower_residual_moglow_pos transflower_residual_moglow_pos transflower_moglow_pos residualflower2_transflower_moglow_pos moglow_moglow_pos
#for exp in moglow_trans_moglow_pos
#for exp in transglower_moglow_pos transglower_residual_moglow_pos transflower_residual_moglow_pos transflower_moglow_pos residualflower2_transflower_moglow_pos moglow_trans_moglow_pos moglow_moglow_pos
#for exp in transglower_aistpp_expmap transglower_residual_aistpp_expmap transflower_residual_aistpp_expmap transflower_aistpp_expmap residualflower2_transflower_aistpp_expmap moglow_aistpp_expmap
#for exp in transflower_residual_aistpp_expmap transflower_aistpp_expmap_future1 transflower_aistpp_expmap_future3 residualflower2_transflower_aistpp_expmap_future1 residualflower2_transflower_aistpp_expmap_future3 moglow_aistpp_expmap 
#for exp in transflower_residual_aistpp_expmap_future3
#for exp in transflower_expmap_use_pos_emb_output moglow_expmap transflower_expmap_studentT_gclp1
#for exp in transflower_expmap_use_pos_emb_output
#for exp in transformer_expmap_no_pos_emb_output
for exp in transformer_expmap
#for exp in transflower_expmap_use_pos_emb_output moglow_expmap transflower_expmap_studentT_gclp1 transflower_residual_expmap_1e4
#for exp in moglow_expmap
do
	sbatch slurm_script_generate.slurm $exp
done

#for exp in transflower_residual_aistpp_expmap_future1 transflower_aistpp_expmap_future1 residualflower2_transflower_aistpp_expmap_future1_future1
#do
#	sbatch slurm_script_generate.slurm $exp
#done
