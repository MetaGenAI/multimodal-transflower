import numpy as np

import matplotlib.pyplot as plt

# a = np.load("inference/generated/transflower_expmap_cr4_bs5_og2_futureN_gauss4_neosraw/predicted_mods/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_1_ID1E66900_streams.combined_streams_scaled.generated.npy")
# a = np.load("inference/generated/transflower_expmap_cr4_bs5_og2_futureN_gauss4_neosraw/predicted_mods/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams.combined_streams_scaled.generated.npy")
a = np.load("data/dekaworld_alex_guille_neosdata/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_1_ID2C00_streams.combined_streams.npy")

# plt.matshow(a[:100,0,6:38])
plt.matshow(a[10000:10100,6:38])
