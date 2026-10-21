rsync -a \
  --info=progress2 \
  --partial \
  --exclude '/data/' \
  iris@irislab.asuscomm.com:/home/iris/Workspace/junsu/DP/ \
  /home/iristele/work/chaewon/DP/
rsync -aR \
  --info=progress2 \
  --partial \
  iris@irislab.asuscomm.com:/home/iris/Workspace/junsu/DP/./data/outputs/2026.10.02/16.46.40_train_diffusion_unet_p2p_image_p2p_image/checkpoints/epoch=0590-train_loss=0.000.ckpt \
  iris@irislab.asuscomm.com:/home/iris/Workspace/junsu/DP/./data/outputs/2026.10.02/20.22.59_train_diffusion_unet_valve_p2p_image_valve_p2p_image/checkpoints/epoch=0510-train_loss=0.000.ckpt \
  /home/iristele/work/chaewon/DP/